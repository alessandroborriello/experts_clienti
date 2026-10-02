"""
Gateway verso MetaTrader5: l'UNICO modulo del progetto che importa
davvero il package `MetaTrader5` ed esegue operazioni reali (letture di
mercato, invio ordini). Tutto il resto (strategie, motore/runner) lavora
sulle dataclass di strategies/base.py e common/risk_rules.py, senza mai
importare MetaTrader5 direttamente - così resta testabile senza un
terminale live (vedi tests/test_runner.py, che usa un gateway finto).

Il pacchetto pip "MetaTrader5" funziona solo su Windows, con un terminale
MetaTrader5 già installato e collegato a un broker. Qui l'import è
"lazy" (dentro connect(), non in testa al file), così importare questo
modulo non fallisce nemmeno dove il pacchetto non è installato (es. in
ambienti di test) - ma USARLO richiede ovviamente Windows + terminale +
pacchetto installato (`pip install MetaTrader5`).

############################################################################
# ATTENZIONE: nessuna riga di questo file è stata eseguita contro un      #
# terminale MT5 reale in fase di sviluppo (l'ambiente di sviluppo non ha  #
# accesso a un terminale Windows/MetaTrader). I nomi di funzioni/campi    #
# usati sono quelli documentati stabilmente da MetaQuotes da anni, ma la  #
# prima esecuzione vera va fatta con cautela: usa engine.runner con       #
# dry_run=True (il default) per vedere cosa farebbe SENZA eseguire nulla, #
# e verifica su un conto demo prima di passare a dry_run=False.           #
############################################################################
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from common.risk_rules import Deal
from strategies.base import AccountSnapshot, MarketSnapshot, OpenPosition


class MT5ConnectionError(RuntimeError):
    pass


@dataclass
class OrderResult:
    success: bool
    ticket: Optional[int]
    retcode: Optional[int]
    comment: str


class MT5Gateway:
    """Wrapper sottile ed esplicito sul package MetaTrader5. Un'istanza =
    una connessione a UN terminale (quello passato a connect())."""

    def __init__(self):
        self._mt5 = None

    # --- connessione ----------------------------------------------------
    def connect(self, path: Optional[str] = None, login: Optional[int] = None,
                password: Optional[str] = None, server: Optional[str] = None,
                timeout_ms: int = 10_000) -> None:
        import MetaTrader5 as mt5   # lazy: fallisce solo se si usa davvero il gateway
        self._mt5 = mt5

        kwargs: dict = {}
        if path:
            kwargs["path"] = path
        if login:
            kwargs["login"] = login
        if password:
            kwargs["password"] = password
        if server:
            kwargs["server"] = server
        if timeout_ms:
            kwargs["timeout"] = timeout_ms

        if not mt5.initialize(**kwargs):
            code, desc = mt5.last_error()
            raise MT5ConnectionError(f"mt5.initialize() fallita: [{code}] {desc}")

    def disconnect(self) -> None:
        if self._mt5 is not None:
            self._mt5.shutdown()

    @property
    def mt5(self):
        if self._mt5 is None:
            raise MT5ConnectionError("Gateway non connesso: chiama connect() prima.")
        return self._mt5

    # --- letture ----------------------------------------------------------
    def ensure_symbol(self, symbol: str) -> None:
        if not self.mt5.symbol_select(symbol, True):
            raise MT5ConnectionError(f"Impossibile selezionare il simbolo {symbol} in Market Watch")

    def get_symbols(self, group: Optional[str] = None) -> list[str]:
        """Elenco dei simboli noti al broker collegato (per popolare una
        tendina invece di farli scrivere a mano). symbols_get() restituisce
        TUTTI i simboli del broker, non solo quelli già nel Market Watch;
        `group` filtra per pattern come fa MetaTrader stesso (es. "*USD*")."""
        raw = self.mt5.symbols_get(group) if group else self.mt5.symbols_get()
        if raw is None:
            return []
        return sorted(s.name for s in raw)

    def server_time_epoch(self, symbol: str) -> float:
        """Ora del server/broker (epoch), letta dall'ultimo tick - stessa
        base temporale di OpenPosition.open_time_epoch."""
        tick = self.mt5.symbol_info_tick(symbol)
        if tick is None:
            raise MT5ConnectionError(f"symbol_info_tick({symbol}) ha restituito None")
        return float(tick.time)

    def get_closed_bars(self, symbol: str, count: int) -> pd.DataFrame:
        """Barre M5 CHIUSE (esclude quella in formazione): start_pos=1 in
        copy_rates_from_pos salta la barra corrente (posizione 0)."""
        rates = self.mt5.copy_rates_from_pos(symbol, self.mt5.TIMEFRAME_M5, 1, count)
        if rates is None or len(rates) == 0:
            raise MT5ConnectionError(f"copy_rates_from_pos({symbol}) ha restituito nessun dato")
        df = pd.DataFrame(rates)
        return df.sort_values("time").reset_index(drop=True)

    def get_account_snapshot(self) -> AccountSnapshot:
        info = self.mt5.account_info()
        if info is None:
            raise MT5ConnectionError("account_info() ha restituito None")
        return AccountSnapshot(balance=float(info.balance), login=int(info.login))

    def get_market_snapshot(self, symbol: str) -> MarketSnapshot:
        info = self.mt5.symbol_info(symbol)
        tick = self.mt5.symbol_info_tick(symbol)
        if info is None or tick is None:
            raise MT5ConnectionError(f"symbol_info/symbol_info_tick({symbol}) ha restituito None")
        return MarketSnapshot(
            bid=float(tick.bid), ask=float(tick.ask),
            point=float(info.point), digits=int(info.digits),
            tick_value=float(info.trade_tick_value), tick_size=float(info.trade_tick_size),
            volume_min=float(info.volume_min), volume_max=float(info.volume_max),
            volume_step=float(info.volume_step),
            stops_level_points=int(info.trade_stops_level),
            filling_mode=int(info.filling_mode),
        )

    def get_open_positions(self, magic: Optional[int] = None) -> list[OpenPosition]:
        raw = self.mt5.positions_get()
        if raw is None:
            return []
        out = []
        for p in raw:
            if magic is not None and p.magic != magic:
                continue
            direction = "BUY" if p.type == self.mt5.POSITION_TYPE_BUY else "SELL"
            out.append(OpenPosition(
                ticket=p.ticket, symbol=p.symbol, magic=p.magic, direction=direction,
                volume=p.volume, price_open=p.price_open, sl=p.sl, tp=p.tp,
                profit=p.profit, swap=p.swap, open_time_epoch=float(p.time),
            ))
        return out

    def get_recent_deals(self, lookback_seconds: float, now_epoch: Optional[float] = None) -> list[Deal]:
        now_epoch = now_epoch if now_epoch is not None else time.time()
        date_from = datetime.fromtimestamp(now_epoch - lookback_seconds, tz=timezone.utc)
        date_to = datetime.fromtimestamp(now_epoch + 60, tz=timezone.utc)
        raw = self.mt5.history_deals_get(date_from, date_to)
        if raw is None:
            return []
        out = []
        for d in raw:
            out.append(Deal(
                magic=d.magic, symbol=d.symbol,
                is_exit=(d.entry == self.mt5.DEAL_ENTRY_OUT),
                profit=d.profit, swap=d.swap, commission=d.commission,
                time_epoch=float(d.time),
            ))
        return out

    # --- scrittura (ordini) ------------------------------------------------
    def resolve_filling_mode(self, market: MarketSnapshot) -> int:
        """Porting di GetFillingMode() dell'EA: stesso bitmask SYMBOL_FILLING_*."""
        mt5 = self.mt5
        fm = market.filling_mode
        if fm & mt5.SYMBOL_FILLING_IOC:
            return mt5.ORDER_FILLING_IOC
        if fm & mt5.SYMBOL_FILLING_FOK:
            return mt5.ORDER_FILLING_FOK
        return mt5.ORDER_FILLING_RETURN

    def send_market_order(self, symbol: str, direction: str, volume: float,
                           sl: float, tp: float, magic: int, comment: str,
                           deviation_points: int = 20) -> OrderResult:
        mt5 = self.mt5
        market = self.get_market_snapshot(symbol)
        order_type = mt5.ORDER_TYPE_BUY if direction == "BUY" else mt5.ORDER_TYPE_SELL
        price = market.ask if direction == "BUY" else market.bid

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": order_type,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": deviation_points,
            "magic": magic,
            "comment": comment,
            "type_filling": self.resolve_filling_mode(market),
        }
        result = mt5.order_send(request)

        # Porting del retry su filling mode non supportato, come nell'EA.
        if result is not None and result.retcode == mt5.TRADE_RETCODE_INVALID_FILL:
            request["type_filling"] = (mt5.ORDER_FILLING_RETURN
                                        if request["type_filling"] != mt5.ORDER_FILLING_RETURN
                                        else mt5.ORDER_FILLING_IOC)
            result = mt5.order_send(request)

        if result is None:
            return OrderResult(False, None, None, "order_send ha restituito None")
        return OrderResult(result.retcode == mt5.TRADE_RETCODE_DONE,
                            getattr(result, "order", None), result.retcode, result.comment)

    def close_position(self, position: OpenPosition, deviation_points: int = 20) -> OrderResult:
        mt5 = self.mt5
        market = self.get_market_snapshot(position.symbol)
        is_buy = position.direction == "BUY"
        close_type = mt5.ORDER_TYPE_SELL if is_buy else mt5.ORDER_TYPE_BUY
        price = market.bid if is_buy else market.ask

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": position.symbol,
            "volume": position.volume,
            "type": close_type,
            "position": position.ticket,
            "price": price,
            "deviation": deviation_points,
            "magic": position.magic,
            "comment": "close by engine",
            "type_filling": self.resolve_filling_mode(market),
        }
        result = mt5.order_send(request)
        if result is None:
            return OrderResult(False, None, None, "order_send ha restituito None")
        return OrderResult(result.retcode == mt5.TRADE_RETCODE_DONE,
                            getattr(result, "order", None), result.retcode, result.comment)

    def modify_position(self, position: OpenPosition, sl: float, tp: float) -> OrderResult:
        request = {
            "action": self.mt5.TRADE_ACTION_SLTP,
            "position": position.ticket,
            "symbol": position.symbol,
            "sl": sl,
            "tp": tp,
        }
        result = self.mt5.order_send(request)
        if result is None:
            return OrderResult(False, None, None, "order_send ha restituito None")
        return OrderResult(result.retcode == self.mt5.TRADE_RETCODE_DONE,
                            getattr(result, "order", None), result.retcode, result.comment)
