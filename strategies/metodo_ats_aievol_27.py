"""
Porting Python di MetodoATS_AIEvol_27.mq5.

Riferimento esatto: MQL5/Experts/Advisors/MetodoATS_AIEvol_27.mq5 (versione
portata il 2 ottobre 2026 - rileggi quel file se nel frattempo e' cambiato).
Stesso timeframe fisso M5, stesso endpoint ML, stessa logica di
apertura/chiusura/protezioni Milfix (breakeven, SL ratio, semaforo, limite
giornaliero).

Questo modulo NON si connette a MetaTrader5: riceve dati già pronti e
restituisce decisioni (Signal/Action). Chi esegue davvero gli ordini è il
motore (prossimo pezzo da costruire). Vedi strategies/base.py per il
contratto e la convenzione sulle barre (.iloc[-1] = ultima barra chiusa).

Quirk preservati volontariamente (presenti anche nell'EA originale, non
sono stati "corretti" per non disallinearsi in silenzio dal modello ML
che e' stato addestrato su quei dati):
  - atr_22_growing_previous, atr_48_growing_previous e
    atr_96_growing_previous vengono SEMPRE inviati come 0: nel codice MQL5
    originale queste variabili sono dichiarate ma non vengono mai
    calcolate (solo le varianti "_mean" lo sono). Se un giorno si
    riaddestra il modello, e' il posto giusto per sistemarli.
  - Le bande di Bollinger qui sono quelle "standard" (nessuno shift);
    l'EA crea l'indicatore con bands_shift=2 (iBands(...,20,2,2,...)), il
    cui effetto esatto sul valore letto via CopyBuffer non e' stato
    verificato contro un terminale live. Vedi common/indicators.py.
  - "rsi" e' RSI(5), non RSI(14): e' cosi' anche nell'EA (iRSI(...,5,...)).
  - GetItalianTime() nell'EA usa un calcolo approssimato dell'ora legale;
    qui si usa invece zoneinfo("Europe/Rome") (vedi common/mt5_time.py),
    corretto al minuto. Per pochi giorni l'anno a cavallo del cambio ora
    i due possono quindi differire di un'ora sui controlli orario.
  - hour/day_of_week/day_of_month inviati al modello ML sono in ora DEL
    SERVER/BROKER (da iTime, senza conversione), NON in ora italiana: e'
    una distinzione voluta dell'EA originale (i controlli di sessione
    invece usano GetItalianTime()) ed e' preservata qui.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd

from common import indicators as ind
from common.lot_sizing import (
    DynamicRiskInputs,
    LotSizeServiceError,
    dynamic_risk_lot,
    fixed_lot,
    profile_table_lot,
)
from common.mt5_time import ItalianMoment, mql5_day_of_week, should_close_friday, should_close_thursday
from strategies.base import (
    Action,
    AccountSnapshot,
    DecisionContext,
    MarketSnapshot,
    OpenPosition,
    Signal,
    Strategy,
)


@dataclass
class MetodoATSConfig:
    """Equivalente degli `input`/parametri globali dell'EA."""
    magic_number: int = 1
    lot_size: float = 0.1
    enable_buy_trades: bool = True
    enable_sell_trades: bool = True
    only_show_signals: bool = False
    base_profit_target: float = 0.5     # % (BaseProfitTarget)
    leverage: float = 1.0               # Leverage (nell'EA e' fissa a 1, non e' un input)
    hours_close: float = 48.0
    monetary_stop_loss: float = 0.0

    enable_breakeven: bool = False
    breakeven_trigger: float = 0.80
    sl_ratio: float = 2.0

    close_friday: bool = True
    friday_close_hour: int = 21
    friday_close_minute: int = 50

    close_thursday: bool = False
    thursday_close_hour: int = 21
    thursday_close_minute: int = 50

    daily_loss_limit_pct: float = 2.5
    semaphore_losses: int = 10

    # --- Lottaggio: tre modalità (vedi common/lot_sizing.py) --------
    lot_sizing_mode: str = "fixed"        # "fixed" | "dynamic_risk" | "profile_table"
    max_risk_pct: float = 1.0             # usato solo da "dynamic_risk" (MaxRiskPct)
    risk_profile: str = "medio"           # usato solo da "profile_table": basso|medio|alto
    risk_profile_service_url: str = ""    # usato solo da "profile_table"
    risk_profile_client_key: Optional[str] = None


def _parse_confidence(raw: Any) -> Optional[float]:
    """Equivalente della conversione fatta da CTradeDashboard::SetConfidence():
    un valore tra 0 e 1 viene letto come frazione e convertito in %."""
    if raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if 0.0 < value <= 1.0:
        return value * 100.0
    return value


class MetodoATSAIEvol27(Strategy):
    name = "MetodoATS_AIEvol_27"

    def __init__(self, config: MetodoATSConfig):
        self.config = config
        self.magic_number = config.magic_number

    # ------------------------------------------------------------------
    def required_history_bars(self) -> int:
        # L'EA richiede almeno 250 barre (bars<250 -> abort) e usa fino a
        # 400 barre per l'EMA(200) sulla serie ATR(22). Margine di
        # sicurezza per la convergenza del lisciamento di Wilder sui
        # periodi più lunghi (96, 200) - vedi nota in common/indicators.py.
        return 450

    # ------------------------------------------------------------------
    def compute_features(self, closed_bars: pd.DataFrame) -> dict[str, Any]:
        if len(closed_bars) < self.required_history_bars():
            raise ValueError(
                f"Servono almeno {self.required_history_bars()} barre M5 chiuse, "
                f"ricevute {len(closed_bars)}"
            )

        close = closed_bars["close"]
        high = closed_bars["high"]
        low = closed_bars["low"]
        open_ = closed_bars["open"]
        volume = closed_bars["tick_volume"]

        f: dict[str, Any] = {}

        # --- ora/giorno della barra "corrente": tempo SERVER (non italiano,
        # a differenza dei controlli di sessione - vedi docstring modulo).
        # E' una barra M5: 5 minuti dopo l'ultima chiusa, come iTime(...,0)
        # letto nell'istante in cui la nuova barra si apre.
        last_bar_time = float(closed_bars["time"].iloc[-1])
        forming_dt = dt.datetime.fromtimestamp(last_bar_time + 5 * 60, tz=dt.timezone.utc)
        f["hour"] = forming_dt.hour
        f["day_of_week"] = mql5_day_of_week(forming_dt)
        f["day_of_month"] = forming_dt.day

        # --- EMA
        ema_8 = float(ind.ema(close, 8).iloc[-1])
        ema_21 = float(ind.ema(close, 21).iloc[-1])
        ema_50 = float(ind.ema(close, 50).iloc[-1])
        ema_200 = float(ind.ema(close, 200).iloc[-1])
        f.update(ema_8=ema_8, ema_21=ema_21, ema_50=ema_50, ema_200=ema_200)
        f["ema_21_up_ema_50"] = int(ema_21 > ema_50)
        f["ema_21_up_ema_200"] = int(ema_21 > ema_200)
        f["ema_8_up_ema_21"] = int(ema_8 > ema_21)
        f["ema_8_up_ema_50"] = int(ema_8 > ema_50)
        f["ema_8_up_ema_200"] = int(ema_8 > ema_200)
        f["ema_crossover"] = int(ema_50 > ema_200)

        # --- Bollinger (20, dev 2) - vedi nota sul bands_shift nel docstring
        upper, mid, lower = ind.bollinger_bands(close, 20, 2.0)
        f["upper_band"] = float(upper.iloc[-1])
        f["middle_band"] = float(mid.iloc[-1])
        f["lower_band"] = float(lower.iloc[-1])

        # --- ADX(14)
        f["adx"] = float(ind.adx(high, low, close, 14).iloc[-1])

        # --- RSI(5) = "rsi", RSI(10), RSI(20)
        rsi_series = ind.rsi(close, 5)
        rsi_now, rsi_prev = float(rsi_series.iloc[-1]), float(rsi_series.iloc[-2])
        f["rsi"] = rsi_now
        f["rsi_overbought"] = int(rsi_now > 70)
        f["rsi_overbought_80"] = int(rsi_now > 80)
        f["rsi_overbought_90"] = int(rsi_now > 90)
        f["rsi_oversold"] = int(rsi_now < 30)
        f["rsi_oversold_20"] = int(rsi_now < 20)
        f["rsi_oversold_10"] = int(rsi_now < 10)
        f["rsi_cross_50_up"] = int(rsi_prev <= 50 and rsi_now > 50)
        f["rsi_cross_50_down"] = int(rsi_prev >= 50 and rsi_now < 50)

        rsi_10_now = float(ind.rsi(close, 10).iloc[-1])
        f["rsi_10"] = rsi_10_now
        f["rsi_10_overbought"] = int(rsi_10_now > 70)
        f["rsi_10_overbought_80"] = int(rsi_10_now > 80)
        f["rsi_10_overbought_90"] = int(rsi_10_now > 90)
        f["rsi_10_oversold"] = int(rsi_10_now < 30)
        f["rsi_10_oversold_20"] = int(rsi_10_now < 20)
        f["rsi_10_oversold_10"] = int(rsi_10_now < 10)

        rsi_20_now = float(ind.rsi(close, 20).iloc[-1])
        f["rsi_20"] = rsi_20_now
        f["rsi_20_overbought"] = int(rsi_20_now > 70)
        f["rsi_20_overbought_80"] = int(rsi_20_now > 80)
        f["rsi_20_overbought_90"] = int(rsi_20_now > 90)
        f["rsi_20_oversold"] = int(rsi_20_now < 30)
        f["rsi_20_oversold_20"] = int(rsi_20_now < 20)
        f["rsi_20_oversold_10"] = int(rsi_20_now < 10)

        # --- ATR(14) "primario" + medie + crescita
        atr14 = ind.atr(high, low, close, 14)
        atr_now, atr_prev = float(atr14.iloc[-1]), float(atr14.iloc[-2])
        f["atr"] = atr_now
        f["atr_growing_previous"] = int(atr_now > atr_prev)
        for p in (5, 10, 20, 50):
            m = float(ind.sma(atr14, p).iloc[-1])
            f[f"atr_mean_{p}"] = m
            f[f"atr_growing_previous_{p}_mean"] = int(atr_now > m)

        # --- ATR(22) + medie + EMA(200) della serie ATR(22) + volatilita'
        atr22 = ind.atr(high, low, close, 22)
        atr_22_now = float(atr22.iloc[-1])
        f["atr_22"] = atr_22_now
        for p in (5, 10, 20, 50):
            m = float(ind.sma(atr22, p).iloc[-1])
            f[f"atr_22_mean_{p}"] = m
            f[f"atr_22_growing_previous_{p}_mean"] = int(atr_22_now > m)
        atr_ema_200 = float(ind.ema(atr22, 200).iloc[-1])
        f["atr_ema_200"] = atr_ema_200
        f["high_volatility"] = int(atr_22_now > atr_ema_200)
        f["range_volatility"] = int(atr_22_now < atr_ema_200)
        f["atr_22_growing_previous"] = 0   # quirk preservato - vedi docstring modulo

        # --- ATR(48) + medie
        atr48 = ind.atr(high, low, close, 48)
        atr_48_now = float(atr48.iloc[-1])
        f["atr_48"] = atr_48_now
        for p in (5, 10, 20, 50):
            m = float(ind.sma(atr48, p).iloc[-1])
            f[f"atr_48_mean_{p}"] = m
            f[f"atr_48_growing_previous_{p}_mean"] = int(atr_48_now > m)
        f["atr_48_growing_previous"] = 0   # quirk preservato - vedi docstring modulo

        # --- ATR(96) + medie
        atr96 = ind.atr(high, low, close, 96)
        atr_96_now = float(atr96.iloc[-1])
        f["atr_96"] = atr_96_now
        for p in (5, 10, 20, 50):
            m = float(ind.sma(atr96, p).iloc[-1])
            f[f"atr_96_mean_{p}"] = m
            f[f"atr_96_growing_previous_{p}_mean"] = int(atr_96_now > m)
        f["atr_96_growing_previous"] = 0   # quirk preservato - vedi docstring modulo

        # --- Pattern sulle candele: barra chiusa [-1] = "c1", [-2] = "c2"
        c2, o2 = float(close.iloc[-2]), float(open_.iloc[-2])
        h2, l2 = float(high.iloc[-2]), float(low.iloc[-2])
        c1, o1 = float(close.iloc[-1]), float(open_.iloc[-1])
        f["bullish_engulfing_full"] = int(c2 < o2 and c1 > o1 and c1 > h2)
        f["bearish_engulfing_full"] = int(c2 > o2 and c1 < o1 and c1 < l2)
        f["big_candle"] = int((c1 - o1) > atr_now)

        # --- Variazione di volume (tick volume)
        vol_now, vol_prev = float(volume.iloc[-1]), float(volume.iloc[-2])
        f["volume_change"] = ((vol_now - vol_prev) / vol_prev) if vol_prev > 0 else 0.0

        # Non inviato al modello: comodo per decide_signal (equivalente di
        # iClose(symbol, PERIOD_M5, 0) letto nell'istante di apertura barra).
        f["_reference_price"] = c1

        return f

    # ------------------------------------------------------------------
    def ml_endpoint_url(self) -> str:
        return "https://fiveminutesmachinelearning-i5x2.onrender.com/predictDomenico"

    # ------------------------------------------------------------------
    def build_ml_payload(self, features: dict[str, Any], account: AccountSnapshot,
                          symbol: str) -> dict[str, Any]:
        f = features
        return {
            "account": str(account.login),      # stringa: come "\"" + account_number + "\"" nell'EA
            "symbol": symbol,
            "hour": f["hour"],
            "day_of_week": f["day_of_week"],
            "day_of_month": f["day_of_month"],
            "ema_8": f["ema_8"],
            "ema_21": f["ema_21"],
            "ema_50": f["ema_50"],
            "ema_200": f["ema_200"],
            "upper_band": f["upper_band"],
            "middle_band": f["middle_band"],
            "Lower_band": f["lower_band"],      # maiuscola voluta, come nell'EA
            "atr": f["atr"],
            "atr_mean_5": f["atr_mean_5"],
            "atr_mean_10": f["atr_mean_10"],
            "atr_mean_20": f["atr_mean_20"],
            "atr_mean_50": f["atr_mean_50"],
            "atr_22": f["atr_22"],
            "atr_ema_200": f["atr_ema_200"],
            "atr_22_mean_5": f["atr_22_mean_5"],
            "atr_22_mean_10": f["atr_22_mean_10"],
            "atr_22_mean_20": f["atr_22_mean_20"],
            "atr_22_mean_50": f["atr_22_mean_50"],
            "atr_48": f["atr_48"],
            "atr_48_mean_5": f["atr_48_mean_5"],
            "atr_48_mean_10": f["atr_48_mean_10"],
            "atr_48_mean_20": f["atr_48_mean_20"],
            "atr_48_mean_50": f["atr_48_mean_50"],
            "atr_96": f["atr_96"],
            "atr_96_mean_5": f["atr_96_mean_5"],
            "atr_96_mean_10": f["atr_96_mean_10"],
            "atr_96_mean_20": f["atr_96_mean_20"],
            "atr_96_mean_50": f["atr_96_mean_50"],
            "adx": f["adx"],
            "rsi": f["rsi"],
            "rsi_10": f["rsi_10"],
            "rsi_20": f["rsi_20"],
            "ema_21_up_ema_50": f["ema_21_up_ema_50"],
            "ema_21_up_ema_200": f["ema_21_up_ema_200"],
            "ema_8_up_ema_21": f["ema_8_up_ema_21"],
            "ema_8_up_ema_50": f["ema_8_up_ema_50"],
            "ema_8_up_ema_200": f["ema_8_up_ema_200"],
            "ema_crossover": f["ema_crossover"],
            "atr_growing_previous": f["atr_growing_previous"],
            "atr_growing_previous_5_mean": f["atr_growing_previous_5_mean"],
            "atr_growing_previous_10_mean": f["atr_growing_previous_10_mean"],
            "atr_growing_previous_20_mean": f["atr_growing_previous_20_mean"],
            "atr_growing_previous_50_mean": f["atr_growing_previous_50_mean"],
            "atr_22_growing_previous": f["atr_22_growing_previous"],
            "atr_22_growing_previous_5_mean": f["atr_22_growing_previous_5_mean"],
            "atr_22_growing_previous_10_mean": f["atr_22_growing_previous_10_mean"],
            "atr_22_growing_previous_20_mean": f["atr_22_growing_previous_20_mean"],
            "atr_22_growing_previous_50_mean": f["atr_22_growing_previous_50_mean"],
            "atr_48_growing_previous": f["atr_48_growing_previous"],
            "atr_48_growing_previous_5_mean": f["atr_48_growing_previous_5_mean"],
            "atr_48_growing_previous_10_mean": f["atr_48_growing_previous_10_mean"],
            "atr_48_growing_previous_20_mean": f["atr_48_growing_previous_20_mean"],
            "atr_48_growing_previous_50_mean": f["atr_48_growing_previous_50_mean"],
            "atr_96_growing_previous": f["atr_96_growing_previous"],
            "atr_96_growing_previous_5_mean": f["atr_96_growing_previous_5_mean"],
            "atr_96_growing_previous_10_mean": f["atr_96_growing_previous_10_mean"],
            "atr_96_growing_previous_20_mean": f["atr_96_growing_previous_20_mean"],
            "atr_96_growing_previous_50_mean": f["atr_96_growing_previous_50_mean"],
            "rsi_overbought": f["rsi_overbought"],
            "rsi_overbought_80": f["rsi_overbought_80"],
            "rsi_overbought_90": f["rsi_overbought_90"],
            "rsi_oversold": f["rsi_oversold"],
            "rsi_oversold_20": f["rsi_oversold_20"],
            "rsi_oversold_10": f["rsi_oversold_10"],
            "rsi_10_overbought": f["rsi_10_overbought"],
            "rsi_10_overbought_80": f["rsi_10_overbought_80"],
            "rsi_10_overbought_90": f["rsi_10_overbought_90"],
            "rsi_10_oversold": f["rsi_10_oversold"],
            "rsi_10_oversold_20": f["rsi_10_oversold_20"],
            "rsi_10_oversold_10": f["rsi_10_oversold_10"],
            "rsi_20_overbought": f["rsi_20_overbought"],
            "rsi_20_overbought_80": f["rsi_20_overbought_80"],
            "rsi_20_overbought_90": f["rsi_20_overbought_90"],
            "rsi_20_oversold": f["rsi_20_oversold"],
            "rsi_20_oversold_20": f["rsi_20_oversold_20"],
            "rsi_20_oversold_10": f["rsi_20_oversold_10"],
            "rsi_cross_50_up": f["rsi_cross_50_up"],
            "rsi_cross_50_down": f["rsi_cross_50_down"],
            "high_volatility": f["high_volatility"],
            "range_volatility": f["range_volatility"],
            "bullish_engulfing_full": f["bullish_engulfing_full"],
            "bearish_engulfing_full": f["bearish_engulfing_full"],
            "big_candle": f["big_candle"],
        }

    # ------------------------------------------------------------------
    def parse_ml_response(self, raw_response: dict[str, Any]) -> dict[str, Any]:
        prediction = raw_response.get("prediction")
        trade_type = -1
        if prediction == 2:
            trade_type = 2
        elif prediction == 0:
            trade_type = 0
        return {
            "trade_type": trade_type,
            "confidence": raw_response.get("confidence"),
            "raw": raw_response,
        }

    # ------------------------------------------------------------------
    def decide_signal(self, prediction: dict[str, Any], ctx: DecisionContext) -> Optional[Signal]:
        trade_type = prediction.get("trade_type", -1)
        if trade_type not in (0, 2):
            return None
        if not ctx.allowed_time:
            return None

        direction = "BUY" if trade_type == 2 else "SELL"

        if direction == "BUY" and not self.config.enable_buy_trades:
            return None
        if direction == "SELL" and not self.config.enable_sell_trades:
            return None

        is_virtual = self.config.only_show_signals

        # Le protezioni Milfix (posizione già aperta, semaforo, limite
        # giornaliero) valgono solo per l'apertura REALE: in modalità
        # "solo segnali" l'EA aggiorna comunque la dashboard, quindi qui
        # continuiamo a calcolare il segnale anche se una di queste
        # scatterebbe (il motore, non virtuale, non lo eseguirà comunque).
        if not is_virtual:
            if ctx.has_open_position:
                return None
            if ctx.semaphore_triggered:
                return None
            if ctx.daily_loss_hit:
                return None

        open_price = ctx.reference_price
        target_pct = self.config.base_profit_target / 100.0
        leverage = self.config.leverage

        if direction == "BUY":
            take_profit = open_price * (1.0 + target_pct * leverage)
            sl_price = 0.0
            if self.config.sl_ratio > 0.0:
                tp_dist = take_profit - open_price
                sl_price = open_price - tp_dist * self.config.sl_ratio
        else:
            take_profit = open_price * (1.0 - target_pct * leverage)
            sl_price = 0.0
            if self.config.sl_ratio > 0.0:
                tp_dist = open_price - take_profit
                sl_price = open_price + tp_dist * self.config.sl_ratio

        if sl_price > 0.0:
            min_dist = ctx.market.stops_level_points * ctx.market.point
            if direction == "BUY":
                sl_price = min(sl_price, open_price - min_dist)
            else:
                sl_price = max(sl_price, open_price + min_dist)

        lot = self._resolve_lot_size(open_price, sl_price, ctx)

        return Signal(
            direction=direction,
            take_profit=take_profit,
            stop_loss=sl_price,
            lot_size=lot,
            confidence=_parse_confidence(prediction.get("confidence")),
            comment="AI Predictor 27",
            is_virtual=is_virtual,
        )

    # ------------------------------------------------------------------
    def _resolve_lot_size(self, open_price: float, sl_price: float, ctx: DecisionContext) -> float:
        mode = self.config.lot_sizing_mode

        if mode == "fixed":
            return fixed_lot(self.config.lot_size)

        if mode == "dynamic_risk":
            inputs = DynamicRiskInputs(
                balance=ctx.account.balance,
                max_risk_pct=self.config.max_risk_pct,
                point=ctx.market.point,
                tick_value=ctx.market.tick_value,
                tick_size=ctx.market.tick_size,
                volume_min=ctx.market.volume_min,
                volume_max=ctx.market.volume_max,
                volume_step=ctx.market.volume_step,
            )
            return dynamic_risk_lot(open_price, sl_price, inputs, fallback_lot=self.config.lot_size)

        if mode == "profile_table":
            try:
                return profile_table_lot(
                    base_url=self.config.risk_profile_service_url,
                    profile=self.config.risk_profile,
                    symbol=ctx.symbol,
                    client_key=self.config.risk_profile_client_key,
                    fallback_lot=self.config.lot_size,
                )
            except LotSizeServiceError:
                return self.config.lot_size

        raise ValueError(f"lot_sizing_mode non valido: {mode!r}")

    # ------------------------------------------------------------------
    def manage_open_positions(self, symbol: str, positions: list[OpenPosition],
                               market: MarketSnapshot, now_epoch: float,
                               italian_moment: ItalianMoment) -> list[Action]:
        actions: list[Action] = []

        def already_closing(ticket: int) -> bool:
            return any(a.ticket == ticket and a.kind == "close" for a in actions)

        mine_any_symbol = [p for p in positions if p.magic == self.magic_number]
        mine_this_symbol = [p for p in mine_any_symbol if p.symbol == symbol]

        # --- SL monetario (CheckMonetaryStopLoss): magic + simbolo
        if self.config.monetary_stop_loss > 0.0:
            for p in mine_this_symbol:
                floating_pl = p.profit + p.swap
                if floating_pl <= -abs(self.config.monetary_stop_loss):
                    actions.append(Action(kind="close", ticket=p.ticket,
                                           reason=f"SL monetario raggiunto ({floating_pl:.2f})"))

        # --- Breakeven (CheckBreakeven): magic + simbolo
        if self.config.enable_breakeven:
            for p in mine_this_symbol:
                if already_closing(p.ticket) or p.tp <= 0.0:
                    continue
                tp_dist = abs(p.tp - p.price_open)
                if tp_dist <= 0.0:
                    continue

                if p.direction == "BUY":
                    trigger = p.price_open + tp_dist * self.config.breakeven_trigger
                    needs_be = market.bid >= trigger and p.sl < p.price_open
                else:
                    trigger = p.price_open - tp_dist * self.config.breakeven_trigger
                    needs_be = market.ask <= trigger and (p.sl <= 0.0 or p.sl > p.price_open)

                if needs_be:
                    new_sl = round(p.price_open, market.digits)
                    actions.append(Action(kind="modify_sl_tp", ticket=p.ticket,
                                           sl=new_sl, tp=p.tp, reason="breakeven"))

        # --- Posizioni troppo vecchie (CloseOldTrades): SOLO magic,
        # qualsiasi simbolo - come nell'EA originale.
        for p in mine_any_symbol:
            if already_closing(p.ticket):
                continue
            hours_open = (now_epoch - p.open_time_epoch) / 3600.0
            if hours_open >= self.config.hours_close:
                actions.append(Action(kind="close", ticket=p.ticket,
                                       reason=f"posizione aperta da {hours_open:.1f}h"))

        # --- Chiusura di venerdì / giovedì (ora italiana): SOLO magic,
        # qualsiasi simbolo - come nell'EA originale.
        close_friday = should_close_friday(italian_moment, self.config.friday_close_hour,
                                            self.config.friday_close_minute,
                                            enabled=self.config.close_friday)
        close_thursday = should_close_thursday(italian_moment, self.config.thursday_close_hour,
                                                self.config.thursday_close_minute,
                                                enabled=self.config.close_thursday)
        if close_friday or close_thursday:
            reason = "chiusura venerdi" if close_friday else "chiusura giovedi"
            for p in mine_any_symbol:
                if already_closing(p.ticket):
                    continue
                actions.append(Action(kind="close", ticket=p.ticket, reason=reason))

        return actions
