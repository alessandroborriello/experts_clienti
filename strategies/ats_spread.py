"""
Porting Python di "ATS Spread.mq5".

Riferimento: MQL5/Experts/Advisors/ATS Spread.mq5 - a differenza di
MetodoATS_AIEvol_27, NON e' un EA scritto a mano: e' generato da fxDreema
(tool visuale a blocchi), con 64 "blocchi" di cui 22 DISABILITATI (lista
esplicita in OnInit, mai più riattivata). Questo porting replica solo il
grafico REALMENTE attivo, non il codice morto - vedi sotto.

Comportamento REALE (confermato leggendo i blocchi attivi, non i soli nomi
dei parametri, che da soli avrebbero fatto supporre qualcosa di diverso):

  - Entrata: Bollinger Bands (20, 2.0) H4 sul "simbolo segnale"
    (simbolo_segnale). Se la chiusura dell'ultima barra chiusa e' sopra la
    banda media -> vende simbolo_1, compra simbolo_2. Se sotto -> il
    contrario. Solo se non c'e' gia' una posizione aperta su quel simbolo
    (guardia debole: non impedisce le medie a martingala una volta aperta).
    Rivalutato OGNI poll entro la finestra oraria (start/end, ora del
    SERVER, non italiana) - non serve "once per bar" perche' la guardia
    "nessuna posizione aperta" e' gia' sufficiente a non riaprire.

  - Martingala a 4 gambe indipendenti (vendita/acquisto x simbolo_1/
    simbolo_2): quando il PREZZO DEL SIMBOLO STESSO incrocia la propria
    Bollinger Band (superiore per le vendite, inferiore per gli acquisti)
    sul timeframe "di grafico" (parametro chart_timeframe - nell'originale
    non e' un input esplicito, dipende dal grafico su cui gira l'EA), e il
    paniere aperto in quella direzione e' in perdita o in pari, apre
    un'altra gamba con lotto a martingala (vedi common.lot_sizing.
    martingale_lot), rispettando una distanza minima dalle gambe gia'
    aperte di `distanza_minima_mediate * 2` PIP (non il valore grezzo: e'
    il fattore 2 che fxDreema applica internamente, vedi v::nearby
    nell'originale). Valutato una sola volta per barra (chart_timeframe)
    DEL SIMBOLO TRADATO - l'originale usa "CurrentSymbol()/
    CurrentTimeframe()" del grafico a cui l'EA e' agganciato, che non e'
    deducibile dal solo codice sorgente: qui si usa il simbolo tradato
    stesso come approssimazione ragionevole (al più disallinea di quale
    esatta barra scatta il gate "una volta per barra", non la logica).

  - Chiusura: NESSUNA chiusura automatica per default (usa_tp_pips_medio=
    False di fabbrica). Gli ordini non hanno mai SL/TP broker-side. Quando
    usa_tp_pips_medio=True, chiude TUTTO il paniere (qualsiasi simbolo,
    stesso magic) non appena: profitto in denaro > 0 E somma dei pip di
    profitto su tutte le gambe >= tp_medio_pips. I blocchi che chiuderebbero
    per incrocio Bollinger su simbolo_segnale sono tra quelli DISABILITATI
    nell'originale: non replicati qui, di proposito (non girano mai).

  - distanza_atr: input dell'EA ma MAI USATO (i blocchi che lo userebbero
    sono anch'essi tra i disabilitati) - omesso qui del tutto, non e'
    configurabile perche' non farebbe comunque nulla.

Approssimazioni/scelte esplicite (si parte da "fedele al comportamento
live", ma alcuni dettagli aritmetici interni di fxDreema non sono stati
riletti riga per riga - vedi anche common.lot_sizing.martingale_lot):
  - "profitto in denaro" di un paniere/posizione = profit + swap (come
    riportato dal broker per le posizioni aperte; MT5 non espone una
    commissione per-posizione aperta, solo sui deal chiusi).
  - 1 pip = 10 * point (vale per i simboli standard a 5/3 decimali usati
    da questa strategia - EURUSD/GBPUSD/EURGBP - non per casi esotici a 6
    decimali che l'originale gestisce con una tabella dedicata).
  - Il lotto calcolato viene arrotondato al volume_step del broker e
    vincolato tra volume_min/volume_max (AlignLots originale non e' stato
    riletto riga per riga).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import pandas as pd

from common import indicators as ind
from common.lot_sizing import martingale_lot
from common.mt5_time import is_within_server_window
from strategies.base import (
    Action,
    MarketSnapshot,
    MultiSymbolMarketData,
    MultiSymbolOpenOrder,
    MultiSymbolStrategy,
)

BOLLINGER_PERIOD = 20
BOLLINGER_DEVIATION = 2.0


def pips_to_price(point: float) -> float:
    """1 pip in prezzo. Vedi nota di modulo: 10*point per i simboli standard
    (5 o 3 decimali) usati da questa strategia."""
    return point * 10.0


@dataclass
class ATSSpreadConfig:
    """Equivalente degli `input` di ATS Spread.mq5 (vedi docstring di modulo
    per cosa NON e' stato portato e perche': distanza_atr, mediaOnLoss/
    OnProfit/addOnLoss/addOnProfit interni al blocco SellNow/BuyNow non
    configurabili dall'EA)."""
    magic_number: int = 2940                 # MagicStart
    simbolo_segnale: str = "EURGBP"          # era "EURGBP tickstory" nel .set di backtest
    simbolo_1: str = "EURUSD"                # era "EURUSD tickstory"
    simbolo_2: str = "GBPUSD"                # era "GBPUSD.r tickstory"
    lotto_1: float = 0.15
    lotto_2: float = 0.1
    lotto_mediata_1: float = 0.01
    lotto_mediata_2: float = 0.01
    distanza_minima_mediate: float = 50.0
    martingala: float = 1.05
    usa_tp_pips_medio: bool = False
    tp_medio_pips: float = 20.0
    start: str = "10:00"
    end: str = "21:00"
    chart_timeframe: str = "H1"              # vedi nota sulla stima di "CurrentTimeframe()"


class ATSSpreadStrategy(MultiSymbolStrategy):
    name = "ATS Spread"

    def __init__(self, config: ATSSpreadConfig):
        self.config = config
        self.magic_number = config.magic_number
        self._last_ladder_bar: dict[tuple[str, str], float] = {}

    # ------------------------------------------------------------------
    def required_bars(self) -> dict[str, tuple[str, int]]:
        cfg = self.config
        return {
            cfg.simbolo_segnale: ("H4", 50),
            cfg.simbolo_1: (cfg.chart_timeframe, 50),
            cfg.simbolo_2: (cfg.chart_timeframe, 50),
        }

    def evaluate(self, data: MultiSymbolMarketData) -> list[Union[MultiSymbolOpenOrder, Action]]:
        cfg = self.config
        actions: list[Union[MultiSymbolOpenOrder, Action]] = []

        actions.extend(self._basket_close_actions(data))
        actions.extend(self._entry_orders(data))
        actions.extend(self._ladder_orders(data, cfg.simbolo_1, "SELL", "upper", cfg.lotto_mediata_1))
        actions.extend(self._ladder_orders(data, cfg.simbolo_2, "SELL", "upper", cfg.lotto_mediata_2))
        actions.extend(self._ladder_orders(data, cfg.simbolo_2, "BUY", "lower", cfg.lotto_mediata_2))
        actions.extend(self._ladder_orders(data, cfg.simbolo_1, "BUY", "lower", cfg.lotto_mediata_1))
        return actions

    # ------------------------------------------------------------------ entrata
    def _entry_orders(self, data: MultiSymbolMarketData) -> list[MultiSymbolOpenOrder]:
        cfg = self.config
        if not is_within_server_window(data.now_epoch, cfg.start, cfg.end):
            return []

        bars = data.bars[cfg.simbolo_segnale]
        _, mid, _ = ind.bollinger_bands(bars["close"], BOLLINGER_PERIOD, BOLLINGER_DEVIATION)
        if pd.isna(mid.iloc[-1]):
            return []   # non ancora abbastanza storico per una banda valida

        close_last = float(bars["close"].iloc[-1])
        mid_last = float(mid.iloc[-1])

        has_pos_1 = any(p.symbol == cfg.simbolo_1 for p in data.positions)
        has_pos_2 = any(p.symbol == cfg.simbolo_2 for p in data.positions)

        orders: list[MultiSymbolOpenOrder] = []
        if close_last > mid_last:
            if not has_pos_1:
                orders.append(MultiSymbolOpenOrder(cfg.simbolo_1, "SELL", cfg.lotto_1, "ATS Spread"))
            if not has_pos_2:
                orders.append(MultiSymbolOpenOrder(cfg.simbolo_2, "BUY", cfg.lotto_2, "ATS Spread"))
        elif close_last < mid_last:
            if not has_pos_1:
                orders.append(MultiSymbolOpenOrder(cfg.simbolo_1, "BUY", cfg.lotto_1, "ATS Spread"))
            if not has_pos_2:
                orders.append(MultiSymbolOpenOrder(cfg.simbolo_2, "SELL", cfg.lotto_2, "ATS Spread"))
        return orders

    # ------------------------------------------------------------------ medie a martingala
    def _ladder_orders(self, data: MultiSymbolMarketData, symbol: str, direction: str,
                        band_side: str, initial_lots: float) -> list[MultiSymbolOpenOrder]:
        cfg = self.config
        bars = data.bars[symbol]
        if len(bars) < 2:
            return []

        upper, _, lower = ind.bollinger_bands(bars["close"], BOLLINGER_PERIOD, BOLLINGER_DEVIATION)
        band = upper if band_side == "upper" else lower
        if pd.isna(band.iloc[-2]) or pd.isna(band.iloc[-1]):
            return []

        last_bar_time = float(bars["time"].iloc[-1])
        state_key = (symbol, direction)
        if self._last_ladder_bar.get(state_key) == last_bar_time:
            return []   # OncePerBar: questa barra e' gia' stata valutata

        # Consumiamo il gate OncePerBar per questa barra qui sotto, anche se
        # non si finisce per aprire nulla - come MDL_OncePerBar nell'originale.
        self._last_ladder_bar[state_key] = last_bar_time

        matching = [p for p in data.positions if p.symbol == symbol and p.direction == direction]
        if not matching:
            return []   # IfOpenedOrders: serve almeno una posizione gia' aperta in questa direzione

        aggregate_profit = sum(p.profit + p.swap for p in matching)
        if aggregate_profit > 0.0:
            return []   # CheckUProfit <= 0: si media solo se il paniere e' in perdita o in pari

        close = bars["close"]
        if band_side == "upper":
            crossed = close.iloc[-2] <= band.iloc[-2] and close.iloc[-1] > band.iloc[-1]
        else:
            crossed = close.iloc[-2] >= band.iloc[-2] and close.iloc[-1] < band.iloc[-1]
        if not crossed:
            return []

        market = data.market[symbol]
        min_distance = pips_to_price(market.point) * (cfg.distanza_minima_mediate * 2.0)
        ref_price = market.bid if direction == "SELL" else market.ask
        if any(abs(ref_price - p.price_open) < min_distance for p in matching):
            return []   # NoNearbyRunning: troppo vicino a una gamba gia' aperta

        recent_first = sorted(matching, key=lambda p: p.open_time_epoch, reverse=True)
        candidates = [(p.profit, p.volume) for p in recent_first]
        volume = martingale_lot(candidates, initial_lots=initial_lots, multiply_on_loss=cfg.martingala)
        volume = self._align_volume(volume, market)

        return [MultiSymbolOpenOrder(symbol, direction, volume, "ATS Spread mediata")]

    @staticmethod
    def _align_volume(volume: float, market: MarketSnapshot) -> float:
        if market.volume_step > 0.0:
            volume = round(volume / market.volume_step) * market.volume_step
        volume = max(volume, market.volume_min)
        volume = min(volume, market.volume_max)
        return round(volume, 8)

    # ------------------------------------------------------------------ chiusura a paniere
    def _basket_close_actions(self, data: MultiSymbolMarketData) -> list[Action]:
        cfg = self.config
        if not cfg.usa_tp_pips_medio or not data.positions:
            return []

        money_profit = sum(p.profit + p.swap for p in data.positions)
        if money_profit <= 0.0:
            return []

        pips_sum = 0.0
        for p in data.positions:
            market = data.market.get(p.symbol)
            if market is None or market.point <= 0.0:
                continue
            price_now = market.bid if p.direction == "BUY" else market.ask
            diff = (price_now - p.price_open) if p.direction == "BUY" else (p.price_open - price_now)
            pips_sum += diff / pips_to_price(market.point)

        if pips_sum < cfg.tp_medio_pips:
            return []

        return [Action(kind="close", ticket=p.ticket, reason="basket: TP medio raggiunto")
                for p in data.positions]
