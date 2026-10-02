"""
Contratto comune che ogni strategia portata da MQL5 deve rispettare, così
il motore (ancora da costruire: il loop che si collega a MetaTrader5 ed
esegue davvero gli ordini) può farle girare tutte nello stesso modo, senza
codice su misura per ciascuna.

Principio di disegno: le strategie NON parlano mai direttamente con
MetaTrader5. Ricevono dati già pronti (barre chiuse, posizioni aperte,
info di mercato) e restituiscono DECISIONI (un Signal da aprire, una lista
di Action su posizioni esistenti). Chi esegue davvero gli ordini è il
motore. Questo rende le strategie testabili senza una connessione MT5 live
(vedi tests/).

Convenzione sulle barre: ogni DataFrame di barre passato a una strategia
contiene SOLO barre chiuse, ordinate per tempo crescente; l'ultima riga
(.iloc[-1]) è la barra più recente già chiusa. Questo corrisponde a:
  - "shift 0" sui buffer degli indicatori MQL5, letti nell'istante in cui
    si apre una nuova barra (quando la barra "0" è appena iniziata e non
    ha ancora informazioni proprie, il suo valore coincide con quello
    dell'ultima barra chiusa);
  - "shift 1" sui prezzi espliciti (iClose(...,1) ecc.) usati dall'EA per
    i pattern sulle candele.
Entrambi i casi mappano su .iloc[-1] di questo DataFrame. ".iloc[-2]" è
quindi l'equivalente di "shift 1" sui buffer indicatori e "shift 2" sui
prezzi espliciti.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd

from common.mt5_time import ItalianMoment


@dataclass
class Signal:
    """Decisione di apertura: l'equivalente di una chiamata a OpenTrade()."""
    direction: str              # "BUY" oppure "SELL"
    take_profit: float
    stop_loss: float            # 0.0 = nessuno stop loss
    lot_size: float
    confidence: Optional[float] = None
    comment: str = ""
    is_virtual: bool = False    # True quando OnlyShowSignals-equivalente e' attivo:
                                 # da mostrare in dashboard/console, NON da eseguire


@dataclass
class Action:
    """Operazione da eseguire su una posizione già aperta (chiusura o
    modifica SL/TP). Il motore la traduce in una vera chiamata order_send/
    position modify su MetaTrader5."""
    kind: str                   # "close" oppure "modify_sl_tp"
    ticket: int
    sl: Optional[float] = None
    tp: Optional[float] = None
    reason: str = ""


@dataclass
class OpenPosition:
    """Snapshot di una posizione aperta, letto da mt5.positions_get()."""
    ticket: int
    symbol: str
    magic: int
    direction: str                # "BUY" oppure "SELL"
    volume: float
    price_open: float
    sl: float
    tp: float
    profit: float
    swap: float
    open_time_epoch: float        # epoch seconds, stessa base di now_utc_epoch sotto


@dataclass
class MarketSnapshot:
    """Dati di mercato correnti per il simbolo, letti da mt5.symbol_info*()."""
    bid: float
    ask: float
    point: float
    digits: int
    tick_value: float
    tick_size: float
    volume_min: float
    volume_max: float
    volume_step: float
    stops_level_points: int
    filling_mode: int             # bitmask SYMBOL_FILLING_* del broker


@dataclass
class AccountSnapshot:
    balance: float
    login: int


@dataclass
class DecisionContext:
    """Tutto quello che serve a decide_signal() per replicare i gate di
    OpenTrade() (PositionExists, direzione abilitata, semaforo, limite
    giornaliero, orario consentito) senza che la strategia debba leggere
    nulla direttamente da MetaTrader5."""
    symbol: str
    market: MarketSnapshot
    account: AccountSnapshot
    reference_price: float        # equivalente di iClose(symbol, PERIOD_M5, 0)
    allowed_time: bool            # equivalente di IsAllowedTime()
    semaphore_triggered: bool     # equivalente di IsSemaphoreTriggered()
    daily_loss_hit: bool          # equivalente di IsDailyLossHit()
    has_open_position: bool       # equivalente di PositionExists()


class Strategy(ABC):
    """Interfaccia comune. Un modulo in strategies/ implementa questa classe."""

    name: str
    magic_number: int

    @abstractmethod
    def required_history_bars(self) -> int:
        """Quante barre M5 CHIUSE servono come minimo per calcolare gli
        indicatori (equivalente del controllo `bars < N` dell'EA)."""

    @abstractmethod
    def compute_features(self, closed_bars: pd.DataFrame) -> dict[str, Any]:
        """Calcola tutte le feature tecniche (equivalente di ComputeIndicators())."""

    @abstractmethod
    def ml_endpoint_url(self) -> str:
        """URL del servizio ML da interrogare (equivalente di GetServerUrl())."""

    @abstractmethod
    def build_ml_payload(self, features: dict[str, Any], account: AccountSnapshot,
                          symbol: str) -> dict[str, Any]:
        """Costruisce il payload JSON da inviare al servizio ML (equivalente
        della costruzione di json_data in SendDataToServer())."""

    @abstractmethod
    def parse_ml_response(self, raw_response: dict[str, Any]) -> dict[str, Any]:
        """Estrae dalla risposta del servizio ML il tipo di trade (2=BUY,
        0=SELL, -1=nessuno) e la confidence. Equivalente della ricerca di
        sottostringhe `"prediction":2` / `"confidence":` nell'EA, ma fatto
        con un vero parser JSON invece di scansionare il testo a mano."""

    @abstractmethod
    def decide_signal(self, prediction: dict[str, Any], ctx: DecisionContext) -> Optional[Signal]:
        """Dato l'esito di parse_ml_response() e il contesto corrente,
        decide se (e come) aprire un'operazione. Equivalente della parte
        finale di SendDataToServer() + OpenTrade(), SENZA eseguire nulla:
        ritorna solo la decisione."""

    @abstractmethod
    def manage_open_positions(self, symbol: str, positions: list[OpenPosition],
                               market: MarketSnapshot, now_epoch: float,
                               italian_moment: ItalianMoment) -> list[Action]:
        """Regole su posizioni già aperte: breakeven, SL monetario, chiusure
        per età/orario (equivalente di CheckBreakeven/CheckMonetaryStopLoss/
        CloseOldTrades/CloseFridayTrades/CloseTradesIfThursday22).

        `positions` deve contenere TUTTE le posizioni con questo magic
        number, su qualsiasi simbolo (non solo `symbol`): nell'EA originale
        le chiusure per età/venerdì/giovedì filtrano solo per magic number,
        senza controllare il simbolo, mentre breakeven e SL monetario
        filtrano sia per magic che per simbolo. L'implementazione deve
        applicare i due filtri in modo diverso per rispettare questa
        asimmetria (vedi MetodoATSAIEvol27 per un esempio).

        `now_epoch` deve essere nella stessa base temporale di
        OpenPosition.open_time_epoch (quella del server/broker)."""
