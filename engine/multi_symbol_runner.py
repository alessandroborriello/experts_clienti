"""
Motore dedicato alle strategie multi-simbolo (strategies.base.
MultiSymbolStrategy, es. ATS Spread) - non riusa StrategyRunner
(engine/runner.py), pensato per UN simbolo alla volta guidato da un
servizio ML esterno.

Qui il ciclo e' piu' semplice e "stupido" apposta: raccoglie barre/mercato
per TUTTI i simboli dichiarati da strategy.required_bars(), piu' tutte le
posizioni aperte con il magic number della strategia (su qualsiasi
simbolo), e passa tutto a evaluate() in un colpo solo. Tutta la logica
(quando aprire, quando chiudere, i gate "once per bar"/distanza minima/
martingala) vive nella strategia, non nel motore - lo stesso principio di
StrategyRunner, solo senza il vincolo di un simbolo singolo.

############################################################################
# ATTENZIONE: come engine/gateway.py e engine/runner.py, anche questo file #
# non e' mai stato eseguito contro un terminale MT5 reale. Prima           #
# esecuzione vera: dry_run=True, idealmente su un conto demo.              #
############################################################################
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional, Protocol

import pandas as pd

from strategies.base import (
    Action,
    MarketSnapshot,
    MultiSymbolMarketData,
    MultiSymbolOpenOrder,
    MultiSymbolStrategy,
    OpenPosition,
)

log = logging.getLogger("engine.multi_symbol_runner")


class MultiSymbolGateway(Protocol):
    """Lo stesso MT5Gateway (vero o finto) usato da engine.runner - qui ne
    serve solo un sottoinsieme, niente account/deal (ATS Spread non li usa,
    vedi strategies/ats_spread.py)."""

    def ensure_symbol(self, symbol: str) -> None: ...
    def server_time_epoch(self, symbol: str) -> float: ...
    def get_closed_bars(self, symbol: str, count: int, timeframe: str = "M5") -> pd.DataFrame: ...
    def get_market_snapshot(self, symbol: str) -> MarketSnapshot: ...
    def get_open_positions(self, magic: Optional[int] = None) -> list[OpenPosition]: ...
    def send_market_order(self, symbol: str, direction: str, volume: float, sl: float, tp: float,
                           magic: int, comment: str): ...
    def close_position(self, position: OpenPosition): ...


@dataclass
class MultiSymbolRunnerConfig:
    poll_seconds: float = 2.0
    dry_run: bool = True   # default sicuro, come RunnerConfig


class MultiSymbolStrategyRunner:
    """Fa girare UNA strategia multi-simbolo. Per farne girare più d'una
    (istanze diverse, magic diversi) serve più di un'istanza di questa
    classe - vedi console.engine_control per come la console le avvia in
    thread separati."""

    def __init__(self, strategy: MultiSymbolStrategy, gateway: MultiSymbolGateway,
                 config: MultiSymbolRunnerConfig):
        self.strategy = strategy
        self.gateway = gateway
        self.config = config
        self._required = strategy.required_bars()

    # ------------------------------------------------------------------
    def run_forever(self) -> None:
        for symbol in self._required:
            self.gateway.ensure_symbol(symbol)
        log.info("Motore multi-simbolo avviato: strategia=%s simboli=%s dry_run=%s",
                  self.strategy.name, list(self._required), self.config.dry_run)
        while True:
            try:
                self.run_once()
            except Exception:
                log.exception("Errore nel ciclo del motore multi-simbolo, continuo al prossimo poll")
            time.sleep(self.config.poll_seconds)

    def run_once(self) -> None:
        symbols = list(self._required.keys())
        now_epoch = self.gateway.server_time_epoch(symbols[0])

        bars = {}
        for symbol, (timeframe, count) in self._required.items():
            bars[symbol] = self.gateway.get_closed_bars(symbol, count, timeframe)

        market = {symbol: self.gateway.get_market_snapshot(symbol) for symbol in symbols}
        positions = self.gateway.get_open_positions(magic=self.strategy.magic_number)
        by_ticket = {p.ticket: p for p in positions}

        data = MultiSymbolMarketData(now_epoch=now_epoch, bars=bars, market=market, positions=positions)
        actions = self.strategy.evaluate(data)

        for action in actions:
            self._execute(action, by_ticket)

    # ------------------------------------------------------------------
    def _execute(self, action, by_ticket: dict[int, OpenPosition]) -> None:
        if isinstance(action, MultiSymbolOpenOrder):
            if self.config.dry_run:
                log.info("[DRY RUN] Apertura %s %s lotti=%.2f (%s)",
                          action.symbol, action.direction, action.volume, action.comment)
                return
            result = self.gateway.send_market_order(
                symbol=action.symbol, direction=action.direction, volume=action.volume,
                sl=0.0, tp=0.0, magic=self.strategy.magic_number, comment=action.comment,
            )
            log.info("Apertura %s %s lotti=%.2f: %s", action.symbol, action.direction,
                      action.volume, result)
            return

        if isinstance(action, Action):
            if action.kind != "close":
                log.warning("Azione sconosciuta dalla strategia multi-simbolo: %s", action.kind)
                return
            if self.config.dry_run:
                log.info("[DRY RUN] Chiusura ticket %s (%s)", action.ticket, action.reason)
                return
            position = by_ticket.get(action.ticket)
            if position is None:
                log.warning("Chiusura su ticket %s ma la posizione non e' (più) nella lista, salto",
                            action.ticket)
                return
            result = self.gateway.close_position(position)
            log.info("Chiusura ticket %s (%s): %s", action.ticket, action.reason, result)
            return

        log.warning("Tipo di azione non riconosciuto dalla strategia multi-simbolo: %r", action)
