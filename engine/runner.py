"""
Il motore: ciclo che collega UNA strategia a UN simbolo tramite un
gateway MT5 (vero: engine.gateway.MT5Gateway; finto: per i test). Legge
dati reali, chiede alla strategia cosa fare, ed esegue quelle decisioni
con ordini veri - a meno di dry_run=True (il default), che logga le
decisioni senza eseguire nulla.

Struttura del ciclo, equivalente a OnTick() dell'EA:
  - gestione delle posizioni già aperte (breakeven, SL monetario,
    chiusure per età/orario): ad OGNI poll, non solo a cambio barra -
    stessa scelta già fatta per MetodoATS_AIEvol_27.mq5 quando abbiamo
    scoperto che i controlli legati solo alla nuova barra possono saltare
    se i tick si diradano vicino all'orario di chiusura;
  - calcolo feature + chiamata al modello ML + eventuale apertura: SOLO
    quando si chiude una nuova barra M5, come ComputeIndicators()/
    SendDataToServer() nell'EA.

############################################################################
# ATTENZIONE: questo file non e' mai stato eseguito contro un terminale  #
# MT5 reale (l'ambiente di sviluppo non ne ha uno). E' stato validato    #
# con un gateway finto (tests/test_runner.py) che verifica la LOGICA del #
# ciclo - quando interrogare il modello, quando eseguire, il filtro      #
# magic/simbolo nelle chiusure - non la correttezza delle vere chiamate  #
# MetaTrader5 (quelle sono in engine/gateway.py, isolate apposta).       #
# Prima esecuzione reale: dry_run=True, e idealmente su un conto demo.   #
############################################################################
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Protocol

import pandas as pd
import requests

from common.mt5_time import ItalianMoment, is_allowed_time
from common.risk_rules import Deal
from strategies.base import AccountSnapshot, MarketSnapshot, OpenPosition, Strategy

log = logging.getLogger("engine.runner")


class OrderResultLike(Protocol):
    success: bool
    ticket: Optional[int]
    retcode: Optional[int]
    comment: str


class Gateway(Protocol):
    """Tutto quello che il motore richiede a un gateway MT5 (vero o finto).
    engine.gateway.MT5Gateway e il FakeGateway dei test implementano
    entrambi questa interfaccia (duck typing, nessuna eredità richiesta)."""

    def ensure_symbol(self, symbol: str) -> None: ...
    def server_time_epoch(self, symbol: str) -> float: ...
    def get_closed_bars(self, symbol: str, count: int) -> pd.DataFrame: ...
    def get_account_snapshot(self) -> AccountSnapshot: ...
    def get_market_snapshot(self, symbol: str) -> MarketSnapshot: ...
    def get_open_positions(self, magic: Optional[int] = None) -> list[OpenPosition]: ...
    def get_recent_deals(self, lookback_seconds: float, now_epoch: Optional[float] = None) -> list[Deal]: ...
    def send_market_order(self, symbol: str, direction: str, volume: float, sl: float, tp: float,
                           magic: int, comment: str) -> OrderResultLike: ...
    def close_position(self, position: OpenPosition) -> OrderResultLike: ...
    def modify_position(self, position: OpenPosition, sl: float, tp: float) -> OrderResultLike: ...


@dataclass
class RunnerConfig:
    symbol: str
    poll_seconds: float = 2.0
    ml_timeout_seconds: float = 5.0
    semaphore_lookback_days: int = 30
    dry_run: bool = True   # default sicuro: logga le decisioni, non esegue nulla


def epoch_to_utc(epoch: float) -> datetime:
    return datetime.fromtimestamp(epoch, tz=timezone.utc)


def day_start_epoch(moment: ItalianMoment) -> float:
    """Inizio della giornata di trading corrente, usato dal limite di
    perdita giornaliera. L'EA parte dalla mezzanotte del SERVER; qui si
    approssima con la mezzanotte italiana per semplicità - la differenza
    sposta solo il momento in cui riparte il conteggio giornaliero (di
    poche ore), non la soglia stessa. Se serve parità esatta con l'EA,
    va passata la mezzanotte server (il gateway puo' fornirla)."""
    local_midnight = moment.dt.replace(hour=0, minute=0, second=0, microsecond=0)
    return local_midnight.timestamp()


class StrategyRunner:
    """Fa girare UNA strategia su UN simbolo. Per più combinazioni
    simbolo/strategia, crea più StrategyRunner (uno ciascuno) - vedi il
    modulo che li istanzia per farli girare in thread/processi separati."""

    def __init__(self, strategy: Strategy, gateway: Gateway, config: RunnerConfig):
        self.strategy = strategy
        self.gateway = gateway
        self.config = config
        self._last_bar_time: Optional[float] = None

    # ------------------------------------------------------------------
    def run_forever(self) -> None:
        self.gateway.ensure_symbol(self.config.symbol)
        log.info("Motore avviato: strategia=%s simbolo=%s dry_run=%s",
                  self.strategy.name, self.config.symbol, self.config.dry_run)
        while True:
            try:
                self.run_once()
            except Exception:
                log.exception("Errore nel ciclo del motore, continuo al prossimo poll")
            time.sleep(self.config.poll_seconds)

    def run_once(self) -> None:
        symbol = self.config.symbol
        now_epoch = self.gateway.server_time_epoch(symbol)
        italian_moment = ItalianMoment.from_utc(epoch_to_utc(now_epoch))

        positions = self.gateway.get_open_positions(magic=self.strategy.magic_number)
        market = self.gateway.get_market_snapshot(symbol)

        # Gestione posizioni aperte: ad ogni poll (vedi docstring modulo).
        self._manage_positions(symbol, positions, market, now_epoch, italian_moment)

        # Feature + ML + apertura: solo a cambio barra M5.
        bars = self.gateway.get_closed_bars(symbol, self.strategy.required_history_bars())
        last_bar_time = float(bars["time"].iloc[-1])

        if self._last_bar_time is not None and last_bar_time == self._last_bar_time:
            return

        self._last_bar_time = last_bar_time
        log.info("Nuova barra M5 su %s (chiusa alle %s UTC), valuto il segnale",
                  symbol, epoch_to_utc(last_bar_time).strftime("%H:%M:%S"))
        self._evaluate_new_bar(symbol, bars, positions, market, italian_moment)

    # ------------------------------------------------------------------
    def _manage_positions(self, symbol: str, positions: list[OpenPosition], market: MarketSnapshot,
                           now_epoch: float, italian_moment: ItalianMoment) -> None:
        actions = self.strategy.manage_open_positions(symbol, positions, market, now_epoch, italian_moment)
        if not actions:
            return

        by_ticket = {p.ticket: p for p in positions}
        for action in actions:
            position = by_ticket.get(action.ticket)
            if position is None:
                log.warning("Action su ticket %s ma la posizione non e' (più) nella lista, salto",
                            action.ticket)
                continue

            if self.config.dry_run:
                log.info("[DRY RUN] %s su ticket %s (%s)", action.kind, action.ticket, action.reason)
                continue

            if action.kind == "close":
                result = self.gateway.close_position(position)
                log.info("Chiusura ticket %s (%s): %s", action.ticket, action.reason, result)
            elif action.kind == "modify_sl_tp":
                result = self.gateway.modify_position(position, action.sl, action.tp)
                log.info("Modifica SL/TP ticket %s (%s): %s", action.ticket, action.reason, result)
            else:
                log.warning("Action.kind sconosciuto: %s", action.kind)

    def _evaluate_new_bar(self, symbol: str, bars: pd.DataFrame, positions: list[OpenPosition],
                           market: MarketSnapshot, italian_moment: ItalianMoment) -> None:
        from strategies.base import DecisionContext   # import locale: evita un ciclo di import

        account = self.gateway.get_account_snapshot()
        features = self.strategy.compute_features(bars)
        payload = self.strategy.build_ml_payload(features, account, symbol)

        url = self.strategy.ml_endpoint_url()
        log.info("Invio richiesta al servizio ML per %s -> %s", symbol, url)
        sent_at = time.monotonic()
        try:
            resp = requests.post(url, json=payload, timeout=self.config.ml_timeout_seconds)
            resp.raise_for_status()
            raw = resp.json()
        except Exception:
            log.exception("Chiamata al servizio ML fallita dopo %.2fs, salto questa barra",
                           time.monotonic() - sent_at)
            return
        log.info("Risposta ML ricevuta per %s in %.2fs", symbol, time.monotonic() - sent_at)

        prediction = self.strategy.parse_ml_response(raw)

        has_open_position = any(p.symbol == symbol for p in positions)
        deals = self.gateway.get_recent_deals(self.config.semaphore_lookback_days * 86400)
        deals_desc = sorted(deals, key=lambda d: d.time_epoch, reverse=True)

        daily_loss_hit, semaphore_triggered = self.strategy.evaluate_risk_gates(
            account, symbol, deals, deals_desc, day_start_epoch(italian_moment),
        )

        ctx = DecisionContext(
            symbol=symbol, market=market, account=account,
            reference_price=features["_reference_price"],
            allowed_time=is_allowed_time(italian_moment),
            semaphore_triggered=semaphore_triggered,
            daily_loss_hit=daily_loss_hit,
            has_open_position=has_open_position,
        )

        signal = self.strategy.decide_signal(prediction, ctx)
        if signal is None:
            log.info("Nessun segnale operativo per %s su questa barra "
                      "(allowed_time=%s semaphore=%s daily_loss=%s has_open_position=%s)",
                      symbol, ctx.allowed_time, ctx.semaphore_triggered,
                      ctx.daily_loss_hit, ctx.has_open_position)
            return

        if signal.is_virtual:
            log.info("[SOLO SEGNALI] %s %s TP=%.5f SL=%.5f lotti=%.2f (confidence=%s)",
                      symbol, signal.direction, signal.take_profit, signal.stop_loss,
                      signal.lot_size, signal.confidence)
            return

        if self.config.dry_run:
            log.info("[DRY RUN] Apertura %s %s TP=%.5f SL=%.5f lotti=%.2f",
                      symbol, signal.direction, signal.take_profit, signal.stop_loss, signal.lot_size)
            return

        result = self.gateway.send_market_order(
            symbol=symbol, direction=signal.direction, volume=signal.lot_size,
            sl=signal.stop_loss, tp=signal.take_profit,
            magic=self.strategy.magic_number, comment=signal.comment,
        )
        log.info("Apertura %s %s: %s", symbol, signal.direction, result)
