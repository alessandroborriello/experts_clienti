"""
Regole di rischio basate sullo storico delle operazioni (deals), condivise
da tutte le strategie.

Porting di IsDailyLossHit() e IsSemaphoreTriggered() di
MetodoATS_AIEvol_27.mq5. Il motore (non ancora costruito) le alimenta con
lo storico letto da mt5.history_deals_get(): qui restano pure funzioni,
testabili senza connessione MT5 live.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Deal:
    """Un deal storico, equivalente ai campi letti con HistoryDealGet*()."""
    magic: int
    symbol: str
    is_exit: bool          # equivalente di DEAL_ENTRY == DEAL_ENTRY_OUT
    profit: float
    swap: float
    commission: float
    time_epoch: float
    volume: float = 0.0    # lottaggio del deal - usato dal martingala di ATS Spread


def is_daily_loss_hit(deals: list[Deal], balance: float, daily_loss_limit_pct: float,
                       magic: int, symbol: str, day_start_epoch: float) -> bool:
    """Porting di IsDailyLossHit(): True se la perdita di oggi (su questo
    magic+simbolo) ha superato daily_loss_limit_pct% del saldo."""
    if daily_loss_limit_pct <= 0.0:
        return False
    limit = balance * daily_loss_limit_pct / 100.0

    today_pnl = 0.0
    for d in deals:
        if d.magic != magic or not d.is_exit or d.symbol != symbol:
            continue
        if d.time_epoch < day_start_epoch:
            continue
        today_pnl += d.profit + d.swap + d.commission

    return today_pnl <= -limit


def is_semaphore_triggered(deals_desc: list[Deal], semaphore_losses: int,
                            magic: int, symbol: str) -> bool:
    """Porting di IsSemaphoreTriggered(): True se le ultime
    `semaphore_losses` operazioni chiuse (su questo magic+simbolo) sono
    state tutte in perdita.

    `deals_desc` deve essere ordinato dal più recente al più vecchio (come
    il ciclo `for i = n_deals-1; i >= 0` dell'EA)."""
    if semaphore_losses <= 0:
        return False

    checked = 0
    for d in deals_desc:
        if checked >= semaphore_losses:
            break
        if d.magic != magic or not d.is_exit or d.symbol != symbol:
            continue
        pnl = d.profit + d.swap + d.commission
        if pnl >= 0.0:
            return False
        checked += 1

    return checked >= semaphore_losses
