#!/usr/bin/env python3
"""
Esempio di avvio del motore con un vero terminale MetaTrader5.

NON eseguito in questo ambiente di sviluppo: serve Windows + un terminale
MetaTrader5 già installato e loggato sul broker + il pacchetto
`pip install MetaTrader5`. Copia/adatta questo file per ogni cliente con
i suoi parametri (magic number, simbolo, lottaggio, ecc.).

Primo avvio consigliato: dry_run=True (il default di RunnerConfig) per
vedere nei log cosa farebbe il motore SENZA eseguire nulla, idealmente
puntato a un conto demo. Passa a dry_run=False solo dopo aver verificato
che le decisioni loggate hanno senso.
"""

import logging

from engine.gateway import MT5Gateway
from engine.runner import RunnerConfig, StrategyRunner
from strategies.metodo_ats_aievol_27 import MetodoATSAIEvol27, MetodoATSConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")


def main() -> None:
    config = MetodoATSConfig(
        magic_number=1,
        lot_size=0.1,
        enable_buy_trades=True,
        enable_sell_trades=True,
        only_show_signals=False,
        base_profit_target=0.5,
        hours_close=48.0,
        sl_ratio=2.0,
        close_friday=True,
        friday_close_hour=21,
        friday_close_minute=50,
        lot_sizing_mode="fixed",        # oppure "dynamic_risk" / "profile_table"
        # risk_profile_service_url="https://<il-tuo-risk-profile-service>",
        # risk_profile="medio",
    )
    strategy = MetodoATSAIEvol27(config)

    gateway = MT5Gateway()
    gateway.connect()   # passa path=/login=/password=/server= se serve un terminale specifico

    runner_config = RunnerConfig(
        symbol="EURUSD",
        poll_seconds=2.0,
        dry_run=True,    # IMPORTANTE: vedi avviso nel docstring del modulo
    )
    runner = StrategyRunner(strategy, gateway, runner_config)

    try:
        runner.run_forever()
    except KeyboardInterrupt:
        logging.info("Interrotto dall'utente, chiudo la connessione...")
    finally:
        gateway.disconnect()


if __name__ == "__main__":
    main()
