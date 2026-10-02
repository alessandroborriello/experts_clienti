"""
Test del motore (engine/runner.py) con un gateway FINTO: verifica la
LOGICA del ciclo (quando interrogare il modello ML, quando eseguire,
dry_run, gestione posizioni ad ogni poll vs. valutazione nuovo segnale
solo a cambio barra) - non la correttezza delle vere chiamate
MetaTrader5 (quelle sono isolate in engine/gateway.py, non testabili qui
senza un terminale live).
"""

import unittest
from unittest.mock import patch, MagicMock

from common.risk_rules import Deal
from engine.runner import RunnerConfig, StrategyRunner
from strategies.base import AccountSnapshot, MarketSnapshot, OpenPosition
from strategies.metodo_ats_aievol_27 import MetodoATSAIEvol27, MetodoATSConfig
from tests.test_metodo_ats_aievol_27 import make_market, make_synthetic_bars


class FakeGateway:
    """Gateway finto: dati preimpostati, nessuna chiamata di rete.
    Registra le chiamate di scrittura (ordini/chiusure/modifiche) così i
    test possono verificare cosa il motore avrebbe eseguito davvero."""

    def __init__(self, bars, market=None, account=None, positions=None, deals=None,
                 server_time=1_704_196_800.0):   # 2024-01-02 12:00 UTC, martedi' - orario sempre consentito
        self.bars = bars
        self.market = market or make_market()
        self.account = account or AccountSnapshot(balance=10_000.0, login=42)
        self.positions = positions or []
        self.deals = deals or []
        self.server_time = server_time

        self.ensure_symbol_calls: list[str] = []
        self.sent_orders: list[dict] = []
        self.closed: list[int] = []
        self.modified: list[tuple] = []

    def ensure_symbol(self, symbol):
        self.ensure_symbol_calls.append(symbol)

    def server_time_epoch(self, symbol):
        return self.server_time

    def get_closed_bars(self, symbol, count):
        return self.bars

    def get_account_snapshot(self):
        return self.account

    def get_market_snapshot(self, symbol):
        return self.market

    def get_open_positions(self, magic=None):
        if magic is None:
            return list(self.positions)
        return [p for p in self.positions if p.magic == magic]

    def get_recent_deals(self, lookback_seconds, now_epoch=None):
        return list(self.deals)

    def send_market_order(self, symbol, direction, volume, sl, tp, magic, comment):
        self.sent_orders.append(dict(symbol=symbol, direction=direction, volume=volume,
                                      sl=sl, tp=tp, magic=magic, comment=comment))
        return MagicMock(success=True, ticket=999, retcode=10009, comment="done")

    def close_position(self, position):
        self.closed.append(position.ticket)
        return MagicMock(success=True, ticket=position.ticket, retcode=10009, comment="closed")

    def modify_position(self, position, sl, tp):
        self.modified.append((position.ticket, sl, tp))
        return MagicMock(success=True, ticket=position.ticket, retcode=10009, comment="modified")


def _ml_response(prediction, confidence=0.8):
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = {"prediction": prediction, "confidence": confidence}
    return resp


class TestRunnerNewBarDetection(unittest.TestCase):
    def setUp(self):
        self.bars = make_synthetic_bars()
        self.strategy = MetodoATSAIEvol27(MetodoATSConfig(only_show_signals=False))
        self.gateway = FakeGateway(self.bars)
        self.runner = StrategyRunner(self.strategy, self.gateway, RunnerConfig(symbol="EURUSD"))

    @patch("engine.runner.requests.post")
    def test_first_call_always_evaluates(self, mock_post):
        mock_post.return_value = _ml_response(prediction=-1)
        self.runner.run_once()
        self.assertEqual(mock_post.call_count, 1)

    @patch("engine.runner.requests.post")
    def test_same_bar_does_not_re_evaluate(self, mock_post):
        mock_post.return_value = _ml_response(prediction=-1)
        self.runner.run_once()
        self.runner.run_once()
        self.runner.run_once()
        self.assertEqual(mock_post.call_count, 1)   # solo la prima volta

    @patch("engine.runner.requests.post")
    def test_new_bar_triggers_new_evaluation(self, mock_post):
        mock_post.return_value = _ml_response(prediction=-1)
        self.runner.run_once()

        more_bars = make_synthetic_bars(n=501)   # una barra in più -> last_bar_time diverso
        self.gateway.bars = more_bars
        self.runner.run_once()

        self.assertEqual(mock_post.call_count, 2)

    def test_manage_positions_runs_every_poll_even_without_new_bar(self):
        position = OpenPosition(ticket=1, symbol="EURUSD", magic=1, direction="BUY",
                                 volume=0.1, price_open=1.1000, sl=0.0, tp=0.0,
                                 profit=-1000.0, swap=0.0, open_time_epoch=0.0)
        cfg = MetodoATSConfig(monetary_stop_loss=10.0)
        strategy = MetodoATSAIEvol27(cfg)
        gateway = FakeGateway(self.bars, positions=[position])
        runner = StrategyRunner(strategy, gateway, RunnerConfig(symbol="EURUSD", dry_run=False))

        with patch("engine.runner.requests.post", return_value=_ml_response(prediction=-1)):
            runner.run_once()
            runner.run_once()   # stessa barra: niente nuova valutazione ML...

        # ...ma la chiusura per SL monetario deve scattare ad ogni poll finché la
        # posizione risulta ancora aperta nel gateway finto (qui scatta già al primo).
        self.assertIn(1, gateway.closed)


class TestRunnerDryRun(unittest.TestCase):
    def setUp(self):
        self.bars = make_synthetic_bars()

    @patch("engine.runner.requests.post")
    def test_dry_run_does_not_execute_signal(self, mock_post):
        mock_post.return_value = _ml_response(prediction=2)   # BUY
        cfg = MetodoATSConfig(only_show_signals=False)
        strategy = MetodoATSAIEvol27(cfg)
        gateway = FakeGateway(self.bars)
        runner = StrategyRunner(strategy, gateway, RunnerConfig(symbol="EURUSD", dry_run=True))

        runner.run_once()

        self.assertEqual(gateway.sent_orders, [])

    @patch("engine.runner.requests.post")
    def test_live_mode_executes_signal(self, mock_post):
        mock_post.return_value = _ml_response(prediction=2)   # BUY
        cfg = MetodoATSConfig(only_show_signals=False)
        strategy = MetodoATSAIEvol27(cfg)
        gateway = FakeGateway(self.bars)
        runner = StrategyRunner(strategy, gateway, RunnerConfig(symbol="EURUSD", dry_run=False))

        runner.run_once()

        self.assertEqual(len(gateway.sent_orders), 1)
        self.assertEqual(gateway.sent_orders[0]["direction"], "BUY")

    @patch("engine.runner.requests.post")
    def test_only_show_signals_never_executes_even_in_live_mode(self, mock_post):
        mock_post.return_value = _ml_response(prediction=2)
        cfg = MetodoATSConfig(only_show_signals=True)
        strategy = MetodoATSAIEvol27(cfg)
        gateway = FakeGateway(self.bars)
        runner = StrategyRunner(strategy, gateway, RunnerConfig(symbol="EURUSD", dry_run=False))

        runner.run_once()

        self.assertEqual(gateway.sent_orders, [])

    @patch("engine.runner.requests.post")
    def test_dry_run_does_not_close_positions(self, mock_post):
        mock_post.return_value = _ml_response(prediction=-1)
        position = OpenPosition(ticket=7, symbol="EURUSD", magic=1, direction="BUY",
                                 volume=0.1, price_open=1.1000, sl=0.0, tp=0.0,
                                 profit=-1000.0, swap=0.0, open_time_epoch=0.0)
        cfg = MetodoATSConfig(monetary_stop_loss=10.0)
        strategy = MetodoATSAIEvol27(cfg)
        gateway = FakeGateway(self.bars, positions=[position])
        runner = StrategyRunner(strategy, gateway, RunnerConfig(symbol="EURUSD", dry_run=True))

        runner.run_once()

        self.assertEqual(gateway.closed, [])


class TestRunnerMlFailure(unittest.TestCase):
    @patch("engine.runner.requests.post", side_effect=ConnectionError("rete giu'"))
    def test_ml_failure_does_not_crash(self, mock_post):
        bars = make_synthetic_bars()
        strategy = MetodoATSAIEvol27(MetodoATSConfig())
        gateway = FakeGateway(bars)
        runner = StrategyRunner(strategy, gateway, RunnerConfig(symbol="EURUSD"))

        runner.run_once()   # non deve lanciare eccezioni

        self.assertEqual(gateway.sent_orders, [])


class TestRunnerSemaphoreAndDailyLoss(unittest.TestCase):
    @patch("engine.runner.requests.post")
    def test_semaphore_blocks_execution(self, mock_post):
        mock_post.return_value = _ml_response(prediction=2)
        cfg = MetodoATSConfig(semaphore_losses=3)
        strategy = MetodoATSAIEvol27(cfg)
        bars = make_synthetic_bars()

        losing_deals = [
            Deal(magic=1, symbol="EURUSD", is_exit=True, profit=-10, swap=0, commission=0,
                 time_epoch=1_704_196_800.0 - i)
            for i in (1, 2, 3)
        ]
        gateway = FakeGateway(bars, deals=losing_deals)
        runner = StrategyRunner(strategy, gateway, RunnerConfig(symbol="EURUSD", dry_run=False))

        runner.run_once()

        self.assertEqual(gateway.sent_orders, [])


if __name__ == "__main__":
    unittest.main()
