"""
Test del motore multi-simbolo (engine/multi_symbol_runner.py) con un
gateway e una strategia FINTI: verifica che il motore raccolga i dati giusti
(barre per simbolo/timeframe dichiarati da required_bars(), mercato per ogni
simbolo, posizioni filtrate per magic) e traduca le azioni della strategia
in chiamate al gateway rispettando dry_run - non la correttezza delle vere
chiamate MetaTrader5 (isolate in engine/gateway.py).
"""

import unittest
from unittest.mock import MagicMock

import pandas as pd

from engine.multi_symbol_runner import MultiSymbolRunnerConfig, MultiSymbolStrategyRunner
from strategies.base import Action, MarketSnapshot, MultiSymbolMarketData, MultiSymbolOpenOrder, OpenPosition


def _market(**overrides):
    base = dict(bid=1.1000, ask=1.1002, point=0.00001, digits=5, tick_value=1.0, tick_size=0.00001,
                volume_min=0.01, volume_max=100.0, volume_step=0.01, stops_level_points=0, filling_mode=0)
    base.update(overrides)
    return MarketSnapshot(**base)


def _bars(n=5, start_time=1_700_000_000):
    return pd.DataFrame({"time": [start_time + i * 3600 for i in range(n)], "close": [1.1] * n})


def _position(ticket, symbol="EURUSD", magic=42, direction="SELL"):
    return OpenPosition(ticket=ticket, symbol=symbol, magic=magic, direction=direction, volume=0.1,
                         price_open=1.1, sl=0.0, tp=0.0, profit=5.0, swap=0.0, open_time_epoch=1_700_000_000.0)


class FakeGateway:
    def __init__(self, positions=None):
        self.positions = positions or []
        self.ensure_symbol_calls = []
        self.get_closed_bars_calls = []
        self.sent_orders = []
        self.closed = []

    def ensure_symbol(self, symbol):
        self.ensure_symbol_calls.append(symbol)

    def server_time_epoch(self, symbol):
        return 1_700_003_600.0

    def get_closed_bars(self, symbol, count, timeframe="M5"):
        self.get_closed_bars_calls.append((symbol, count, timeframe))
        return _bars()

    def get_market_snapshot(self, symbol):
        return _market()

    def get_open_positions(self, magic=None):
        if magic is None:
            return list(self.positions)
        return [p for p in self.positions if p.magic == magic]

    def send_market_order(self, symbol, direction, volume, sl, tp, magic, comment):
        self.sent_orders.append(dict(symbol=symbol, direction=direction, volume=volume,
                                      sl=sl, tp=tp, magic=magic, comment=comment))
        return MagicMock(success=True, ticket=999)

    def close_position(self, position):
        self.closed.append(position.ticket)
        return MagicMock(success=True, ticket=position.ticket)


class FakeStrategy:
    """Strategia finta: ritorna qualunque lista di azioni le venga passata in
    `next_actions`, e registra l'ultimo MultiSymbolMarketData ricevuto così i
    test possono verificare cosa il motore raccoglie davvero."""

    name = "Fake"
    magic_number = 42

    def __init__(self, required_bars, next_actions=None):
        self._required_bars = required_bars
        self.next_actions = next_actions or []
        self.received: list[MultiSymbolMarketData] = []

    def required_bars(self):
        return self._required_bars

    def evaluate(self, data):
        self.received.append(data)
        return self.next_actions


class TestDataGathering(unittest.TestCase):
    def test_fetches_bars_per_symbol_with_declared_timeframe_and_count(self):
        gateway = FakeGateway()
        strategy = FakeStrategy({"EURGBP": ("H4", 50), "EURUSD": ("H1", 30)})
        runner = MultiSymbolStrategyRunner(strategy, gateway, MultiSymbolRunnerConfig())

        runner.run_once()

        self.assertIn(("EURGBP", 50, "H4"), gateway.get_closed_bars_calls)
        self.assertIn(("EURUSD", 30, "H1"), gateway.get_closed_bars_calls)
        self.assertEqual(len(strategy.received), 1)
        data = strategy.received[0]
        self.assertEqual(set(data.bars.keys()), {"EURGBP", "EURUSD"})
        self.assertEqual(set(data.market.keys()), {"EURGBP", "EURUSD"})

    def test_positions_filtered_by_strategy_magic_number(self):
        gateway = FakeGateway(positions=[_position(1, magic=42), _position(2, magic=99)])
        strategy = FakeStrategy({"EURUSD": ("M5", 10)})
        runner = MultiSymbolStrategyRunner(strategy, gateway, MultiSymbolRunnerConfig())

        runner.run_once()

        tickets = {p.ticket for p in strategy.received[0].positions}
        self.assertEqual(tickets, {1})


class TestDryRun(unittest.TestCase):
    def test_dry_run_does_not_send_orders_or_close_positions(self):
        gateway = FakeGateway(positions=[_position(1)])
        actions = [MultiSymbolOpenOrder("EURUSD", "BUY", 0.1, "test"),
                   Action(kind="close", ticket=1, reason="test")]
        strategy = FakeStrategy({"EURUSD": ("M5", 10)}, next_actions=actions)
        runner = MultiSymbolStrategyRunner(strategy, gateway, MultiSymbolRunnerConfig(dry_run=True))

        runner.run_once()

        self.assertEqual(gateway.sent_orders, [])
        self.assertEqual(gateway.closed, [])


class TestLiveExecution(unittest.TestCase):
    def test_open_order_sent_with_strategy_magic_number(self):
        gateway = FakeGateway()
        actions = [MultiSymbolOpenOrder("EURUSD", "BUY", 0.25, "entrata")]
        strategy = FakeStrategy({"EURUSD": ("M5", 10)}, next_actions=actions)
        runner = MultiSymbolStrategyRunner(strategy, gateway, MultiSymbolRunnerConfig(dry_run=False))

        runner.run_once()

        self.assertEqual(len(gateway.sent_orders), 1)
        order = gateway.sent_orders[0]
        self.assertEqual(order["symbol"], "EURUSD")
        self.assertEqual(order["direction"], "BUY")
        self.assertEqual(order["volume"], 0.25)
        self.assertEqual(order["magic"], 42)

    def test_close_action_resolves_ticket_to_full_position(self):
        gateway = FakeGateway(positions=[_position(7, symbol="GBPUSD")])
        actions = [Action(kind="close", ticket=7, reason="basket TP")]
        strategy = FakeStrategy({"GBPUSD": ("M5", 10)}, next_actions=actions)
        runner = MultiSymbolStrategyRunner(strategy, gateway, MultiSymbolRunnerConfig(dry_run=False))

        runner.run_once()

        self.assertEqual(gateway.closed, [7])

    def test_close_action_on_unknown_ticket_does_not_crash(self):
        gateway = FakeGateway(positions=[])
        actions = [Action(kind="close", ticket=123, reason="gia' sparita")]
        strategy = FakeStrategy({"EURUSD": ("M5", 10)}, next_actions=actions)
        runner = MultiSymbolStrategyRunner(strategy, gateway, MultiSymbolRunnerConfig(dry_run=False))

        runner.run_once()   # non deve sollevare eccezioni

        self.assertEqual(gateway.closed, [])

    def test_unknown_action_kind_does_not_crash(self):
        gateway = FakeGateway()
        actions = [Action(kind="modify_sl_tp", ticket=1)]
        strategy = FakeStrategy({"EURUSD": ("M5", 10)}, next_actions=actions)
        runner = MultiSymbolStrategyRunner(strategy, gateway, MultiSymbolRunnerConfig(dry_run=False))

        runner.run_once()   # non deve sollevare eccezioni

        self.assertEqual(gateway.sent_orders, [])
        self.assertEqual(gateway.closed, [])


if __name__ == "__main__":
    unittest.main()
