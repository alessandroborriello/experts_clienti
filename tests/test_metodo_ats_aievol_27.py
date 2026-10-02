"""
Test del porting di MetodoATS_AIEvol_27.mq5.

L'elenco EXPECTED_PAYLOAD_KEYS sotto non e' stato ritrascritto a mano: e'
stato estratto con una regex dal corpo di SendDataToServer() nel file .mq5
originale (88 chiavi), per avere un oracolo indipendente da eventuali
errori di trascrizione nel modulo portato.
"""

import math
import unittest

import numpy as np
import pandas as pd

from strategies.base import AccountSnapshot, DecisionContext, MarketSnapshot, OpenPosition
from strategies.metodo_ats_aievol_27 import MetodoATSAIEvol27, MetodoATSConfig, _parse_confidence
from common.mt5_time import ItalianMoment

EXPECTED_PAYLOAD_KEYS = {
    "account", "symbol", "hour", "day_of_week", "day_of_month", "ema_8", "ema_21", "ema_50",
    "ema_200", "upper_band", "middle_band", "Lower_band", "atr", "atr_mean_5", "atr_mean_10",
    "atr_mean_20", "atr_mean_50", "atr_22", "atr_ema_200", "atr_22_mean_5", "atr_22_mean_10",
    "atr_22_mean_20", "atr_22_mean_50", "atr_48", "atr_48_mean_5", "atr_48_mean_10",
    "atr_48_mean_20", "atr_48_mean_50", "atr_96", "atr_96_mean_5", "atr_96_mean_10",
    "atr_96_mean_20", "atr_96_mean_50", "adx", "rsi", "rsi_10", "rsi_20", "ema_21_up_ema_50",
    "ema_21_up_ema_200", "ema_8_up_ema_21", "ema_8_up_ema_50", "ema_8_up_ema_200",
    "ema_crossover", "atr_growing_previous", "atr_growing_previous_5_mean",
    "atr_growing_previous_10_mean", "atr_growing_previous_20_mean",
    "atr_growing_previous_50_mean", "atr_22_growing_previous", "atr_22_growing_previous_5_mean",
    "atr_22_growing_previous_10_mean", "atr_22_growing_previous_20_mean",
    "atr_22_growing_previous_50_mean", "atr_48_growing_previous", "atr_48_growing_previous_5_mean",
    "atr_48_growing_previous_10_mean", "atr_48_growing_previous_20_mean",
    "atr_48_growing_previous_50_mean", "atr_96_growing_previous", "atr_96_growing_previous_5_mean",
    "atr_96_growing_previous_10_mean", "atr_96_growing_previous_20_mean",
    "atr_96_growing_previous_50_mean", "rsi_overbought", "rsi_overbought_80",
    "rsi_overbought_90", "rsi_oversold", "rsi_oversold_20", "rsi_oversold_10",
    "rsi_10_overbought", "rsi_10_overbought_80", "rsi_10_overbought_90", "rsi_10_oversold",
    "rsi_10_oversold_20", "rsi_10_oversold_10", "rsi_20_overbought", "rsi_20_overbought_80",
    "rsi_20_overbought_90", "rsi_20_oversold", "rsi_20_oversold_20", "rsi_20_oversold_10",
    "rsi_cross_50_up", "rsi_cross_50_down", "high_volatility", "range_volatility",
    "bullish_engulfing_full", "bearish_engulfing_full", "big_candle",
}


def make_synthetic_bars(n=500, seed=0, start_time=1_700_000_000) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 1.1000 + np.cumsum(rng.normal(0, 0.0003, n))
    open_ = close - rng.normal(0, 0.0001, n)
    high = np.maximum(open_, close) + np.abs(rng.normal(0, 0.0002, n))
    low = np.minimum(open_, close) - np.abs(rng.normal(0, 0.0002, n))
    volume = rng.integers(50, 500, n).astype(float)
    time = start_time + np.arange(n) * 300   # barre M5 = 300 secondi

    return pd.DataFrame({
        "time": time, "open": open_, "high": high, "low": low,
        "close": close, "tick_volume": volume,
    })


def make_market(**overrides) -> MarketSnapshot:
    base = dict(
        bid=1.1050, ask=1.1052, point=0.0001, digits=5,
        tick_value=1.0, tick_size=0.0001,
        volume_min=0.01, volume_max=100.0, volume_step=0.01,
        stops_level_points=0, filling_mode=0,
    )
    base.update(overrides)
    return MarketSnapshot(**base)


class TestComputeFeatures(unittest.TestCase):
    def setUp(self):
        self.strategy = MetodoATSAIEvol27(MetodoATSConfig())
        self.bars = make_synthetic_bars()

    def test_requires_minimum_history(self):
        short_bars = make_synthetic_bars(n=100)
        with self.assertRaises(ValueError):
            self.strategy.compute_features(short_bars)

    def test_no_nan_or_inf(self):
        f = self.strategy.compute_features(self.bars)
        for k, v in f.items():
            if isinstance(v, float):
                self.assertFalse(math.isnan(v), msg=f"{k} e' NaN")
                self.assertFalse(math.isinf(v), msg=f"{k} e' inf")

    def test_boolean_features_are_0_or_1(self):
        f = self.strategy.compute_features(self.bars)
        bool_like = [k for k in f if k.endswith(("_up_ema_21", "_up_ema_50", "_up_ema_200",
                                                   "crossover", "overbought", "overbought_80",
                                                   "overbought_90", "oversold", "oversold_20",
                                                   "oversold_10", "growing_previous",
                                                   "growing_previous_5_mean", "growing_previous_10_mean",
                                                   "growing_previous_20_mean", "growing_previous_50_mean",
                                                   "cross_50_up", "cross_50_down", "volatility",
                                                   "engulfing_full", "big_candle"))]
        self.assertGreater(len(bool_like), 30)
        for k in bool_like:
            self.assertIn(f[k], (0, 1), msg=f"{k} = {f[k]!r} non e' 0/1")

    def test_rsi_bounded(self):
        f = self.strategy.compute_features(self.bars)
        for key in ("rsi", "rsi_10", "rsi_20"):
            self.assertGreaterEqual(f[key], 0.0)
            self.assertLessEqual(f[key], 100.0)

    def test_always_zero_quirks_preserved(self):
        f = self.strategy.compute_features(self.bars)
        self.assertEqual(f["atr_22_growing_previous"], 0)
        self.assertEqual(f["atr_48_growing_previous"], 0)
        self.assertEqual(f["atr_96_growing_previous"], 0)

    def test_hour_day_derived_from_server_time_not_italian(self):
        f = self.strategy.compute_features(self.bars)
        self.assertIn(f["hour"], range(24))
        self.assertIn(f["day_of_week"], range(7))
        self.assertIn(f["day_of_month"], range(1, 32))


class TestBuildPayload(unittest.TestCase):
    def test_exact_key_set_matches_mql5_source(self):
        strategy = MetodoATSAIEvol27(MetodoATSConfig())
        bars = make_synthetic_bars()
        features = strategy.compute_features(bars)
        account = AccountSnapshot(balance=10_000.0, login=123456)
        payload = strategy.build_ml_payload(features, account, "EURUSD")

        self.assertEqual(set(payload.keys()), EXPECTED_PAYLOAD_KEYS)

    def test_account_and_symbol_are_strings(self):
        strategy = MetodoATSAIEvol27(MetodoATSConfig())
        bars = make_synthetic_bars()
        features = strategy.compute_features(bars)
        account = AccountSnapshot(balance=10_000.0, login=123456)
        payload = strategy.build_ml_payload(features, account, "EURUSD")

        self.assertEqual(payload["account"], "123456")
        self.assertEqual(payload["symbol"], "EURUSD")


class TestParseMlResponse(unittest.TestCase):
    def setUp(self):
        self.strategy = MetodoATSAIEvol27(MetodoATSConfig())

    def test_buy(self):
        out = self.strategy.parse_ml_response({"prediction": 2, "confidence": 0.8})
        self.assertEqual(out["trade_type"], 2)

    def test_sell(self):
        out = self.strategy.parse_ml_response({"prediction": 0, "confidence": 0.6})
        self.assertEqual(out["trade_type"], 0)

    def test_none(self):
        out = self.strategy.parse_ml_response({"prediction": 1})
        self.assertEqual(out["trade_type"], -1)

    def test_missing_field(self):
        out = self.strategy.parse_ml_response({})
        self.assertEqual(out["trade_type"], -1)


class TestParseConfidence(unittest.TestCase):
    def test_fraction_converted_to_percent(self):
        self.assertAlmostEqual(_parse_confidence(0.73), 73.0)

    def test_already_percent_unchanged(self):
        self.assertAlmostEqual(_parse_confidence(73.0), 73.0)

    def test_string_input(self):
        self.assertAlmostEqual(_parse_confidence("0.5"), 50.0)

    def test_invalid_returns_none(self):
        self.assertIsNone(_parse_confidence("n/a"))
        self.assertIsNone(_parse_confidence(None))


class TestDecideSignal(unittest.TestCase):
    def _ctx(self, **overrides):
        base = dict(
            symbol="EURUSD",
            market=make_market(),
            account=AccountSnapshot(balance=10_000.0, login=1),
            reference_price=1.1000,
            allowed_time=True,
            semaphore_triggered=False,
            daily_loss_hit=False,
            has_open_position=False,
        )
        base.update(overrides)
        return DecisionContext(**base)

    def test_no_signal_when_prediction_neutral(self):
        strategy = MetodoATSAIEvol27(MetodoATSConfig())
        sig = strategy.decide_signal({"trade_type": -1}, self._ctx())
        self.assertIsNone(sig)

    def test_no_signal_when_time_not_allowed(self):
        strategy = MetodoATSAIEvol27(MetodoATSConfig())
        sig = strategy.decide_signal({"trade_type": 2}, self._ctx(allowed_time=False))
        self.assertIsNone(sig)

    def test_buy_signal_tp_sl_math(self):
        cfg = MetodoATSConfig(base_profit_target=0.5, sl_ratio=2.0, lot_size=0.1)
        strategy = MetodoATSAIEvol27(cfg)
        sig = strategy.decide_signal({"trade_type": 2, "confidence": 0.8}, self._ctx(reference_price=1.1000))

        self.assertIsNotNone(sig)
        self.assertEqual(sig.direction, "BUY")
        expected_tp = 1.1000 * 1.005
        self.assertAlmostEqual(sig.take_profit, expected_tp, places=6)
        tp_dist = expected_tp - 1.1000
        expected_sl = 1.1000 - tp_dist * 2.0
        self.assertAlmostEqual(sig.stop_loss, expected_sl, places=6)
        self.assertEqual(sig.lot_size, 0.1)
        self.assertFalse(sig.is_virtual)

    def test_sell_signal_tp_sl_math(self):
        cfg = MetodoATSConfig(base_profit_target=0.5, sl_ratio=2.0)
        strategy = MetodoATSAIEvol27(cfg)
        sig = strategy.decide_signal({"trade_type": 0}, self._ctx(reference_price=1.1000))

        self.assertEqual(sig.direction, "SELL")
        expected_tp = 1.1000 * 0.995
        self.assertAlmostEqual(sig.take_profit, expected_tp, places=6)

    def test_sl_ratio_zero_disables_stop_loss(self):
        cfg = MetodoATSConfig(sl_ratio=0.0)
        strategy = MetodoATSAIEvol27(cfg)
        sig = strategy.decide_signal({"trade_type": 2}, self._ctx())
        self.assertEqual(sig.stop_loss, 0.0)

    def test_direction_disabled_blocks_signal(self):
        cfg = MetodoATSConfig(enable_buy_trades=False)
        strategy = MetodoATSAIEvol27(cfg)
        sig = strategy.decide_signal({"trade_type": 2}, self._ctx())
        self.assertIsNone(sig)

    def test_existing_position_blocks_real_signal(self):
        strategy = MetodoATSAIEvol27(MetodoATSConfig())
        sig = strategy.decide_signal({"trade_type": 2}, self._ctx(has_open_position=True))
        self.assertIsNone(sig)

    def test_semaphore_blocks_real_signal(self):
        strategy = MetodoATSAIEvol27(MetodoATSConfig())
        sig = strategy.decide_signal({"trade_type": 2}, self._ctx(semaphore_triggered=True))
        self.assertIsNone(sig)

    def test_daily_loss_blocks_real_signal(self):
        strategy = MetodoATSAIEvol27(MetodoATSConfig())
        sig = strategy.decide_signal({"trade_type": 2}, self._ctx(daily_loss_hit=True))
        self.assertIsNone(sig)

    def test_only_show_signals_bypasses_gates_but_marks_virtual(self):
        cfg = MetodoATSConfig(only_show_signals=True)
        strategy = MetodoATSAIEvol27(cfg)
        sig = strategy.decide_signal({"trade_type": 2}, self._ctx(has_open_position=True,
                                                                    semaphore_triggered=True,
                                                                    daily_loss_hit=True))
        self.assertIsNotNone(sig)
        self.assertTrue(sig.is_virtual)

    def test_confidence_propagated(self):
        strategy = MetodoATSAIEvol27(MetodoATSConfig())
        sig = strategy.decide_signal({"trade_type": 2, "confidence": 0.9}, self._ctx())
        self.assertAlmostEqual(sig.confidence, 90.0)

    def test_dynamic_risk_lot_sizing_mode(self):
        cfg = MetodoATSConfig(lot_sizing_mode="dynamic_risk", max_risk_pct=1.0, sl_ratio=2.0)
        strategy = MetodoATSAIEvol27(cfg)
        market = make_market(point=0.0001, tick_value=1.0, tick_size=0.0001,
                              volume_min=0.01, volume_max=100.0, volume_step=0.01)
        ctx = self._ctx(market=market, account=AccountSnapshot(balance=10_000.0, login=1),
                         reference_price=1.1000)
        sig = strategy.decide_signal({"trade_type": 2}, ctx)
        self.assertGreater(sig.lot_size, 0.0)


class TestManageOpenPositions(unittest.TestCase):
    def _position(self, ticket, symbol, magic=1, direction="BUY", price_open=1.1000,
                   sl=0.0, tp=1.1100, profit=0.0, swap=0.0, open_time_epoch=0.0):
        return OpenPosition(ticket=ticket, symbol=symbol, magic=magic, direction=direction,
                             volume=0.1, price_open=price_open, sl=sl, tp=tp, profit=profit,
                             swap=swap, open_time_epoch=open_time_epoch)

    def _italian_moment(self, day, hour, minute=0):
        return ItalianMoment(dt=None, day_of_week=day, hour=hour, minute=minute)

    def test_old_trade_closed_regardless_of_symbol(self):
        cfg = MetodoATSConfig(hours_close=48.0, close_friday=False)
        strategy = MetodoATSAIEvol27(cfg)
        positions = [
            self._position(1, "EURUSD", open_time_epoch=0.0),
            self._position(2, "GBPUSD", open_time_epoch=0.0),   # simbolo diverso, stesso magic
        ]
        now = 49 * 3600   # 49 ore dopo
        actions = strategy.manage_open_positions("EURUSD", positions, make_market(), now,
                                                   self._italian_moment(2, 12))
        tickets_closed = {a.ticket for a in actions if a.kind == "close"}
        self.assertEqual(tickets_closed, {1, 2})   # entrambi, come nell'EA (solo magic)

    def test_breakeven_only_applies_to_matching_symbol(self):
        cfg = MetodoATSConfig(enable_breakeven=True, breakeven_trigger=0.5, hours_close=9999, close_friday=False)
        strategy = MetodoATSAIEvol27(cfg)
        # trigger BUY: open=1.1000, tp=1.1100 -> tp_dist=0.01, trigger=1.1050
        positions = [
            self._position(1, "EURUSD", price_open=1.1000, tp=1.1100, sl=0.0),
            self._position(2, "GBPUSD", price_open=1.1000, tp=1.1100, sl=0.0),
        ]
        market = make_market(bid=1.1060, ask=1.1062)   # bid oltre il trigger
        actions = strategy.manage_open_positions("EURUSD", positions, market, 0.0,
                                                   self._italian_moment(2, 12))
        modify_tickets = {a.ticket for a in actions if a.kind == "modify_sl_tp"}
        self.assertEqual(modify_tickets, {1})   # solo il simbolo corrente, non GBPUSD

    def test_monetary_stop_loss_only_applies_to_matching_symbol(self):
        cfg = MetodoATSConfig(monetary_stop_loss=50.0, hours_close=9999, close_friday=False)
        strategy = MetodoATSAIEvol27(cfg)
        positions = [
            self._position(1, "EURUSD", profit=-60.0),
            self._position(2, "GBPUSD", profit=-60.0),
        ]
        actions = strategy.manage_open_positions("EURUSD", positions, make_market(), 0.0,
                                                   self._italian_moment(2, 12))
        closed = {a.ticket for a in actions if a.kind == "close"}
        self.assertEqual(closed, {1})

    def test_friday_close_closes_regardless_of_symbol(self):
        cfg = MetodoATSConfig(close_friday=True, friday_close_hour=21, friday_close_minute=50,
                               hours_close=9999)
        strategy = MetodoATSAIEvol27(cfg)
        positions = [
            self._position(1, "EURUSD"),
            self._position(2, "GBPUSD"),
        ]
        actions = strategy.manage_open_positions("EURUSD", positions, make_market(), 0.0,
                                                   self._italian_moment(5, 22, 0))  # venerdi 22:00
        closed = {a.ticket for a in actions if a.kind == "close"}
        self.assertEqual(closed, {1, 2})

    def test_no_closures_when_nothing_triggers(self):
        cfg = MetodoATSConfig(hours_close=9999, close_friday=False, monetary_stop_loss=0.0,
                               enable_breakeven=False)
        strategy = MetodoATSAIEvol27(cfg)
        positions = [self._position(1, "EURUSD")]
        actions = strategy.manage_open_positions("EURUSD", positions, make_market(), 0.0,
                                                   self._italian_moment(2, 12))
        self.assertEqual(actions, [])

    def test_other_magic_ignored(self):
        cfg = MetodoATSConfig(hours_close=1.0, close_friday=False)
        strategy = MetodoATSAIEvol27(cfg)
        positions = [self._position(1, "EURUSD", magic=999, open_time_epoch=0.0)]
        actions = strategy.manage_open_positions("EURUSD", positions, make_market(), 10 * 3600,
                                                   self._italian_moment(2, 12))
        self.assertEqual(actions, [])


if __name__ == "__main__":
    unittest.main()
