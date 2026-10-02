import unittest

import numpy as np
import pandas as pd

from common import indicators as ind


class TestEMA(unittest.TestCase):
    def test_constant_series_converges_to_the_constant(self):
        s = pd.Series([10.0] * 50)
        out = ind.ema(s, 8)
        self.assertAlmostEqual(out.iloc[-1], 10.0, places=6)

    def test_matches_known_small_example(self):
        # EMA(3) manuale su [1,2,3,4,5] con alpha=2/(3+1)=0.5, seed=prezzo[0]
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        out = ind.ema(s, 3)
        expected = [1.0, 1.5, 2.25, 3.125, 4.0625]
        for got, exp in zip(out.tolist(), expected):
            self.assertAlmostEqual(got, exp, places=6)


class TestSMA(unittest.TestCase):
    def test_basic(self):
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        out = ind.sma(s, 3)
        self.assertTrue(np.isnan(out.iloc[0]))
        self.assertAlmostEqual(out.iloc[2], 2.0, places=6)   # media di 1,2,3
        self.assertAlmostEqual(out.iloc[4], 4.0, places=6)   # media di 3,4,5


class TestRSI(unittest.TestCase):
    def test_monotonic_up_series_is_overbought(self):
        s = pd.Series(np.arange(1.0, 60.0))   # sempre in salita
        out = ind.rsi(s, 14)
        self.assertGreater(out.iloc[-1], 95.0)

    def test_monotonic_down_series_is_oversold(self):
        s = pd.Series(np.arange(60.0, 1.0, -1.0))
        out = ind.rsi(s, 14)
        self.assertLess(out.iloc[-1], 5.0)

    def test_flat_series_is_neutral(self):
        s = pd.Series([10.0] * 30)
        out = ind.rsi(s, 14)
        self.assertAlmostEqual(out.iloc[-1], 50.0, places=3)

    def test_bounded_0_100(self):
        rng = np.random.default_rng(42)
        s = pd.Series(100 + np.cumsum(rng.normal(0, 1, 200)))
        out = ind.rsi(s, 14).dropna()
        self.assertTrue((out >= 0).all())
        self.assertTrue((out <= 100).all())


class TestATR(unittest.TestCase):
    def test_zero_range_gives_zero_atr(self):
        n = 30
        close = pd.Series([10.0] * n)
        high = close.copy()
        low = close.copy()
        out = ind.atr(high, low, close, 14)
        self.assertAlmostEqual(out.iloc[-1], 0.0, places=6)

    def test_constant_range_converges_to_that_range(self):
        n = 60
        close = pd.Series([10.0] * n)
        high = pd.Series([11.0] * n)
        low = pd.Series([9.0] * n)
        out = ind.atr(high, low, close, 14)
        # range = high-low = 2.0 ad ogni barra -> ATR deve convergere a 2.0
        self.assertAlmostEqual(out.iloc[-1], 2.0, places=3)


class TestBollinger(unittest.TestCase):
    def test_flat_series_zero_width(self):
        s = pd.Series([10.0] * 30)
        upper, mid, lower = ind.bollinger_bands(s, 20, 2.0)
        self.assertAlmostEqual(upper.iloc[-1], 10.0, places=6)
        self.assertAlmostEqual(mid.iloc[-1], 10.0, places=6)
        self.assertAlmostEqual(lower.iloc[-1], 10.0, places=6)

    def test_upper_above_lower_on_noisy_series(self):
        rng = np.random.default_rng(1)
        s = pd.Series(100 + np.cumsum(rng.normal(0, 1, 60)))
        upper, mid, lower = ind.bollinger_bands(s, 20, 2.0)
        self.assertGreater(upper.iloc[-1], mid.iloc[-1])
        self.assertGreater(mid.iloc[-1], lower.iloc[-1])


class TestADX(unittest.TestCase):
    def test_strong_trend_has_high_adx(self):
        n = 100
        close = pd.Series(np.linspace(100, 200, n))   # trend lineare forte
        high = close + 0.5
        low = close - 0.5
        out = ind.adx(high, low, close, 14).dropna()
        self.assertGreater(out.iloc[-1], 30.0)

    def test_flat_market_has_low_adx(self):
        n = 100
        rng = np.random.default_rng(7)
        close = pd.Series(100 + rng.normal(0, 0.01, n))   # rumore minimo, nessun trend
        high = close + 0.02
        low = close - 0.02
        out = ind.adx(high, low, close, 14).dropna()
        self.assertLess(out.iloc[-1], 30.0)


if __name__ == "__main__":
    unittest.main()
