import unittest
from datetime import datetime, timezone

import pandas as pd

from strategies.ats_spread import ATSSpreadConfig, ATSSpreadStrategy, pips_to_price
from strategies.base import Action, MarketSnapshot, MultiSymbolMarketData, MultiSymbolOpenOrder, OpenPosition

SEGNALE = "EURGBP"
SIM1 = "EURUSD"
SIM2 = "GBPUSD"


def _epoch(hour, minute=0):
    return datetime(2026, 10, 2, hour, minute, tzinfo=timezone.utc).timestamp()


def _bars(closes, start_time=1_700_000_000, step=3600):
    n = len(closes)
    return pd.DataFrame({
        "time": [start_time + i * step for i in range(n)],
        "close": list(closes),
    })


def _flat_then(values, flat=1.1000, flat_count=23):
    return _bars([flat] * flat_count + list(values))


def _market(point=0.00001, bid=1.1000, ask=1.1002, volume_min=0.01, volume_max=100.0, volume_step=0.01):
    return MarketSnapshot(bid=bid, ask=ask, point=point, digits=5, tick_value=1.0, tick_size=point,
                           volume_min=volume_min, volume_max=volume_max, volume_step=volume_step,
                           stops_level_points=0, filling_mode=0)


def _position(ticket, symbol, direction, volume, price_open, profit, swap=0.0, magic=2940,
              open_time_epoch=1_700_000_000.0):
    return OpenPosition(ticket=ticket, symbol=symbol, magic=magic, direction=direction, volume=volume,
                         price_open=price_open, sl=0.0, tp=0.0, profit=profit, swap=swap,
                         open_time_epoch=open_time_epoch)


def _data(now_epoch, segnale_bars, sim1_bars, sim2_bars, positions=None, market_overrides=None):
    market = {SEGNALE: _market(), SIM1: _market(), SIM2: _market()}
    if market_overrides:
        market.update(market_overrides)
    return MultiSymbolMarketData(
        now_epoch=now_epoch,
        bars={SEGNALE: segnale_bars, SIM1: sim1_bars, SIM2: sim2_bars},
        market=market,
        positions=positions or [],
    )


def _strategy(**overrides):
    cfg = ATSSpreadConfig(simbolo_segnale=SEGNALE, simbolo_1=SIM1, simbolo_2=SIM2, **overrides)
    return ATSSpreadStrategy(cfg)


class TestPipsToPrice(unittest.TestCase):
    def test_standard_five_digit_symbol(self):
        self.assertAlmostEqual(pips_to_price(0.00001), 0.0001)


class TestEntrySignal(unittest.TestCase):
    def _flat_sim_bars(self):
        return _bars([1.1000] * 24)

    def test_close_above_mid_sells_sim1_buys_sim2(self):
        strat = _strategy()
        segnale = _flat_then([1.2000])   # ben sopra la media -> close > mid
        data = _data(_epoch(15), segnale, self._flat_sim_bars(), self._flat_sim_bars())

        orders = [a for a in strat.evaluate(data) if isinstance(a, MultiSymbolOpenOrder)]
        by_symbol = {o.symbol: o for o in orders}
        self.assertEqual(by_symbol[SIM1].direction, "SELL")
        self.assertEqual(by_symbol[SIM2].direction, "BUY")
        self.assertAlmostEqual(by_symbol[SIM1].volume, 0.15)
        self.assertAlmostEqual(by_symbol[SIM2].volume, 0.1)

    def test_close_below_mid_buys_sim1_sells_sim2(self):
        strat = _strategy()
        segnale = _flat_then([1.0000])   # ben sotto la media -> close < mid
        data = _data(_epoch(15), segnale, self._flat_sim_bars(), self._flat_sim_bars())

        orders = [a for a in strat.evaluate(data) if isinstance(a, MultiSymbolOpenOrder)]
        by_symbol = {o.symbol: o for o in orders}
        self.assertEqual(by_symbol[SIM1].direction, "BUY")
        self.assertEqual(by_symbol[SIM2].direction, "SELL")

    def test_outside_time_window_no_entry(self):
        strat = _strategy(start="10:00", end="21:00")
        segnale = _flat_then([1.2000])
        data = _data(_epoch(22), segnale, self._flat_sim_bars(), self._flat_sim_bars())

        orders = [a for a in strat.evaluate(data) if isinstance(a, MultiSymbolOpenOrder)
                  and a.comment == "ATS Spread"]
        self.assertEqual(orders, [])

    def test_existing_position_on_symbol_blocks_new_entry_there_only(self):
        strat = _strategy()
        segnale = _flat_then([1.2000])
        positions = [_position(1, SIM1, "SELL", 0.15, 1.1000, profit=-5.0)]
        data = _data(_epoch(15), segnale, self._flat_sim_bars(), self._flat_sim_bars(), positions=positions)

        orders = [a for a in strat.evaluate(data) if isinstance(a, MultiSymbolOpenOrder)
                  and a.comment == "ATS Spread"]
        symbols = {o.symbol for o in orders}
        self.assertNotIn(SIM1, symbols)
        self.assertIn(SIM2, symbols)


class TestAveragingLadder(unittest.TestCase):
    def _segnale_bars(self):
        # fuori orario cosi' l'ingresso non interferisce con i test della scaletta
        return _bars([1.1000] * 24)

    def test_no_open_position_no_averaging(self):
        strat = _strategy()
        sim1_bars = _flat_then([1.0990, 1.2000])   # incrocio netto sopra la banda
        data = _data(_epoch(23), self._segnale_bars(), sim1_bars, _bars([1.1000] * 24))

        orders = [a for a in strat.evaluate(data) if isinstance(a, MultiSymbolOpenOrder)
                  and a.comment == "ATS Spread mediata"]
        self.assertEqual(orders, [])

    def test_aggregate_profit_positive_blocks_averaging(self):
        strat = _strategy()
        sim1_bars = _flat_then([1.0990, 1.2000])
        positions = [_position(1, SIM1, "SELL", 0.15, 1.1500, profit=50.0)]
        data = _data(_epoch(23), self._segnale_bars(), sim1_bars, _bars([1.1000] * 24), positions=positions)

        orders = [a for a in strat.evaluate(data) if isinstance(a, MultiSymbolOpenOrder)
                  and a.comment == "ATS Spread mediata"]
        self.assertEqual(orders, [])

    def test_no_crossover_no_averaging(self):
        strat = _strategy()
        sim1_bars = _bars([1.1000] * 24)   # nessun incrocio
        positions = [_position(1, SIM1, "SELL", 0.15, 1.1000, profit=-5.0)]
        data = _data(_epoch(23), self._segnale_bars(), sim1_bars, _bars([1.1000] * 24), positions=positions)

        orders = [a for a in strat.evaluate(data) if isinstance(a, MultiSymbolOpenOrder)
                  and a.comment == "ATS Spread mediata"]
        self.assertEqual(orders, [])

    def test_too_close_to_existing_leg_blocks_averaging(self):
        strat = _strategy(distanza_minima_mediate=50.0)   # soglia effettiva 100 pip
        sim1_bars = _flat_then([1.0990, 1.2000])
        # prezzo di riferimento (bid, vendita) 1.1000 - 1.1500 e' lontanissimo
        # dalla banda ma vicinissimo (0 pip) al prezzo di apertura della gamba
        positions = [_position(1, SIM1, "SELL", 0.15, 1.1000, profit=-5.0)]
        data = _data(_epoch(23), self._segnale_bars(), sim1_bars, _bars([1.1000] * 24),
                     positions=positions, market_overrides={SIM1: _market(bid=1.1000, ask=1.1002)})

        orders = [a for a in strat.evaluate(data) if isinstance(a, MultiSymbolOpenOrder)
                  and a.comment == "ATS Spread mediata"]
        self.assertEqual(orders, [])

    def test_crossover_with_enough_distance_opens_martingale_leg(self):
        strat = _strategy(martingala=1.05, lotto_mediata_1=0.01, distanza_minima_mediate=10.0)
        sim1_bars = _flat_then([1.0990, 1.2000])   # incrocio sopra la banda
        # posizione aperta lontana (1500 pip) dal prezzo corrente di riferimento
        positions = [_position(1, SIM1, "SELL", 0.034, 1.3500, profit=-5.0)]
        data = _data(_epoch(23), self._segnale_bars(), sim1_bars, _bars([1.1000] * 24),
                     positions=positions,
                     market_overrides={SIM1: _market(bid=1.3450, ask=1.3452, volume_step=0.001)})

        orders = [a for a in strat.evaluate(data) if isinstance(a, MultiSymbolOpenOrder)
                  and a.comment == "ATS Spread mediata"]
        self.assertEqual(len(orders), 1)
        self.assertEqual(orders[0].symbol, SIM1)
        self.assertEqual(orders[0].direction, "SELL")
        # 0.034*1.05=0.0357, arrotondato al volume_step (0.001) del broker
        self.assertAlmostEqual(orders[0].volume, 0.036, places=6)

    def test_once_per_bar_gate_blocks_second_evaluation_same_bar(self):
        strat = _strategy(distanza_minima_mediate=10.0)
        sim1_bars = _flat_then([1.0990, 1.2000])
        positions = [_position(1, SIM1, "SELL", 0.03, 1.3500, profit=-5.0)]
        data = _data(_epoch(23), self._segnale_bars(), sim1_bars, _bars([1.1000] * 24),
                     positions=positions, market_overrides={SIM1: _market(bid=1.3450, ask=1.3452)})

        first = [a for a in strat.evaluate(data) if isinstance(a, MultiSymbolOpenOrder)
                 and a.comment == "ATS Spread mediata"]
        second = [a for a in strat.evaluate(data) if isinstance(a, MultiSymbolOpenOrder)
                  and a.comment == "ATS Spread mediata"]
        self.assertEqual(len(first), 1)
        self.assertEqual(second, [])


class TestBasketClose(unittest.TestCase):
    def _flat(self):
        return _bars([1.1000] * 24)

    def test_disabled_by_default_never_closes(self):
        strat = _strategy()   # usa_tp_pips_medio=False di default
        positions = [_position(1, SIM1, "SELL", 0.1, 1.2000, profit=1000.0)]
        data = _data(_epoch(15), self._flat(), self._flat(), self._flat(), positions=positions,
                     market_overrides={SIM1: _market(bid=1.0000, ask=1.0002)})

        closes = [a for a in strat.evaluate(data) if isinstance(a, Action)]
        self.assertEqual(closes, [])

    def test_enabled_but_money_not_positive_no_close(self):
        strat = _strategy(usa_tp_pips_medio=True, tp_medio_pips=20.0)
        positions = [_position(1, SIM1, "SELL", 0.1, 1.2000, profit=-10.0)]
        data = _data(_epoch(15), self._flat(), self._flat(), self._flat(), positions=positions,
                     market_overrides={SIM1: _market(bid=1.0000, ask=1.0002)})

        closes = [a for a in strat.evaluate(data) if isinstance(a, Action)]
        self.assertEqual(closes, [])

    def test_enabled_money_positive_but_pips_below_target_no_close(self):
        strat = _strategy(usa_tp_pips_medio=True, tp_medio_pips=500.0)
        # SELL aperta a 1.2000, bid ora 1.1990 -> 10 pip di profitto, sotto i 500 richiesti
        positions = [_position(1, SIM1, "SELL", 0.1, 1.2000, profit=10.0)]
        data = _data(_epoch(15), self._flat(), self._flat(), self._flat(), positions=positions,
                     market_overrides={SIM1: _market(bid=1.1990, ask=1.1992)})

        closes = [a for a in strat.evaluate(data) if isinstance(a, Action)]
        self.assertEqual(closes, [])

    def test_enabled_both_conditions_met_closes_everything(self):
        strat = _strategy(usa_tp_pips_medio=True, tp_medio_pips=50.0)
        positions = [
            _position(1, SIM1, "SELL", 0.1, 1.2000, profit=100.0),
            _position(2, SIM2, "BUY", 0.1, 1.0000, profit=20.0),
        ]
        data = _data(_epoch(15), self._flat(), self._flat(), self._flat(), positions=positions,
                     market_overrides={
                         SIM1: _market(bid=1.1900, ask=1.1902),   # SELL a 1.2000 -> 100 pip
                         SIM2: _market(bid=1.0050, ask=1.0052),   # BUY a 1.0000 -> 50 pip
                     })

        closes = [a for a in strat.evaluate(data) if isinstance(a, Action)]
        self.assertEqual({a.ticket for a in closes}, {1, 2})
        self.assertTrue(all(a.kind == "close" for a in closes))


if __name__ == "__main__":
    unittest.main()
