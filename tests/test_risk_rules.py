import unittest

from common.risk_rules import Deal, is_daily_loss_hit, is_semaphore_triggered


class TestDailyLossHit(unittest.TestCase):
    def test_disabled_when_limit_zero(self):
        self.assertFalse(is_daily_loss_hit([], 10_000, 0.0, magic=1, symbol="EURUSD", day_start_epoch=0))

    def test_not_hit_when_profit_positive(self):
        deals = [Deal(magic=1, symbol="EURUSD", is_exit=True, profit=50, swap=0, commission=0, time_epoch=100)]
        self.assertFalse(is_daily_loss_hit(deals, 10_000, 2.5, magic=1, symbol="EURUSD", day_start_epoch=0))

    def test_hit_when_loss_exceeds_limit(self):
        # limite = 10000 * 2.5% = 250
        deals = [Deal(magic=1, symbol="EURUSD", is_exit=True, profit=-300, swap=0, commission=0, time_epoch=100)]
        self.assertTrue(is_daily_loss_hit(deals, 10_000, 2.5, magic=1, symbol="EURUSD", day_start_epoch=0))

    def test_ignores_deals_before_today(self):
        deals = [Deal(magic=1, symbol="EURUSD", is_exit=True, profit=-300, swap=0, commission=0, time_epoch=50)]
        self.assertFalse(is_daily_loss_hit(deals, 10_000, 2.5, magic=1, symbol="EURUSD", day_start_epoch=100))

    def test_ignores_other_magic_or_symbol(self):
        deals = [
            Deal(magic=2, symbol="EURUSD", is_exit=True, profit=-1000, swap=0, commission=0, time_epoch=100),
            Deal(magic=1, symbol="GBPUSD", is_exit=True, profit=-1000, swap=0, commission=0, time_epoch=100),
        ]
        self.assertFalse(is_daily_loss_hit(deals, 10_000, 2.5, magic=1, symbol="EURUSD", day_start_epoch=0))

    def test_ignores_entry_deals(self):
        deals = [Deal(magic=1, symbol="EURUSD", is_exit=False, profit=-1000, swap=0, commission=0, time_epoch=100)]
        self.assertFalse(is_daily_loss_hit(deals, 10_000, 2.5, magic=1, symbol="EURUSD", day_start_epoch=0))

    def test_includes_swap_and_commission(self):
        # profit=-100, swap=-100, commission=-60 -> somma -260, limite 250 -> hit
        deals = [Deal(magic=1, symbol="EURUSD", is_exit=True, profit=-100, swap=-100, commission=-60, time_epoch=100)]
        self.assertTrue(is_daily_loss_hit(deals, 10_000, 2.5, magic=1, symbol="EURUSD", day_start_epoch=0))


class TestSemaphoreTriggered(unittest.TestCase):
    def test_disabled_when_zero(self):
        self.assertFalse(is_semaphore_triggered([], 0, magic=1, symbol="EURUSD"))

    def test_triggered_after_n_consecutive_losses(self):
        deals_desc = [
            Deal(magic=1, symbol="EURUSD", is_exit=True, profit=-10, swap=0, commission=0, time_epoch=t)
            for t in (5, 4, 3)
        ]
        self.assertTrue(is_semaphore_triggered(deals_desc, 3, magic=1, symbol="EURUSD"))

    def test_not_triggered_if_a_win_breaks_the_streak(self):
        deals_desc = [
            Deal(magic=1, symbol="EURUSD", is_exit=True, profit=-10, swap=0, commission=0, time_epoch=5),
            Deal(magic=1, symbol="EURUSD", is_exit=True, profit=+5, swap=0, commission=0, time_epoch=4),
            Deal(magic=1, symbol="EURUSD", is_exit=True, profit=-10, swap=0, commission=0, time_epoch=3),
        ]
        self.assertFalse(is_semaphore_triggered(deals_desc, 3, magic=1, symbol="EURUSD"))

    def test_not_enough_history_not_triggered(self):
        deals_desc = [
            Deal(magic=1, symbol="EURUSD", is_exit=True, profit=-10, swap=0, commission=0, time_epoch=5),
        ]
        self.assertFalse(is_semaphore_triggered(deals_desc, 3, magic=1, symbol="EURUSD"))

    def test_ignores_other_symbol(self):
        deals_desc = [
            Deal(magic=1, symbol="GBPUSD", is_exit=True, profit=-10, swap=0, commission=0, time_epoch=t)
            for t in (5, 4, 3)
        ]
        self.assertFalse(is_semaphore_triggered(deals_desc, 3, magic=1, symbol="EURUSD"))


if __name__ == "__main__":
    unittest.main()
