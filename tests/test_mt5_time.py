import unittest
from datetime import datetime, timezone

from common.mt5_time import (
    ItalianMoment,
    is_within_server_window,
    mql5_day_of_week,
    should_close_friday,
    should_close_thursday,
    is_allowed_time,
    to_italian_time,
)


class TestDayOfWeek(unittest.TestCase):
    def test_matches_mql5_convention(self):
        # 2026-10-05 e' un lunedi
        monday = datetime(2026, 10, 5, 12, 0, tzinfo=timezone.utc)
        self.assertEqual(mql5_day_of_week(monday), 1)
        sunday = datetime(2026, 10, 4, 12, 0, tzinfo=timezone.utc)
        self.assertEqual(mql5_day_of_week(sunday), 0)
        saturday = datetime(2026, 10, 3, 12, 0, tzinfo=timezone.utc)
        self.assertEqual(mql5_day_of_week(saturday), 6)
        friday = datetime(2026, 10, 2, 12, 0, tzinfo=timezone.utc)
        self.assertEqual(mql5_day_of_week(friday), 5)


class TestItalianConversion(unittest.TestCase):
    def test_dst_summer(self):
        # Agosto -> ora legale, UTC+2
        utc_dt = datetime(2026, 8, 15, 10, 0, tzinfo=timezone.utc)
        local = to_italian_time(utc_dt)
        self.assertEqual(local.hour, 12)

    def test_dst_winter(self):
        # Gennaio -> ora solare, UTC+1
        utc_dt = datetime(2026, 1, 15, 10, 0, tzinfo=timezone.utc)
        local = to_italian_time(utc_dt)
        self.assertEqual(local.hour, 11)


class TestIsAllowedTime(unittest.TestCase):
    def _moment(self, day, hour, minute=0):
        return ItalianMoment(dt=None, day_of_week=day, hour=hour, minute=minute)

    def test_weekend_blocked(self):
        self.assertFalse(is_allowed_time(self._moment(0, 12)))   # domenica
        self.assertFalse(is_allowed_time(self._moment(6, 12)))   # sabato

    def test_friday_blocked_all_day(self):
        self.assertFalse(is_allowed_time(self._moment(5, 8)))
        self.assertFalse(is_allowed_time(self._moment(5, 23)))

    def test_monday_before_9_blocked(self):
        self.assertFalse(is_allowed_time(self._moment(1, 8, 59)))
        self.assertTrue(is_allowed_time(self._moment(1, 9, 0)))

    def test_thursday_after_21_blocked(self):
        self.assertFalse(is_allowed_time(self._moment(4, 21)))
        self.assertTrue(is_allowed_time(self._moment(4, 20, 59)))

    def test_normal_midweek_allowed(self):
        self.assertTrue(is_allowed_time(self._moment(2, 12)))    # martedi
        self.assertTrue(is_allowed_time(self._moment(3, 12)))    # mercoledi


class TestCloseFriday(unittest.TestCase):
    def _moment(self, day, hour, minute=0):
        return ItalianMoment(dt=None, day_of_week=day, hour=hour, minute=minute)

    def test_before_threshold_not_closed(self):
        self.assertFalse(should_close_friday(self._moment(5, 21, 49), 21, 50))

    def test_at_threshold_closed(self):
        self.assertTrue(should_close_friday(self._moment(5, 21, 50), 21, 50))

    def test_after_threshold_closed(self):
        self.assertTrue(should_close_friday(self._moment(5, 23, 0), 21, 50))

    def test_wrong_day_not_closed(self):
        self.assertFalse(should_close_friday(self._moment(4, 23, 0), 21, 50))

    def test_disabled_never_closes(self):
        self.assertFalse(should_close_friday(self._moment(5, 23, 0), 21, 50, enabled=False))


class TestCloseThursday(unittest.TestCase):
    def _moment(self, day, hour, minute=0):
        return ItalianMoment(dt=None, day_of_week=day, hour=hour, minute=minute)

    def test_disabled_by_default(self):
        # replica il default dell'EA v27: CloseThursday = false
        self.assertFalse(should_close_thursday(self._moment(4, 23, 0), 21, 50, enabled=False))

    def test_enabled_works_like_friday_logic(self):
        self.assertTrue(should_close_thursday(self._moment(4, 21, 50), 21, 50, enabled=True))
        self.assertFalse(should_close_thursday(self._moment(4, 21, 49), 21, 50, enabled=True))


class TestServerWindow(unittest.TestCase):
    def _epoch(self, hour, minute):
        # tratta l'epoch COME l'ora del server, senza conversioni di fuso
        # (vedi docstring di is_within_server_window)
        return datetime(2026, 10, 2, hour, minute, tzinfo=timezone.utc).timestamp()

    def test_inside_window(self):
        self.assertTrue(is_within_server_window(self._epoch(15, 0), "10:00", "21:00"))

    def test_before_window(self):
        self.assertFalse(is_within_server_window(self._epoch(9, 59), "10:00", "21:00"))

    def test_after_window(self):
        self.assertFalse(is_within_server_window(self._epoch(21, 1), "10:00", "21:00"))

    def test_inclusive_bounds(self):
        self.assertTrue(is_within_server_window(self._epoch(10, 0), "10:00", "21:00"))
        self.assertTrue(is_within_server_window(self._epoch(21, 0), "10:00", "21:00"))


if __name__ == "__main__":
    unittest.main()
