import unittest
from unittest.mock import patch, MagicMock

from common.lot_sizing import (
    DynamicRiskInputs,
    LotSizeServiceError,
    dynamic_risk_lot,
    fixed_lot,
    martingale_lot,
    profile_table_lot,
)


class TestFixedLot(unittest.TestCase):
    def test_returns_input_unchanged(self):
        self.assertEqual(fixed_lot(0.37), 0.37)


class TestDynamicRiskLot(unittest.TestCase):
    def _inputs(self, **overrides):
        base = dict(
            balance=10_000.0,
            max_risk_pct=1.0,
            point=0.0001,
            tick_value=1.0,
            tick_size=0.0001,
            volume_min=0.01,
            volume_max=100.0,
            volume_step=0.01,
        )
        base.update(overrides)
        return DynamicRiskInputs(**base)

    def test_matches_hand_calculation(self):
        # risk_money = 10000*1% = 100
        # sl_points = |1.1050 - 1.1000| / 0.0001 = 50
        # value_per_point = tick_value/tick_size*point = 1.0/0.0001*0.0001 = 1.0
        # lots = 100 / (50*1.0) = 2.0
        lots = dynamic_risk_lot(1.1050, 1.1000, self._inputs(), fallback_lot=0.1)
        self.assertAlmostEqual(lots, 2.0, places=6)

    def test_respects_volume_step_and_floor(self):
        inputs = self._inputs(volume_step=0.5)
        lots = dynamic_risk_lot(1.1050, 1.1000, inputs, fallback_lot=0.1)
        # 2.0 e' già multiplo di 0.5, resta 2.0
        self.assertAlmostEqual(lots, 2.0, places=6)

    def test_clamped_to_volume_max(self):
        inputs = self._inputs(volume_max=1.0)
        lots = dynamic_risk_lot(1.1050, 1.1000, inputs, fallback_lot=0.1)
        self.assertEqual(lots, 1.0)

    def test_clamped_to_volume_min(self):
        inputs = self._inputs(max_risk_pct=0.0001)
        lots = dynamic_risk_lot(1.1050, 1.1000, inputs, fallback_lot=0.1)
        self.assertEqual(lots, inputs.volume_min)

    def test_no_sl_falls_back(self):
        lots = dynamic_risk_lot(1.1050, 0.0, self._inputs(), fallback_lot=0.1)
        self.assertEqual(lots, 0.1)

    def test_zero_tick_size_falls_back(self):
        inputs = self._inputs(tick_size=0.0)
        lots = dynamic_risk_lot(1.1050, 1.1000, inputs, fallback_lot=0.1)
        self.assertEqual(lots, 0.1)


class TestMartingaleLot(unittest.TestCase):
    def test_no_candidates_returns_initial(self):
        self.assertEqual(martingale_lot([], initial_lots=0.01, multiply_on_loss=1.05), 0.01)

    def test_most_recent_loss_multiplies_its_own_lots(self):
        # l'ultima operazione pertinente (0.03 lotti) era in perdita -> 0.03*1.05
        lots = martingale_lot([(-5.0, 0.03), (-2.0, 0.01)], initial_lots=0.01, multiply_on_loss=1.05)
        self.assertAlmostEqual(lots, 0.0315, places=6)

    def test_most_recent_profit_resets_to_initial(self):
        lots = martingale_lot([(12.0, 0.0315), (-5.0, 0.03)], initial_lots=0.01, multiply_on_loss=1.05)
        self.assertEqual(lots, 0.01)

    def test_exact_zero_pnl_is_skipped_like_nonexistent(self):
        # la più recente ha pnl 0.0 (ignorata, come GetBetTradesInfo): si guarda la precedente
        lots = martingale_lot([(0.0, 0.05), (-5.0, 0.03)], initial_lots=0.01, multiply_on_loss=1.05)
        self.assertAlmostEqual(lots, 0.0315, places=6)

    def test_all_zero_pnl_falls_back_to_initial(self):
        lots = martingale_lot([(0.0, 0.05), (0.0, 0.03)], initial_lots=0.01, multiply_on_loss=1.05)
        self.assertEqual(lots, 0.01)


class TestProfileTableLot(unittest.TestCase):
    @patch("common.lot_sizing.requests.get")
    def test_happy_path(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"status": "ok", "lot_size": 0.08}
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp

        lots = profile_table_lot("https://example.com", "alto", "EURUSD")
        self.assertEqual(lots, 0.08)
        mock_get.assert_called_once()

    @patch("common.lot_sizing.requests.get")
    def test_not_found_without_fallback_raises(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"status": "not_found", "lot_size": None}
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp

        with self.assertRaises(LotSizeServiceError):
            profile_table_lot("https://example.com", "alto", "USDCAD")

    @patch("common.lot_sizing.requests.get")
    def test_not_found_with_fallback_returns_fallback(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"status": "not_found", "lot_size": None}
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp

        lots = profile_table_lot("https://example.com", "alto", "USDCAD", fallback_lot=0.01)
        self.assertEqual(lots, 0.01)

    @patch("common.lot_sizing.requests.get")
    def test_network_error_without_fallback_raises(self, mock_get):
        mock_get.side_effect = ConnectionError("boom")
        with self.assertRaises(LotSizeServiceError):
            profile_table_lot("https://example.com", "alto", "EURUSD")

    @patch("common.lot_sizing.requests.get")
    def test_network_error_with_fallback_returns_fallback(self, mock_get):
        mock_get.side_effect = ConnectionError("boom")
        lots = profile_table_lot("https://example.com", "alto", "EURUSD", fallback_lot=0.02)
        self.assertEqual(lots, 0.02)


if __name__ == "__main__":
    unittest.main()
