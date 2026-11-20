from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sustai_energy.study import (  # noqa: E402
    build_daily_supervised,
    conformal_adjustment,
    optimise_battery,
)


class FeatureTests(unittest.TestCase):
    def test_daily_features_are_causal(self) -> None:
        index = pd.date_range("2025-01-01", periods=24 * 15, freq="h")
        series = pd.Series(np.arange(len(index), dtype=float), index=index)
        frame = build_daily_supervised(series)
        row = frame.iloc[0]
        origin_position = series.index.get_loc(row["origin_time"])
        target_position = series.index.get_loc(row["target_time"])
        self.assertGreater(target_position, origin_position)
        self.assertEqual(row["last_observation"], series.iloc[origin_position])
        self.assertEqual(
            row["same_hour_yesterday"],
            series.iloc[origin_position + int(row["horizon"]) - 24],
        )
        self.assertLessEqual(
            origin_position + int(row["horizon"]) - 24, origin_position
        )

    def test_conformal_adjustment_is_nonnegative(self) -> None:
        actual = np.array([1.0, 2.0, 5.0])
        lower = np.array([0.5, 1.5, 3.0])
        upper = np.array([1.5, 2.5, 4.0])
        self.assertGreaterEqual(conformal_adjustment(actual, lower, upper, 0.1), 0)


class OptimisationTests(unittest.TestCase):
    def test_battery_constraints_and_terminal_soc(self) -> None:
        demand = np.array([[10.0] * 24])
        price = np.array([0.1] * 8 + [0.2] * 8 + [0.4] * 8)
        carbon = np.full(24, 0.2)
        result = optimise_battery(
            demand,
            np.array([1.0]),
            price,
            carbon,
            capacity=40.0,
            power=10.0,
            efficiency=0.95,
            demand_charge=0.45,
            carbon_price=0.12,
        )
        charge = np.asarray(result["charge"])
        discharge = np.asarray(result["discharge"])
        soc = np.asarray(result["soc"])
        self.assertTrue(np.all(charge >= -1e-8))
        self.assertTrue(np.all(discharge >= -1e-8))
        self.assertTrue(np.all(demand[0] + charge - discharge >= -1e-7))
        self.assertAlmostEqual(soc[0], 20.0, places=5)
        self.assertAlmostEqual(soc[-1], 20.0, places=5)
        self.assertTrue(np.all((soc >= -1e-7) & (soc <= 40.0 + 1e-7)))


if __name__ == "__main__":
    unittest.main()

