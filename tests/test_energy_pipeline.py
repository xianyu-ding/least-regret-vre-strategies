"""Offline tests for the supplied-data figure inputs; never calls an API."""
import importlib.util
from pathlib import Path
import sys
import unittest
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from energy_data import (load_mcdm_data, numeric, is_cost_indicator,
                         vector_normalize, validate_combined, SCENARIOS)

class EnergyInputTests(unittest.TestCase):
    def test_supplied_nine_criteria(self):
        data = load_mcdm_data(ROOT)
        self.assertEqual(data.shape, (6, 10))
        self.assertEqual(data.Scenario.tolist(), SCENARIOS)
        self.assertTrue(np.isfinite(data.drop(columns='Scenario').to_numpy(dtype=float)).all())
        self.assertTrue(pd.api.types.is_numeric_dtype(data.Cost))
        self.assertAlmostEqual(data.Cost.iloc[0], 441320618646.37, places=2)

    def test_comma_grouped_cost(self):
        self.assertEqual(numeric(pd.Series(['441,320,618,646.37']), 'Cost').iloc[0], 441320618646.37)

    def test_invalid_numeric_fails(self):
        for value in ['not-a-number', 'NaN', 'inf', '-1']:
            with self.subTest(value=value), self.assertRaises(ValueError):
                numeric(pd.Series([value]), 'Cost')

    def test_missing_required_co2_fails(self):
        if (ROOT/'data/raw/environment/co2_df.csv').exists():
            self.assertEqual(load_mcdm_data(ROOT, include_co2=True).shape, (6, 11))
        else:
            with self.assertRaisesRegex(FileNotFoundError, 'co2_df.csv'):
                load_mcdm_data(ROOT, include_co2=True)

    def test_missing_all_inputs_fail(self):
        with self.assertRaisesRegex(FileNotFoundError, 'lole_df.csv'):
            load_mcdm_data(ROOT/'tests'/'nonexistent_input_root')

    def test_explicit_direction_including_space(self):
        self.assertTrue(is_cost_indicator('Land occupation_indicator'))
        self.assertTrue(is_cost_indicator('land_occupation_indicator'))
        with self.assertRaises(ValueError):
            is_cost_indicator('unknown_indicator')

    def test_zero_vector_is_finite(self):
        self.assertTrue(np.isfinite(vector_normalize([0,0,0])).all())
        np.testing.assert_array_equal(vector_normalize([0,0,0]), [1,1,1])

    def test_combined_no_silent_missing_scenarios(self):
        data = load_mcdm_data(ROOT)
        validated = validate_combined(data)
        self.assertEqual(validated.scenario.tolist(), SCENARIOS)
        with self.assertRaises(ValueError):
            validate_combined(data.iloc[:-1])

if __name__ == '__main__':
    unittest.main()
