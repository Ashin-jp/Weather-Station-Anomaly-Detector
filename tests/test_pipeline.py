import unittest
import numpy as np
import pandas as pd
from src.data_generator import ensure_dataset, apply_fault, FAULTS
from src.features import extract_features, validate_data, FEATURES
from src.model import load_model, audit_data


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = ensure_dataset().query("date >= '2023-01-01' and date <= '2023-03-01'").reset_index(drop=True)
        cls.model = load_model()

    def test_features_and_raw_preservation(self):
        for mode in FAULTS:
            with self.subTest(mode=mode):
                data = apply_fault(self.data, 'NY_JFK', '2023-01-20', mode, .4 if mode == 'drift' else 25, 8)
                original = data.copy(deep=True)
                features = extract_features(data)
                self.assertEqual(list(features.columns), FEATURES)
                self.assertTrue(np.isfinite(features.to_numpy()).all())
                result = audit_data(data, self.model)
                pd.testing.assert_frame_equal(original, data)
                pd.testing.assert_frame_equal(result[original.columns], original)
                self.assertTrue((result.final_alert == (result.rule_flag | result.ml_flag)).all())

    def test_single_station_and_all_missing(self):
        data = self.data[self.data.station_id == 'NY_JFK'].copy()
        data[['tmax', 'tmin', 'rainfall']] = np.nan
        self.assertTrue(np.isfinite(extract_features(data)).all().all())
        self.assertTrue(audit_data(data, self.model).rule_flag.all())
        self.assertEqual(len(extract_features(data.head(1))), 1)

    def test_gap_resets_runs(self):
        data = self.data[self.data.station_id == 'NY_JFK'].head(5).copy()
        data['tmax'] = 28.
        data = data.drop(data.index[2])
        self.assertEqual(extract_features(data).tmax_consecutive_identical.tolist(), [1, 2, 1, 2])

    def test_schema_rejections(self):
        with self.assertRaises(ValueError):
            validate_data(pd.concat([self.data, self.data]))
        with self.assertRaises(ValueError):
            validate_data(self.data.drop(columns='rainfall'))
        data = self.data.copy()
        data.loc[0, 'tmax'] = np.inf
        with self.assertRaises(ValueError):
            validate_data(data)

    def test_scenario_isolation_and_order(self):
        data = self.data.copy()
        data['scenario_id'] = 'first'
        other = data.copy()
        other['scenario_id'] = 'second'
        other['tmax'] += 100
        combined = pd.concat([data, other], ignore_index=True)
        pd.testing.assert_frame_equal(extract_features(data), extract_features(combined).iloc[:len(data)])
        shuffled = self.data.sample(frac=1, random_state=10)
        pd.testing.assert_frame_equal(extract_features(self.data), extract_features(shuffled).sort_index())


if __name__ == '__main__':
    unittest.main()
