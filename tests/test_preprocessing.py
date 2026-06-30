"""
Tests for preprocessing.py — label encoding and standardization (Step 2).

Run all tests:   make test
Run just this:   ./venv/bin/python -m unittest tests.test_preprocessing -v
"""

import os
import sys
import unittest

import numpy as np

# Make the project root importable when running this file directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing import (encode_labels, compute_standardization, standardize,
                           load_raw, split_features_labels, FEATURE_NAMES)


class TestEncodeLabels(unittest.TestCase):
    def test_mapping_and_shape(self):
        y = np.array(["B", "M", "B", "M"])
        onehot = encode_labels(y)
        self.assertEqual(onehot.shape, (4, 2))          # (n, 2) for softmax
        np.testing.assert_array_equal(onehot[0], [1, 0])  # B -> [1,0]
        np.testing.assert_array_equal(onehot[1], [0, 1])  # M -> [0,1]

    def test_rows_are_probability_distributions(self):
        y = np.array(["M", "B", "M"])
        onehot = encode_labels(y)
        np.testing.assert_allclose(onehot.sum(axis=1), 1.0)  # each row sums to 1


class TestStandardization(unittest.TestCase):
    def test_train_becomes_mean0_std1(self):
        rng = np.random.default_rng(0)
        X = rng.normal(loc=50, scale=7, size=(200, 5))   # arbitrary scale
        mu, sigma = compute_standardization(X)
        Xs = standardize(X, mu, sigma)
        np.testing.assert_allclose(Xs.mean(axis=0), 0.0, atol=1e-12)
        np.testing.assert_allclose(Xs.std(axis=0), 1.0, atol=1e-12)

    def test_no_leakage_valid_uses_train_stats(self):
        # mu/sigma come from TRAIN only; applying them to VALID must NOT
        # force valid's mean to 0 — proving train stats are reused, not recomputed.
        rng = np.random.default_rng(1)
        X_train = rng.normal(10, 2, size=(100, 3))
        X_valid = rng.normal(30, 5, size=(40, 3))        # different distribution
        mu, sigma = compute_standardization(X_train)
        X_valid_s = standardize(X_valid, mu, sigma)
        # valid mean should be far from 0 because it used train's mu, not its own
        self.assertTrue(np.all(np.abs(X_valid_s.mean(axis=0)) > 1.0))

    def test_constant_feature_does_not_divide_by_zero(self):
        X = np.ones((10, 2))                             # zero variance
        mu, sigma = compute_standardization(X)
        self.assertTrue(np.all(sigma == 1.0))            # guarded to 1
        self.assertFalse(np.any(np.isnan(standardize(X, mu, sigma))))


class TestLoading(unittest.TestCase):
    DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "data", "data.csv")

    @unittest.skipUnless(os.path.exists(DATA), "data/data.csv not present")
    def test_shapes_from_real_file(self):
        df = load_raw(self.DATA)
        self.assertEqual(df.shape[1], 32)                # id + label + 30 features
        X, y = split_features_labels(df)
        self.assertEqual(X.shape[1], 30)
        self.assertEqual(len(FEATURE_NAMES), 30)
        self.assertTrue(set(np.unique(y)).issubset({"M", "B"}))


if __name__ == "__main__":
    unittest.main(verbosity=2)
