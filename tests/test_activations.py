"""
Tests for activations.py — sigmoid, its derivative, and softmax (Step 3).

Run just this:  ./venv/bin/python -m unittest tests.test_activations -v
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from activations import sigmoid, sigmoid_derivative, softmax


class TestSigmoid(unittest.TestCase):
    def test_range_and_midpoint(self):
        z = np.array([[-2.0, -0.5, 0.0, 0.5, 2.0]])
        s = sigmoid(z)
        self.assertTrue(np.all((s >= 0) & (s <= 1)))     # squashed to [0,1]
        self.assertTrue(np.all(np.diff(s) > 0))          # monotonic increasing
        self.assertTrue(np.isclose(sigmoid(np.array([[0.0]]))[0, 0], 0.5))

    def test_numerically_stable_at_extremes(self):
        # Must not overflow / produce NaN for very large |z|.
        z = np.array([[-1000.0, 1000.0]])
        s = sigmoid(z)
        self.assertFalse(np.any(np.isnan(s)))
        np.testing.assert_allclose(s, [[0.0, 1.0]])      # saturates cleanly

    def test_derivative_matches_numerical_gradient(self):
        # Gold-standard check: analytic derivative vs central finite difference.
        z = np.array([[-1.3, 0.0, 0.7, 2.0]])
        eps = 1e-6
        numerical = (sigmoid(z + eps) - sigmoid(z - eps)) / (2 * eps)
        analytic = sigmoid_derivative(z)
        np.testing.assert_allclose(analytic, numerical, atol=1e-7)

    def test_derivative_max_at_zero(self):
        # sigmoid'(0) = 0.25 is the maximum slope.
        self.assertTrue(np.isclose(sigmoid_derivative(np.array([[0.0]]))[0, 0], 0.25))


class TestSoftmax(unittest.TestCase):
    def test_rows_sum_to_one_and_positive(self):
        z = np.array([[2.0, 1.0], [-3.0, 4.0], [0.0, 0.0]])
        p = softmax(z)
        np.testing.assert_allclose(p.sum(axis=1), 1.0)
        self.assertTrue(np.all(p > 0))

    def test_equal_scores_give_uniform(self):
        p = softmax(np.array([[5.0, 5.0]]))
        np.testing.assert_allclose(p, [[0.5, 0.5]])

    def test_shift_invariance(self):
        # softmax(z) == softmax(z + c): adding a constant changes nothing.
        z = np.array([[1.0, 2.0, 3.0]])
        np.testing.assert_allclose(softmax(z), softmax(z + 100.0))

    def test_stable_for_large_values(self):
        p = softmax(np.array([[1000.0, 999.0]]))
        self.assertFalse(np.any(np.isnan(p)))
        np.testing.assert_allclose(p.sum(axis=1), 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
