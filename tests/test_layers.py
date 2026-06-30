"""
Tests for layers.py — the DenseLayer (Step 3).

Run just this:  ./venv/bin/python -m unittest tests.test_layers -v
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from layers import DenseLayer


class TestDenseLayer(unittest.TestCase):
    def test_parameter_shapes_and_zero_bias(self):
        layer = DenseLayer(30, 24, rng=np.random.default_rng(0))
        self.assertEqual(layer.W.shape, (30, 24))        # (n_in, n_out)
        self.assertEqual(layer.b.shape, (1, 24))         # (1, n_out)
        self.assertTrue(np.all(layer.b == 0))            # biases start at 0

    def test_he_uniform_within_bounds(self):
        n_in = 30
        layer = DenseLayer(n_in, 24, initializer="heUniform",
                           rng=np.random.default_rng(0))
        limit = np.sqrt(6.0 / n_in)
        self.assertTrue(np.all(np.abs(layer.W) <= limit))

    def test_weights_not_all_equal(self):
        # Symmetry must be broken: weights are random, not constant.
        layer = DenseLayer(10, 10, rng=np.random.default_rng(0))
        self.assertGreater(np.unique(layer.W).size, 1)

    def test_reproducible_with_seed(self):
        # Same seed -> identical weights (needed for reproducible training).
        a = DenseLayer(8, 4, rng=np.random.default_rng(123))
        b = DenseLayer(8, 4, rng=np.random.default_rng(123))
        np.testing.assert_array_equal(a.W, b.W)

    def test_forward_shape_and_sigmoid_range(self):
        rng = np.random.default_rng(0)
        layer = DenseLayer(30, 24, activation="sigmoid", rng=rng)
        X = rng.standard_normal((5, 30))                 # batch of 5
        A = layer.forward(X)
        self.assertEqual(A.shape, (5, 24))
        self.assertTrue(np.all((A > 0) & (A < 1)))       # sigmoid outputs

    def test_forward_caches_for_backprop(self):
        rng = np.random.default_rng(0)
        layer = DenseLayer(4, 3, rng=rng)
        X = rng.standard_normal((6, 4))
        layer.forward(X)
        # Backprop (Step 6) needs these cached intermediates.
        np.testing.assert_array_equal(layer.A_prev, X)
        self.assertEqual(layer.Z.shape, (6, 3))
        self.assertEqual(layer.A.shape, (6, 3))

    def test_softmax_output_layer_sums_to_one(self):
        rng = np.random.default_rng(0)
        out = DenseLayer(24, 2, activation="softmax", rng=rng)
        A = rng.random((5, 24))
        P = out.forward(A)
        self.assertEqual(P.shape, (5, 2))
        np.testing.assert_allclose(P.sum(axis=1), 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
