"""
DenseLayer — a fully connected (dense) layer: every input is connected to
every neuron in the layer.

A layer stores its learnable parameters (weight matrix W and bias vector b)
and knows how to produce its output from the previous layer's output.

Shape convention (one sample per row, m = batch size):
    A_prev : (m, n_in)
    W      : (n_in, n_out)
    b      : (1, n_out)
    Z = A_prev @ W + b   ->  (m, n_out)
    A = activation(Z)    ->  (m, n_out)
"""

import numpy as np

from activations import activate


class DenseLayer:
    def __init__(self, n_in, n_out, activation="sigmoid",
                 initializer="heUniform", rng=None):
        self.n_in = n_in
        self.n_out = n_out
        self.activation = activation
        rng = rng if rng is not None else np.random.default_rng()

        self.W = self._init_weights(n_in, n_out, initializer, rng)
        self.b = np.zeros((1, n_out))          # biases start at 0

        # Caches filled during the forward pass, reused by backprop (Step 6).
        self.A_prev = None
        self.Z = None
        self.A = None

    @staticmethod
    def _init_weights(n_in, n_out, initializer, rng):
        """
        Random initialization that breaks symmetry while keeping the signal
        variance controlled (scaled by fan_in = n_in).

        heUniform : U(-sqrt(6/n_in), +sqrt(6/n_in))
        heNormal  : N(0, 2/n_in)
        """
        if initializer == "heUniform":
            limit = np.sqrt(6.0 / n_in)
            return rng.uniform(-limit, limit, size=(n_in, n_out))
        if initializer == "heNormal":
            std = np.sqrt(2.0 / n_in)
            return rng.normal(0.0, std, size=(n_in, n_out))
        raise ValueError(f"unknown initializer: {initializer}")

    def forward(self, A_prev):
        """
        Compute this layer's output and cache the intermediate values that
        backprop will need. Returns A = activation(A_prev @ W + b).
        """
        self.A_prev = A_prev
        self.Z = A_prev @ self.W + self.b
        self.A = activate(self.Z, self.activation)
        return self.A
