"""
Activation functions and their derivatives.

An activation function is applied element-wise to a layer's weighted sums Z.
It introduces non-linearity: without it, stacking layers would collapse into a
single linear map and the network could not learn curved decision boundaries.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Sigmoid  -  used on the hidden layers
# ---------------------------------------------------------------------------
def sigmoid(z):
    """
    sigmoid(z) = 1 / (1 + e^-z), squashing any real number into (0, 1).

    Numerically stable: e^-z overflows for very negative z, so we use the
    algebraically identical form e^z / (1 + e^z) where z < 0.
    """
    pos = z >= 0
    out = np.empty_like(z, dtype=float)
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    exp_z = np.exp(z[~pos])
    out[~pos] = exp_z / (1.0 + exp_z)
    return out


def sigmoid_derivative(z):
    """
    sigmoid'(z) = sigmoid(z) * (1 - sigmoid(z)).
    Needed during backpropagation to send the error gradient back through a
    sigmoid layer.
    """
    s = sigmoid(z)
    return s * (1.0 - s)


# ---------------------------------------------------------------------------
# Softmax  -  used on the OUTPUT layer (2 neurons -> probability distribution)
# ---------------------------------------------------------------------------
def softmax(z):
    """
    softmax(z)_i = e^z_i / sum_j e^z_j, computed per row (per sample).

    Converts a row of raw scores into a probability distribution that is all
    positive and sums to 1. We subtract each row's max before exponentiating
    (shift-invariance of softmax) to avoid overflow of e^z for large z.
    """
    z_shift = z - np.max(z, axis=1, keepdims=True)
    exp = np.exp(z_shift)
    return exp / np.sum(exp, axis=1, keepdims=True)


# Lookup tables so a layer can refer to its activation by name (as in the
# subject's config-style API: activation='sigmoid', activation='softmax').
ACTIVATIONS = {
    "sigmoid": sigmoid,
    "softmax": softmax,
}
DERIVATIVES = {
    "sigmoid": sigmoid_derivative,
    # softmax's derivative is handled jointly with the loss in backprop
    # (it collapses to the clean term (y_hat - y)); see Step 6.
}


def activate(z, name):
    """Apply the named activation function to Z."""
    return ACTIVATIONS[name](z)
