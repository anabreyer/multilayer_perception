"""
Shared preprocessing utilities used by the split, train, and predict programs.

Keeps three concerns in one place so every program treats the data identically:
  - loading the headerless CSV with correct column names,
  - one-hot encoding the M/B label for the 2-output softmax,
  - standardization (z-score) with TRAIN-ONLY statistics (no data leakage).
"""

import numpy as np
import pandas as pd

# The 10 base measurements, each reported as _mean, _se, _worst -> 30 features.
BASE = ["radius", "texture", "perimeter", "area", "smoothness",
        "compactness", "concavity", "concave_points", "symmetry",
        "fractal_dimension"]
FEATURE_NAMES = (
    [f"{b}_mean" for b in BASE]
    + [f"{b}_se" for b in BASE]
    + [f"{b}_worst" for b in BASE]
)
COLUMNS = ["id", "diagnosis"] + FEATURE_NAMES

# Label convention: B = class 0, M = class 1 (M is the "positive" class).
CLASS_TO_INDEX = {"B": 0, "M": 1}


def load_raw(path):
    """Read the headerless CSV and attach our column names."""
    return pd.read_csv(path, header=None, names=COLUMNS)


def split_features_labels(df):
    """Return X (n, 30) float matrix and y (n,) array of 'M'/'B' strings."""
    X = df[FEATURE_NAMES].to_numpy(dtype=float)
    y = df["diagnosis"].to_numpy()
    return X, y


def encode_labels(y):
    """
    One-hot encode 'M'/'B' into shape (n, 2) for the softmax output.
      B -> [1, 0]   (column 0)
      M -> [0, 1]   (column 1)
    A one-hot row is just a probability distribution that puts 100% on the
    true class, so it can be compared directly against the softmax output.
    """
    idx = np.array([CLASS_TO_INDEX[label] for label in y])
    onehot = np.zeros((len(y), 2))
    onehot[np.arange(len(y)), idx] = 1.0
    return onehot


def compute_standardization(X):
    """
    Learn the per-feature mean and std from the TRAINING data only.
    Returns (mu, sigma), each shape (30,). These are saved with the model
    so validation/test data get the exact same transform (no leakage).
    """
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma[sigma == 0] = 1.0   # guard: a constant feature would divide by zero
    return mu, sigma


def standardize(X, mu, sigma):
    """Apply the z-score transform z = (x - mu) / sigma column-wise."""
    return (X - mu) / sigma
