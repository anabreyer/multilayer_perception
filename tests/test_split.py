"""
Tests for split.py — the stratified train/validation split (Step 2).

These run on the real dataset if present (data/data.csv), otherwise on a
synthetic dataframe with the same structure.

Run just this:  ./venv/bin/python -m unittest tests.test_split -v
"""

import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from split import stratified_split
from preprocessing import load_raw, COLUMNS

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "data", "data.csv")


def make_df():
    """Real data if available, else a synthetic stand-in with id/diagnosis/features."""
    if os.path.exists(DATA):
        return load_raw(DATA)
    rng = np.random.default_rng(0)
    n = 500
    labels = np.where(rng.random(n) < 0.4, "M", "B")     # ~40% M
    data = {COLUMNS[0]: np.arange(n), COLUMNS[1]: labels}
    for col in COLUMNS[2:]:
        data[col] = rng.standard_normal(n)
    return pd.DataFrame(data)


class TestStratifiedSplit(unittest.TestCase):
    def setUp(self):
        self.df = make_df()

    def test_sizes_add_up(self):
        train, valid = stratified_split(self.df, ratio=0.8, seed=42)
        self.assertEqual(len(train) + len(valid), len(self.df))
        # ~80% in train (allow rounding per class)
        self.assertAlmostEqual(len(train) / len(self.df), 0.8, delta=0.02)

    def test_no_overlap(self):
        train, valid = stratified_split(self.df, ratio=0.8, seed=42)
        shared = set(train["id"]) & set(valid["id"])
        self.assertEqual(len(shared), 0)                 # no patient in both

    def test_class_ratio_preserved(self):
        train, valid = stratified_split(self.df, ratio=0.8, seed=42)
        full_ratio = (self.df["diagnosis"] == "M").mean()
        for part in (train, valid):
            part_ratio = (part["diagnosis"] == "M").mean()
            self.assertAlmostEqual(part_ratio, full_ratio, delta=0.03)

    def test_reproducible_with_seed(self):
        t1, v1 = stratified_split(self.df, ratio=0.8, seed=42)
        t2, v2 = stratified_split(self.df, ratio=0.8, seed=42)
        np.testing.assert_array_equal(t1["id"].to_numpy(), t2["id"].to_numpy())
        np.testing.assert_array_equal(v1["id"].to_numpy(), v2["id"].to_numpy())

    def test_different_seed_changes_split(self):
        t1, _ = stratified_split(self.df, ratio=0.8, seed=42)
        t2, _ = stratified_split(self.df, ratio=0.8, seed=7)
        self.assertFalse(np.array_equal(t1["id"].to_numpy(), t2["id"].to_numpy()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
