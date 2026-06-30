"""
Step 1 — Data Exploration for the Multilayer Perceptron project.

Goal: understand the raw dataset BEFORE preprocessing, so every later
decision (standardization, train/valid split, network shape) is justified.

Run:  ./venv/bin/python explore.py
"""

import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Dataset path is configurable so a different dataset can be explored without
# editing code:  ./venv/bin/python explore.py --dataset data/other.csv
parser = argparse.ArgumentParser(description="Explore the breast-cancer dataset.")
parser.add_argument("--dataset", default="data/data.csv", help="input CSV path")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# 1. LOAD THE DATA
# ---------------------------------------------------------------------------
# The CSV has NO header row, so pandas would otherwise treat the first
# patient as column names. We pass header=None and build our own names.
#
# The 30 features are 10 base measurements, each reported 3 ways:
#   - the MEAN over the nucleus,
#   - the STANDARD ERROR (se),
#   - the "WORST" (mean of the 3 largest values).
BASE = ["radius", "texture", "perimeter", "area", "smoothness",
        "compactness", "concavity", "concave_points", "symmetry",
        "fractal_dimension"]
feature_names = (
    [f"{b}_mean" for b in BASE]
    + [f"{b}_se"   for b in BASE]
    + [f"{b}_worst" for b in BASE]
)
columns = ["id", "diagnosis"] + feature_names

df = pd.read_csv(args.dataset, header=None, names=columns)

# ---------------------------------------------------------------------------
# 2. BASIC SHAPE, MISSING VALUES, CLASS BALANCE
# ---------------------------------------------------------------------------
print("=" * 60)
print("SHAPE:", df.shape, "(rows = patients, cols = id + label + 30 features)")
print("=" * 60)

print("\nMISSING VALUES per column (should all be 0):")
print(int(df.isna().sum().sum()), "missing values total")

print("\nCLASS BALANCE (the label we want to predict):")
counts = df["diagnosis"].value_counts()
for label, n in counts.items():
    name = "Malignant" if label == "M" else "Benign"
    print(f"  {label} ({name}): {n}  ({100*n/len(df):.1f}%)")

# ---------------------------------------------------------------------------
# 3. FEATURE SCALES  ->  motivates STANDARDIZATION
# ---------------------------------------------------------------------------
# A neural network multiplies features by weights and sums them. If one
# feature ranges 0-2500 (area) and another 0-0.2 (smoothness), the big one
# dominates the weighted sum purely because of its units, not its importance.
print("\n" + "=" * 60)
print("FEATURE SCALES (min / mean / max) — note how different they are:")
print("=" * 60)
stats = df[feature_names].agg(["min", "mean", "max"]).T
# Show a few telling examples spanning the scale range.
for f in ["area_mean", "radius_mean", "smoothness_mean",
          "fractal_dimension_mean", "area_worst"]:
    r = stats.loc[f]
    print(f"  {f:24s} min={r['min']:9.4f}  mean={r['mean']:9.4f}  max={r['max']:9.4f}")

# ---------------------------------------------------------------------------
# 4. CLASS SEPARABILITY  ->  is prediction even possible?
# ---------------------------------------------------------------------------
# For each feature, compare its average value for Malignant vs Benign.
# A large gap means the feature carries signal about the diagnosis.
print("\n" + "=" * 60)
print("MOST DISCRIMINATIVE FEATURES (mean M vs mean B, standardized gap):")
print("=" * 60)
M = df[df["diagnosis"] == "M"][feature_names]
B = df[df["diagnosis"] == "B"][feature_names]
# Standardized mean difference (Cohen's-d-like): gap measured in std-devs.
pooled_std = df[feature_names].std()
sep = ((M.mean() - B.mean()) / pooled_std).abs().sort_values(ascending=False)
for f in sep.head(8).index:
    print(f"  {f:24s} M={M[f].mean():8.3f}  B={B[f].mean():8.3f}  gap={sep[f]:.2f} std")

# ---------------------------------------------------------------------------
# 5. PLOTS
# ---------------------------------------------------------------------------
# (a) Class balance bar chart
fig, ax = plt.subplots(figsize=(5, 4))
ax.bar(["Benign", "Malignant"], [counts.get("B", 0), counts.get("M", 0)],
       color=["#4c9f70", "#c0504d"])
ax.set_title("Class balance")
ax.set_ylabel("number of patients")
fig.tight_layout()
fig.savefig("explore_class_balance.png", dpi=110)

# (b) Distributions of the 6 most discriminative features, split by class.
top6 = sep.head(6).index
fig, axes = plt.subplots(2, 3, figsize=(13, 7))
for ax, f in zip(axes.ravel(), top6):
    ax.hist(B[f], bins=30, alpha=0.6, label="Benign", color="#4c9f70")
    ax.hist(M[f], bins=30, alpha=0.6, label="Malignant", color="#c0504d")
    ax.set_title(f)
    ax.legend()
fig.suptitle("Feature distributions by diagnosis (separation = learnable signal)")
fig.tight_layout()
fig.savefig("explore_top_features.png", dpi=110)

# (c) Raw feature scales (boxplot of the 10 *_mean features) — visual proof
#     that scales differ by orders of magnitude before standardization.
fig, ax = plt.subplots(figsize=(12, 5))
mean_feats = [f"{b}_mean" for b in BASE]
ax.boxplot([df[f] for f in mean_feats], tick_labels=BASE, showfliers=False)
ax.set_yscale("log")
ax.set_title("Raw scales of the 10 '_mean' features (log y) — why we must standardize")
ax.set_ylabel("value (log scale)")
plt.xticks(rotation=45, ha="right")
fig.tight_layout()
fig.savefig("explore_scales.png", dpi=110)

print("\nSaved plots: explore_class_balance.png, explore_top_features.png, explore_scales.png")
