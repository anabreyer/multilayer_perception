# Step 1 — Data Exploration

> **Eval goal of this step:** be able to describe the dataset and *justify* every
> preprocessing decision (especially standardization) with evidence from the data.

Script: [`explore.py`](../explore.py) · Plots: `explore_scales.png`,
`explore_top_features.png`, `explore_class_balance.png`

---

## Why we explore before building anything

You cannot preprocess data you do not understand, and you cannot defend your
preprocessing in the eval without evidence. Exploration answers four questions,
each of which drives a later decision:

| Question | Decision it drives |
|----------|-------------------|
| What are the features and their scales? | **Whether/how to standardize** |
| Are there missing values? | Whether we need cleaning |
| Is the data balanced (M vs B)? | Whether accuracy is a fair metric |
| Which features separate M from B? | Sanity check that learning is possible |

---

## The dataset

- **569 patients (rows) × 32 columns.** No header row in the CSV.
- **Column 1** = `id` (patient ID — carries no predictive info, we **drop** it).
- **Column 2** = `diagnosis` — the **label** we predict: `M` (malignant) or `B` (benign).
- **Columns 3–32** = **30 numeric features**.

The 30 features are **10 base measurements of the cell nucleus**, each reported **3 ways**:

```
10 measurements: radius, texture, perimeter, area, smoothness,
                 compactness, concavity, concave_points, symmetry, fractal_dimension
3 statistics each: _mean , _se (standard error) , _worst (mean of 3 largest)
10 × 3 = 30 features
```

These come from a digitized image of a **fine-needle aspiration (FNA)** of a breast mass.

---

## Findings (the real numbers)

### 1. Shape & integrity
- **0 missing values** across the whole table → no cleaning / imputation needed.

### 2. Class balance
| Label | Meaning | Count | Percent |
|-------|---------|-------|---------|
| B | Benign | 357 | 62.7% |
| M | Malignant | 212 | 37.3% |

> **Eval point — the baseline.** Because it is not 50/50, a dumb model that
> *always predicts Benign* already scores **62.7% accuracy**. Our network must
> beat this convincingly. This is also why we monitor **loss**, not accuracy alone.

### 3. Feature scales → motivates standardization
Sample of raw ranges:

| Feature | min | mean | max |
|---|---|---|---|
| area_mean | 143.5 | 654.9 | 2501.0 |
| radius_mean | 6.98 | 14.13 | 28.11 |
| smoothness_mean | 0.053 | 0.096 | 0.163 |
| fractal_dimension_mean | 0.050 | 0.063 | 0.097 |
| area_worst | 185.2 | 880.6 | 4254.0 |

The boxplot `explore_scales.png` (note the **logarithmic** y-axis) shows the
`_mean` features span **~4 orders of magnitude**: `area` ≈ 145–2500 while
`smoothness`/`fractal_dimension` ≈ 0.05–0.3.

**Why this matters mathematically.** A neuron computes a weighted sum:

$$ z = \sum_k x_k \, w_k + b $$

If `area` (~1000) and `smoothness` (~0.1) start with equal weights, `area`
contributes **~10,000×** more to `z` — purely because of its **units**, not its
importance. The network would effectively ignore small-scale features and
training would be slow/unstable.

> ➡️ **This is the evidence that justifies standardization in Step 2**:
> transform every feature to mean 0, std 1 via `z = (x − μ) / σ`.

### 4. Class separability → is prediction even possible?
For each feature we compared the average value for Malignant vs Benign, measured
in standard deviations (a Cohen's-d-style "gap"). Top discriminators:

| Feature | Mean (Malignant) | Mean (Benign) | Gap (std) |
|---|---|---|---|
| concave_points_worst | 0.182 | 0.074 | 1.64 |
| perimeter_worst | 141.37 | 87.01 | 1.62 |
| concave_points_mean | 0.088 | 0.026 | 1.60 |
| radius_worst | 21.14 | 13.38 | 1.60 |
| perimeter_mean | 115.37 | 78.08 | 1.53 |
| area_worst | 1422.29 | 558.90 | 1.52 |

The histograms in `explore_top_features.png` show these classes are **clearly
shifted apart** (signal is real and learnable) but **overlap in the middle**
(not trivially separable by one feature → we need a multi-feature, multi-layer
model). Pattern: **larger, more irregular nuclei → malignant**, matching medical
intuition (cancer cells are bigger with irregular boundaries).

---

## Decisions carried into the next steps
1. **No missing-data handling needed.**
2. **Standardization is mandatory** (proven by the scale plot) — Step 2.
3. **Monitor loss + accuracy together**, since 62.7% accuracy is a free baseline.

---

## What `explore.py` does, block by block
1. **Load** the CSV with `header=None` and assign our own 32 column names.
2. **Integrity**: print shape, count missing values.
3. **Class balance**: `value_counts()` on the `diagnosis` column.
4. **Scales**: aggregate min/mean/max per feature to expose the unit mismatch.
5. **Separability**: standardized mean difference per feature, sorted, to find
   the most discriminative ones.
6. **Plots**: class-balance bar chart, per-class histograms of the top features,
   and a log-scale boxplot of raw feature scales.
