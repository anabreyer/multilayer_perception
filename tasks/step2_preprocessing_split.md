# Step 2 — Preprocessing & the Split Program

> **Eval goal:** explain label encoding, standardization (and *why train-only
> statistics*), and the train/validation split — and defend each choice.

Files: [`preprocessing.py`](../preprocessing.py) (shared module),
[`split.py`](../split.py) (Program 1 of 3).
Outputs: `data_train.csv` (456 rows), `data_valid.csv` (113 rows).

---

## ① Label encoding — `M`/`B` → one-hot

The output layer is **2 neurons + softmax**, producing `[P(benign), P(malignant)]`.
The true label must be the same shape, so we one-hot encode:

$$ B \rightarrow [1, 0] \qquad M \rightarrow [0, 1] $$

Convention fixed in code: **B = class 0, M = class 1** (M is the "positive" class —
the thing we detect; this matters for binary cross-entropy in prediction).
A one-hot row is a probability distribution that puts 100% on the true class,
so it can be compared directly to the softmax output.

Code: `encode_labels(y)` in `preprocessing.py`.

---

## ② Standardization — the z-score

Step 1 proved features span ~4 orders of magnitude. We rescale each feature:

$$ z = \frac{x - \mu}{\sigma} $$

- `μ` = the feature's mean, `σ` = its standard deviation.
- After this every feature has **mean 0, std 1**, so none dominates the weighted
  sum `z = Σ xₖwₖ + b` just because of its units.

### ⚠️ The data-leakage rule (key eval point)
Compute `μ` and `σ` from the **training set ONLY**, then reuse those exact
values to standardize the validation set (and later any test set). Computing them
on the full dataset would leak information about validation samples into the
preprocessing, inflating the validation score — a classic mistake.

Consequence: `μ, σ` are **learned parameters saved with the model**, so the
prediction program applies the identical transform. This is why the **split
program keeps the data raw** and standardization is applied at training time.

Code: `compute_standardization(X_train)` → `(mu, sigma)`; `standardize(X, mu, sigma)`.

---

## ③ Train / validation split — why hold data out

Judging a model on data it trained on rewards memorization and hides
**overfitting**. So we hold out a validation set the model never trains on — an
honest estimate of performance on unseen patients.

Design choices and their justification:

| Choice | Value | Why |
|--------|-------|-----|
| Shuffle before split | yes | the raw file may be ordered |
| Seed | 42 (configurable) | reproducible split (spec allows a seed) |
| Ratio | 80% / 20% → 456 / 113 | standard; enough train data, meaningful valid set |
| Stratified | yes | data is 62.7% B / 37.3% M; keeps that ratio in both sets so validation isn't skewed |

`stratified_split` shuffles **each class separately**, takes the first `ratio`
fraction of each for training, then re-shuffles the combined sets so classes are
interleaved.

---

## Verification performed
- **Stratified:** train 62.7% B / 37.3% M, valid 62.8% B / 37.2% M (matches original).
- **Reproducible:** identical MD5 checksums when re-run with the same seed.
- **No overlap:** 0 patient IDs shared between `data_train.csv` and `data_valid.csv`.

---

## How the pieces will be used later
- `split.py` → produces the two CSVs (run once).
- Training program: `load_raw` → `split_features_labels` → `encode_labels` for y;
  `compute_standardization` on train X, `standardize` on train & valid X; save
  `μ, σ` in the model file.
- Prediction program: load `μ, σ` from the model, `standardize` the input, predict.
