# Multilayer Perceptron — Project Plan

**Goal:** Build an MLP from scratch (Python + NumPy) to classify breast tumors as
Malignant (M) or Benign (B) from 30 cell-nucleus features. Learn the math well
enough to explain feedforward, backpropagation, and gradient descent in the eval.

**Language:** Python + NumPy (linalg + matplotlib only — no ML libraries).
**Explanation depth:** Deep math + code, step by step. Pause at each step for notes.

## Dataset facts
- 569 rows, 32 columns, NO header row.
- Col 1 = ID (drop). Col 2 = diagnosis label (M/B). Cols 3–32 = 30 numeric features.
- 30 features = 10 measurements × {mean, standard error, worst}.
- Class balance: 357 B / 212 M.
- Raw data → MUST standardize features before training.

## Build steps (pause + explain at each)
- [x] 1. Data exploration — load, plot feature distributions M vs B, understand data
      Findings: 569×32, 0 missing, 357B/212M (62.7%/37.3% → baseline 62.7%).
      Scales span 4 orders of magnitude (area~1000 vs smoothness~0.1) → MUST standardize.
      Top discriminators: concave_points_worst, perimeter_worst, radius_worst (size+irregularity).
- [x] 2. Preprocessing + split program — encode labels, standardize, train/valid split (seeded)
      preprocessing.py (load/encode/standardize) + split.py (Program 1/3).
      80/20 stratified seeded split -> data_train.csv (456) / data_valid.csv (113).
      Verified: stratified, reproducible (md5), zero overlap. Standardize = train-only μ,σ (no leakage).
- [x] 3. Building blocks — DenseLayer, weight init (He), activations (sigmoid, softmax)
      activations.py (sigmoid+deriv, softmax, stable forms) + layers.py (DenseLayer, He init, forward).
      Convention: samples as rows; Z = A_prev@W + b. He uniform ±√(6/n_in), bias=0.
      Verified: sigmoid deriv vs numerical match 10 decimals; softmax sums to 1; shapes correct.
- [ ] 4. Feedforward — forward pass to get predictions
- [ ] 5. Loss — categorical / binary cross-entropy
- [ ] 6. Backpropagation — gradients via chain rule (hardest math)
- [ ] 7. Gradient descent + training loop — epochs, batches, lr, loss/accuracy curves, save model
- [ ] 8. Prediction program — load weights, predict, report binary cross-entropy
- [ ] 9. (Bonus, optional) Adam, early stopping, extra metrics, multi-curve compare

## Dataset location & swapping for evaluation
- Dataset lives in `data/data.csv` (all programs default to this path).
- To use a different dataset given at eval, EITHER:
  - drop it in as `data/data.csv` and re-run, OR
  - override the path: `make split DATASET=data/their_file.csv`
- Swap flow:  `make fclean` (or delete split CSVs) -> `make split [DATASET=...]`
  -> `make train` -> `make predict`. Only split reads the raw dataset;
  train/predict read the regenerated data_train.csv / data_valid.csv.

## Tests (run `make test`)
- tests/ uses built-in unittest (no extra deps). 26 tests passing so far.
- test_preprocessing (encode, standardize, no-leakage), test_activations
  (sigmoid+deriv vs numerical, softmax), test_layers (shapes, He, seed),
  test_split (no overlap, stratified, reproducible).
- Add matching tests as each step is built (feedforward, loss, gradient-check backprop).

## Deliverables (per spec)
1. Split program  2. Training program  3. Prediction program
(or one program with a mode switch)

## Review
_(to be filled in as we go)_
