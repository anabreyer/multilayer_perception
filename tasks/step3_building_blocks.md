# Step 3 — Network Building Blocks

> **Eval goal:** explain the layer/weight/bias matrices and their shapes, why
> weight initialization matters (symmetry & saturation), He initialization, and
> the sigmoid & softmax activation functions (with derivatives).

Files: [`activations.py`](../activations.py), [`layers.py`](../layers.py).

---

## Shape convention (one sample per row, m = batch size)

| Object | Symbol | Shape |
|--------|--------|-------|
| Input activations | A^[l-1] | (m, n_in) |
| Weight matrix | W^[l] | (n_in, n_out) |
| Bias vector | b^[l] | (1, n_out) |
| Pre-activation | Z^[l] | (m, n_out) |
| Output | A^[l] | (m, n_out) |

Core layer equation:

$$ Z^{[l]} = A^{[l-1]} W^{[l]} + b^{[l]}, \qquad A^{[l]} = f(Z^{[l]}) $$

One matrix multiply computes the weighted sum `Σ xₖwₖ + b` for **every neuron
and every sample at once**.

---

## ① Weight initialization

**Why not zeros?** Identical weights → every neuron in the layer computes the
same Z, gets the same gradient, updates identically → they stay clones forever
(**symmetry problem**). The layer could only represent one feature. Need random
weights to break symmetry.

**Why not large random?** Large Z pushes sigmoid into its flat tails (output ≈ 0
or 1) where the derivative ≈ 0 → **vanishing gradients**, learning stalls.

**He initialization** scales spread by `fan_in = n_in` to keep signal variance
stable across layers. Subject uses **He uniform**:

$$ W \sim \mathcal{U}\!\left(-\sqrt{\tfrac{6}{n_{in}}},\; +\sqrt{\tfrac{6}{n_{in}}}\right) $$

Biases initialized to **0** (weights already broke symmetry).

> Nuance: He init is derived for ReLU; **Xavier/Glorot** `sqrt(6/(n_in+n_out))`
> is the sigmoid-tuned variant. We follow the subject's choice of heUniform.

Code: `DenseLayer._init_weights` (supports `heUniform`, `heNormal`).

---

## ② Sigmoid (hidden layers)

$$ \sigma(z) = \frac{1}{1+e^{-z}} \in (0,1) $$
$$ \sigma'(z) = \sigma(z)\,(1-\sigma(z)) $$

- Squashes any real number to (0,1); `σ(0)=0.5`.
- Numerically stable implementation: use `e^z/(1+e^z)` for z<0 to avoid `e^-z`
  overflow.
- Saturates to exactly 0/1 at large |z| in floating point — this *is* the
  vanishing-gradient cause.

Code: `sigmoid`, `sigmoid_derivative` in `activations.py`.

---

## ③ Softmax (output layer, 2 neurons)

$$ \text{softmax}(z)_i = \frac{e^{z_i}}{\sum_j e^{z_j}} $$

- Converts a row of scores into a probability distribution (positive, sums to 1)
  → output reads as `[P(benign), P(malignant)]`.
- Stable form: subtract `max(z)` per row before exponentiating (softmax is
  shift-invariant) to avoid overflow.
- Its derivative is handled jointly with the loss in backprop, where it
  collapses to the clean term `(ŷ − y)` — see Step 6.

Code: `softmax` in `activations.py`.

---

## Verification performed (scratchpad test_step3.py)
- Sigmoid: in [0,1], monotonic, `σ(0)=0.5`, no overflow at ±1000, saturates to 0/1.
- Sigmoid derivative: **analytic vs numerical gradient match to 10 decimals**
  (0.2217128733) — the gold-standard derivative check, reused for backprop.
- Softmax: rows sum to 1, all positive, equal scores → 50/50, stable at [1000,999].
- DenseLayer: W (30,24), b (1,24) zero, weights within ±√(6/30)=±0.4472,
  forward (5,30)→(5,24)→(5,2) with final probs summing to 1.

---

## What's used later
- Step 4 (feedforward): chain `DenseLayer.forward` across all layers.
- Step 6 (backprop): uses `sigmoid_derivative` and the cached `A_prev, Z, A`.
