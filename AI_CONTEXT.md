# AI Handoff & Context — Multilayer Perceptron Project

> **Read this file first.** It lets any AI assistant continue this project exactly
> as the previous chat did. It captures the goal, the working method, every design
> decision, the current progress, and what to do next.
>
> **IMPORTANT — keep this file current:** after completing each step (and any
> meaningful change), UPDATE this file: move the step from "Remaining" to "Done",
> update "Current status / next action", and note any new design decisions or
> files. This is the single source of truth across chat sessions.

---

## 1. Who the user is and what they need

- This is a **42 school machine-learning project**. The user is **learning ML**
  and will defend the project in a **peer evaluation**, where they must **explain
  the math and the code line by line** — especially feedforward, backpropagation,
  and gradient descent.
- Therefore: **the user must understand everything before we move on.** Never
  rush ahead. Never implement future steps early.
- The user does not yet know the math deeply — explanations must be clear,
  intuitive first, then rigorous.

## 2. Working method (follow this exactly)

For **every step**:
1. **Explain before coding**, in this order:
   **(a) intuition** (what & why) → **(b) math** (formulas, derived; for backprop
   show the chain rule) → **(c) code** (written so each line maps to the math,
   with comments the user can read aloud in the eval).
2. **Implement** the step (Python + NumPy only — see constraints).
3. **Verify** it works: run it, show real output, and **add matching unit tests**
   under `tests/` (the user runs them live in the eval).
4. **Write a study sheet** `tasks/stepN_<name>.md` summarizing intuition + math +
   code + verification for that step (these are the user's eval notes).
5. **Update** `tasks/todo.md` (check the box, add a short result) **and this file**.
6. **Pause** and let the user take notes / ask questions before the next step.

Tone: concrete, no fluff. Use real numbers from the data. Reference files as
clickable relative links. Flag "eval talking points" and common-mistake traps.

## 3. Hard constraints (from the subject)

- **No ML libraries.** Implement the network, activations, loss, backprop, and
  gradient descent **from scratch**. NumPy (linear algebra) and matplotlib
  (plots) are allowed; pandas is used for CSV loading/exploration.
- Network must have **at least 2 hidden layers**.
- Output layer uses **softmax**.
- Must produce **two learning-curve plots**: loss and accuracy.
- Three deliverable programs: **split**, **train**, **predict** (we keep them
  separate). Prediction reports **binary cross-entropy**.
- Makefile must install deps and only redo work as needed. Cross-platform.

## 4. Design decisions already made (do not silently change)

- **Language:** Python 3 + NumPy. Virtualenv in `venv/` (created by the Makefile).
- **Output design:** **2-output softmax** (not 1-output sigmoid). Labels are
  one-hot: `B -> [1,0]`, `M -> [0,1]`. Convention **B=class 0, M=class 1**
  (M is the positive class). Loss = categorical cross-entropy; the prediction
  program also reports binary cross-entropy as the subject requires.
- **Matrix/shape convention:** one **sample per row**, batch size `m`.
  - `A_prev (m, n_in)`, `W (n_in, n_out)`, `b (1, n_out)`.
  - `Z = A_prev @ W + b` → `(m, n_out)`; `A = activation(Z)`.
- **Weight init:** He uniform `U(-sqrt(6/n_in), +sqrt(6/n_in))` (subject's
  choice for the sigmoid layers); biases start at 0. `heNormal` also available.
- **Standardization:** z-score `(x-mu)/sigma` with **TRAIN-ONLY** mu, sigma
  (no data leakage); these get saved with the model for predict to reuse.
- **Split:** stratified, seeded, 80/20.
- **Dataset location:** `data/data.csv`. All programs default to it and accept
  `--dataset` / Makefile `DATASET=` override (handles a new eval dataset).
- **Tests:** Python built-in `unittest` under `tests/`, run via `make test`.
- **Hidden architecture (planned default):** input(30) → dense(24, sigmoid) →
  dense(24, sigmoid) → dense(2, softmax). Configurable via args/config.

## 5. File inventory (current)

```
data/data.csv              raw dataset (569×32, no header)
multilayer_perceptron.md   the subject/spec
requirements.txt           numpy, pandas, matplotlib
Makefile                   venv/explore/split/train/predict/test/clean/fclean/re
explore.py                 Step 1 — data exploration (--dataset)
preprocessing.py           shared: load, encode_labels, compute_standardization, standardize
split.py                   Program 1/3 — stratified seeded split (--dataset/--ratio/--seed)
activations.py             sigmoid(+deriv), softmax (stable forms)
layers.py                  DenseLayer (He init, forward with caching)
tests/                     unittest suite (preprocessing, activations, layers, split)
tasks/todo.md              live plan + progress + results
tasks/step1..stepN_*.md    per-step study sheets (eval notes)
AI_CONTEXT.md              THIS FILE — handoff/source of truth
```

Files NOT yet created (future steps): `network.py` (the Network/model class),
`losses.py` (cross-entropy), `train.py` (Program 2/3), `predict.py` (Program 3/3),
model save file (e.g. `model.npy`), curve plots.

## 6. Progress — Done

- **Step 1 — Data exploration.** 569×32, 0 missing, 357 B / 212 M
  (62.7%/37.3% → baseline accuracy 62.7%). Features span ~4 orders of magnitude
  → standardization required. Plots + `explore.py`. Sheet: `tasks/step1_data_exploration.md`.
- **Step 2 — Preprocessing + split.** `preprocessing.py` + `split.py`. One-hot
  encoding, train-only standardization (no leakage), stratified seeded 80/20 →
  `data_train.csv` (456) / `data_valid.csv` (113). Sheet: `tasks/step2_preprocessing_split.md`.
- **Step 3 — Building blocks.** `activations.py` + `layers.py`. Shape convention,
  He init, sigmoid (+derivative verified vs numerical gradient), softmax.
  Sheet: `tasks/step3_building_blocks.md`.
- **Tooling.** Makefile (self-installing venv, cross-platform, clean/fclean),
  `data/` folder + `DATASET` override, `tests/` suite via `make test` (26 tests
  passing).

## 7. Progress — Remaining

- [ ] **Step 4 — Feedforward.** Build `network.py` with a `Network` class
  (`createNetwork`-style builder) that chains `DenseLayer.forward` to map
  `X → Ŷ`. Run a forward pass on real `data_train.csv`; show the untrained net
  outputs valid probabilities (~50/50). Add tests (output shape, rows sum to 1).
- [ ] **Step 5 — Loss.** `losses.py`: categorical cross-entropy
  `L = -(1/N) Σ Σ y log(ŷ)` with epsilon clipping; relate to the subject's
  binary cross-entropy. Add tests (perfect pred → ~0 loss, etc.).
- [ ] **Step 6 — Backpropagation.** The hard math. Chain rule through the net;
  show why softmax+cross-entropy gives the clean output gradient `(Ŷ − Y)`;
  per-layer gradients dW, db, and propagation with `sigmoid_derivative`.
  **Add a gradient-checking test** (analytic vs numerical) — the key proof.
- [ ] **Step 7 — Gradient descent + training loop.** `train.py` (Program 2/3):
  mini-batches, epochs, learning rate, `W -= lr * dW`; per-epoch train/val
  loss+accuracy printout (subject's format); two learning-curve plots; save
  model (topology + weights + mu/sigma). Configurable via args/config.
- [ ] **Step 8 — Prediction.** `predict.py` (Program 3/3): load model, standardize
  with saved mu/sigma, predict, report **binary cross-entropy** + accuracy.
- [ ] **Step 9 — Bonus (optional, only if mandatory is perfect):** Adam/momentum,
  early stopping, extra metrics, multi-curve comparison, metric history.

## 8. Current status / next action

- **Last completed:** Step 3 (building blocks) + persistent test suite + handoff file.
- **Next action:** **Step 4 — Feedforward.** Wait for the user to say go, then
  follow the Working Method (Section 2): explain intuition→math→code, build
  `network.py`, run on real data, add tests, write `tasks/step4_feedforward.md`,
  update `tasks/todo.md` and this file, then pause.

## 9. How to run things

```bash
make            # set up venv + install deps
make explore    # Step 1 plots
make split      # Step 2 -> data_train.csv / data_valid.csv
make test       # run the unit-test suite
make train      # (once train.py exists)
make predict    # (once predict.py exists)
make clean / make fclean / make re
# swap dataset:  make split DATASET=data/other.csv
```
