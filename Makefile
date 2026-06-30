# Multilayer Perceptron — Makefile (works on macOS and Linux)
#
# Dependencies are installed automatically into a local virtual environment
# the first time you run any target, and only reinstalled when requirements.txt
# changes (the venv/.installed stamp tracks this).
#
# Common targets:
#   make            -> set up the virtualenv (alias of `make venv`)
#   make explore    -> Step 1: data exploration + plots
#   make split      -> Step 2: split dataset into data_train.csv / data_valid.csv
#   make train      -> Step 7: train the network (coming soon)
#   make predict    -> Step 8: predict + evaluate (coming soon)
#   make test       -> run the unit-test suite
#   make clean      -> remove generated plots, caches
#   make fclean     -> clean + remove split data, model, and the virtualenv
#   make re         -> fclean then rebuild the venv

# ---- configuration --------------------------------------------------------
PYTHON  := python3
VENV    := venv
PY      := $(VENV)/bin/python
PIP     := $(VENV)/bin/pip
STAMP   := $(VENV)/.installed
# Default dataset lives in data/. Override on the command line to use another:
#   make split DATASET=data/other.csv
DATASET := data/data.csv
SEED    := 42
RATIO   := 0.8

# ---- default --------------------------------------------------------------
all: venv

# ---- virtualenv (self-installing dependencies) ----------------------------
# The stamp file depends on requirements.txt, so editing requirements
# triggers a reinstall, but unchanged requirements do no work.
venv: $(STAMP)

$(STAMP): requirements.txt
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	touch $(STAMP)

# ---- pipeline steps -------------------------------------------------------
explore: $(STAMP)
	$(PY) explore.py --dataset "$(DATASET)"

split: $(STAMP)
	$(PY) split.py --dataset "$(DATASET)" --ratio $(RATIO) --seed $(SEED)

train: $(STAMP) data_train.csv
	$(PY) train.py

predict: $(STAMP)
	$(PY) predict.py

# Run the unit-test suite (uses Python's built-in unittest, no extra deps).
test: $(STAMP)
	$(PY) -m unittest discover -s tests -p "test_*.py" -v

# data_train.csv is produced by `split`; declaring it lets `train` auto-split.
data_train.csv data_valid.csv:
	$(MAKE) split

# ---- cleaning -------------------------------------------------------------
clean:
	rm -f explore_*.png loss_curve.png accuracy_curve.png learning_curves.png
	rm -rf __pycache__ .pytest_cache
	find . -name '*.pyc' -delete

fclean: clean
	rm -f data_train.csv data_valid.csv
	rm -f model.npy saved_model.npy
	rm -rf $(VENV)

re: fclean all

.PHONY: all venv explore split train predict test clean fclean re
