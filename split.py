"""
Program 1/3 — Split the dataset into a training set and a validation set.

The split keeps the data RAW (no standardization here): standardization must use
training-only statistics, so it is applied later, at training time. This program
only shuffles and partitions the rows.

Usage:
    ./venv/bin/python split.py --dataset "data (1).csv" --ratio 0.8 --seed 42
Outputs:
    data_train.csv , data_valid.csv   (same column layout as the input, no header)
"""

import argparse
import numpy as np
import pandas as pd
from preprocessing import load_raw


def stratified_split(df, ratio, seed):
    """
    Split while preserving each class's proportion in both parts (stratified).

    Why stratified: the data is 62.7% B / 37.3% M. A plain random split could,
    by chance, skew the validation set's class balance and give a misleading
    score. Splitting each class separately guarantees both sets keep the ratio.
    """
    rng = np.random.default_rng(seed)        # seeded RNG -> reproducible split
    train_parts, valid_parts = [], []

    for label in ["B", "M"]:
        sub = df[df["diagnosis"] == label]
        order = rng.permutation(len(sub))    # shuffle this class's rows
        sub = sub.iloc[order]
        n_train = int(round(len(sub) * ratio))
        train_parts.append(sub.iloc[:n_train])
        valid_parts.append(sub.iloc[n_train:])

    train = pd.concat(train_parts)
    valid = pd.concat(valid_parts)

    # Shuffle again so the two classes are interleaved, not block-ordered.
    train = train.iloc[rng.permutation(len(train))]
    valid = valid.iloc[rng.permutation(len(valid))]
    return train, valid


def main():
    parser = argparse.ArgumentParser(description="Split dataset into train/valid.")
    parser.add_argument("--dataset", default="data/data.csv", help="input CSV path")
    parser.add_argument("--ratio", type=float, default=0.8,
                        help="fraction of data used for training (default 0.8)")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for a reproducible split")
    args = parser.parse_args()

    df = load_raw(args.dataset)
    train, valid = stratified_split(df, args.ratio, args.seed)

    train.to_csv("data_train.csv", header=False, index=False)
    valid.to_csv("data_valid.csv", header=False, index=False)

    def ratio_of(d):
        m = (d["diagnosis"] == "M").mean()
        return f"{100*(1-m):.1f}% B / {100*m:.1f}% M"

    print(f"Total samples : {len(df)}")
    print(f"Train         : {len(train):>3}  ({ratio_of(train)})  -> data_train.csv")
    print(f"Valid         : {len(valid):>3}  ({ratio_of(valid)})  -> data_valid.csv")


if __name__ == "__main__":
    main()
