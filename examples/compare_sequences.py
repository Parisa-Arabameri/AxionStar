#!/usr/bin/env python3
"""Compare mass-radius sequences from different regimes.

Example:

    python examples/compare_sequences.py \
        outputs/leading_first/sequence.csv \
        outputs/second_order/sequence.csv \
        outputs/full_gr_real/sequence.csv

Every solver writes the same leading columns, so this script only needs
``mass`` and ``radius_95`` to compare regimes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def label_from_table(df: pd.DataFrame, path: Path) -> str:
    if {"regime", "model", "lambda_input"}.issubset(df.columns) and len(df) > 0:
        return f"{df['regime'].iloc[0]} / {df['model'].iloc[0]} / lambda={df['lambda_input'].iloc[0]}"
    return path.parent.name


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot mass-radius curves from sequence CSV files.")
    parser.add_argument("csv", nargs="+", help="One or more sequence.csv files.")
    parser.add_argument("--output", default="comparison_mass_radius.png", help="Output figure name.")
    args = parser.parse_args()

    plt.figure(figsize=(7, 5))
    for item in args.csv:
        path = Path(item)
        df = pd.read_csv(path)
        if "accepted" in df.columns:
            df = df[df["accepted"].astype(bool)]
        if df.empty:
            continue
        plt.plot(df["radius_95"], df["mass"], "o-", label=label_from_table(df, path))

    plt.xlabel("R95")
    plt.ylabel("Mass")
    plt.title("Axion-star mass-radius comparison")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(args.output, dpi=200)
    print(f"Saved {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
