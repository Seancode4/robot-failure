"""Paired comparison of two scored constraint runs (no API calls).

    python scripts/compare_constraints.py constraints_plain constraints_structured

Per model, on examples parsed in both runs: strict and clear violation rates
and the difference (B - A).
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from experiments.constraints import score_log

METRICS = {"violation": "Strict violation", "clear_violation": "Clear violation"}


def paired(a_name: str, b_name: str) -> pd.DataFrame:
    a, b = score_log(a_name), score_log(b_name)
    m = a.merge(b, on=["id", "model"], suffixes=("_a", "_b"))
    return m[m.parsed_a & m.parsed_b].reset_index(drop=True)


def compare(m: pd.DataFrame, a_name: str, b_name: str) -> pd.DataFrame:
    """Return the paired stats table."""
    rows = []
    for model, g in m.groupby("model", sort=False):
        short = model.split("/")[-1]
        for col, label in METRICS.items():
            va = g[f"{col}_a"].astype(bool).to_numpy()
            vb = g[f"{col}_b"].astype(bool).to_numpy()
            rows.append({
                "model": short, "metric": label, "n_paired": len(g),
                f"{a_name}_%": round(100 * va.mean(), 1),
                f"{b_name}_%": round(100 * vb.mean(), 1),
                "diff_pts": round(100 * (vb.mean() - va.mean()), 1),
            })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    args = ap.parse_args()
    m = paired(args.a, args.b)
    stats = compare(m, args.a, args.b)
    pd.set_option("display.width", 250)
    print(stats.to_string(index=False))


if __name__ == "__main__":
    main()
