"""Point-level error analysis (Level 1, rule-based; no API calls).

    python scripts/point_errors.py constraints_plain constraints_structured

Writes one row per violating point to results/<name>_violating_points.csv
(columns in experiments/constraints.py) and prints, per model and run:
the two reporting groups, "likely real error" (buckets B + C) and
"contested" (T + D), as raw counts and per 100 parsed answers.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from experiments.constraints import point_table, score_log

BUCKETS = ["B", "C", "T", "D"]  # see experiments/constraints.py
GROUPS = {"likely real error (B+C)": ["B", "C"], "contested (T+D)": ["T", "D"]}


def main(names):
    pd.set_option("display.width", 250)
    tables, n_answers = [], {}
    for name in names:
        t = point_table(name)
        t.to_csv(ROOT / "results" / f"{name}_violating_points.csv", index=False)
        t["run"] = name
        tables.append(t)
        sc = score_log(name)
        for m, g in sc.groupby("model"):
            n_answers[(m.split("/")[-1], name)] = int(g.parsed.sum())
    t = pd.concat(tables)
    t["model"] = t["model"].str.split("/").str[-1]
    counts = t.groupby(["model", "run", "bucket"]).size().unstack("bucket", fill_value=0)
    counts = counts.reindex(columns=list(BUCKETS), fill_value=0)

    groups = pd.DataFrame({g: counts[b].sum(axis=1) for g, b in GROUPS.items()})
    groups["n_answers"] = [n_answers[i] for i in groups.index]
    for g in GROUPS:
        groups[f"{g} /100"] = (100 * groups[g] / groups.n_answers).round(1)
    print("violating points by reporting group (raw counts and per 100 parsed answers):")
    print(groups.to_string())


if __name__ == "__main__":
    main(sys.argv[1:])
