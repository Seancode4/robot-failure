"""Level 2 summary: named-the-forbidden-object vs named-something-else.

    python scripts/same_object.py constraints_plain constraints_structured [--run-judge]
    python scripts/same_object.py constraints_er2_plain constraints_er2_structured --out same_object_points_er2

Counts bucket B/C violating points by the judge's answer, per model and run,
as totals and per 100 responses parsed in both runs, and the change
(second run minus first).
Writes results/<out>.csv (default same_object_points).
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from experiments.constraints import score_log
from experiments.same_object import judged_points, run_judge

KINDS = {"yes": "named forbidden object", "no": "named something else", "unsure": "unsure"}


def per_example_counts(j: pd.DataFrame, run: str, model: str, ids: list[str]) -> pd.DataFrame:
    sub = j[(j.run == run) & (j.model == model)]
    c = sub.groupby(["example_id", "same_object"]).size().unstack(fill_value=0)
    return c.reindex(index=ids, columns=list(KINDS), fill_value=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--run-judge", action="store_true")
    ap.add_argument("--out", default="same_object_points", help="output name in results/ (no .csv)")
    args = ap.parse_args()
    names = [args.a, args.b]
    if args.run_judge:
        run_judge(names)
    j = judged_points(names)
    j.to_csv(ROOT / "results" / f"{args.out}.csv", index=False)
    bad = j.same_object.isin(["invalid", "missing"]).sum()
    print(f"{len(j)} judged points ({bad} invalid/missing)")

    scored = {r: score_log(r) for r in names}
    rows = []
    groups = [(m, [m]) for m in dict.fromkeys(j.model)] + [("all models", list(dict.fromkeys(j.model)))]
    for label, model_list in groups:
        counts = {(r, k): 0 for r in names for k in KINDS}
        n_resp = 0
        for m in model_list:
            # responses parsed in both runs (the paired set used everywhere else)
            ids = sorted(set.intersection(*[set(scored[r][(scored[r].model == m) & scored[r].parsed].id)
                                            for r in names]))
            ca, cb = (per_example_counts(j, r, m, ids) for r in names)
            n_resp += len(ids)
            for k in KINDS:
                counts[(args.a, k)] += int(ca[k].sum())
                counts[(args.b, k)] += int(cb[k].sum())
        for k, kname in KINDS.items():
            rows.append({
                "model": label.split("/")[-1], "kind": kname, "n_responses": n_resp,
                f"{args.a}": counts[(args.a, k)], f"{args.b}": counts[(args.b, k)],
                "per100_a": round(100 * counts[(args.a, k)] / n_resp, 1),
                "per100_b": round(100 * counts[(args.b, k)] / n_resp, 1),
                "change_per100": round(100 * (counts[(args.b, k)] - counts[(args.a, k)]) / n_resp, 1),
            })
    pd.set_option("display.width", 250)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
