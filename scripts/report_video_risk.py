"""Score a video risk run and write its result tables (no API calls).

    python scripts/report_video_risk.py              # run name video_risk
    python scripts/report_video_risk.py --name video_risk

Writes, for run <name>:
    results/<name>_scored.csv    one row per answer (no output/reasoning text)
    results/<name>_summary.csv   per model x condition: n, accuracy, share of
                                 "Yes" answers

Unparsed answers and failed calls (e.g. Gemini content blocks) are excluded
and counted in the printed notes.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from common.analysis import load_log
from experiments.video_risk import score_log


def summary(df: pd.DataFrame) -> pd.DataFrame:
    return (df.groupby(["model", "condition"])
              .agg(n=("correct", "size"), acc=("correct", "mean"), pred_yes=("pred", lambda p: (p == "Yes").mean()))
              .reset_index())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="video_risk")
    args = ap.parse_args()

    scored = score_log(args.name)
    out = ROOT / "results"
    scored.drop(columns=["output", "reasoning", "note"], errors="ignore").to_csv(
        out / f"{args.name}_scored.csv", index=False)
    summ = summary(scored[scored["parsed"]])
    summ.to_csv(out / f"{args.name}_summary.csv", index=False)

    log = load_log(args.name)
    failed = log[log["error"].notna()].groupby(["model", "condition"]).size()
    unparsed = scored[~scored["parsed"]].groupby(["model", "condition"]).size()
    pd.set_option("display.width", 200)
    print(summ.round(3).to_string(index=False), "\n")
    print("failed calls (excluded):", failed.to_dict() or "none")
    print("unparsed answers (excluded):", unparsed.to_dict() or "none")
    print(f"wrote results/{args.name}_{{scored,summary}}.csv")


if __name__ == "__main__":
    main()
