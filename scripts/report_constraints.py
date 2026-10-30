"""Per-model summary of one scored constraint run (no API calls).

    python scripts/report_constraints.py constraints_plain

Rates are over parsed responses; unparsed ones are counted separately.
Also writes the scored rows to results/<name>_scored.csv.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from experiments import settings as S
from experiments.constraints import score_log

ROOT = Path(__file__).resolve().parents[1]


def summarize(name: str) -> pd.DataFrame:
    df = score_log(name)
    df.drop(columns=["output", "reasoning"], errors="ignore").to_csv(ROOT / "results" / f"{name}_scored.csv", index=False)
    rows = []
    for m, g in df.groupby("model", sort=False):
        p = g[g.parsed]
        rows.append({
            "model": m.split("/")[-1],
            "n_parsed": len(p),
            "unparsed": int((~g.parsed).sum()),
            "strict_%": round(100 * p.violation.astype(float).mean(), 1),
            "paper_%": S.PAPER_VIOLATION_RATE.get(m),
            "clear_%": round(100 * p.clear_violation.astype(float).mean(), 1),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    for name in sys.argv[1:]:
        print(f"== {name}")
        print(summarize(name).to_string(index=False))
