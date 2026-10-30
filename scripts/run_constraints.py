"""Run the constraint experiment and score it.

    python scripts/run_constraints.py --name constraints_plain --conditions plain
    python scripts/run_constraints.py --name pilot --limit 10

Calls are cached and the log is resumable, so rerunning only pays for
missing or failed calls. Scored rows go to results/<name>_scored.csv and
unparsed responses to results/unparsed/<name>.jsonl.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import models
from common.data import load_constraints
from common.runner import run
from experiments import settings as S
from experiments.constraints import CONDITIONS, build_input, score_log

PAPER_MODELS = [S.GPT5, S.GEMINI25_PRO, S.OPUS41]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--models", nargs="+", default=PAPER_MODELS)
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS), choices=CONDITIONS)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    run(args.name, load_constraints(), args.models, args.conditions, build_input,
        settings={m: S.MAIN[m] for m in args.models}, limit=args.limit, workers=args.workers, meta=S.META)
    print("model calls:", models.STATS)
    df = score_log(args.name)
    path = Path(__file__).resolve().parents[1] / "results" / f"{args.name}_scored.csv"
    df.drop(columns=["output"]).to_csv(path, index=False)
    print(f"scored {len(df)} rows -> {path}")


if __name__ == "__main__":
    main()
