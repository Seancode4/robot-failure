"""Run the video risk experiment (frames vs text description) and score it.

    python scripts/run_video_risk.py --name video_risk_test --ids vid_183 vid_080
    python scripts/run_video_risk.py --name video_risk

Text needs the descriptions from scripts/run_video_describe.py first. Calls
are cached and the log is resumable. Scored rows go to
results/<name>_scored.csv.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import models
from common.data import load_videos
from common.runner import run
from experiments import settings as S
from experiments.video_risk import CONDITIONS, DESCRIPTIONS_LOG, build_input, score_log

PAPER_MODELS = [S.GPT5, S.GEMINI25_PRO, S.OPUS41]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--models", nargs="+", default=PAPER_MODELS)
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS), choices=CONDITIONS)
    ap.add_argument("--ids", nargs="+", help="only these video ids (for pilots)")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    videos = load_videos()
    if args.ids:
        videos = [v for v in videos if v["id"] in args.ids]
    meta = {**S.META, "video_input": "16 JPEG frames as stored (1280x720), each preceded by "
                                     "'Frame at <t>s:', prompt last",
            "image_setting": "each model at its API default image setting (GPT-5: no detail field, "
                             "resolves to auto = high, 910 tokens/frame; Gemini 2.5 Pro: "
                             "MEDIA_RESOLUTION_MEDIUM, 258 tokens/frame; Opus 4.1: unchanged, "
                             "~1,200 tokens/frame), checked 2026-10-27",
            "descriptions": f"results/logs/{DESCRIPTIONS_LOG}.jsonl ({S.DESCRIBER})"}
    run(args.name, videos, args.models, args.conditions, build_input,
        settings={m: S.MAIN[m] for m in args.models}, limit=args.limit, workers=args.workers, meta=meta)
    print("model calls:", models.STATS)
    df = score_log(args.name)
    path = Path(__file__).resolve().parents[1] / "results" / f"{args.name}_scored.csv"
    df.drop(columns=["output", "reasoning"], errors="ignore").to_csv(path, index=False)
    print(f"scored {len(df)} rows -> {path}")
    print(df.pivot_table(index="model", columns="condition", values="correct", aggfunc="mean"))


if __name__ == "__main__":
    main()
