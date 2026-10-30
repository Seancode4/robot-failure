"""Describe every video with the describer model (16 labeled frames in one call).

    python scripts/run_video_describe.py --name video_describe --ids vid_048
    python scripts/run_video_describe.py --name video_describe

Calls are cached and the log is resumable. Descriptions are in
results/logs/<name>.jsonl (field ``output``).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import models
from common.data import load_videos
from common.runner import run
from experiments import settings as S
from experiments.video_describe import CONDITIONS, build_input

META = {
    "temperature": "0",
    "thinking": "none (Instruct model)",
    "input": "16 JPEG frames as stored (1280x720), each preceded by 'Frame at <t>s:', prompt last",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--ids", nargs="+", help="only these video ids (for pilots)")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    videos = load_videos()
    if args.ids:
        videos = [v for v in videos if v["id"] in args.ids]
    run(args.name, videos, [S.DESCRIBER], CONDITIONS, build_input, settings=S.DESCRIBER_SETTINGS,
        limit=args.limit, workers=args.workers, meta=META)
    print("model calls:", models.STATS)


if __name__ == "__main__":
    main()
