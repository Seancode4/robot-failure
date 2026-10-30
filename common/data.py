"""Load the ASIMOV-2.0 datasets as plain Python objects.

TensorFlow is only imported the first time a dataset is exported from TFDS.
After that, examples are read from a local cache in data/cache/<dataset>/:
the original JPEG bytes (never re-encoded) plus an examples.json with the text
fields. No TensorFlow objects leave this module.

Example ids are stable: ``<prefix><index>`` in TFDS order with
``shuffle_files=False`` (e.g. ``con_000``, ``vid_000``).
"""

from __future__ import annotations

import ast
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
TFDS_DIR = ROOT / "data" / "tfds"
CACHE_DIR = ROOT / "data" / "cache"

CONSTRAINTS = "asimov_v2_constraints_with_rationale"
VIDEOS = "asimov_v2_videos"


# --------------------------------------------------------------------------- #
# Bounding boxes
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Box:
    """Axis-aligned box in normalized [0, 1] image coordinates (x right, y down)."""

    label: str
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    def to_pixels(self, width: int, height: int) -> tuple[float, float, float, float]:
        """Return (x_min, y_min, x_max, y_max) in pixels."""
        return (self.x_min * width, self.y_min * height, self.x_max * width, self.y_max * height)

    def contains(self, x: float, y: float, strict: bool = True) -> bool:
        """Whether normalized point (x, y) lies inside the box (interior if ``strict``)."""
        if strict:
            return self.x_min < x < self.x_max and self.y_min < y < self.y_max
        return self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max


_CORNER_RE = re.compile(r"x:\s*([-\d.eE+]+)\s*y:\s*([-\d.eE+]+)")


def parse_boxes(raw: str) -> list[Box]:
    """Parse an ASIMOV bounding-box field into a list of Boxes.

    The raw field is a Python dict repr mapping label -> corner string, e.g.::

        {'Water': '[x: 0.50593019\\ny: 0.67884323\\n, x: 0.60593019\\ny: 0.77884323\\n]'}

    Each corner string holds two (x, y) points: (x_min, y_min), (x_max, y_max),
    normalized to [0, 1] by image width/height. Raises ValueError on anything
    that does not match this format, so format drift is caught loudly.
    """
    if not raw.strip():
        return []
    mapping = ast.literal_eval(raw)
    if not isinstance(mapping, dict):
        raise ValueError(f"expected dict repr, got {type(mapping).__name__}")
    boxes = []
    for label, corners in mapping.items():
        pts = _CORNER_RE.findall(corners)
        if len(pts) != 2:
            raise ValueError(f"expected 2 corners for {label!r}, got {corners!r}")
        (x1, y1), (x2, y2) = [(float(x), float(y)) for x, y in pts]
        if not (0 <= x1 <= x2 <= 1 and 0 <= y1 <= y2 <= 1):
            raise ValueError(f"corners out of order/range for {label!r}: {corners!r}")
        boxes.append(Box(label, x1, y1, x2, y2))
    return boxes


def yx1000_to_normalized(point: list[float]) -> tuple[float, float]:
    """Convert a ``[y, x]`` 0-1000 point into normalized (x, y) in [0, 1], matching Box."""
    y, x = point
    return (x / 1000, y / 1000)


# --------------------------------------------------------------------------- #
# TFDS export + local cache
# --------------------------------------------------------------------------- #

def _timestamp_to_seconds(ts: str) -> float:
    """'MM:SS.mmm' -> seconds."""
    minutes, seconds = ts.split(":")
    return int(minutes) * 60 + float(seconds)


def _parse_answer(raw: str) -> dict:
    """Parse a fenced JSON answer such as '```json{...}```'. Raises on failure."""
    match = re.search(r"\{.*\}", raw, re.S)
    if not match:
        raise ValueError(f"no JSON object in answer: {raw!r}")
    return json.loads(match.group(0))


def _export(name: str) -> None:
    """Read a dataset from TFDS once and write images + examples.json to the cache."""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    import tensorflow_datasets as tfds  # heavy import, only needed once

    out = CACHE_DIR / name
    (out / "images").mkdir(parents=True, exist_ok=True)
    skip = tfds.decode.SkipDecoding()
    decoders = {"image": skip} if name == CONSTRAINTS else {"video": {"frames": skip}} if name == VIDEOS else None
    ds = tfds.load(name, split="val", data_dir=str(TFDS_DIR), shuffle_files=False,
                   download=True, decoders=decoders)
    prefix = {CONSTRAINTS: "con", VIDEOS: "vid"}[name]
    records = []
    for i, ex in enumerate(tfds.as_numpy(ds)):
        rec = {"id": f"{prefix}_{i:03d}", "index": i, "prompt": ex["prompt"].decode()}
        if name == CONSTRAINTS:
            path = out / "images" / f"{rec['id']}.jpg"
            path.write_bytes(ex["image"])
            rec["image"] = str(path.relative_to(out))
            rec["violating_raw"] = ex["violating_objects_bounding_boxes"].decode()
            rec["non_violating_raw"] = ex["non_violating_objects_bounding_boxes"].decode()
        else:
            rec["answer_raw"] = ex["answer"].decode()
        if name == VIDEOS:
            rec["timestamps"] = [t.decode() for t in ex["video"]["timestamps"]]
            rec["frames"] = []
            for j, frame in enumerate(ex["video"]["frames"]):
                path = out / "images" / f"{rec['id']}_f{j:02d}.jpg"
                path.write_bytes(frame)
                rec["frames"].append(str(path.relative_to(out)))
        records.append(rec)
    (out / "examples.json").write_text(json.dumps(records, indent=1))


def _records(name: str) -> tuple[Path, list[dict]]:
    path = CACHE_DIR / name / "examples.json"
    if not path.exists():
        _export(name)
    return path.parent, json.loads(path.read_text())


def _open(path: Path) -> Image.Image:
    """Open an image lazily; keeps .filename/.format so models.py can send original bytes."""
    img = Image.open(path)
    img.load()
    return img


# --------------------------------------------------------------------------- #
# Public loaders
# --------------------------------------------------------------------------- #

def load_constraints() -> list[dict]:
    """ASIMOV-2.0 constraints (164 examples).

    Keys: id, prompt, constraint (text between the CONSTRAINTS markers),
    image (PIL), width, height, violating / non_violating (list[Box]),
    violating_raw / non_violating_raw (original strings).
    """
    base, records = _records(CONSTRAINTS)
    out = []
    for r in records:
        img = _open(base / r["image"])
        m = re.search(r"<CONSTRAINTS START>(.*?)<CONSTRAINTS END>", r["prompt"], re.S)
        out.append({
            "id": r["id"],
            "prompt": r["prompt"],
            "constraint": m.group(1).strip() if m else None,
            "image": img,
            "width": img.width,
            "height": img.height,
            "violating": parse_boxes(r["violating_raw"]),
            "non_violating": parse_boxes(r["non_violating_raw"]),
            "violating_raw": r["violating_raw"],
            "non_violating_raw": r["non_violating_raw"],
        })
    return out


def load_videos() -> list[dict]:
    """ASIMOV-2.0 videos (287 examples, 16 frames each).

    Frames are NOT decoded up front (287 x 16 frames would be ~12 GB of pixels).
    Keys: id, prompt (raw; it is only a timestamp),
    frame_paths (list[Path]), timestamps (list[str]), timestamps_s (list[float]),
    answer (parsed dict), answer_raw. Use ``load_frames(example, indices)``.
    """
    base, records = _records(VIDEOS)
    return [{
        "id": r["id"],
        "prompt": r["prompt"],
        "frame_paths": [base / p for p in r["frames"]],
        "timestamps": r["timestamps"],
        "timestamps_s": [_timestamp_to_seconds(t) for t in r["timestamps"]],
        "answer": _parse_answer(r["answer_raw"]),
        "answer_raw": r["answer_raw"],
    } for r in records]


def load_frames(example: dict, indices: list[int] | None = None) -> list[Image.Image]:
    """Decode the frames of a video example (all frames if ``indices`` is None)."""
    paths = example["frame_paths"]
    if indices is not None:
        paths = [paths[i] for i in indices]
    return [_open(p) for p in paths]
