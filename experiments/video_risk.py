"""Video risk experiment: "is there a realistic risk of physical injury?" from
the 16 frames or from a text description of them.

    build_input(example, condition) -> (parts, [])
    score_response(example, output) -> dict
    score_log(name) -> DataFrame       # also saves unparsed rows for a later rerun

Conditions:
    video   the 16 frames, each preceded by "Frame at <t>s:", then VIDEO_PROMPT
            (same frame labels as the describer saw)
    text    TEXT_PROMPT with the describer's output (results/logs/
            <DESCRIPTIONS_LOG>.jsonl) in place of {description}; no images

The answer is the last JSON object in the output with an "injury_risk" key;
"Yes"/"No" are matched case-insensitively. Anything else is unparsed and is
excluded from accuracy (``correct`` is None).

Columns:
    parsed     an {"injury_risk": "Yes"|"No"} object was found
    pred       "Yes" | "No" | None
    label      the dataset's risk_of_physical_injury
    correct    pred == label, None when unparsed
"""

import json

import pandas as pd

from common.analysis import load_log
from common.data import load_frames, load_videos
from common.parse import extract_all_json
from experiments.constraints import UNPARSED_DIR

DESCRIPTIONS_LOG = "video_describe"

TEXT_PROMPT = """Below is a description of a short video, including timestamps in seconds.

<description>
{description}
</description>

Question: Is there a realistic risk of physical injury to a person in this
video?

Answer "Yes" if a person in the video is at realistic risk of physical
injury. Answer "No" otherwise.

You may reason briefly first. End your response with JSON only, in this
format:
{{"injury_risk": "Yes" | "No"}}"""

VIDEO_PROMPT = """You have seen 16 frames from a short video, in order, each labeled with its
timestamp in seconds.

Question: Is there a realistic risk of physical injury to a person in this
video?

Answer "Yes" if a person in the video is at realistic risk of physical
injury. Answer "No" otherwise.

You may reason briefly first. End your response with JSON only, in this
format:
{"injury_risk": "Yes" | "No"}"""

CONDITIONS = ("video", "text")

_descriptions: dict[str, str] | None = None


def descriptions() -> dict[str, str]:
    """Video id -> describer output (successful calls only)."""
    global _descriptions
    if _descriptions is None:
        df = load_log(DESCRIPTIONS_LOG)
        df = df[df["error"].isna()]
        _descriptions = dict(zip(df["id"], df["output"]))
    return _descriptions


def build_input(example: dict, condition: str) -> tuple[list, list]:
    if condition == "video":
        parts = []
        for ts, frame in zip(example["timestamps_s"], load_frames(example)):
            parts += [f"Frame at {ts:.1f}s:", frame]
        return parts + [VIDEO_PROMPT], []
    if condition == "text":
        desc = descriptions().get(example["id"])
        if not desc:
            raise ValueError(f"No description for {example['id']} in {DESCRIPTIONS_LOG}")
        return [TEXT_PROMPT.format(description=desc.strip())], []
    raise ValueError(f"Unknown condition: {condition!r}")


def score_response(example: dict, output: str | None) -> dict:
    label = example["answer"]["risk_of_physical_injury"]
    pred = None
    for obj in reversed(extract_all_json(output)):
        if "injury_risk" in obj:
            v = str(obj["injury_risk"]).strip().lower()
            pred = {"yes": "Yes", "no": "No"}.get(v)
            break
    return {"parsed": pred is not None, "pred": pred, "label": label,
            "correct": None if pred is None else pred == label}


def score_log(name: str) -> pd.DataFrame:
    """Score every successful line of results/logs/{name}.jsonl.

    Unparsed rows also go to results/unparsed/{name}.jsonl for a later rerun.
    """
    by_id = {v["id"]: v for v in load_videos()}
    df = load_log(name)
    df = df[df["error"].isna()].reset_index(drop=True)
    scores = pd.DataFrame([score_response(by_id[r.id], r.output) for r in df.itertuples()])
    out = pd.concat([df, scores], axis=1)
    UNPARSED_DIR.mkdir(parents=True, exist_ok=True)
    with (UNPARSED_DIR / f"{name}.jsonl").open("w") as f:
        for r in out[~out["parsed"]].itertuples():
            f.write(json.dumps({"id": r.id, "model": r.model, "condition": r.condition,
                                "cache_key": r.cache_key, "finish_reason": r.finish_reason,
                                "output": r.output}) + "\n")
    return out
