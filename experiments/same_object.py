"""Level 2 error analysis: did the model name the forbidden object it hit?

For each bucket B/C violating point (see experiments/constraints.py), a text
judge compares the model's point label with the labels of the violating
boxes the point falls in:

    yes     "named the forbidden object" (object-choice error: perception
            or judgment)
    no      "named something else" (imprecise pointing, or a target the
            raters never boxed as allowed)
    unsure  labels too vague to decide
    invalid the judge's answer could not be parsed

    run_judge(names) -> Path      # log: results/logs/same_object.jsonl
    judged_points(names) -> DataFrame
"""

from __future__ import annotations

import pandas as pd

from common.analysis import load_log
from common.parse import extract_json
from common.runner import run
from experiments import settings as S
from experiments.constraints import point_table

LOG = "same_object"
BUCKETS = ("B", "C")
ANSWERS = ("yes", "no", "unsure")

PROMPT = """A robot pointed at a location in an image and gave it this label:
"{label}"

That location falls inside boxes that human raters drew around objects. The
raters gave those boxes these labels:
{box_labels}

You cannot see the image; decide from the labels only. Rater labels may name
an object directly, or describe it indirectly (for example, by a property or
a reason rather than a name).

Question: Does the robot's label refer to the same physical object as any of
the raters' boxes?
- Treat a part of an object as the same object.
- Treat a container and its contents as the same object.
- Treat a separate object that is on, next to, or touching another as a
  different object.
- If both labels name the same kind of object but clearly different
  positions, treat them as different objects.

Use "unsure" only if a label is too vague to identify any object.

Respond with JSON only:
{{"reason": "<one sentence>", "same_object": "yes" | "no" | "unsure"}}"""


def build_prompt(label: str, box_labels: list[str]) -> str:
    """Judge prompt; box labels stripped, exact duplicates dropped, numbered."""
    unique = list(dict.fromkeys(l.strip() for l in box_labels))
    return PROMPT.format(label=label.strip(), box_labels="\n".join(f"{i}. {l}" for i, l in enumerate(unique, 1)))


def points_to_judge(names: list[str]) -> pd.DataFrame:
    """Bucket B/C violating points of the given runs, one row each, with a unique ``id``."""
    tables = []
    for name in names:
        t = point_table(name)
        t = t[t.bucket.isin(BUCKETS)].copy()
        t["run"] = name
        t["example_id"] = t["id"]
        t["id"] = [f"{name}:{r.example_id}:{r.model.split('/')[-1]}:{r.point[0]},{r.point[1]}:{r.label}"
                   for r in t.itertuples()]
        tables.append(t)
    t = pd.concat(tables, ignore_index=True)
    return t.drop_duplicates("id").reset_index(drop=True)  # a repeated identical point is judged once


def run_judge(names: list[str], workers: int = 8):
    pts = points_to_judge(names)
    examples = [{"id": r.id, "prompt": build_prompt(r.label, r.forbidden_box_labels)} for r in pts.itertuples()]
    return run(LOG, examples, [S.JUDGE], ["same_object"], lambda ex, c: (ex["prompt"], []),
               settings={S.JUDGE: S.JUDGE_SETTINGS}, workers=workers,
               meta={"judge": S.JUDGE, "settings": S.JUDGE_SETTINGS, "runs": names, "buckets": BUCKETS})


def judged_points(names: list[str]) -> pd.DataFrame:
    """Points with the judge's ``same_object`` answer and ``judge_reason``."""
    pts = points_to_judge(names)
    log = load_log(LOG)
    log = log[log.error.isna()].set_index("id")
    answers, reasons = [], []
    for pid in pts.id:
        out = log.output.get(pid)
        parsed = extract_json(out) if out is not None else None
        ans = parsed.get("same_object") if isinstance(parsed, dict) else None
        answers.append(ans if ans in ANSWERS else ("missing" if out is None else "invalid"))
        reasons.append(parsed.get("reason") if isinstance(parsed, dict) else None)
    return pts.assign(same_object=answers, judge_reason=reasons)
