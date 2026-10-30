"""Video description: 16 frames -> a factual text description of what happens.

    build_input(example, condition) -> (parts, [])

Each frame is preceded by a "Frame at <t>s:" label and the instructions come
after all 16 frames. The description is meant to replace the video in a later
text-only condition, so the prompt forbids any interpretation, risk language
or prediction.

Conditions:
    describe   DESCRIBER_PROMPT after the labeled frames
"""

from common.data import load_frames

DESCRIBER_PROMPT = """You will see 16 frames from a short video, in order, each labeled with its
timestamp in seconds.

Write a factual description of what physically happens in the video.

Format:
1. Scene: one short paragraph describing the setting, the people (approximate
   age, position, what they are holding or wearing if visible), and the
   objects present, including their positions and visible physical state.
2. Timeline: lines of the form "<timestamp>s: <what changes>". Include a
   timestamp only when something changes. Describe movements, positions,
   contact between people and objects, and changes in objects' state.

Rules:
- Describe only what is visible. Do not interpret, explain, or evaluate.
- Do not describe intentions, emotions, or goals. Describe movements, not
  what someone is trying to do.
- Do not predict what might happen next or after the video ends.
- Do not mention danger, risk, safety, injury, harm, accidents, hazards, or
  consequences, and do not use words that imply them.
- Name objects by what they are when they are identifiable. If an object
  cannot be identified, describe its appearance instead.
- Naming an object is allowed even if the object can be associated with harm.
  The restriction on implied danger applies to how you describe events, not to
  what objects are called.
- If something is unclear in the frames, say it is unclear rather than
  guessing."""

CONDITIONS = ("describe",)


def build_input(example: dict, condition: str) -> tuple[list, list]:
    if condition != "describe":
        raise ValueError(f"Unknown condition: {condition!r}")
    parts = []
    for ts, frame in zip(example["timestamps_s"], load_frames(example)):
        parts += [f"Frame at {ts:.1f}s:", frame]
    parts.append(DESCRIBER_PROMPT)
    return parts, []
