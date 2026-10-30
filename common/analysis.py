"""Load run logs."""

from __future__ import annotations

import json
import pandas as pd

from .runner import log_path


def load_log(name: str, keep: str = "last") -> pd.DataFrame:
    """Load results/logs/{name}.jsonl as a DataFrame.

    Retried triples appear more than once; ``keep="last"`` keeps the latest line
    per (id, model, condition), ``keep="all"`` keeps every line.
    """
    rows = []
    for line in log_path(name).read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # torn final line from an interrupted run
    df = pd.DataFrame(rows)
    if keep == "last" and not df.empty:
        df = df.drop_duplicates(["id", "model", "condition"], keep="last").reset_index(drop=True)
    return df
