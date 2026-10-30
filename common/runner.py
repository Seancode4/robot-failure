"""Experiment loop: models x conditions x examples -> one JSON line per call, resumable."""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

from tqdm import tqdm

from .models import call_model_full

LOG_DIR = Path(__file__).resolve().parents[1] / "results" / "logs"


def log_path(name: str) -> Path:
    return LOG_DIR / f"{name}.jsonl"


def meta_path(name: str) -> Path:
    return LOG_DIR / f"{name}.meta.jsonl"


def _done(path: Path) -> set[tuple[str, str, str]]:
    """Successfully logged (id, model, condition) triples. Tolerates a torn last line.

    Error lines are not counted as done, so a restart retries them.
    """
    done = set()
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not r.get("error"):
                done.add((r["id"], r["model"], r["condition"]))
    return done


def run(name: str, examples: list[dict], models: Iterable[str], conditions: Iterable[str],
        build_input: Callable[[dict, str], tuple], settings: dict | None = None,
        limit: int | None = None, workers: int = 1, meta: dict | None = None,
        cache_tag: str | None = None) -> Path:
    """Run every (model, condition, example) not yet in results/logs/{name}.jsonl.

    ``build_input(example, condition)`` returns ``(text, images)``; ``text`` may
    also be a list of interleaved str / PIL.Image parts (see models.py).
    ``settings`` is either one dict applied to all models, or
    ``{model_name: dict}`` for per-model settings. ``limit`` keeps only the
    first N examples (for pilots). ``workers`` > 1 runs calls in threads.
    Each invocation appends one line to results/logs/{name}.meta.jsonl with
    the models, conditions, settings and free-form ``meta`` (e.g. notes on
    defaults and known differences from a paper). ``cache_tag`` is passed to
    call_model_full to draw fresh samples for inputs that are already cached.

    Each log line: id, model, condition, settings, output, cached, cache_key,
    finish_reason, usage, reasoning (thinking text, if returned), error, timestamp. A failed call is logged with
    ``error`` set and ``output`` null, and the run continues; failed triples
    are retried on the next run (analysis.load_log keeps the latest line).
    """
    models, conditions = list(models), list(conditions)
    examples = examples[:limit] if limit else examples
    path = log_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    done = _done(path)
    todo = [(m, c, ex) for m in models for c in conditions for ex in examples
            if (ex["id"], m, c) not in done]
    per_model = isinstance(settings, dict) and settings and all(k in models for k in settings)
    with meta_path(name).open("a") as f:
        f.write(json.dumps({"timestamp": datetime.now(timezone.utc).isoformat(), "models": models,
                            "conditions": conditions, "n_examples": len(examples), "settings": settings, "cache_tag": cache_tag,
                            "meta": meta}) + "\n")
    lock = threading.Lock()

    def one(model, condition, ex):
        s = (settings.get(model, {}) if per_model else settings) or {}
        rec = {"id": ex["id"], "model": model, "condition": condition, "cache_tag": cache_tag}
        try:
            text, images = build_input(ex, condition)
            r = call_model_full(model, text, images, cache_tag=cache_tag, **s)
            rec.update(settings=r["settings"], output=r["output"], cached=r["cached"], cache_key=r["key"],
                       finish_reason=r.get("finish_reason"), usage=r.get("usage"),
                       reasoning=r.get("reasoning"), error=None)
        except Exception as e:
            rec.update(settings=s, output=None, cached=False, error=f"{type(e).__name__}: {e}")
        rec["timestamp"] = datetime.now(timezone.utc).isoformat()
        with lock, path.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        return rec

    desc = f"{name} ({len(done)} already done)"
    with tqdm(total=len(todo), desc=desc) as bar:
        if workers <= 1:
            for job in todo:
                r = one(*job)
                bar.update()
                if r["error"]:
                    tqdm.write(f"[runner] {r['id']} {r['model']} {r['condition']}: {r['error'][:200]}")
        else:
            with ThreadPoolExecutor(workers) as pool:
                for fut in as_completed([pool.submit(one, *job) for job in todo]):
                    r = fut.result()
                    bar.update()
                    if r["error"]:
                        tqdm.write(f"[runner] {r['id']} {r['model']} {r['condition']}: {r['error'][:200]}")
    return path
