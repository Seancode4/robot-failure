"""Robust JSON extraction from free-form model output. Never raises."""

from __future__ import annotations

import ast
import json
import re

_FENCE_RE = re.compile(r"```(?:json|JSON|python)?\s*(.*?)```", re.S)
_DECODER = json.JSONDecoder()


def _loads(s: str):
    """Parse strict JSON, falling back to a Python literal (single quotes, True/None)."""
    try:
        return json.loads(s)
    except (ValueError, TypeError):
        pass
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
        return None


def _scan(text: str, opener: str) -> list:
    """All top-level JSON values starting with ``opener`` found anywhere in ``text``."""
    found, i = [], 0
    while True:
        i = text.find(opener, i)
        if i < 0:
            return found
        try:
            value, end = _DECODER.raw_decode(text, i)
            found.append(value)
            i = end
        except ValueError:
            # try a Python-literal parse of the bracket-balanced span
            closer = "}" if opener == "{" else "]"
            depth, j = 0, i
            for j in range(i, len(text)):
                depth += text[j] == opener
                depth -= text[j] == closer
                if depth == 0:
                    break
            value = _loads(text[i:j + 1]) if depth == 0 else None
            if value is not None:
                found.append(value)
                i = j + 1
            else:
                i += 1


def extract_all_json(text: str | None, expect: type = dict) -> list:
    """Every JSON value of type ``expect`` (dict or list) in ``text``, in order.

    Fenced code blocks are preferred: if any fence parses, only fenced values
    are returned. Otherwise the whole text is scanned.
    """
    if not text:
        return []
    opener = "{" if expect is dict else "["
    try:
        fenced = []
        for block in _FENCE_RE.findall(text):
            value = _loads(block.strip())
            if value is None:
                value = next(iter(_scan(block, opener)), None)
            if isinstance(value, expect):
                fenced.append(value)
        if fenced:
            return fenced
        return [v for v in _scan(text, opener) if isinstance(v, expect)]
    except Exception:  # never raise: a parse failure is a result
        return []


def extract_json(text: str | None, expect: type = dict):
    """Return the last JSON value of type ``expect`` in ``text``, or None.

    Handles ```json fences (with or without newlines), prose before/after the
    JSON, and Python-style dicts with single quotes. ``expect=list`` extracts a
    top-level JSON array instead (e.g. ``[{"point": ..., "label": ...}]``).
    The last match is returned because models usually put the final answer
    after any reasoning.
    """
    values = extract_all_json(text, expect)
    return values[-1] if values else None
