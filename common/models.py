"""Unified, disk-cached model calls for the Gemini API and OpenRouter.

    call_model_full(model, text, images=(), **settings) -> dict

Routing is by model-name prefix: ``gemini-*`` -> google-genai,
``openrouter/<author>/<slug>`` -> OpenRouter (OpenAI-compatible Chat Completions),
e.g. ``openrouter/anthropic/claude-opus-4.1`` for models retired upstream.

Settings (all optional; unsupported ones raise ValueError instead of being
silently dropped, so logged settings always reflect what was sent):

    temperature        Never sent unless given, so the provider default applies.
    max_output_tokens  both providers.
    thinking_budget    Gemini: int token budget (0 disables thinking on
                       models that allow it, -1 = dynamic).
                       OpenRouter: reasoning.max_tokens.
    thinking           OpenRouter only: True/False -> reasoning.enabled (turn
                       thinking on/off at the provider's default budget).
    effort             Gemini thinking_level / OpenRouter reasoning.effort
                       ("low", "medium", ..., "none").
    providers          OpenRouter only: list of upstream provider slugs the
                       request may go to (``provider.only``, no fallbacks).
                       Defaults to OPENROUTER_PINS[model]; a model without a
                       pin must be given one explicitly. Requests also set
                       ``require_parameters`` so a provider that would ignore
                       a setting is never picked.

Content order: ``text`` may be a str (images are placed BEFORE the text) or a
list of str / PIL.Image parts for interleaving (e.g. timestamp labels between
video frames); in that case ``images`` must be empty.

Every call is cached in results/cache/<sha256>.json, keyed on (model, content
parts incl. image bytes, settings, optional ``cache_tag``). A cache hit makes
no API request. ``cache_tag`` is never sent to the API; a new tag (e.g.
"rerun1") gets a fresh sample for the same input without touching the
original cache entry.
``STATS`` counts api_calls and cache_hits for the current process.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import random
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence, Union

from PIL import Image

try:  # optional .env support (gitignored); real env vars take precedence
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)
except ImportError:  # pragma: no cover
    pass

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "results" / "cache"

MAX_RETRIES = 6
STATS = {"api_calls": 0, "cache_hits": 0}
_STATS_LOCK = threading.Lock()

Part = Union[str, Image.Image]

ALLOWED = {
    "gemini": {"temperature", "max_output_tokens", "thinking_budget", "effort"},
    "openrouter": {"temperature", "max_output_tokens", "thinking_budget", "effort", "thinking", "providers"},
}
DEFAULTS = {
    "gemini": {},
    "openrouter": {},
}

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
# Upstream provider per OpenRouter model, so results don't depend on routing.
# Slugs from GET /api/v1/models/<author>/<slug>/endpoints (2026-10-26).
OPENROUTER_PINS = {
    "openrouter/anthropic/claude-opus-4.1": ["amazon-bedrock"],  # only endpoint
    "openrouter/openai/gpt-5": ["openai"],  # Azure endpoints drop max_tokens
    "openrouter/google/gemini-2.5-pro": ["google-ai-studio"],
    # DeepSeek's own endpoint (may train on requests; needs the OpenRouter
    # privacy toggle "Allow paid endpoints that train on request data")
    "openrouter/deepseek/deepseek-v4-pro-0813": ["deepseek"],
    # fp8; DeepInfra (cheaper) was 429 engine_overloaded on 2026-10-27; no Alibaba endpoint
    "openrouter/qwen/qwen3-vl-235b-a22b-instruct": ["parasail/fp8"],
}


class RetryableError(Exception):
    """Wraps a transient provider error (rate limit, 5xx, network)."""


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def provider_for(model: str) -> str:
    """Route a model name to a provider by prefix."""
    if model.startswith("openrouter/"):
        return "openrouter"
    if model.startswith("gemini"):
        return "gemini"
    raise ValueError(f"Unknown model prefix: {model!r}")


def encode_image(img: Image.Image) -> tuple[bytes, str]:
    """Return (bytes, mime type) for an image.

    An image opened unmodified from a JPEG/PNG file is sent as its original
    file bytes (no re-encoding). Anything else is encoded as lossless PNG.
    """
    fmt = (img.format or "").upper()
    path = getattr(img, "filename", "")
    if fmt in ("JPEG", "PNG") and path and Path(path).exists():
        return Path(path).read_bytes(), f"image/{fmt.lower()}"
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return buf.getvalue(), "image/png"


def _normalize_parts(text: str | Sequence[Part], images: Sequence[Image.Image]) -> list:
    """Return a list of ("text", str) / ("image", bytes, mime) tuples."""
    if isinstance(text, str):
        parts: list[Part] = [*images, text]
    else:
        if images:
            raise ValueError("Pass images either via `images` or inside `text` parts, not both.")
        parts = list(text)
    out = []
    for p in parts:
        if isinstance(p, str):
            out.append(("text", p))
        elif isinstance(p, Image.Image):
            out.append(("image", *encode_image(p)))
        else:
            raise TypeError(f"Unsupported content part: {type(p).__name__}")
    return out


def _cache_key(model: str, parts: list, settings: dict, cache_tag: str | None = None) -> str:
    h = hashlib.sha256()
    key = {"model": model, "settings": settings}
    if cache_tag is not None:  # untagged keys stay identical to older caches
        key["cache_tag"] = cache_tag
    h.update(json.dumps(key, sort_keys=True).encode())
    for p in parts:
        if p[0] == "text":
            h.update(b"\x00text\x00" + p[1].encode())
        else:
            h.update(b"\x00image\x00" + p[2].encode() + b"\x00" + hashlib.sha256(p[1]).digest())
    return h.hexdigest()


def _bump(stat: str) -> None:
    with _STATS_LOCK:
        STATS[stat] += 1


def _require_env(*names: str) -> str:
    for n in names:
        if os.environ.get(n):
            return os.environ[n]
    raise RuntimeError(f"Missing API key: set one of {', '.join(names)}")


# --------------------------------------------------------------------------- #
# Providers. Each returns (text, metadata) and raises RetryableError on
# transient failures; anything else propagates.
# --------------------------------------------------------------------------- #

_clients: dict = {}


def _client(provider: str):
    if provider not in _clients:
        if provider == "gemini":
            from google import genai
            _clients[provider] = genai.Client(api_key=_require_env("GEMINI_API_KEY", "GOOGLE_API_KEY"))
        else:
            import openai
            _clients[provider] = openai.OpenAI(base_url=OPENROUTER_BASE_URL,
                                               api_key=_require_env("OPENROUTER_API_KEY"), max_retries=0)
    return _clients[provider]


def _is_transient_status(code) -> bool:
    return code in (408, 409, 429) or (isinstance(code, int) and code >= 500)


def _call_gemini(model: str, parts: list, s: dict) -> tuple[str, dict]:
    from google.genai import errors, types

    contents = [types.Part.from_text(text=p[1]) if p[0] == "text"
                else types.Part.from_bytes(data=p[1], mime_type=p[2]) for p in parts]
    thinking = None
    if "thinking_budget" in s or "effort" in s:
        thinking = types.ThinkingConfig(thinking_budget=s.get("thinking_budget"),
                                        thinking_level=s.get("effort"))
    config = types.GenerateContentConfig(temperature=s.get("temperature"),
                                         max_output_tokens=s.get("max_output_tokens"),
                                         thinking_config=thinking)
    try:
        resp = _client("gemini").models.generate_content(model=model, contents=contents, config=config)
    except errors.APIError as e:
        if _is_transient_status(e.code):
            raise RetryableError(str(e)) from e
        raise
    except Exception as e:  # network-level errors from httpx
        if type(e).__module__.startswith(("httpx", "httpcore")):
            raise RetryableError(repr(e)) from e
        raise
    u = resp.usage_metadata
    cand = resp.candidates[0] if resp.candidates else None
    meta = {
        "finish_reason": str(cand.finish_reason) if cand and cand.finish_reason else None,
        "block_reason": str(resp.prompt_feedback.block_reason) if resp.prompt_feedback else None,
        "model_version": resp.model_version,
        "usage": {"input_tokens": u.prompt_token_count, "output_tokens": u.candidates_token_count,
                  "thinking_tokens": u.thoughts_token_count, "total_tokens": u.total_token_count} if u else None,
    }
    return resp.text or "", meta


def _call_openrouter(model: str, parts: list, s: dict) -> tuple[str, dict]:
    import openai

    content = [{"type": "text", "text": p[1]} if p[0] == "text" else
               {"type": "image_url", "image_url": {"url": f"data:{p[2]};base64,{base64.b64encode(p[1]).decode()}"}}
               for p in parts]
    kwargs = {"model": model.removeprefix("openrouter/"), "messages": [{"role": "user", "content": content}]}
    if "temperature" in s:
        kwargs["temperature"] = s["temperature"]
    if "max_output_tokens" in s:
        kwargs["max_tokens"] = s["max_output_tokens"]
    extra = {"provider": {"only": s["providers"], "allow_fallbacks": False, "require_parameters": True}}
    reasoning = {k: s[n] for n, k in (("effort", "effort"), ("thinking_budget", "max_tokens"),
                                      ("thinking", "enabled")) if n in s}
    if reasoning:
        extra["reasoning"] = reasoning
    try:
        resp = _client("openrouter").chat.completions.create(**kwargs, extra_body=extra)
    except openai.APIStatusError as e:
        if _is_transient_status(e.status_code):
            raise RetryableError(str(e)) from e
        raise
    except openai.APIConnectionError as e:
        raise RetryableError(repr(e)) from e
    # OpenRouter reports upstream failures after a provider accepted the
    # request as HTTP 200 with an `error` object (top level or per choice).
    extra_out = resp.model_extra or {}
    choice = resp.choices[0] if resp.choices else None
    err = extra_out.get("error") or (choice and (choice.model_extra or {}).get("error"))
    if err or choice is None:
        code = err.get("code") if isinstance(err, dict) else None
        msg = f"OpenRouter error from {extra_out.get('provider')}: {err!r}"
        if code is None or _is_transient_status(code):
            raise RetryableError(msg)
        raise RuntimeError(msg)
    u = resp.usage
    u_extra = (u.model_extra or {}) if u else {}
    meta = {
        "finish_reason": choice.finish_reason,
        "native_finish_reason": (choice.model_extra or {}).get("native_finish_reason"),
        "model_version": resp.model,
        "served_by": extra_out.get("provider"),
        "generation_id": resp.id,
        # thinking text when the provider returns it (GPT-5: summary only;
        # Gemini: thought summary; Claude: full thinking); None otherwise
        "reasoning": (choice.message.model_extra or {}).get("reasoning"),
        "usage": {"input_tokens": u.prompt_tokens, "output_tokens": u.completion_tokens,
                  "thinking_tokens": getattr(u.completion_tokens_details, "reasoning_tokens", None),
                  "total_tokens": u.total_tokens, "cost_usd": u_extra.get("cost")} if u else None,
    }
    return choice.message.content or "", meta


_PROVIDERS = {"gemini": _call_gemini, "openrouter": _call_openrouter}


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def resolve_settings(model: str, **settings) -> dict:
    """Merge provider defaults with ``settings`` and validate. None values are dropped."""
    provider = provider_for(model)
    merged = {**DEFAULTS[provider], **{k: v for k, v in settings.items() if v is not None}}
    if provider == "openrouter":
        merged.setdefault("providers", OPENROUTER_PINS.get(model))
        if not merged["providers"]:
            raise ValueError(f"No upstream provider pinned for {model!r}: pass providers=[...] "
                             "or add it to OPENROUTER_PINS")
        merged["providers"] = list(merged["providers"])
    bad = set(merged) - ALLOWED[provider]
    if bad:
        raise ValueError(f"Settings {sorted(bad)} not supported for {provider} model {model!r}")
    return merged


def call_model_full(model: str, text: str | Sequence[Part], images: Sequence[Image.Image] = (),
                    use_cache: bool = True, cache_tag: str | None = None, **settings) -> dict:
    """Call ``model`` with text (+ images); return the cache record (output + metadata + 'cached')."""
    provider = provider_for(model)
    s = resolve_settings(model, **settings)
    parts = _normalize_parts(text, images)
    key = _cache_key(model, parts, s, cache_tag)
    path = CACHE_DIR / f"{key}.json"
    if use_cache and path.exists():
        _bump("cache_hits")
        return {**json.loads(path.read_text()), "cached": True}

    for attempt in range(MAX_RETRIES + 1):
        try:
            _bump("api_calls")
            t0 = time.time()
            output, meta = _PROVIDERS[provider](model, parts, s)
            break
        except RetryableError as e:
            if attempt == MAX_RETRIES:
                raise
            delay = min(60.0, 2 ** attempt) * (1 + random.random())
            print(f"[models] transient error from {model} ({e}); retry {attempt + 1}/{MAX_RETRIES} in {delay:.1f}s")
            time.sleep(delay)

    record = {
        "key": key, "model": model, "provider": provider, "settings": s, "cache_tag": cache_tag, "output": output,
        "n_images": sum(p[0] == "image" for p in parts), "latency_s": round(time.time() - t0, 3),
        "timestamp": datetime.now(timezone.utc).isoformat(), **meta,
    }
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".tmp{threading.get_ident()}")
    tmp.write_text(json.dumps(record, indent=1))
    tmp.replace(path)  # atomic, safe under threads
    return {**record, "cached": False}
