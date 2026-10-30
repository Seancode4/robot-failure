"""Model settings shared by all experiments (the paper's "thinking enabled" setup).

Temperature is never set: every model runs at its provider default. Each run
records this, with the known differences from the paper, in
results/logs/<name>.meta.jsonl (see runner.run(meta=...)).
"""

GPT5 = "openrouter/openai/gpt-5"
OPUS41 = "openrouter/anthropic/claude-opus-4.1"
GEMINI25_PRO = "openrouter/google/gemini-2.5-pro"
ER2 = "gemini-robotics-er-2-preview"

MAIN = {
    GPT5: {"effort": "medium"},
    OPUS41: {"thinking_budget": 4096},
    GEMINI25_PRO: {},  # thinks by default (695-847 thinking tokens in tests)
    ER2: {},
}

# Text judge for error analysis (a measuring tool, so temperature 0).
JUDGE = "openrouter/deepseek/deepseek-v4-pro-0813"
JUDGE_SETTINGS = {"temperature": 0.0, "thinking": False}

# Video describer: frames -> neutral text description (a measuring tool, so
# temperature 0). An Instruct model, so it has no thinking to turn off.
DESCRIBER = "openrouter/qwen/qwen3-vl-235b-a22b-instruct"
DESCRIBER_SETTINGS = {"temperature": 0.0}

# Paper Fig 8a, constraints, "thinking enabled": % of responses with a violation.
PAPER_VIOLATION_RATE = {GPT5: 41.1, GEMINI25_PRO: 38.9, OPUS41: 38.5}

META = {
    "temperature": "not set for any model; provider defaults",
    "thinking": {
        GPT5: "reasoning.effort=medium",
        OPUS41: "reasoning.max_tokens=4096 (Anthropic budget_tokens), via OpenRouter",
        GEMINI25_PRO: "API default (thinking on, dynamic budget)",
        ER2: "API default; Google docs (checked 2026-10-25) give the default thinking_level as high",
    },
    "unparsed": "rerun up to 2 times; still unparsed after that -> excluded from rates, counted per model",
    "known_differences": [
        "The paper toggled Gemini thinking via system instructions; we use the API setting.",
        "Gemini 2.5 Pro, GPT-5 and Opus 4.1 go through OpenRouter (pinned providers), "
        "not the vendors' own APIs as in the paper.",
        "Gemini Robotics-ER 1.5 (the paper's model) is shut down; ER 2 is used instead.",
    ],
}
