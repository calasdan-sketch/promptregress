# promptregress

`pytest` for prompts, local-first. Prompt drift — not model decay — causes
most LLM-ops failures. Version your prompts as files, keep golden cases in
YAML, run them in CI, and diff versions A/B before you ship.

## Install

```bash
git clone <repo> && cd prompt-regress
pip install .   # puts the `promptregress` console script on PATH
```

Or run without installing: `python3 -m promptregress` from the repo root.

## Quick start

```bash
pip install .                                            # puts `promptregress` on PATH
promptregress init                                       # scaffold prompts/ with an example
promptregress --demo run --prompt summarize --version v1 # zero-setup trial, no model needed
promptregress --demo diff v1 v2 --prompt summarize        # A/B on the same golden cases
promptregress list
```

`--demo` runs the suite against a built-in fake model, so you can trial the
whole workflow with zero dependencies — no Ollama, no API key. Drop the flag
once `PROMPTREGRESS_BASE_URL` points at your real endpoint (see below).

(Haven't run `pip install .` yet? Prefix every command with `python3 -m`
from the repo checkout, e.g. `python3 -m promptregress init`.)

## Layout

```
prompts/cls.v1.txt          prompt version 1 ({{input}} placeholder)
prompts/cls.v2.txt          prompt version 2
prompts/cls.cases.yaml      golden cases shared across versions
```

## Golden cases

```yaml
cases:
  - name: angry review
    input: "This broke after two days. Terrible quality."
    assert:
      - contains: negative
      - regex: (?i)\bpositive\b|\bnegative\b|\bneutral\b
      - max_tokens: 10
  - name: structured output
    input: "It's fine."
    assert:
      - json_valid: true
      - schema:
          type: object
          required: [sentiment]
          properties:
            sentiment: {type: string, enum: [positive, negative, neutral]}
```

## Assertions (deterministic, no LLM)

`contains` · `not_contains` · `regex` · `not_regex` · `starts_with` ·
`ends_with` · `json_valid` · `schema` (mini validator: type, required,
properties, items, enum) · `max_tokens` · `min_tokens` (whitespace estimate,
documented as approximate).

Plus an optional LLM judge:

```yaml
assert:
  - judge:
      rubric: "Output must be exactly one word: positive, negative, or neutral."
```

## Model endpoint

Zero dependencies — plain stdlib HTTP against any OpenAI-compatible API:

| Env | Default | Notes |
|---|---|---|
| `PROMPTREGRESS_BASE_URL` | `http://localhost:11434/v1` | Ollama, LM Studio, freellmapi… |
| `PROMPTREGRESS_API_KEY` | `ollama` | |
| `PROMPTREGRESS_MODEL` | `llama3.1` | |

Flags `--model`, `--base-url`, `--api-key` override env. Temperature is 0
for determinism. If the endpoint is unreachable, `run` errors honestly
instead of pretending.

## Caching

Responses are cached in `.promptregress/cache/` keyed by prompt text + case
+ model — `diff v1 v2` never pays twice and CI re-runs are cheap.
LLM-judge verdicts are cached alongside, keyed by prompt version + case +
output hash, so a re-run never re-pays the judge either (transient judge
outages are reported, never cached). `--no-cache` forces fresh calls.

## CI

```bash
promptregress run --prompt cls --version v2 --format json
```

Exit 1 on any failing case (human and `--format json` alike), 2 on errors
(missing prompt, malformed cases YAML, unreachable model).

## Why not Langfuse / LangSmith / Braintrust?

They're platforms with hosted infra and team workflows. This is one file of
YAML and a CLI — the solo-dev "just run my 30 golden cases in CI" slot.

## Name note

The research sketch called this `promptest` — taken on PyPI, npm, and
GitHub (17 repos). `promptregress` is clear on PyPI and npm.

## Tests

37/37 pass: `python3 -m unittest discover -s tests`
(free — tests run against a fake in-process model server, no API spend)
