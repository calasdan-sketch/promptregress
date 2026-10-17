# promptregress

`pytest` for prompts, local-first. Prompt drift — not model decay — causes
most LLM-ops failures. Version your prompts as files, keep golden cases in
YAML, run them in CI, and diff versions A/B before you ship.

Here is one caught before shipping — `sentiment.v2.txt` reworded the
classifier and it started chatting instead of answering in one word:

```
$ promptregress diff v1 v2 --prompt sentiment
case                v1      v2
angry review      pass    pass
happy review      pass    pass
neutral review    pass    FAIL  ← CHANGED

CHANGED neutral review:
  v1 output: 'neutral'
  v2 output: "I'd call this one neutral, leaning slightly positive: the customer seems satisfied but not enthusiastic."

1/3 case(s) changed verdict v1 -> v2
```

```
$ promptregress run --prompt sentiment --version v2
sentiment.v2: ..F

  FAIL neutral review
    ✗ {'max_tokens': 10} -> ~26 tokens (max 10)
    output: "I'd call this one neutral, leaning slightly positive: the customer seems satisfied but not enthusiastic."

2/3 passed
```

Exit code 1, so CI blocks the merge. (Both outputs above are what
`--demo` prints — try them before pointing it at a model.)

## Install

Zero dependencies — Python 3.10+ stdlib only.

```bash
pip install git+https://github.com/calasdan-sketch/promptregress
```

Or from a clone:

```bash
git clone https://github.com/calasdan-sketch/promptregress && cd promptregress
pip install .
```

Either way pip puts the `promptregress` command on your PATH, on every OS.
Or run without installing: `python3 -m promptregress` from the repo root.

## Quick start

```bash
promptregress init                                       # scaffold prompts/ with an example
promptregress --demo run --prompt sentiment --version v1 # zero-setup trial, no model needed
promptregress --demo diff v1 v2 --prompt sentiment       # A/B on the same golden cases
promptregress run --all                                  # every prompt x every version (CI)
promptregress list
```

`--demo` runs the suite against a built-in fake model, so you can trial the
whole workflow with zero dependencies — no Ollama, no API key. It always
uses the fake model: a `PROMPTREGRESS_BASE_URL` left in your environment
(or a `--base-url`) is ignored with a notice, never silently hit. Drop the
flag once `PROMPTREGRESS_BASE_URL` points at your real endpoint (see
below).

The example `sentiment` prompt is a one-word sentiment classifier
(`sentiment.v1.txt`, `sentiment.v2.txt`, `sentiment.cases.yaml`); a `v1`
run prints `sentiment.v1: ...` followed by `3/3 passed`, and `v2` shows the
regression at the top of this page (the fake model "chats" on the neutral
review for v2, so `diff v1 v2` flips one verdict and `run --version v2`
exits 1). `init` also writes `.promptregress/.gitignore`, so the response
cache never ends up in a commit.

(Haven't installed it yet? Prefix every command with `python3 -m`
from the repo checkout, e.g. `python3 -m promptregress init`.)

## Layout

```
prompts/<name>.v1.txt       prompt version 1 ({{input}} placeholder)
prompts/<name>.v2.txt       prompt version 2
prompts/<name>.cases.yaml   golden cases shared across versions
.promptregress/             response cache (git-ignored by `init`)
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
            sentiment:
              type: string
              enum: [positive, negative, neutral]
  - name: long review
    input: |
      Arrived late.
      Box was crushed and the manual was missing.
    assert:
      - contains: negative
```

Multi-line inputs use YAML block scalars: literal `|` keeps line breaks
(and trailing spaces), folded `>` joins lines with spaces; add `-` to strip
or `+` to keep trailing newlines (`|-`, `>+`). Double-quoted strings take
the JSON escapes (`"line1\nline2"`, `\t`, `\"`, `\\`, `\u00e9`); single-quoted
strings are literal, except that `''` is one apostrophe (`'it''s'` is
`it's`). The cases file is read by a small built-in YAML subset
parser (no PyYAML needed); anchors, aliases, tags, flow maps (`{...}` as a
value) and explicit indentation indicators (`|2`) are rejected with a
line-numbered error.

Mind YAML's plain scalars in assertions: `contains: yes` is the boolean
`true`, `contains: ~` is null. Those (and an invalid `regex`) are reported
per case as `bad assertion: …` with the fix — quote the text:
`contains: "yes"`. Plain numbers are fine (`contains: 1` means the text
"1"; `max_tokens: "10"` is accepted as 10).

## Assertions (deterministic, no LLM)

`contains` · `not_contains` · `regex` · `not_regex` · `starts_with` ·
`ends_with` · `json_valid` · `schema` (mini validator: type, required,
properties, items, enum) · `max_tokens` · `min_tokens` (a rough estimate of
one token per 4 characters — a guardrail, not a tokenizer).

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
| `PROMPTREGRESS_API_KEY` | `ollama` | prefer the env var: a key given as `--api-key` is visible to other users via `ps` and your shell history |
| `PROMPTREGRESS_MODEL` | `llama3.1` | |
| `PROMPTREGRESS_TIMEOUT` | `60` | seconds per request |

Flags `--model`, `--base-url`, `--api-key`, `--timeout` (and `--demo`)
override env; they are accepted before or after the subcommand
(`promptregress --base-url … run …` and `promptregress run … --base-url …`
mean the same). Temperature is 0 for determinism. If the model cannot be
used, `run`/`diff` exit 2 with the real reason instead of an all-FAIL
table — `model unreachable at …`, `answered HTTP 401 Unauthorized: Invalid
API key`, `answered HTTP 404 Not Found`, `timed out after 60s`, `answered
without any text (content is null; it refused: …)` — and stop at that
case: a hung endpoint costs one timeout, not one per case. Replies that
are cut off, not UTF-8 or JSON, missing `choices`, or carrying no text (a
refusal or a tool call instead of an answer) are reported the same way,
never as a crash; message content given as a list of text parts is joined.

## Caching

Responses are cached in `.promptregress/cache/` keyed by prompt text + case
+ model + base URL — `diff v1 v2` never pays twice and CI re-runs are cheap,
and pointing at a different endpoint never reuses another provider's answers.
LLM-judge verdicts are cached alongside, keyed by prompt version + case +
output hash + judge model/base URL, so a re-run never re-pays the judge
either (transient judge outages are reported, never cached). `--no-cache`
forces fresh calls and writes nothing; `--demo` never reads or writes the
cache, so fake-model answers can't make a later real run pass. The cache
directory carries its own `.gitignore` (written by `init` and by the first
cached run), so it stays out of your commits.

## CI

```bash
promptregress run --all            # every prompt against each of its versions
```

```
sentiment.v1: ...

3/3 passed

sentiment.v2: ..F

  FAIL neutral review
    ✗ {'max_tokens': 10} -> ~26 tokens (max 10)
    output: "I'd call this one neutral, leaning slightly positive: the customer seems satisfied but not enthusiastic."

2/3 passed

suite         passed
sentiment.v1  3/3
sentiment.v2  2/3  FAIL

2 suite(s), 1 with failures; 5/6 cases passed
```

Exit 1 on any failing case (human and `--format json` alike — the JSON
form lists every suite with its cases), 2 on errors (missing prompt,
malformed cases YAML, model unreachable or rejecting the call). A single
version is `promptregress run --prompt sentiment --version v2`.

Piping is safe: when the reader quits early (`promptregress run --all |
head -5`) the output just stops, with no traceback, and the exit code is
still the verdict (a failing run exits 1, never a false 0; in a shell
pipeline `set -o pipefail` lets you see it).

Ready-to-copy GitHub Actions workflow:
[`examples/promptregress-example.yml`](examples/promptregress-example.yml).
It runs in `--demo` mode only, so CI never depends on a live model — but
demo mode answers with a fake model that only knows the scaffolded
`sentiment` example, so it checks the install and wiring, not your
prompts. To gate on real prompts, point `PROMPTREGRESS_BASE_URL` (and
`PROMPTREGRESS_API_KEY`) at a model reachable from the runner via repo
secrets and run `promptregress run --all` without `--demo` (commented step
in the file).

This repo's own CI ([`.github/workflows/ci.yml`](.github/workflows/ci.yml))
installs the package and runs the test suite on Ubuntu, Windows and macOS
with Python 3.11, 3.12 and 3.13, plus Python 3.10 (the oldest supported)
on Ubuntu.

## Why not Langfuse / LangSmith / Braintrust?

They're platforms with hosted infra and team workflows. This is one file of
YAML and a CLI — the solo-dev "just run my 30 golden cases in CI" slot.

## Tests

111/111 pass: `python3 -m unittest discover -s tests` (on Python 3.10 one of
them skips: it reads `pyproject.toml` with `tomllib`, new in 3.11)
(free — tests run against a fake in-process model server, no API spend)

## License

MIT — see [LICENSE](LICENSE).
