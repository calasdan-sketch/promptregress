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

All of it is looked for in the current directory, or in `--root DIR` (given
before the subcommand: `promptregress --root path/to/project run --all`).

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
YAML's escapes: `\n` `\t` `\r` `\"` `\\` `\/` `\0` `\a` `\b` `\v` `\f` `\e`
`\N` `\_` `\L` `\P`, a backslash and a space, and `\xE9`, `\u00e9` and
`\U0001F600` (2, 4 and 8 hex digits: how PyYAML writes an accented letter or
an emoji). Any other backslash there is a line-numbered error, never kept as
text: a Windows path needs `\\` for each backslash (`"C:\\Users\\dan"`), or
single quotes (`'C:\Users\dan'`), and half of a surrogate pair (`\ud83d` on
its own) is not a character. Single-quoted strings are literal, except that
`''` is one apostrophe (`'it''s'` is `it's`); plain and block scalars have no
escapes at all. The cases file is read by a small built-in YAML subset
parser (no PyYAML needed). It reads block mappings and lists - a list may
start at the indent of its key (`cases:` and `- name: ...` below it, as
PyYAML writes it) or be indented under it - `[a, b]` lists of scalars, block
scalars, and plain or quoted values. A key is written without quotes, in
ASCII letters, digits, spaces and `_ . - /`: a quoted key (`"input": ...`)
is refused with that fix, as are anchors, aliases, tags, flow maps (`{...}`
as a value), a nested `[...]`, explicit indentation indicators (`|2`), a
key given twice and a value continued on the next line (PyYAML wraps a long
string that way unless it is dumped with a large `width`). Each is a
line-numbered error.

A comment is a `#` that starts a line or follows a space, outside quotes:
`input: It's fine   # TODO` sends `It's fine`. A quote only opens a quoted
string at the start of a value, so the apostrophe in a word does not hide
the comment (and `5 o'clock # soon` is `5 o'clock`). A NUL character in the
file is refused.

Mind YAML's plain scalars: `contains: yes` is the boolean `true`,
`contains: ~` is null, `input: no` is `false`. Those are refused, with the
fix - quote the text: `contains: "yes"`. Plain numbers are fine and keep the
text you wrote: `input: 42` sends `42`, `contains: 3.10` looks for `3.10`
(not `3.1`), `starts_with: 007` for `007` (`max_tokens: "10"` is accepted as
10). A file saved with a UTF-8 byte-order mark (Notepad's "UTF-8") reads
like any other.

A case has `name` (optional: `case-0`, `case-1` ... when left out), `input`,
`vars` (optional extra template variables: `{{tone}}` in the prompt is
replaced by the text of `tone`) and `assert` - nothing else. A key such as
`varss:` would otherwise be dropped without a word; notes go in a `#`
comment.

The prompt takes a case's input where it says `{{input}}`, spelt exactly so -
`{{ input }}`, `{{Input}}` and `{input}` are not replaced - and `{{tone}}`
takes the `tone` the case declares under `vars`. Any other `{{name}}` goes to
the model as written, and a prompt with no `{{input}}` never sees the inputs
at all. `run` and `diff` say so on stderr, one `warning:` line per problem
naming the prompt file, and carry on: the verdict and the exit code are what
they would have been. Cases that all say `input: ""` need no `{{input}}`.

### Checked before anything runs

`run`, `diff` and `list` read the whole cases file first, every case and
every assertion, and a file that cannot be used as a suite exits 2 with one
line per problem, each naming the file, the case and the key. Nothing is
sent to the model, and `run --all` reads every suite before it runs the
first:

```
error: prompts/sentiment.cases.yaml: case 1 ('happy review'): assert[0]: unknown assertion 'contain' (did you mean 'contains'?) - the assertions are contains, not_contains, ...
prompts/sentiment.cases.yaml: case 2 ('neutral review'): assert: has no assertions - a case that checks nothing always passes
```

What is refused: a case without `input` or `assert`; an unknown case key; a
`name`, `input` or `vars` value that is a boolean or a list; an `input` or a
`vars` value that is empty (write `""` if empty is what you mean); an empty
`assert:`; an assertion that is not a one-key mapping or whose name does not
exist (`contain:`); an operand that is a boolean or null where text is meant;
an empty `contains: ""` (it would match every reply); a `regex` that does not
compile; a token count that is not a whole number of 0 or more; a `schema`
the validator cannot enforce (below); a `judge` without a rubric, or with any
other key. At most 20 problems are listed. A case built by hand through the
Python API skips the loader, and the runner holds it to the same rules: a bad
assertion fails that case with `bad assertion: ...`, it never crashes the run.

## Assertions (deterministic, no LLM)

`contains` · `not_contains` · `regex` · `not_regex` · `starts_with` ·
`ends_with` · `json_valid` · `schema` · `max_tokens` · `min_tokens`.

- `json_valid: true` wants a reply that parses as JSON; `json_valid: false`
  wants one that does not. Anything but true or false is an error.
- `schema` is a mini validator: `type` (one JSON type name, or a list of
  them: `type: [integer, "null"]`), `enum`, `required`, `properties` and
  `items`. `required`, `properties` and `items` apply to any object or array,
  with or without a `type` beside them. `enum` compares as JSON does: `true`
  is not the number 1 (nor `false` the number 0), `1` equals `1.0`, and
  lists and objects are compared item by item the same way, in any key
  order. A keyword beyond those (`minimum`, `pattern`,
  `additionalProperties`, `oneOf` ...) would pass every reply whatever it
  said, so it is an error, not ignored - and so is a type name JSON does not
  have (`str`, `int`: write `string`, `integer`). The annotations `title`,
  `description`, `$schema`, `default` and `examples` are accepted and
  ignored.
- `max_tokens` / `min_tokens` use a rough estimate of one token per 4
  characters - a guardrail, not a tokenizer. An empty or blank reply is 0
  tokens, so `min_tokens: 1` fails it.

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
| `PROMPTREGRESS_TIMEOUT` | `60` | seconds for one request, from connecting to the last byte of the reply: a deadline, however slowly the server sends |
| `PROMPTREGRESS_MAX_REPLY_BYTES` | `8388608` | the largest reply body accepted (8 MiB); a bigger one is an error and is never read into memory |

Flags `--model`, `--base-url`, `--api-key`, `--timeout`, `--max-reply-bytes`
(and `--demo`) override env; they are accepted before or after the subcommand
(`promptregress --base-url … run …` and `promptregress run … --base-url …`
mean the same). Temperature is 0 for determinism.

The API key goes out in one `Authorization: Bearer` header and nowhere
else. Whitespace around it is ignored (a key copied from a file ends in a
newline); a key with a line break, a control character or a non-ASCII
character inside is refused with exit 2, and no message ever shows the key,
not even where a server echoes it back in its error text. Redirects are
never followed: urllib would turn the POST into a GET and send your key to
wherever `Location` points. A 301/302/303/307/308 stops the run with
`model at … redirected the request to … the API key was not sent there`;
point `--base-url` at the new address.

If the model cannot be used, `run`/`diff` exit 2 with the real reason
instead of an all-FAIL table — `model unreachable at …`, `answered HTTP 401
Unauthorized: Invalid API key`, `answered HTTP 404 Not Found`, `timed out
after 60s`, `answered without any text (content is null; it refused: …)` —
and stop at that case: a hung endpoint costs one timeout, not one per
case. Replies that are cut off, not UTF-8 or JSON, missing `choices`, or
carrying no text (a refusal or a tool call instead of an answer) are
reported the same way, never as a crash; message content given as a list
of text parts is joined.

`--timeout` is a deadline for the whole request, not a limit per read: a
server that sends one byte every few seconds is cut off after `--timeout`
seconds in all, like one that sends nothing. A reply larger than
`--max-reply-bytes` is an error too, and a body that announces more is
refused without being read, so a runaway server cannot fill your memory.
Both must be positive numbers (`--timeout` at most a day); anything else is
a usage error, exit 2 - in `PROMPTREGRESS_TIMEOUT` and
`PROMPTREGRESS_MAX_REPLY_BYTES` as well (a `PROMPTREGRESS_TIMEOUT` that was
not a number used to be ignored without a word).

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

The cache only saves money, so it never fails a run. When it cannot be
written - a read-only or root-owned checkout, a file where
`.promptregress/cache` belongs, a full disk - the command prints one
`warning: cache disabled: ...` line on stderr and carries on without it:
the same verdicts and exit code, only the next run pays again. Nothing is
cached while the `.gitignore` cannot be written.

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

A single version is `promptregress run --prompt sentiment --version v2`.
`--format json` lists every suite with its cases.

### Exit codes

The same for the human report and `--format json`:

| Code | Meaning |
|---|---|
| `0` | `run`: every case passed. `list`, `init`: done. `diff`: the comparison was made, whatever it shows |
| `1` | `run`: at least one case failed |
| `2` | usage or configuration error, nothing judged: a prompt or cases file that is missing or malformed; a cases file with no cases (`case:` typed for `cases:`, `cases: []`, an empty file) or with a case or an assertion that cannot be used (see "Checked before anything runs"); a file that cannot be read or written (a prompt file you may not open; `init` in a read-only folder); an API key that cannot be sent (see Model endpoint); a model that cannot be used (unreachable, rejecting the call, redirecting). `list` exits 2 too when a cases file is invalid or unreadable. A response cache that cannot be written is not an error (see Caching) |

A suite with nothing in it never passes: a CI job that ran no cases would
be green and test nothing.

`diff` reports, it does not judge: it exits 0 whenever the comparison was
made, flipped verdicts included - the example at the top of this page exits
0 though a case regressed, because a flip can be a fix as well as a
regression. Gate a build with `run` (or `run --all`); to act on a flip, read
`changed` per case in `diff --format json`.

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

## Limitations

What this tool does not do, so you do not find out the hard way:

- The deadline ends the request on time, but a DNS lookup (or a connect to
  an address that never answers) cannot be interrupted: `timed out after
  ...` is reported on time and the lookup finishes by itself in the
  background.
- A server's error text is scrubbed of your API key as you gave it. A
  server that echoes the key altered (cut, encoded) is shown as it sent it,
  and keys shorter than 8 characters (placeholders such as `ollama`) are
  not scrubbed at all.
- The key travels in a header: over `http://` it is not encrypted. That is
  fine for localhost; use `https://` for anything else.
- `max_tokens` / `min_tokens` count one token per 4 characters, not real
  tokens (see Assertions).
- `schema` is a mini validator and the cases file is read by a YAML subset
  parser. What they do not understand is refused with an error, never
  ignored.
- Requests run one after the other; there is no parallel mode.
- `diff` shows flipped verdicts but exits 0 (see Exit codes); `run` is what
  fails a build.

## Why not Langfuse / LangSmith / Braintrust?

They're platforms with hosted infra and team workflows. This is one file of
YAML and a CLI — the solo-dev "just run my 30 golden cases in CI" slot.

## Tests

`python3 -m unittest discover -s tests`. A few tests skip, and say why: the
check of `pyproject.toml` needs `tomllib` (Python 3.11 and later), and the
ones that depend on file permissions skip on Windows and when you run as
root. The tests are free: they run against fake model servers on 127.0.0.1
and in temporary directories, with no API spend.

## License

MIT — see [LICENSE](LICENSE).
