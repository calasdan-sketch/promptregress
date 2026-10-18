"""Prompts + cases: versioned files, {{var}} templating, YAML cases.

Layout:
  prompts/<name>.v1.txt        prompt version 1 ({{input}} placeholder)
  prompts/<name>.v2.txt        prompt version 2
  prompts/<name>.cases.yaml    golden cases shared across versions

Case YAML:
  cases:
    - name: short review          # optional; case-0, case-1 ... if left out
      input: "This broke after two days."
      vars:                       # optional extra template vars: {{tone}}
        tone: terse
      assert:
        - contains: negative
        - json_valid: true

A case has those four keys and no others (a typo such as `varss:` would
otherwise be dropped without a word). load_cases() checks the whole file -
every case, every assertion - and returns the cases in working form: texts
as text (`input: 3.10` is "3.10"), assertions as asserts.check() has them. A
file that cannot be used as a suite is a ValueError with one line per
problem, each naming the file, the case and the key: a configuration error
(exit 2) before any model is called.
"""

from __future__ import annotations

import difflib
import hashlib
import os
import re

from . import asserts, miniyaml

PROMPTS_DIR = "prompts"


def list_prompts(root: str, prompts_dir: str = PROMPTS_DIR
                 ) -> dict[str, list[str]]:
    """name -> sorted version list, e.g. {'sentiment': ['v1', 'v2']}."""
    d = os.path.join(root, prompts_dir)
    found: dict[str, list[str]] = {}
    if not os.path.isdir(d):
        return found
    for fn in os.listdir(d):
        m = re.match(r"^(.+)\.(v\d+)\.txt$", fn)
        if m:
            found.setdefault(m.group(1), []).append(m.group(2))
    return {k: sorted(v, key=lambda x: int(x[1:]))
            for k, v in sorted(found.items())}


def prompt_path(root: str, name: str, version: str,
                prompts_dir: str = PROMPTS_DIR) -> str:
    return os.path.join(root, prompts_dir, f"{name}.{version}.txt")


def load_prompt(root: str, name: str, version: str,
                prompts_dir: str = PROMPTS_DIR) -> str:
    path = prompt_path(root, name, version, prompts_dir)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"no such prompt: {name}.{version} ({path})")
    with open(path, encoding="utf-8") as f:
        return f.read()


def render(template: str, input_text: str, vars: dict | None = None) -> str:
    out = template.replace("{{input}}", input_text)
    for key, val in (vars or {}).items():
        out = out.replace("{{" + key + "}}", str(val))
    return out


#: Looks like a placeholder: {{input}}, {{ tone }}, {{Input}}. Double braces
#: around anything else ({{"label": "x"}}) are not a name.
_NAME_RE = re.compile(r"\{\{\s*[A-Za-z_][A-Za-z0-9_.-]*\s*\}\}")
#: At most this many left-over names are listed in one warning line.
MAX_LEFTOVERS = 5


def template_problems(template: str, cases: list[dict]) -> list[str]:
    """What is wrong with the way `template` takes its cases: short
    sentences, each meant to follow "<prompt file>: " (empty when all is well).

    render() replaces the exact text {{input}} and, for each key of a case's
    vars, {{key}}. Anything else - {{ input }}, {{Input}}, {input}, a name no
    var declares - reaches the model as it is written, and a prompt with no
    {{input}} never sees the case input at all: the suite then tests nothing
    and passes. Nothing here changes a verdict; the caller warns."""
    problems: list[str] = []
    if "{{input}}" not in template and any(
            str(case.get("input", "")).strip() for case in cases):
        problems.append(
            "no {{input}} in the prompt, so the input of the cases is never "
            "sent to the model (the placeholder is exactly {{input}}: not "
            "{{ input }}, {{Input}} or {input})")
    left: list[str] = []
    for case in cases:
        # what render() leaves of the template for this case
        declared = {key: "" for key in (case.get("vars") or {})}
        for token in _NAME_RE.findall(render(template, "", declared)):
            if token not in left:
                left.append(token)
    if left:
        shown = ", ".join(left[:MAX_LEFTOVERS])
        if len(left) > MAX_LEFTOVERS:
            shown += f" and {len(left) - MAX_LEFTOVERS} more"
        problems.append(
            f"{shown} {'is' if len(left) == 1 else 'are'} left as written in "
            "the prompt: only {{input}} and a {{name}} the case declares "
            "under vars are replaced")
    return problems


CASE_KEYS = ("name", "input", "vars", "assert")
#: At most this many problems are listed, so a file that is wrong everywhere
#: does not flood a CI log.
MAX_PROBLEMS = 20


def _check_case(case: dict, index: int) -> tuple[dict, str, list[str]]:
    """(the case in working form, how to name it in a message, its problems).
    The working form has exactly name, input, vars and assert: every text a
    str (a number is the text it was written as), every assertion as
    asserts.check() returns it. A problem is one line that names the key."""
    problems: list[str] = []
    for key in case:
        if key not in CASE_KEYS:
            close = difflib.get_close_matches(str(key), CASE_KEYS, n=1)
            problems.append(
                f"unknown key {str(key)!r}"
                + (f" (did you mean {close[0]!r}?)" if close else
                   f" (a case has {', '.join(CASE_KEYS)}; put notes in a "
                   f"# comment)"))
    missing = [k for k in ("input", "assert") if k not in case]
    if missing:
        problems.append("needs 'input' and 'assert' (no "
                        + " or ".join(repr(k) for k in missing) + " here)")

    name, label = f"case-{index}", ""
    if case.get("name") is not None:       # `name:` alone is no name either
        try:
            given = asserts.as_text(case["name"], "name")
        except ValueError as e:
            problems.append(str(e))
        else:
            if given.strip():
                name, label = given, f" ({given!r})"

    text = ""
    if "input" in case:
        if case["input"] is None:
            problems.append('input: is empty - write input: "" if an empty '
                            'input is what you mean')
        else:
            try:
                text = asserts.as_text(case["input"], "input")
            except ValueError as e:
                problems.append(str(e))

    variables: dict[str, str] = {}
    declared = case.get("vars")
    if isinstance(declared, dict):
        for key, value in declared.items():
            try:
                variables[str(key)] = asserts.as_text(value, f"vars.{key}")
            except ValueError as e:
                problems.append(str(e))
    elif declared is not None:
        problems.append(f"vars: must be a mapping of name to text, got "
                        f"{asserts.describe(declared)}")

    checked: list[dict] = []
    if "assert" in case:
        listed = case["assert"]
        if listed is None or listed == []:
            problems.append("assert: has no assertions - a case that checks "
                            "nothing always passes")
        elif not isinstance(listed, list):
            problems.append(f"assert: must be a list of assertions (each one "
                            f"starts with `- `), got "
                            f"{asserts.describe(listed)}")
        else:
            for j, item in enumerate(listed):
                try:
                    checked.append(asserts.check(item))
                except ValueError as e:
                    problems.append(f"assert[{j}]: {e}")
    return ({"name": name, "input": text, "vars": variables,
             "assert": checked}, label, problems)


def load_cases(root: str, name: str,
               prompts_dir: str = PROMPTS_DIR) -> list[dict]:
    """The cases of prompt `name`, checked and in working form (_check_case).

    FileNotFoundError when there is no cases file. ValueError when the file
    cannot be used as a suite - malformed YAML, no cases, a case or an
    assertion that is not one - with one line per problem, each naming the
    file, the case and the key."""
    path = os.path.join(root, prompts_dir, f"{name}.cases.yaml")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"no cases file: {path}")
    try:
        # utf-8-sig: a byte-order mark (Notepad's "UTF-8") is not part of
        # the first key
        with open(path, encoding="utf-8-sig") as f:
            text = f.read()
    except UnicodeDecodeError as e:
        raise ValueError(f"{path}: not UTF-8 text ({e.reason} at byte "
                         f"{e.start}) - save the file as UTF-8")
    try:
        data = miniyaml.safe_load(text) or {}
    except ValueError as e:
        raise ValueError(f"{path}: malformed YAML: {e}")
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top-level must be a mapping "
                         f"with a 'cases' list")
    if "cases" not in data:
        # a typo ("case:", "test:") must not turn into a suite that runs
        # nothing and passes: say what the file has instead
        found = ", ".join(repr(str(k)) for k in data) or "nothing"
        raise ValueError(f"{path}: no 'cases:' list found (the file has "
                         f"{found}) - it must start with `cases:`")
    cases = data["cases"]
    if cases is None or cases == []:
        raise ValueError(f"{path}: 'cases' has no cases - a suite that "
                         f"runs nothing would pass, so add at least one")
    if not isinstance(cases, list):
        raise ValueError(f"{path}: 'cases' must be a list")
    problems: list[str] = []
    checked: list[dict] = []
    for i, raw in enumerate(cases):
        if not isinstance(raw, dict):
            problems.append(f"{path}: case {i} needs 'input' and 'assert', "
                            f"got {asserts.describe(raw)}")
            continue
        case, label, found = _check_case(raw, i)
        checked.append(case)
        problems.extend(f"{path}: case {i}{label}: {p}" for p in found)
    if len(problems) > MAX_PROBLEMS:
        more = len(problems) - MAX_PROBLEMS
        problems = problems[:MAX_PROBLEMS] + [
            f"{path}: ... and {more} more problem(s)"]
    if problems:
        raise ValueError("\n".join(problems))
    return checked


def cache_key(name: str, version: str, prompt_text: str,
              case: dict, model: str, base_url: str = "") -> str:
    """base_url is part of the provider's identity: the same model name at
    a different address (fake demo server, other host) is a different
    model and must never share cached outputs."""
    h = hashlib.sha256()
    for part in (name, version, prompt_text, json_dumps(case), model,
                 base_url):
        h.update(part.encode())
        h.update(b"\x00")
    return h.hexdigest()[:32]


def json_dumps(obj) -> str:
    import json
    return json.dumps(obj, sort_keys=True)
