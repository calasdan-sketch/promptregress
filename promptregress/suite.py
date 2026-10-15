"""Prompts + cases: versioned files, {{var}} templating, YAML cases.

Layout:
  prompts/<name>.v1.txt        prompt version 1 ({{input}} placeholder)
  prompts/<name>.v2.txt        prompt version 2
  prompts/<name>.cases.yaml    golden cases shared across versions

Case YAML:
  cases:
    - name: short review
      input: "This broke after two days."
      vars: {tone: terse}          # optional extra template vars
      assert:
        - contains: negative
        - json_valid: true
"""

from __future__ import annotations

import hashlib
import os
import re

from . import miniyaml

PROMPTS_DIR = "prompts"


def list_prompts(root: str, prompts_dir: str = PROMPTS_DIR
                 ) -> dict[str, list[str]]:
    """name -> sorted version list, e.g. {'summarize': ['v1', 'v2']}."""
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


def load_prompt(root: str, name: str, version: str,
                prompts_dir: str = PROMPTS_DIR) -> str:
    path = os.path.join(root, prompts_dir, f"{name}.{version}.txt")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"no such prompt: {name}.{version} ({path})")
    with open(path, encoding="utf-8") as f:
        return f.read()


def render(template: str, input_text: str, vars: dict | None = None) -> str:
    out = template.replace("{{input}}", input_text)
    for key, val in (vars or {}).items():
        out = out.replace("{{" + key + "}}", str(val))
    return out


def load_cases(root: str, name: str,
               prompts_dir: str = PROMPTS_DIR) -> list[dict]:
    path = os.path.join(root, prompts_dir, f"{name}.cases.yaml")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"no cases file: {path}")
    with open(path, encoding="utf-8") as f:
        text = f.read()
    try:
        data = miniyaml.safe_load(text) or {}
    except ValueError as e:
        raise ValueError(f"{path}: malformed YAML: {e}")
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top-level must be a mapping "
                         f"with a 'cases' list")
    cases = data.get("cases", [])
    if not isinstance(cases, list):
        raise ValueError(f"{path}: 'cases' must be a list")
    for i, c in enumerate(cases):
        if not isinstance(c, dict) or "input" not in c or "assert" not in c:
            raise ValueError(f"{path}: case {i} needs 'input' and 'assert'")
        c.setdefault("name", f"case-{i}")
        c.setdefault("vars", {})
    return cases


def cache_key(name: str, version: str, prompt_text: str,
              case: dict, model: str) -> str:
    h = hashlib.sha256()
    h.update(name.encode())
    h.update(version.encode())
    h.update(prompt_text.encode())
    h.update(json_dumps(case).encode())
    h.update(model.encode())
    return h.hexdigest()[:32]


def json_dumps(obj) -> str:
    import json
    return json.dumps(obj, sort_keys=True)
