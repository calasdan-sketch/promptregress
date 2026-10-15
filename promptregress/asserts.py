"""Deterministic assertions for prompt outputs. No LLM needed.

Supported assertions (each is a one-key dict in the case's `assert` list):

  contains: str            output contains str
  not_contains: str        output does not contain str
  regex: pattern           re.search finds pattern
  not_regex: pattern        re.search finds nothing
  starts_with: str / ends_with: str
  json_valid: true         output parses as JSON
  schema: {...}            parsed JSON validates (mini validator below)
  max_tokens: n / min_tokens: n   whitespace-token estimate (documented)

Each returns (passed: bool, detail: str).
"""

from __future__ import annotations

import json
import re


def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token. Documented as approximate;
    use only for guardrail assertions, never billing."""
    return max(1, len(text) // 4)


def _check_schema(data, schema: dict, path: str = "$") -> list[str]:
    """Minimal JSON-schema validator. Returns list of error strings."""
    errors: list[str] = []
    if not isinstance(schema, dict):
        return [f"{path}: schema must be an object"]
    stype = schema.get("type")
    if stype:
        ok = {
            "object": lambda v: isinstance(v, dict),
            "array": lambda v: isinstance(v, list),
            "string": lambda v: isinstance(v, str),
            "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
            "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
            "boolean": lambda v: isinstance(v, bool),
            "null": lambda v: v is None,
        }.get(stype, lambda v: True)(data)
        if not ok:
            errors.append(f"{path}: expected {stype}, got {type(data).__name__}")
            return errors
    if "enum" in schema and data not in schema["enum"]:
        errors.append(f"{path}: {data!r} not in enum {schema['enum']}")
    if stype == "object" and isinstance(data, dict):
        for req in schema.get("required", []):
            if req not in data:
                errors.append(f"{path}: missing required key {req!r}")
        for key, subschema in schema.get("properties", {}).items():
            if key in data:
                errors.extend(_check_schema(data[key], subschema, f"{path}.{key}"))
    if stype == "array" and isinstance(data, list) and "items" in schema:
        for i, item in enumerate(data):
            errors.extend(_check_schema(item, schema["items"], f"{path}[{i}]"))
    return errors


def evaluate(assertion: dict, output: str) -> tuple[bool, str]:
    if not isinstance(assertion, dict) or len(assertion) != 1:
        raise ValueError(f"assertion must be a one-key dict, got {assertion!r}")
    kind, want = next(iter(assertion.items()))

    if kind == "contains":
        ok = want in output
        return ok, f"contains {want!r}" if ok else f"missing {want!r}"
    if kind == "not_contains":
        ok = want not in output
        return ok, f"absent {want!r}" if ok else f"unexpectedly contains {want!r}"
    if kind == "regex":
        ok = re.search(want, output) is not None
        return ok, f"matches /{want}/" if ok else f"no match for /{want}/"
    if kind == "not_regex":
        ok = re.search(want, output) is None
        return ok, f"no match for /{want}/" if ok else f"unexpected match /{want}/"
    if kind == "starts_with":
        ok = output.startswith(want)
        return ok, "starts correctly" if ok else f"does not start with {want!r}"
    if kind == "ends_with":
        ok = output.rstrip().endswith(want)
        return ok, "ends correctly" if ok else f"does not end with {want!r}"
    if kind == "json_valid":
        try:
            json.loads(output)
            return True, "valid JSON"
        except json.JSONDecodeError as e:
            return False, f"invalid JSON: {e}"
    if kind == "schema":
        try:
            data = json.loads(output)
        except json.JSONDecodeError as e:
            return False, f"schema check needs JSON, got: {e}"
        errors = _check_schema(data, want)
        return (not errors,
                "schema valid" if not errors else "; ".join(errors[:3]))
    if kind == "max_tokens":
        n = estimate_tokens(output)
        return n <= want, f"~{n} tokens (max {want})"
    if kind == "min_tokens":
        n = estimate_tokens(output)
        return n >= want, f"~{n} tokens (min {want})"
    raise ValueError(f"unknown assertion: {kind!r}")
