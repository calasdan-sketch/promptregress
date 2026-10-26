"""Deterministic assertions for prompt outputs. No LLM needed.

Supported assertions (each is a one-key dict in the case's `assert` list):

  contains: str            output contains str
  not_contains: str        output does not contain str
  regex: pattern           re.search finds pattern
  not_regex: pattern        re.search finds nothing
  starts_with: str / ends_with: str
  json_valid: true         output parses as JSON (false: it does NOT)
  schema: {...}            parsed JSON validates (mini validator below)
  max_tokens: n / min_tokens: n   ~chars/4 token estimate (documented)

Each returns (passed: bool, detail: str). A malformed assertion (unknown
kind, YAML null/bool where text is needed, a non-integer token count, an
invalid regex, a schema the validator cannot enforce) raises ValueError
with a message the runner shows as "bad assertion: ..." — never a TypeError
or re.error traceback.

A plain number given as text is the text it was written as: `contains: 3.10`
looks for "3.10", `starts_with: 007` for "007" (see miniyaml.scalar_text).

check() holds an assertion to all of those rules and returns it in working
form (text as text, counts as ints). The cases loader runs it on every
assertion of a file, so a bad one is a configuration error before any model
is called; evaluate() runs it again, so a case built by hand gets the same
verdict.
"""

from __future__ import annotations

import difflib
import json
import re

from . import miniyaml


def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token. Documented as approximate;
    use only for guardrail assertions, never billing. An empty or blank text
    is no tokens at all (it used to count as one, so `min_tokens: 1` passed
    an empty reply)."""
    if not text.strip():
        return 0
    return max(1, len(text) // 4)


#: The JSON types the schema validator knows, and the Python-flavoured names
#: people write instead of them (for the hint in the error).
SCHEMA_TYPES = ("object", "array", "string", "number", "integer", "boolean",
                "null")
_TYPE_HINTS = {"str": "string", "text": "string", "int": "integer",
               "bool": "boolean", "float": "number", "double": "number",
               "dict": "object", "map": "object", "list": "array",
               "none": "null", "nil": "null"}
#: What the validator enforces. Any other keyword would pass every reply, so
#: it is an error (check_schema), except these notes that say nothing a
#: reply could break.
SCHEMA_KEYWORDS = ("type", "enum", "required", "properties", "items")
_SCHEMA_NOTES = ("title", "description", "$schema", "$id", "$comment",
                 "default", "examples")

_TYPE_TESTS = {
    "object": lambda v: isinstance(v, dict),
    "array": lambda v: isinstance(v, list),
    "string": lambda v: isinstance(v, str),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    # JSON Schema: 1.0 is an integer too (Python's json reads `1.0` as a float)
    "integer": lambda v: (isinstance(v, int) and not isinstance(v, bool))
    or (isinstance(v, float) and v.is_integer()),
    "boolean": lambda v: isinstance(v, bool),
    "null": lambda v: v is None,
}


def _no_constant(name: str):
    raise ValueError(f"{name} is not valid JSON")


#: Nesting deeper than this is not accepted as JSON. The limit is ours, not the
#: interpreter's: how deep json.loads can go differs between Python versions
#: (3.14 no longer fails at a deep reply), so a RecursionError cannot be relied on.
MAX_JSON_DEPTH = 500


def _too_deep(text: str) -> bool:
    """Linear scan: does [ / { nesting (outside strings) exceed MAX_JSON_DEPTH?"""
    depth = 0
    in_string = False
    escaped = False
    for ch in text:
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch in "[{":
            depth += 1
            if depth > MAX_JSON_DEPTH:
                return True
        elif ch in "]}":
            depth -= 1
    return False


def loads(text: str):
    """json.loads, strictly: NaN, Infinity and -Infinity are Python
    extensions, not JSON (a model that writes them has not written JSON), and
    nesting deeper than MAX_JSON_DEPTH is refused on every Python version."""
    if _too_deep(text):
        raise ValueError(f"nested deeper than {MAX_JSON_DEPTH} levels")
    return json.loads(text, parse_constant=_no_constant)


def describe(value) -> str:
    """A value as a message names it: text in quotes, null as null, anything
    else by its type."""
    if isinstance(value, str):
        shown = repr(value)
        return shown if len(shown) <= 60 else shown[:57] + "..."
    return "null" if value is None else type(value).__name__


def check_schema(schema, path: str = "schema") -> None:
    """ValueError unless `schema` is one the validator can enforce. It checks
    type (one name or a list of them), enum, required, properties and items;
    a keyword beyond those (minimum, pattern, additionalProperties ...) would
    quietly pass every reply, so it is refused, as is a type it does not
    know (`str`, `int`: JSON says string, integer)."""
    if not isinstance(schema, dict):
        raise ValueError(f"{path}: must be a mapping, got {describe(schema)}")
    for key in schema:
        if key not in SCHEMA_KEYWORDS and key not in _SCHEMA_NOTES:
            raise ValueError(
                f"{path}: '{key}' is not supported - this validator checks "
                f"{', '.join(SCHEMA_KEYWORDS)} and would pass every reply "
                f"whatever '{key}' says, so it is refused instead of "
                f"ignored (annotations such as title and description are "
                f"fine)")
    if "type" in schema:
        names = schema["type"]
        if isinstance(names, list):
            if not names:
                raise ValueError(f"{path}.type: needs at least one type")
        else:
            names = [names]
        for name in names:
            if not isinstance(name, str) or name not in SCHEMA_TYPES:
                hint = _TYPE_HINTS.get(name.lower()) \
                    if isinstance(name, str) else None
                raise ValueError(
                    f"{path}.type: unknown type {name!r} (use one of "
                    f"{', '.join(SCHEMA_TYPES)})"
                    + (f" - did you mean '{hint}'?" if hint else ""))
    if "enum" in schema and not isinstance(schema["enum"], list):
        raise ValueError(f"{path}.enum: must be a list, got "
                         f"{describe(schema['enum'])}")
    if "required" in schema:
        required = schema["required"]
        if not isinstance(required, list) or not all(
                isinstance(r, str) for r in required):
            raise ValueError(f"{path}.required: must be a list of key names "
                             f"(quote a name that looks like a number)")
    if "properties" in schema:
        props = schema["properties"]
        if not isinstance(props, dict):
            raise ValueError(f"{path}.properties: must be a mapping of key "
                             f"to schema, got {describe(props)}")
        for key, sub in props.items():
            check_schema(sub, f"{path}.properties.{key}")
    if "items" in schema:
        check_schema(schema["items"], f"{path}.items")


def _same_json(a, b) -> bool:
    """Whether two JSON values are equal: true is no number (Python says
    True == 1, JSON does not), 1 is 1.0, and lists and objects are compared
    item by item the same way. Key order does not matter."""
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same_json(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(map(_same_json, a, b))
    if isinstance(a, (dict, list)) or isinstance(b, (dict, list)):
        return False
    return a == b


def _check_schema(data, schema: dict, path: str = "$") -> list[str]:
    """Minimal JSON-schema validator (see check_schema for what it checks).
    Returns list of error strings. required / properties apply to any object
    and items to any array, whether or not `type` is given beside them."""
    errors: list[str] = []
    names = schema.get("type")
    if names:
        names = names if isinstance(names, list) else [names]
        if not any(_TYPE_TESTS[n](data) for n in names):
            errors.append(f"{path}: expected {' or '.join(names)}, "
                          f"got {type(data).__name__}")
            return errors
    if "enum" in schema and not any(_same_json(data, member)
                                    for member in schema["enum"]):
        errors.append(f"{path}: {data!r} not in enum {schema['enum']}")
    if isinstance(data, dict):
        for req in schema.get("required", []):
            if req not in data:
                errors.append(f"{path}: missing required key {req!r}")
        for key, subschema in schema.get("properties", {}).items():
            if key in data:
                errors.extend(_check_schema(data[key], subschema, f"{path}.{key}"))
    if isinstance(data, list) and "items" in schema:
        for i, item in enumerate(data):
            errors.extend(_check_schema(item, schema["items"], f"{path}[{i}]"))
    return errors


TEXT_KINDS = ("contains", "not_contains", "regex", "not_regex",
              "starts_with", "ends_with")
COUNT_KINDS = ("max_tokens", "min_tokens", "max_total_tokens")
#: Budgets: the reply must stay under a cost in US dollars / a time in seconds.
BUDGET_KINDS = ("max_cost_usd", "max_seconds")
#: Every assertion name. `judge` is put to a model by the runner; evaluate()
#: decides all the others itself.
KINDS = (TEXT_KINDS + ("json_valid", "schema") + COUNT_KINDS + BUDGET_KINDS
         + ("judge",))


def as_text(value, where: str) -> str:
    """`value` as the text it stands for, or ValueError naming `where` (the
    key it came from). YAML turns `contains: 1` into an int - fine, that is
    the text "1", as written - but `contains: yes` into a boolean and
    `contains: ~` into null: those are almost never what was meant, so say so
    instead of searching for "True"."""
    text = miniyaml.scalar_text(value)
    if text is not None:
        return text           # a number is the text it was written as
    if isinstance(value, bool):
        raise ValueError(
            f"{where}: {value!r} is a YAML boolean (yes/no/true/false/on/off)"
            f" - quote the text you mean, e.g. {where}: \"yes\"")
    if value is None:
        raise ValueError(f"{where}: needs text, got null (empty or ~) - "
                         f"quote it, e.g. {where}: \"~\"")
    raise ValueError(f"{where}: needs text, got {type(value).__name__} "
                     f"{value!r}")


def _text(kind: str, want) -> str:
    """The text operand of a string assertion. Empty text is in every reply,
    so `contains: ""` would pass them all (and `not_contains: ""` fail them
    all): that is a mistake, not a test."""
    text = as_text(want, kind)
    if not text:
        raise ValueError(
            f"{kind}: is empty, so it would "
            + ("fail every reply - write the text that must not appear"
               if kind in ("not_contains", "not_regex") else
               "match every reply - write the text to look for"))
    return text


def _flag(kind: str, want) -> bool:
    """The true / false operand of json_valid."""
    if isinstance(want, bool):
        return want
    raise ValueError(f"{kind}: needs true or false, got {want!r}")


def _count(kind: str, want) -> int:
    """The whole-number operand of a token-count assertion ("10" -> 10)."""
    number = None
    if isinstance(want, bool) or want is None:
        pass
    elif isinstance(want, int):
        number = int(want)
    elif isinstance(want, float) and want.is_integer():
        number = int(want)
    elif isinstance(want, str) and re.fullmatch(r"[0-9]+", want.strip()):
        number = int(want.strip())
    if number is None or number < 0:
        raise ValueError(f"{kind}: needs a whole number (0 or more), "
                         f"got {want!r}")
    return number


def _amount(kind: str, want) -> float:
    """The positive-number operand of a budget assertion."""
    if (isinstance(want, bool) or not isinstance(want, (int, float))
            or not 0 < want < float("inf")):
        if isinstance(want, str):
            try:
                value = float(want.strip())
            except ValueError:
                value = None
            if value is not None and 0 < value < float("inf"):
                return value
        raise ValueError(f"{kind}: needs a number above 0, got {want!r}")
    return float(want)


def _pattern(kind: str, want) -> re.Pattern:
    text = _text(kind, want)
    try:
        return re.compile(text)
    except (re.error, RecursionError, OverflowError) as e:
        raise ValueError(f"{kind}: invalid regex /{text}/: {e}")


def _rubric(value, where: str) -> str:
    if value is None:
        raise ValueError(f"{where}: needs a rubric, got nothing - say what "
                         f"the output must be to pass")
    text = as_text(value, where)
    if not text.strip():
        raise ValueError(f"{where}: is empty - say what the output must be "
                         f"to pass")
    return text


def _judge(want):
    """The operand of a judge: the rubric as text, or {rubric: text}."""
    if not isinstance(want, dict):
        return _rubric(want, "judge")
    if "rubric" not in want:
        found = ", ".join(repr(str(k)) for k in want) or "nothing"
        raise ValueError(f"judge: needs a 'rubric' (found only {found}) - "
                         f"write `judge: text`, or `rubric: text` under "
                         f"`judge:`")
    extra = [str(k) for k in want if k != "rubric"]
    if extra:
        raise ValueError(f"judge: unknown key {extra[0]!r} - a judge takes "
                         f"only 'rubric'")
    return {"rubric": _rubric(want["rubric"], "judge.rubric")}


def check(assertion) -> dict:
    """The assertion in working form - a text operand as text, a count as an
    int, a judge's rubric as text - or ValueError saying what is wrong with
    it (the key is named, the case and file are the caller's to add). Running
    it on its own result changes nothing."""
    if not isinstance(assertion, dict):
        raise ValueError(f"must be a mapping like `contains: text`, got "
                         f"{describe(assertion)}")
    if len(assertion) != 1:
        found = ", ".join(str(k) for k in assertion)
        raise ValueError(
            "is empty - write one `kind: value` pair" if not assertion else
            f"must have exactly one key, got {len(assertion)} ({found}) - "
            f"start a new list item with `- ` for each assertion")
    ((kind, want),) = assertion.items()
    if kind not in KINDS:
        close = difflib.get_close_matches(str(kind), KINDS, n=1)
        raise ValueError(
            f"unknown assertion {str(kind)!r}"
            + (f" (did you mean {close[0]!r}?)" if close else "")
            + f" - the assertions are {', '.join(KINDS)}")
    if kind in TEXT_KINDS:
        text = _text(kind, want)
        if kind in ("regex", "not_regex"):
            _pattern(kind, text)         # a pattern that cannot compile
        return {kind: text}
    if kind in COUNT_KINDS:
        return {kind: _count(kind, want)}
    if kind in BUDGET_KINDS:
        return {kind: _amount(kind, want)}
    if kind == "json_valid":
        return {kind: _flag(kind, want)}
    if kind == "schema":
        check_schema(want)
        return {kind: want}
    return {kind: _judge(want)}


def evaluate(assertion: dict, output: str,
             context: "dict | None" = None) -> tuple[bool, str]:
    """Decide one assertion about a reply: (passed, detail). ValueError when
    the assertion itself cannot be used (see check).

    `context` (the runner fills it in) carries what the budget assertions
    need: `prompt` (the text that was sent), `seconds` (how long the model
    took) and `price_in` / `price_out` (US dollars per million tokens)."""
    context = context or {}
    kind, want = next(iter(check(assertion).items()))

    if kind == "contains":
        ok = want in output
        return ok, f"contains {want!r}" if ok else f"missing {want!r}"
    if kind == "not_contains":
        ok = want not in output
        return ok, f"absent {want!r}" if ok else f"unexpectedly contains {want!r}"
    if kind == "regex":
        pat = _pattern(kind, want)
        ok = pat.search(output) is not None
        return ok, (f"matches /{pat.pattern}/" if ok
                    else f"no match for /{pat.pattern}/")
    if kind == "not_regex":
        pat = _pattern(kind, want)
        ok = pat.search(output) is None
        return ok, (f"no match for /{pat.pattern}/" if ok
                    else f"unexpected match /{pat.pattern}/")
    if kind == "starts_with":
        ok = output.startswith(want)
        return ok, "starts correctly" if ok else f"does not start with {want!r}"
    if kind == "ends_with":
        ok = output.rstrip().endswith(want)
        return ok, "ends correctly" if ok else f"does not end with {want!r}"
    if kind == "json_valid":
        wanted = _flag(kind, want)
        try:
            loads(output)
            problem = None
        except (ValueError, RecursionError) as e:   # nested too deep: no JSON
            problem = str(e)
        if wanted:
            return (True, "valid JSON") if problem is None else \
                (False, f"invalid JSON: {problem}")
        return (True, "not JSON, as required") if problem is not None else \
            (False, "valid JSON, but json_valid: false asks for text that "
                    "is not")
    if kind == "schema":
        check_schema(want)
        try:
            data = loads(output)
        except (ValueError, RecursionError) as e:
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
    if kind == "max_total_tokens":
        sent = estimate_tokens(context.get("prompt") or "")
        got = estimate_tokens(output)
        return (sent + got <= want,
                f"~{sent + got} tokens in all (~{sent} sent + ~{got} back; max {want})")
    if kind == "max_cost_usd":
        price_in, price_out = context.get("price_in"), context.get("price_out")
        if price_in is None and price_out is None:
            return False, ("cannot price the call: give --price-in / --price-out "
                           "(US dollars per million tokens), or drop max_cost_usd")
        sent = estimate_tokens(context.get("prompt") or "")
        got = estimate_tokens(output)
        cost = (sent * (price_in or 0.0) + got * (price_out if price_out is not None
                                                   else (price_in or 0.0))) / 1e6
        return cost <= want, f"~${cost:.6f} estimated (max ${want:g}; token counts are rough)"
    if kind == "max_seconds":
        seconds = context.get("seconds")
        if seconds is None:
            return False, "no timing for this reply (cache entry written by an older version); run with --no-cache"
        return seconds <= want, f"{seconds:.2f}s (max {want:g}s)"
    raise ValueError(f"{kind}: is put to a model by the runner; it is not "
                     f"decided here")
