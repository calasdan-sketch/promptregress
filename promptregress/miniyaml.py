"""Minimal YAML subset parser (stdlib only).

Handles the subset these tools need:
  - block mappings and lists, nested by indentation
  - inline-map list items ("- key: value")
  - flow-style lists of scalars ("[a, b]")
  - block scalars: literal "|" and folded ">", with "-" (strip) and "+"
    (keep) chomping, as a mapping value or list item
  - scalars: quoted strings (double quotes take the JSON escapes:
    backslash-n, backslash-t, escaped quote and backslash, backslash-uXXXX;
    single quotes are literal, except that a doubled '' is one '),
    ints, floats (ASCII decimal forms; a number remembers the text it was
    written as, see scalar_text), bools, null
  - "#" comments: a "#" that starts a line or follows whitespace, outside
    quotes. A quote only opens a quoted string at the START of a scalar, so
    the apostrophe in `It's fine   # TODO` does not hide the comment

NOT supported (raises ValueError): anchors, aliases, tags, block-scalar
indentation indicators ("|2"), flow maps, duplicate keys. Configs using them get a
clear error naming the feature and the line number.

Strictness notes:
  - Every ValueError carries a line number: "miniyaml: <what> (line N)".
  - Unclosed flow lists ("a: [x"), embedded mapping indicators
    ("a: b: c"), unterminated quotes, and nested flow collections are
    rejected instead of silently misparsed.
  - Nesting deeper than MAX_DEPTH (50) raises ValueError, never
    RecursionError.
"""

from __future__ import annotations

import json
import re

_KEY_RE = re.compile(r"^([A-Za-z0-9_.\-/ ]+):(?:\s|$)")
# "key: |", "- >-", "- key: |+" ... -> (prefix, style, header flags)
_BLOCK_RE = re.compile(r"^(- +(?:[A-Za-z0-9_.\-/ ]+: +)?|[A-Za-z0-9_.\-/ ]+: +)"
                       r"([|>])([-+0-9]*)$")
_BLOCK_TAG = "\x00block:"

# YAML's core schema: ASCII decimal digits only. Python's int() and float()
# also take "1_000", "nan", "Infinity" and the digits of other scripts, which
# are words, not numbers.
_INT_RE = re.compile(r"[-+]?[0-9]+")
_FLOAT_RE = re.compile(r"[-+]?(?:[0-9]+\.[0-9]*|\.[0-9]+|[0-9]+(?=[eE]))"
                       r"(?:[eE][-+]?[0-9]+)?")

#: Deepest block-nesting level accepted. Deeper input raises ValueError
#: (the parser's error contract) instead of RecursionError.
MAX_DEPTH = 50


def _err(msg: str, lineno: int) -> ValueError:
    return ValueError(f"miniyaml: {msg} (line {lineno})")


# What leads a line's scalar: indentation, one "- ", one "key: ".
_LEAD_RE = re.compile(r"[ \t]*(?:-(?:[ \t]+|$))?"
                      r"(?:[A-Za-z0-9_.\-/ ]+:(?:[ \t]+|$))?")


def _quoted_end(s: str, i: int) -> int | None:
    """Index just past the quoted string that opens at s[i], or None when it
    never closes. Double quotes take backslash escapes; in single quotes a
    doubled '' is an apostrophe, not the end."""
    quote, n = s[i], len(s)
    i += 1
    while i < n:
        ch = s[i]
        if quote == '"' and ch == "\\":
            i += 2
        elif ch != quote:
            i += 1
        elif quote == "'" and s[i + 1:i + 2] == "'":
            i += 2
        else:
            return i + 1
    return None


def _flow_end(s: str, i: int) -> int:
    """Index just past the flow list that opens at s[i]: len(s) when it never
    closes, the index of a comment's "#" when one cuts it short. Inside it a
    quote only opens a quoted item at the start of an item."""
    n = len(s)
    i += 1
    at_item_start = True
    while i < n:
        ch = s[i]
        if ch in " \t":
            i += 1
            continue
        if ch == "#" and s[i - 1] in " \t":
            return i
        if at_item_start and ch in ("'", '"'):
            end = _quoted_end(s, i)
            if end is None:
                return n
            i, at_item_start = end, False
            continue
        if ch == "]":
            return i + 1
        at_item_start = ch == ","
        i += 1
    return n


def _strip_comment(line: str) -> str:
    """The line without its trailing comment: a "#" that starts the line or
    follows whitespace, outside a quoted string. In YAML a quote only opens a
    quoted string at the start of a scalar -- the apostrophe in `It's fine
    # TODO` and in `5 o'clock` is a character like any other -- so a line is
    read as what leads it (indent, "- ", "key:") and then one scalar: quoted,
    a flow list, or plain text."""
    i = _LEAD_RE.match(line).end()
    first = line[i:i + 1]
    if first in ("'", '"'):
        i = _quoted_end(line, i)
        if i is None:
            return line          # unterminated: the value check says so
    elif first == "[":
        i = _flow_end(line, i)
    while True:
        j = line.find("#", i)
        if j < 0:
            return line
        if j == 0 or line[j - 1] in " \t":
            return line[:j]
        i = j + 1


class _RawInt(int):
    """An int that remembers the text it was written as ("007", "+5")."""
    raw = ""


class _RawFloat(float):
    """A float that remembers the text it was written as ("3.10", "1e5")."""
    raw = ""


def scalar_text(value) -> str | None:
    """The text a scalar was written as, for the places that need text: what
    a plain number looked like in the file, not what Python prints for it
    (3.10 -> "3.10", not "3.1"; 007 -> "007"). None for anything that is
    neither text nor a number: a boolean, null, a list, a mapping."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (_RawInt, _RawFloat)):
        return value.raw
    if isinstance(value, (str, int, float)):
        return str(value)
    return None


class _Block:
    """Stands for the n-th block scalar until the whole text is parsed. A
    class of our own: no text in the file can be mistaken for it (a quoted
    "\u0000block:0" decodes to the same characters as the placeholder)."""

    def __init__(self, index: int):
        self.index = index


def _scalar(s: str):
    s = s.strip()
    if s.startswith(_BLOCK_TAG):      # our own placeholder (see safe_load)
        return _Block(int(s[len(_BLOCK_TAG):]))
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'":
        inner = s[1:-1]
        if s[0] == '"':
            # Double-quoted scalars use the JSON escape set (\n, \t, \",
            # \\, \uXXXX); anything JSON rejects (a YAML-only escape such
            # as \e) keeps the old \" and \\ handling.
            try:
                return json.loads(s)
            except ValueError:
                inner = inner.replace('\\"', '"').replace("\\\\", "\\")
        else:
            # Single quotes are literal; their only escape is '' -> '.
            inner = inner.replace("''", "'")
        return inner
    low = s.lower()
    if low in ("null", "~", "none", ""):
        return None
    if low in ("true", "yes", "on"):
        return True
    if low in ("false", "no", "off"):
        return False
    if _INT_RE.fullmatch(s):
        try:
            number = _RawInt(int(s))
        except ValueError:            # more digits than int() will take
            return s
        number.raw = s
        return number
    if _FLOAT_RE.fullmatch(s):
        number = _RawFloat(float(s))
        number.raw = s
        return number
    return s


def _check_no_mapping_indicator(v: str, lineno: int):
    """Reject ':' inside a plain scalar ("a: b: c") — PyYAML rejects it too.

    v is an unquoted scalar here; stray quote chars are tracked so a
    colon inside e.g. "it's: fine" does not false-positive.
    """
    in_s = in_d = False
    for i, ch in enumerate(v):
        if ch == "'" and not in_d:
            in_s = not in_s
        elif ch == '"' and not in_s:
            in_d = not in_d
        elif ch == ":" and not in_s and not in_d:
            nxt = v[i + 1] if i + 1 < len(v) else ""
            if nxt in ("", " ", "\t"):
                raise _err(
                    "unexpected ':' in scalar "
                    f"(nested mappings need their own lines): {v!r}",
                    lineno,
                )


def _check_value(val: str, lineno: int, in_flow: bool = False):
    """Reject unsupported/malformed value shapes before _scalar() sees them.

    Anchors (&), aliases (*), and tags (!) are named explicitly so a
    config using them fails loudly instead of silently dropping content.
    """
    v = val.strip()
    if not v:
        return
    c0 = v[0]
    if c0 == "&":
        raise _err("YAML anchors (&) are not supported", lineno)
    if c0 == "*":
        raise _err("YAML aliases (*) are not supported", lineno)
    if c0 == "!":
        raise _err("YAML tags (!) are not supported", lineno)
    if c0 in ("|", ">"):
        raise _err("multi-line (|, >) scalars are not supported", lineno)
    if c0 in "\"'":
        # Quoted scalars are opaque — nothing else to check.
        if len(v) < 2 or v[-1] != c0:
            raise _err(f"unterminated quoted string: {v!r}", lineno)
        return
    if c0 == "{":
        raise _err("flow-style maps ({...}) are not supported", lineno)
    if c0 == "[":
        if in_flow:
            raise _err("nested flow collections are not supported", lineno)
        if not v.endswith("]"):
            raise _err(f"unclosed flow list: {v!r}", lineno)
        return  # balanced "[...]"; items are checked per-item in _flow_list
    if in_flow and any(ch in v for ch in "[]{}"):
        raise _err("nested flow collections are not supported", lineno)
    _check_no_mapping_indicator(v, lineno)


def _flow_list(s: str, lineno: int) -> list:
    """Parse [a, b, "c"] — scalars only."""
    inner = s.strip()[1:-1].strip()
    if not inner:
        return []
    # a quote opens a quoted item only at the start of an item
    raw_items, start = [], 0
    while True:
        i = start
        while i < len(inner) and inner[i] in " \t":
            i += 1
        if inner[i:i + 1] in ("'", '"'):
            i = _quoted_end(inner, i)
            if i is None:
                raise _err(f"unterminated quote in flow list: {s}", lineno)
        comma = inner.find(",", i)
        if comma < 0:
            raw_items.append(inner[start:])
            break
        raw_items.append(inner[start:comma])
        start = comma + 1
    out = []
    for item in raw_items:
        _check_value(item, lineno, in_flow=True)
        out.append(_scalar(item))
    return out


def _fold(lines: list[str]) -> str:
    """Folded (>) style: single breaks between plain lines become spaces;
    blank lines and more-indented lines keep their breaks."""
    out, prev, empties = [], None, 0
    for ln in lines:
        if not ln:
            empties += 1
            continue
        if prev is None:
            out.append("\n" * empties)
        elif empties == 0 and ln[0] != " " and prev[0] != " ":
            out.append(" ")
        elif ln[0] != " " and prev[0] != " ":
            out.append("\n" * empties)
        else:
            out.append("\n" * (empties + 1))
        out.append(ln)
        prev, empties = ln, 0
    return "".join(out)


def _block_scalar(raw_lines, i, parent_indent, style, chomp):
    """Read block-scalar body lines after raw_lines[i-1]. Returns
    (value, index of the first line after the block)."""
    body, n = [], len(raw_lines)
    while i < n:
        # Only line breaks are stripped: trailing spaces are content in a
        # block scalar (PyYAML keeps them), and a whitespace-only line is
        # an empty line, never the end of the block.
        ln = raw_lines[i].rstrip("\r\n")
        if ln.strip() and len(ln) - len(ln.lstrip(" ")) <= parent_indent:
            break
        body.append(ln)
        i += 1
    content = [ln for ln in body if ln.strip()]
    ind = min((len(ln) - len(ln.lstrip(" ")) for ln in content[:1]),
              default=0)
    lines = []
    for k, ln in enumerate(body):
        if ln.strip() and len(ln) - len(ln.lstrip(" ")) < ind:
            raise _err("block scalar line is less indented than its first "
                       "line", i - len(body) + k + 1)
        lines.append(ln[ind:])
    trailing = 0
    while lines and not lines[-1]:
        lines.pop()
        trailing += 1
    text = "\n".join(lines) if style == "|" else _fold(lines)
    if chomp == "-" or not lines:
        return (text + "\n" * trailing if chomp == "+" else text), i
    return text + "\n" + ("\n" * trailing if chomp == "+" else ""), i


def _resolve(obj, blocks):
    if isinstance(obj, dict):
        return {k: _resolve(v, blocks) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve(v, blocks) for v in obj]
    if isinstance(obj, _Block):
        return blocks[obj.index]
    return obj


def safe_load(text: str):
    # items: (line number, indent, content)
    items, blocks = [], []
    raw_lines = text.splitlines()
    # the placeholder for a block scalar contains a NUL; none may be in the text
    for n, ln in enumerate(raw_lines, 1):
        if "\x00" in ln:
            raise _err("NUL characters are not allowed", n)
    i = 0
    while i < len(raw_lines):
        raw = raw_lines[i]
        i += 1
        lineno = i
        line = _strip_comment(raw).rstrip()
        if not line.strip():
            continue
        if "\t" in line[: len(line) - len(line.lstrip())]:
            raise _err("tabs not allowed for indentation", lineno)
        indent = len(line) - len(line.lstrip(" "))
        content = line.strip()
        if content in ("---", "..."):
            continue
        m = _BLOCK_RE.match(content)
        if m:
            prefix, style, chomp = m.groups()
            if chomp not in ("", "-", "+"):
                raise _err("block scalar indentation indicators "
                           f"({style}{chomp}) are not supported", lineno)
            # "- key: |": the body must be indented past the key, not
            # just past the dash.
            parent = indent
            if prefix.startswith("-") and ":" in prefix:
                parent += len(prefix) - len(prefix[1:].lstrip(" "))
            value, i = _block_scalar(raw_lines, i, parent, style, chomp)
            content = f"{prefix}{_BLOCK_TAG}{len(blocks)}"
            blocks.append(value)
        items.append((lineno, indent, content))
    if not items:
        return None
    val, pos = _parse_block(items, 0, items[0][1], 0)
    if pos != len(items):
        lineno, _, content = items[pos]
        raise _err(f"unexpected content: {content!r}", lineno)
    return _resolve(val, blocks) if blocks else val


def _parse_block(items, pos, indent, depth):
    if depth > MAX_DEPTH:
        raise _err(
            f"maximum nesting depth ({MAX_DEPTH}) exceeded", items[pos][0]
        )
    first = items[pos][2]
    if first == "-" or first.startswith("- "):
        return _parse_list(items, pos, indent, depth)
    return _parse_map(items, pos, indent, depth)


def _parse_map(items, pos, indent, depth):
    d: dict = {}
    while pos < len(items):
        lineno, ind, line = items[pos]
        if ind < indent:
            break
        if ind > indent:
            raise _err(f"bad indentation: {line!r}", lineno)
        if line == "-" or line.startswith("- "):
            raise _err(f"list item inside mapping: {line!r}", lineno)
        if line[0] in "&*!":
            # Anchor/alias/tag where a key was expected — name it.
            _check_value(line, lineno)
        m = _KEY_RE.match(line)
        if not m:
            raise _err(f"expected 'key: value', got: {line!r}", lineno)
        key = m.group(1).strip()
        rest = line[m.end():].strip()
        pos += 1
        if key in d:
            raise _err(f"duplicate key: {key!r}", lineno)
        if rest:
            _check_value(rest, lineno)
            if rest.startswith("["):
                d[key] = _flow_list(rest, lineno)
            else:
                d[key] = _scalar(rest)
        else:
            if pos < len(items) and items[pos][1] > indent:
                val, pos = _parse_block(items, pos, items[pos][1], depth + 1)
                d[key] = val
            else:
                d[key] = None
    return d, pos


def _parse_list(items, pos, indent, depth):
    lst: list = []
    while pos < len(items):
        lineno, ind, line = items[pos]
        if ind < indent:
            break
        if ind > indent:
            raise _err(f"bad indentation: {line!r}", lineno)
        if not (line == "-" or line.startswith("- ")):
            break  # end of list; parent map continues
        content = line[1:].strip()
        pos += 1
        if not content:
            if pos < len(items) and items[pos][1] > indent:
                val, pos = _parse_block(items, pos, items[pos][1], depth + 1)
                lst.append(val)
            else:
                lst.append(None)
            continue
        if _KEY_RE.match(content):
            sub = [(lineno, indent + 2, content)]
            while pos < len(items) and items[pos][1] > indent:
                sub.append(items[pos])
                pos += 1
            val, _ = _parse_map(sub, 0, indent + 2, depth + 1)
            lst.append(val)
        elif content.startswith("[") and content.endswith("]"):
            _check_value(content, lineno)
            lst.append(_flow_list(content, lineno))
        else:
            _check_value(content, lineno)
            lst.append(_scalar(content))
    return lst, pos


# alias mirroring the PyYAML call sites being replaced
def load(text: str):
    return safe_load(text)
