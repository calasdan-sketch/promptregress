"""Minimal YAML subset parser (stdlib only).

Handles the subset these tools need:
  - block mappings and lists, nested by indentation
  - inline-map list items ("- key: value")
  - flow-style lists of scalars ("[a, b]")
  - block scalars: literal "|" and folded ">", with "-" (strip) and "+"
    (keep) chomping, as a mapping value or list item
  - scalars: quoted strings, ints, floats, bools, null
  - "#" comments (outside quotes)

SHARED COPY: the canonical version lives here in adr_drift/miniyaml.py.
pr_verify and promptregress carry verbatim copies (see their headers).
Do not fork — fix the canonical copy and re-copy.
(promptregress copy is ahead: it adds | and > block scalars — port them
to the canonical copy.)

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

import re

_KEY_RE = re.compile(r"^([A-Za-z0-9_.\-/ ]+):(?:\s|$)")
# "key: |", "- >-", "- key: |+" ... -> (prefix, style, header flags)
_BLOCK_RE = re.compile(r"^(- +(?:[A-Za-z0-9_.\-/ ]+: +)?|[A-Za-z0-9_.\-/ ]+: +)"
                       r"([|>])([-+0-9]*)$")
_BLOCK_TAG = "\x00block:"

#: Deepest block-nesting level accepted. Deeper input raises ValueError
#: (the parser's error contract) instead of RecursionError.
MAX_DEPTH = 50


def _err(msg: str, lineno: int) -> ValueError:
    return ValueError(f"miniyaml: {msg} (line {lineno})")


def _strip_comment(line: str) -> str:
    in_s = in_d = False
    for i, ch in enumerate(line):
        if ch == "'" and not in_d:
            in_s = not in_s
        elif ch == '"' and not in_s:
            in_d = not in_d
        elif ch == "#" and not in_s and not in_d:
            if i == 0 or line[i - 1] in " \t":
                return line[:i]
    return line


def _scalar(s: str):
    s = s.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'":
        inner = s[1:-1]
        if s[0] == '"':
            inner = inner.replace('\\"', '"').replace("\\\\", "\\")
        return inner
    low = s.lower()
    if low in ("null", "~", "none", ""):
        return None
    if low in ("true", "yes", "on"):
        return True
    if low in ("false", "no", "off"):
        return False
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
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
    raw_items, cur, in_s, in_d = [], "", False, False
    for ch in inner:
        if ch == "'" and not in_d:
            in_s = not in_s
            cur += ch
        elif ch == '"' and not in_s:
            in_d = not in_d
            cur += ch
        elif ch == "," and not in_s and not in_d:
            raw_items.append(cur)
            cur = ""
        else:
            cur += ch
    if in_s or in_d:
        raise _err(f"unterminated quote in flow list: {s}", lineno)
    raw_items.append(cur)
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
        ln = raw_lines[i].rstrip()
        if ln and len(ln) - len(ln.lstrip(" ")) <= parent_indent:
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
    if isinstance(obj, str) and obj.startswith(_BLOCK_TAG):
        return blocks[int(obj[len(_BLOCK_TAG):])]
    return obj


def safe_load(text: str):
    # items: (line number, indent, content)
    items, blocks = [], []
    raw_lines = text.splitlines()
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
