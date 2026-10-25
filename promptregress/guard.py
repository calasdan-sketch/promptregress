"""The prompt-change guard: a prompt that changed needs its cases to change.

A prompt file is prompts/<name>.vN.txt; the cases that exercise it are in
prompts/<name>.cases.yaml. Changing a prompt without touching its cases is
how a prompt gets edited and never re-judged by anything new, so the guard
fails then. It does not run a model.
"""

from __future__ import annotations

import posixpath
import re
import subprocess

from . import suite as suite_mod

_PROMPT_RE = re.compile(r"^(?P<name>.+)\.(?P<version>v\d+)\.txt$")
_CASES_RE = re.compile(r"^(?P<name>.+)\.cases\.yaml$")


class GuardError(Exception):
    """The guard could not be run (git failed, a bad ref ...): exit 2."""


def normalize(path: str) -> str:
    """A changed-file path as a clean relative '/' path ("./a\\b" -> "a/b")."""
    p = path.strip().replace("\\", "/")
    while p.startswith("./"):
        p = p[2:]
    return posixpath.normpath(p) if p else ""


def split_changed(values: "list[str]") -> "list[str]":
    """The paths in one or more --changed-files values (commas, and also
    newlines, so `git diff --name-only` can be pasted in)."""
    out: "list[str]" = []
    for value in values:
        for part in re.split(r"[,\n]", value):
            p = normalize(part)
            if p and p != "." and p not in out:
                out.append(p)
    return out


def check(changed: "list[str]", cases_exist=None) -> "list[dict]":
    """The prompts that changed without their cases.

    `changed` are paths relative to the project root (see split_changed).
    `cases_exist(name)` says whether prompts/<name>.cases.yaml exists, for
    the message only. -> one dict per prompt file at fault:
    {"prompt_file", "name", "version", "cases_file", "cases_exist"}, sorted.
    """
    prefix = suite_mod.PROMPTS_DIR + "/"
    changed_cases = set()
    changed_prompts = []
    for path in changed:
        if not path.startswith(prefix):
            continue
        rest = path[len(prefix):]
        if "/" in rest:
            continue    # the layout is flat: prompts/<file>
        m = _CASES_RE.match(rest)
        if m:
            changed_cases.add(m.group("name"))
            continue
        m = _PROMPT_RE.match(rest)
        if m:
            changed_prompts.append((path, m.group("name"),
                                    m.group("version")))
    problems = []
    for path, name, version in changed_prompts:
        if name in changed_cases:
            continue
        problems.append({
            "prompt_file": path, "name": name, "version": version,
            "cases_file": f"{prefix}{name}.cases.yaml",
            "cases_exist": (cases_exist(name) if cases_exist else None),
        })
    return sorted(problems, key=lambda d: d["prompt_file"])


def git_changed(root: str, since: str = "HEAD") -> "list[str]":
    """Files changed under `root` against git ref `since` (working tree
    included), plus new files git does not track yet. Paths are relative to
    `root`; deleted files are left out. Raises GuardError."""
    def run(*args: str) -> str:
        try:
            p = subprocess.run(["git", "-C", root, *args],
                               capture_output=True, timeout=120)
        except FileNotFoundError:
            raise GuardError("git is not installed, so the changed files "
                             "cannot be listed; pass them with "
                             "--changed-files") from None
        except subprocess.TimeoutExpired:
            raise GuardError("git took too long to list the changed "
                             "files") from None
        if p.returncode:
            why = p.stderr.decode("utf-8", "replace").strip().splitlines()
            raise GuardError(why[-1] if why else f"git {args[0]} failed")
        return p.stdout.decode("utf-8", "replace")

    if since.startswith("-"):
        raise GuardError(f"{since!r} is not a git ref")
    run("rev-parse", "--is-inside-work-tree")
    # -z: names are not quoted or escaped, whatever characters they have
    tracked = run("diff", "--name-only", "-z", "--relative",
                  "--diff-filter=d", since, "--").split("\0")
    new = run("ls-files", "--others", "--exclude-standard", "-z").split("\0")
    out: "list[str]" = []
    for name in tracked + new:
        p = normalize(name)
        if p and p != "." and p not in out:
            out.append(p)
    return out
