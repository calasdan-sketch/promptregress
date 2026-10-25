"""Runner: execute cases against a prompt version, with caching.

Cache (.promptregress/cache/) is keyed by prompt text + case + model +
provider base URL, so
`diff v1 v2` never pays for the same call twice and CI re-runs are cheap.

LLM-judge verdicts are cached too, keyed by (prompt version, case name,
output hash, judge model + base URL) — a second identical run never re-pays the judge. Transient
judge outages are never cached: only real verdicts are stored.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from . import asserts as asserts_mod
from . import model as model_mod
from . import suite as suite_mod

CACHE_ROOT = ".promptregress"
CACHE_DIR = CACHE_ROOT + "/cache"
# Dropped into .promptregress/ so model outputs never land in a commit;
# the file ignores everything beside itself.
CACHE_GITIGNORE = ("# promptregress response cache: safe to delete, "
                   "never commit\n*\n!.gitignore\n")


def error_text(e: BaseException) -> str:
    """One line for the user. An error of ours (a ValueError, a
    FileNotFoundError("no cases file: ...")) reads as it was written; a
    read or write that failed in the system names the path and the reason
    ("prompts/a.v1.txt: Permission denied"), not Python's "[Errno 13] ..."."""
    if isinstance(e, OSError) and e.errno is not None:
        why = e.strerror or str(e)
        return f"{e.filename}: {why}" if e.filename else why
    return str(e)


def ensure_cache_ignored(root: str) -> bool:
    """Create .promptregress/.gitignore if missing. True when written.
    Raises OSError when it cannot be written."""
    d = os.path.join(root, CACHE_ROOT)
    p = os.path.join(d, ".gitignore")
    if os.path.exists(p):
        return False
    os.makedirs(d, exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        f.write(CACHE_GITIGNORE)
    return True


@dataclasses.dataclass
class AssertionResult:
    assertion: dict
    passed: bool
    detail: str


@dataclasses.dataclass
class CaseResult:
    name: str
    passed: bool
    output: str
    assertions: list[AssertionResult]
    cached: bool = False
    error: str = ""
    seconds: float = 0.0   # wall time of the case (reports only)

    def as_dict(self) -> dict:
        return {
            "name": self.name, "passed": self.passed,
            "output": self.output, "cached": self.cached,
            "error": self.error,
            "assertions": [
                {"assertion": a.assertion, "passed": a.passed,
                 "detail": a.detail} for a in self.assertions],
        }


def _cache_path(root: str, key: str) -> str:
    return os.path.join(root, CACHE_DIR, key + ".json")


def _cache_read(root: str, key: str) -> dict | None:
    """The cached JSON object for key, or None when absent/unreadable.
    Shared by response and judge-verdict entries."""
    p = _cache_path(root, key)
    if not os.path.isfile(p):
        return None
    try:
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    return data if isinstance(data, dict) else None


#: Roots whose cache could not be written, with the reason. The first
#: failure of a command prints one warning and ends that root's cache
#: writes; cli.main() empties this at the start of every command.
_cache_off: dict[str, str] = {}
#: Cases may run on several threads (--jobs): one lock for the cache's
#: first-failure bookkeeping, so the warning is printed once.
_cache_lock = threading.Lock()


def reset_cache_warnings() -> None:
    """Forget which roots had their cache switched off (a command warns
    once; the next command starts afresh)."""
    _cache_off.clear()


def _cache_write(root: str, key: str, payload: dict) -> bool:
    """Store `payload` under `key`. True when it was stored.

    The cache only saves money, so a write that fails - a file where the
    folder belongs, a read-only or full disk, a checkout owned by another
    user - never fails or crashes a run: the first failure prints one
    warning on stderr, the rest are skipped, and the command goes on
    without a cache. The .gitignore is written first and nothing is cached
    without it: model output must not end up in a commit."""
    if root in _cache_off:
        return False
    p = _cache_path(root, key)
    try:
        with _cache_lock:
            os.makedirs(os.path.dirname(p), exist_ok=True)
            ensure_cache_ignored(root)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(payload, f)
    except OSError as e:
        with _cache_lock:
            if root in _cache_off:
                return False
            why = _cache_off[root] = error_text(e)
            print(f"warning: cache disabled: cannot write {why} "
                  "(the run goes on without it)", file=sys.stderr)
        return False
    return True


def _provider(model_kw: dict) -> tuple[str, str]:
    """(model, base_url) actually used for model_kw, env defaults applied."""
    return model_mod.endpoint(model_kw.get("base_url"),
                              model_kw.get("model"))


def _judge_key(prompt_name: str, version: str, case_name: str,
               output: str, rubric: str, provider: str = "") -> str:
    """Cache key for a judge verdict. Includes the rubric so two cases
    that share a name but judge different things never collide, and the
    judge's model + base URL so another judge's verdict is never reused."""
    h = hashlib.sha256()
    for part in (prompt_name, version, case_name, rubric, provider):
        h.update(part.encode("utf-8"))
        h.update(b"\x00")
    h.update(hashlib.sha256(output.encode("utf-8")).digest())
    return "judge-" + h.hexdigest()[:32]


def run_case(root: str, prompt_name: str, version: str, prompt_text: str,
             case: dict, model_kw: dict, use_cache: bool = True,
             judge_model_kw: dict | None = None) -> CaseResult:
    started = time.perf_counter()
    result = _run_case(root, prompt_name, version, prompt_text, case,
                       model_kw, use_cache, judge_model_kw)
    result.seconds = time.perf_counter() - started
    return result


def _run_case(root: str, prompt_name: str, version: str, prompt_text: str,
              case: dict, model_kw: dict, use_cache: bool = True,
              judge_model_kw: dict | None = None) -> CaseResult:
    model_name, base_url = _provider(model_kw)
    key = suite_mod.cache_key(prompt_name, version, prompt_text, case,
                              model_name, base_url)
    cached = False
    if use_cache:
        hit = _cache_read(root, key)
        if hit is not None and isinstance(hit.get("output"), str):
            output, cached = hit["output"], True
    if not cached:
        rendered = suite_mod.render(prompt_text, case["input"],
                                    case.get("vars"))
        try:
            output = model_mod.complete(rendered, **model_kw)
        except model_mod.ModelError as e:
            return CaseResult(name=case["name"], passed=False, output="",
                              assertions=[], error=str(e))
        if use_cache:  # same contract as judge verdicts: off = no I/O
            _cache_write(root, key, {"output": output})

    results: list[AssertionResult] = []
    for assertion in case["assert"]:
        try:
            assertion = asserts_mod.check(assertion)
        except ValueError as e:
            # load_cases refuses such a case; one built by hand (the API) is
            # reported here as a failed assertion instead of crashing the run
            results.append(AssertionResult(assertion, False,
                                           f"bad assertion: {e}"))
            continue
        if "judge" in assertion:
            spec = assertion["judge"]
            rubric = spec["rubric"] if isinstance(spec, dict) else spec
            jkey = _judge_key(prompt_name, version, case["name"],
                              output, rubric,
                              "|".join(_provider(judge_model_kw or model_kw)))
            hit = _cache_read(root, jkey) if use_cache else None
            if hit is not None and "passed" in hit and "detail" in hit:
                passed, detail = bool(hit["passed"]), str(hit["detail"])
            else:
                try:
                    passed, detail = model_mod.judge(
                        output, rubric, **(judge_model_kw or model_kw))
                except model_mod.ModelError as e:
                    # transient outage — report, never cache
                    passed, detail = False, f"judge unavailable: {e}"
                else:
                    if use_cache:
                        _cache_write(root, jkey,
                                     {"passed": passed, "detail": detail})
            results.append(AssertionResult(assertion, passed, detail))
            continue
        try:
            passed, detail = asserts_mod.evaluate(assertion, output)
        except ValueError as e:
            passed, detail = False, f"bad assertion: {e}"
        results.append(AssertionResult(assertion, passed, detail))
    return CaseResult(name=case["name"],
                      passed=all(r.passed for r in results),
                      output=output, assertions=results, cached=cached)


def _warn_about_template(root: str, prompt_name: str, version: str,
                         prompt_text: str, cases: list[dict]) -> None:
    """One stderr line per problem, each naming the prompt file. The run
    goes on and its verdict is what it was: a prompt may not want an input."""
    path = suite_mod.prompt_path(root, prompt_name, version)
    for problem in suite_mod.template_problems(prompt_text, cases):
        print(f"warning: {path}: {problem}", file=sys.stderr)


#: The most cases run at once: more is a typo, or a way to flatten a server.
MAX_JOBS = 64


def run_suite(root: str, prompt_name: str, version: str,
              cases: list[dict] | None = None,
              model_kw: dict | None = None,
              use_cache: bool = True,
              fail_fast: bool = True,
              jobs: int = 1) -> tuple[list[CaseResult], str]:
    """Run every case of prompt_name against `version`.

    fail_fast (default): the first case the model could not answer
    (unreachable, timeout, HTTP 401/404 ...) ends the run — the remaining
    cases would only repeat the same error, each paying the timeout
    again. The returned list is then shorter than the case list; its
    last entry carries the error.

    jobs > 1 sends that many requests at once (a thread pool). The results
    come back in case order whatever order the answers arrived in, and
    fail_fast cuts at the first failing case in case order, so a run reads
    the same as one with jobs=1: cases that were already in flight past
    that point are dropped from the list.
    """
    prompt_text = suite_mod.load_prompt(root, prompt_name, version)
    cases = cases if cases is not None else suite_mod.load_cases(root,
                                                                 prompt_name)
    _warn_about_template(root, prompt_name, version, prompt_text, cases)
    if not 1 <= jobs <= MAX_JOBS:
        raise ValueError(f"jobs must be between 1 and {MAX_JOBS}, "
                         f"got {jobs}")
    kw = model_kw or {}
    results: list[CaseResult] = []
    if jobs == 1 or len(cases) < 2:
        for c in cases:
            r = run_case(root, prompt_name, version, prompt_text, c, kw,
                         use_cache)
            results.append(r)
            if fail_fast and r.error:
                break
        return results, prompt_text
    pool = ThreadPoolExecutor(max_workers=min(jobs, len(cases)),
                              thread_name_prefix="promptregress-case")
    try:
        futures = [pool.submit(run_case, root, prompt_name, version,
                               prompt_text, c, kw, use_cache)
                   for c in cases]
        for future in futures:
            r = future.result()
            results.append(r)
            if fail_fast and r.error:
                break
    except BaseException:
        # Ctrl-C or a bug: do not sit through the requests still in flight
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    pool.shutdown(wait=True, cancel_futures=True)
    return results, prompt_text
