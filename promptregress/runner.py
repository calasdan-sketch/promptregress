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

from . import asserts as asserts_mod
from . import model as model_mod
from . import suite as suite_mod

CACHE_ROOT = ".promptregress"
CACHE_DIR = CACHE_ROOT + "/cache"
# Dropped into .promptregress/ so model outputs never land in a commit;
# the file ignores everything beside itself.
CACHE_GITIGNORE = ("# promptregress response cache: safe to delete, "
                   "never commit\n*\n!.gitignore\n")


def ensure_cache_ignored(root: str) -> bool:
    """Create .promptregress/.gitignore if missing. True when written."""
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


def _cache_write(root: str, key: str, payload: dict) -> None:
    p = _cache_path(root, key)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    ensure_cache_ignored(root)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(payload, f)


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


def run_suite(root: str, prompt_name: str, version: str,
              cases: list[dict] | None = None,
              model_kw: dict | None = None,
              use_cache: bool = True,
              fail_fast: bool = True) -> tuple[list[CaseResult], str]:
    """Run every case of prompt_name against `version`.

    fail_fast (default): the first case the model could not answer
    (unreachable, timeout, HTTP 401/404 ...) ends the run — the remaining
    cases would only repeat the same error, each paying the timeout
    again. The returned list is then shorter than the case list; its
    last entry carries the error.
    """
    prompt_text = suite_mod.load_prompt(root, prompt_name, version)
    cases = cases if cases is not None else suite_mod.load_cases(root,
                                                                 prompt_name)
    results = []
    for c in cases:
        r = run_case(root, prompt_name, version, prompt_text, c,
                     model_kw or {}, use_cache)
        results.append(r)
        if fail_fast and r.error:
            break
    return results, prompt_text
