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

CACHE_DIR = ".promptregress/cache"


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


def _cache_get(root: str, key: str) -> str | None:
    p = _cache_path(root, key)
    if os.path.isfile(p):
        try:
            with open(p, encoding="utf-8") as f:
                return json.load(f)["output"]
        except (json.JSONDecodeError, KeyError, OSError):
            return None
    return None


def _cache_put(root: str, key: str, output: str) -> None:
    p = _cache_path(root, key)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump({"output": output}, f)


def _provider(model_kw: dict) -> tuple[str, str]:
    """(model, base_url) actually used for model_kw, env defaults applied."""
    cfg = model_mod._cfg(model_kw.get("base_url"), None,
                         model_kw.get("model"))
    return cfg["model"], cfg["base_url"]


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


def _judge_cache_get(root: str, key: str) -> tuple[bool, str] | None:
    p = _cache_path(root, key)
    if os.path.isfile(p):
        try:
            with open(p, encoding="utf-8") as f:
                d = json.load(f)
            return bool(d["passed"]), str(d["detail"])
        except (json.JSONDecodeError, KeyError, OSError, TypeError):
            return None
    return None


def _judge_cache_put(root: str, key: str, passed: bool,
                     detail: str) -> None:
    p = _cache_path(root, key)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump({"passed": passed, "detail": detail}, f)


def run_case(root: str, prompt_name: str, version: str, prompt_text: str,
             case: dict, model_kw: dict, use_cache: bool = True,
             judge_model_kw: dict | None = None) -> CaseResult:
    model_name, base_url = _provider(model_kw)
    key = suite_mod.cache_key(prompt_name, version, prompt_text, case,
                              model_name, base_url)
    cached = False
    if use_cache:
        hit = _cache_get(root, key)
        if hit is not None:
            output, cached = hit, True
    if not cached:
        rendered = suite_mod.render(prompt_text, case["input"],
                                    case.get("vars"))
        try:
            output = model_mod.complete(rendered, **model_kw)
        except model_mod.ModelError as e:
            return CaseResult(name=case["name"], passed=False, output="",
                              assertions=[], error=str(e))
        if use_cache:  # same contract as judge verdicts: off = no I/O
            _cache_put(root, key, output)

    results: list[AssertionResult] = []
    for assertion in case["assert"]:
        if isinstance(assertion, dict) and "judge" in assertion:
            spec = assertion["judge"]
            rubric = spec["rubric"] if isinstance(spec, dict) else spec
            jkey = _judge_key(prompt_name, version, case["name"],
                              output, rubric,
                              "|".join(_provider(judge_model_kw or model_kw)))
            hit = _judge_cache_get(root, jkey) if use_cache else None
            if hit is not None:
                passed, detail = hit
            else:
                try:
                    passed, detail = model_mod.judge(
                        output, rubric, **(judge_model_kw or model_kw))
                except model_mod.ModelError as e:
                    # transient outage — report, never cache
                    passed, detail = False, f"judge unavailable: {e}"
                else:
                    if use_cache:
                        _judge_cache_put(root, jkey, passed, detail)
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
              use_cache: bool = True) -> tuple[list[CaseResult], str]:
    prompt_text = suite_mod.load_prompt(root, prompt_name, version)
    cases = cases if cases is not None else suite_mod.load_cases(root,
                                                                 prompt_name)
    results = [run_case(root, prompt_name, version, prompt_text, c,
                        model_kw or {}, use_cache) for c in cases]
    return results, prompt_text
