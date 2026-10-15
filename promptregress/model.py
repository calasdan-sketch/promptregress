"""Model client: OpenAI-compatible chat completions via stdlib only.

Zero dependencies. Works with Ollama, LM Studio, freellmapi, OpenAI,
or any OpenAI-compatible endpoint.

Config (env or flags):
  PROMPTREGRESS_BASE_URL  default http://localhost:11434/v1 (Ollama)
  PROMPTREGRESS_API_KEY   default "ollama" (Ollama needs none)
  PROMPTREGRESS_MODEL     default llama3.1
"""

from __future__ import annotations

import json
import os
import urllib.request
import urllib.error


class ModelError(RuntimeError):
    pass


def _cfg(base_url=None, api_key=None, model=None):
    return {
        "base_url": (base_url or os.environ.get("PROMPTREGRESS_BASE_URL")
                     or "http://localhost:11434/v1").rstrip("/"),
        "api_key": api_key or os.environ.get("PROMPTREGRESS_API_KEY") or "ollama",
        "model": model or os.environ.get("PROMPTREGRESS_MODEL") or "llama3.1",
    }


def complete(prompt: str, *, base_url=None, api_key=None, model=None,
             temperature: float = 0.0, timeout: int = 120) -> str:
    cfg = _cfg(base_url, api_key, model)
    body = json.dumps({
        "model": cfg["model"],
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
    }).encode()
    req = urllib.request.Request(
        cfg["base_url"] + "/chat/completions", data=body,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {cfg['api_key']}"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode())
    except urllib.error.URLError as e:
        raise ModelError(
            f"cannot reach model at {cfg['base_url']} "
            f"(set PROMPTREGRESS_BASE_URL): {e.reason}")
    except (json.JSONDecodeError, KeyError) as e:
        raise ModelError(f"bad response from model: {e}")
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as e:
        raise ModelError(f"unexpected response shape: {e}")


JUDGE_TEMPLATE = """You are a strict test judge. Answer with exactly one word: PASS or FAIL.

Rubric:
{rubric}

Output to judge:
---
{output}
---
"""


def judge(output: str, rubric: str, **model_kw) -> tuple[bool, str]:
    """LLM-as-judge. Returns (passed, detail). Raises ModelError if down."""
    verdict = complete(JUDGE_TEMPLATE.format(rubric=rubric, output=output),
                       **model_kw).strip().upper()
    if verdict.startswith("PASS"):
        return True, "judge: PASS"
    if verdict.startswith("FAIL"):
        # keep the judge's reasoning (if any) after FAIL
        rest = verdict[4:].strip(" :-")
        return False, f"judge: FAIL{(' — ' + rest[:200]) if rest else ''}"
    return False, f"judge gave non-binary verdict: {verdict[:100]!r}"
