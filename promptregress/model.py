"""Model client: OpenAI-compatible chat completions via stdlib only.

Zero dependencies. Works with Ollama, LM Studio, freellmapi, OpenAI,
or any OpenAI-compatible endpoint.

Config (env or flags):
  PROMPTREGRESS_BASE_URL  default http://localhost:11434/v1 (Ollama)
  PROMPTREGRESS_API_KEY   default "ollama" (Ollama needs none)
  PROMPTREGRESS_MODEL     default llama3.1
  PROMPTREGRESS_TIMEOUT   default 60 (seconds per request; --timeout)
"""

from __future__ import annotations

import json
import os
import urllib.request
import urllib.error


DEFAULT_TIMEOUT = 60.0


class ModelError(RuntimeError):
    pass


def _http_detail(e: urllib.error.HTTPError) -> str:
    """The server's own error text: OpenAI-style {"error": {"message"}}
    when present, else the (whitespace-collapsed) body."""
    try:
        body = e.read(4096).decode("utf-8", "replace")
    except Exception:  # noqa: BLE001 - the status line is still useful
        return ""
    try:
        err = json.loads(body).get("error")
        msg = err.get("message") if isinstance(err, dict) else err
        if isinstance(msg, str) and msg.strip():
            return msg.strip()
    except (ValueError, AttributeError):
        pass
    if body.lstrip().startswith("<"):
        return ""  # an HTML error page adds nothing to the status line
    return " ".join(body.split())[:200]


def _cfg(base_url=None, api_key=None, model=None, timeout=None):
    if timeout is None:
        try:
            timeout = float(os.environ.get("PROMPTREGRESS_TIMEOUT")
                            or DEFAULT_TIMEOUT)
        except ValueError:
            timeout = DEFAULT_TIMEOUT
    return {
        "base_url": (base_url or os.environ.get("PROMPTREGRESS_BASE_URL")
                     or "http://localhost:11434/v1").rstrip("/"),
        "api_key": api_key or os.environ.get("PROMPTREGRESS_API_KEY") or "ollama",
        "model": model or os.environ.get("PROMPTREGRESS_MODEL") or "llama3.1",
        "timeout": timeout,
    }


def _fmt_seconds(t: float) -> str:
    return f"{t:g}s"


def complete(prompt: str, *, base_url=None, api_key=None, model=None,
             temperature: float = 0.0, timeout: float | None = None) -> str:
    """One chat completion. `timeout` (seconds, connect and read) defaults
    to PROMPTREGRESS_TIMEOUT or 60: a hung endpoint fails one request,
    not the whole suite times 120 s."""
    cfg = _cfg(base_url, api_key, model, timeout)
    timeout = cfg["timeout"]
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
    except urllib.error.HTTPError as e:
        # A 401 (bad key) or 404 (wrong path/model) is not "unreachable":
        # surface the status and the server's message.
        detail = _http_detail(e)
        raise ModelError(
            f"model at {cfg['base_url']} answered HTTP {e.code} {e.reason}"
            + (f": {detail}" if detail else ""))
    except urllib.error.URLError as e:
        raise ModelError(
            f"model unreachable at {cfg['base_url']}: {e.reason} "
            f"(set PROMPTREGRESS_BASE_URL / --base-url)")
    except TimeoutError:  # read timeout is raised bare, not as URLError
        raise ModelError(
            f"model at {cfg['base_url']} timed out after "
            f"{_fmt_seconds(timeout)} (raise it with --timeout / "
            f"PROMPTREGRESS_TIMEOUT)")
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
