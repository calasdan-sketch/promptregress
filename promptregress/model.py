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

import http.client
import json
import os
import urllib.request
import urllib.error


DEFAULT_TIMEOUT = 60.0


class ModelError(RuntimeError):
    pass


def _error_text(err) -> str:
    """The message of an OpenAI-style error value, {"message": "..."} or a
    bare string; "" when there is none."""
    msg = err.get("message") if isinstance(err, dict) else err
    return msg.strip() if isinstance(msg, str) else ""


def _http_detail(e: urllib.error.HTTPError) -> str:
    """The server's own error text: OpenAI-style {"error": {"message"}}
    when present, else the (whitespace-collapsed) body."""
    try:
        body = e.read(4096).decode("utf-8", "replace")
    except Exception:  # noqa: BLE001 - the status line is still useful
        return ""
    try:
        msg = _error_text(json.loads(body).get("error"))
        if msg:
            return msg
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


_JSON_KINDS = {dict: "an object", list: "an array", str: "a string",
               bool: "a boolean", int: "a number", float: "a number",
               type(None): "null"}


def _kind(value) -> str:
    """Plain-English JSON type of a decoded value, for error messages."""
    return _JSON_KINDS.get(type(value), type(value).__name__)


def _reply_text(data, where: str) -> str:
    """The assistant's text out of a chat-completions reply.

    Anything else is a ModelError that says what came back instead: a
    refusal or tool call (`content: null`), content with no text in it, no
    choices, an error sent with HTTP 200, a body that is not an object.
    Nothing but a string ever leaves here, so no later step (an assertion,
    the judge, the cache) meets None or a list.
    """
    def odd(what: str) -> ModelError:
        return ModelError(f"model at {where} sent an unexpected reply: {what}")

    if not isinstance(data, dict):
        raise odd(f"expected a JSON object, got {_kind(data)}")
    choices = data.get("choices")
    if not choices:
        # some gateways send an error body with HTTP 200
        err = _error_text(data.get("error"))
        if err:
            raise ModelError(
                f"model at {where} returned an error: {err[:200]}")
        raise odd("no choices")
    if not isinstance(choices, list):
        raise odd(f'"choices" is {_kind(choices)}, expected a list')
    first = choices[0]
    message = first.get("message") if isinstance(first, dict) else None
    if not isinstance(message, dict):
        raise odd("the first choice has no message")
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):  # content parts: keep the text ones
        texts = [p["text"] for p in content
                 if isinstance(p, dict) and isinstance(p.get("text"), str)
                 and p.get("type", "text") in ("text", "output_text")]
        if texts:
            return "".join(texts)
        raise ModelError(f"model at {where} answered without any text "
                         f"(content has no text parts)")
    if content is None:
        refusal = message.get("refusal")
        if isinstance(refusal, str) and refusal.strip():
            why = f"; it refused: {refusal.strip()[:200]}"
        elif message.get("tool_calls"):
            why = "; it answered with a tool call instead"
        else:
            why = ""
        state = "null" if "content" in message else "missing"
        raise ModelError(f"model at {where} answered without any text "
                         f"(content is {state}{why})")
    raise odd(f"message content is {_kind(content)}, expected text")


def complete(prompt: str, *, base_url=None, api_key=None, model=None,
             temperature: float = 0.0, timeout: float | None = None) -> str:
    """One chat completion. `timeout` (seconds, connect and read) defaults
    to PROMPTREGRESS_TIMEOUT or 60: a hung endpoint fails one request,
    not the whole suite times 120 s.

    Returns the answer text. Everything else that can come back (an HTTP
    error, a cut-off or non-UTF-8 or non-JSON body, a reply with no text)
    raises ModelError with a plain message."""
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
            raw = resp.read()
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
    except http.client.IncompleteRead as e:
        raise ModelError(
            f"model at {cfg['base_url']} hung up before the whole reply "
            f"arrived (got {len(e.partial)} bytes)")
    except (OSError, http.client.HTTPException) as e:
        # reset mid-reply, hung up without answering, not HTTP at all ...
        raise ModelError(
            f"model at {cfg['base_url']} failed: "
            f"{' '.join(str(e).split()) or type(e).__name__}")
    try:
        data = json.loads(raw.decode("utf-8"))
    except UnicodeDecodeError:  # a ValueError: must come first
        raise ModelError(
            f"model at {cfg['base_url']} sent a reply that is not UTF-8 text")
    except ValueError as e:
        raise ModelError(f"model at {cfg['base_url']} sent a reply that is "
                         f"not valid JSON: {e}")
    return _reply_text(data, cfg["base_url"])


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
