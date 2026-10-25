"""Model client: OpenAI-compatible chat completions via stdlib only.

Zero dependencies. Works with Ollama, LM Studio, freellmapi, OpenAI,
or any OpenAI-compatible endpoint.

Config (env or flags):
  PROMPTREGRESS_BASE_URL  default http://localhost:11434/v1 (Ollama)
  PROMPTREGRESS_API_KEY   default "ollama" (Ollama needs none)
  PROMPTREGRESS_MODEL     default llama3.1
  PROMPTREGRESS_TIMEOUT   default 60 (seconds for one request, from connecting
                          to the last byte of the reply: a deadline, not a
                          limit per read; --timeout)
  PROMPTREGRESS_MAX_REPLY_BYTES
                          default 8388608 (the largest reply body accepted;
                          --max-reply-bytes)

The API key goes out in one header and nowhere else: never in a message,
never to another address (redirects are refused, not followed).
"""

from __future__ import annotations

import contextlib
import http.client
import json
import os
import re
import socket
import threading
import time
import urllib.parse
import urllib.request
import urllib.error


DEFAULT_TIMEOUT = 60.0
MAX_TIMEOUT = 24 * 60 * 60.0       # a day; more than that is a typo
DEFAULT_MAX_REPLY_BYTES = 8 * 1024 * 1024
MAX_REPLY_LIMIT = 1 << 30          # a GiB; more is a typo (and read(n) takes
                                   # a C ssize_t)
WORKER_NAME = "promptregress-request"


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


DEFAULT_API_KEY = "ollama"  # a placeholder: Ollama and LM Studio need none

# What can travel in an Authorization header: printable ASCII. A line break
# or other control character is a header-injection hazard (http.client
# refuses it and quotes the whole header in its error), anything that is not
# ASCII is not a key any provider issues.
_KEY_OK = re.compile(r"[\x20-\x7e]+")

# Keys shorter than this are placeholders ("ollama", "demo"), not secrets,
# and replacing them in a server's message would only mangle the message.
_SECRET_MIN_LEN = 8


def resolve_api_key(api_key=None) -> str:
    """The key to use: the flag, else PROMPTREGRESS_API_KEY, else the
    placeholder. Surrounding whitespace is dropped (a key copied from a file
    or `echo`ed into a secret ends in a newline) and a blank value counts as
    not set."""
    for value in (api_key, os.environ.get("PROMPTREGRESS_API_KEY")):
        if isinstance(value, str) and value.strip():
            return value.strip()
    return DEFAULT_API_KEY


def check_key(key: str) -> None:
    """ModelError when `key` cannot be sent as an HTTP header value. The
    message says what is wrong and never contains any part of the key."""
    if not _KEY_OK.fullmatch(key):
        raise ModelError(
            "the API key (--api-key / PROMPTREGRESS_API_KEY) cannot be "
            "used: it contains a line break, a control character or a "
            "character that is not plain ASCII, which cannot go into an "
            "HTTP header (the key is not shown). Check where it was copied "
            "from")


def _scrub(text: str, key: str) -> str:
    """`text` with the API key, should a server have echoed it, hidden."""
    if len(key) < _SECRET_MIN_LEN or key == DEFAULT_API_KEY:
        return text
    return text.replace(key, "<api key>")


def parse_seconds(value) -> float:
    """`value` (a number, or the text of one) as a deadline in seconds.
    ValueError, in words, unless it is above 0 and at most MAX_TIMEOUT."""
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        seconds = float("nan")
    if not 0 < seconds <= MAX_TIMEOUT:        # false for nan, too
        raise ValueError(
            f"needs a number of seconds above 0 and at most "
            f"{MAX_TIMEOUT:g} (a day), got {value!r}")
    return seconds


def parse_byte_limit(value) -> int:
    """`value` (a whole number, or the text of one) as a size limit in bytes.
    ValueError, in words, unless it is from 1 to MAX_REPLY_LIMIT."""
    number = None
    if isinstance(value, int) and not isinstance(value, bool):
        number = value
    elif isinstance(value, str) and re.fullmatch(r"[0-9]+", value.strip()):
        number = int(value.strip())
    if number is None or not 1 <= number <= MAX_REPLY_LIMIT:
        raise ValueError(f"needs a whole number of bytes from 1 to "
                         f"{MAX_REPLY_LIMIT}, got {value!r}")
    return number


def _setting(value, env_name: str, default, parse, name: str):
    """The setting: `value` if given, else the environment variable, else
    `default`. A bad value is a ModelError that names where it came from."""
    source = name
    if value is None:
        raw = os.environ.get(env_name)
        if raw is None or not raw.strip():
            return default
        value, source = raw, env_name
    try:
        return parse(value)
    except ValueError as e:
        raise ModelError(f"{source} {e}")


def resolve_timeout(timeout=None) -> float:
    """The deadline for one request: the argument, else
    PROMPTREGRESS_TIMEOUT, else DEFAULT_TIMEOUT."""
    return _setting(timeout, "PROMPTREGRESS_TIMEOUT", DEFAULT_TIMEOUT,
                    parse_seconds, "timeout")


def resolve_max_reply_bytes(limit=None) -> int:
    """The largest reply body accepted: the argument, else
    PROMPTREGRESS_MAX_REPLY_BYTES, else DEFAULT_MAX_REPLY_BYTES."""
    return _setting(limit, "PROMPTREGRESS_MAX_REPLY_BYTES",
                    DEFAULT_MAX_REPLY_BYTES, parse_byte_limit,
                    "max_reply_bytes")


def _without_userinfo(url: str) -> str:
    """`url` with any user:password@ cut out, so an error message can show
    what was typed without repeating a secret."""
    try:
        parts = urllib.parse.urlsplit(url)
        if parts.username is None and parts.password is None \
                and "@" not in parts.netloc:
            return url[:200]
        host = parts.netloc.rpartition("@")[2]
        return urllib.parse.urlunsplit(parts._replace(netloc=host))[:200]
    except ValueError:
        return "<unreadable URL>"


def check_base_url(url: str) -> str:
    """`url` if it is an http(s) address with a host and no credentials,
    else a ValueError that says what to do. urllib would also serve file:,
    ftp: and data: (a crash or a read of the local disk), and would send a
    user:password@ part to the wrong host and echo it back in errors."""
    try:
        parts = urllib.parse.urlsplit(url)
        host, _ = parts.hostname, parts.port
    except ValueError:
        raise ValueError(
            f"--base-url {_without_userinfo(url)!r} is not a valid URL "
            "(expected http://host[:port]/v1)") from None
    if parts.scheme.lower() not in ("http", "https") or not host:
        raise ValueError(
            f"--base-url {_without_userinfo(url)!r} must be an http:// or "
            "https:// address, such as http://localhost:11434/v1")
    if "@" in parts.netloc:
        raise ValueError(
            f"--base-url {_without_userinfo(url)!r} must not carry a "
            "user name or password; put the key in PROMPTREGRESS_API_KEY")
    return url


def endpoint(base_url=None, model=None) -> tuple[str, str]:
    """(model, base_url) with the environment's defaults applied: who will
    answer. A base URL that is not a plain http(s) address is a ValueError."""
    url = (base_url or os.environ.get("PROMPTREGRESS_BASE_URL")
           or "http://localhost:11434/v1").rstrip("/")
    return (model or os.environ.get("PROMPTREGRESS_MODEL") or "llama3.1",
            check_base_url(url))


def _cfg(base_url=None, api_key=None, model=None, timeout=None,
         max_reply_bytes=None):
    try:
        model, base_url = endpoint(base_url, model)
    except ValueError as e:
        raise ModelError(str(e)) from None
    return {
        "base_url": base_url,
        "api_key": resolve_api_key(api_key),
        "model": model,
        "timeout": resolve_timeout(timeout),
        "max_reply_bytes": resolve_max_reply_bytes(max_reply_bytes),
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


class _NoRedirects(urllib.request.HTTPRedirectHandler):
    """Follow no redirect, ever. urllib would turn the POST into a GET (301,
    302, 303) and send every header, the API key included, to wherever the
    Location points - another host, or plain http. A chat completion cannot
    be answered by a GET anyway, so a redirect only means the base URL is
    wrong. Declining makes urllib raise the 3xx as an HTTPError, which
    `complete` words (see _redirect_message)."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


_REDIRECT_CODES = (301, 302, 303, 307, 308)


def _shown_url(location: str, base: str) -> str:
    """A redirect target as it is safe to print: scheme://host[:port]/path,
    with no credentials, query string or fragment (a signed URL keeps its
    secret there) and every character that could drive a terminal percent-
    encoded."""
    try:
        parts = urllib.parse.urlsplit(urllib.parse.urljoin(base, location))
    except ValueError:
        return "(an address that cannot be read)"
    host = parts.netloc.rpartition("@")[2]
    shown = (f"{parts.scheme}://{host}" if host else
             f"{parts.scheme}:" if parts.scheme else "") + parts.path
    return urllib.parse.quote(shown, safe="/:@!$&'()*+,;=-._~%[]")


def _redirect_message(err: urllib.error.HTTPError, base: str,
                      request_url: str) -> str | None:
    """The ModelError text for a 3xx that was not followed, or None when
    `err` is not a redirect that points somewhere."""
    headers = getattr(err, "headers", None)
    location = headers.get("Location") or headers.get("URI") if headers else ""
    if err.code not in _REDIRECT_CODES or not location:
        return None
    return (f"model at {base} redirected the request to "
            f"{_shown_url(location, request_url)} (HTTP {err.code}); the API "
            f"key was not sent there. promptregress does not follow "
            f"redirects - point --base-url / PROMPTREGRESS_BASE_URL at the "
            f"new address (what comes before /chat/completions)")


class _Gate:
    """The sockets one request has open, so that its deadline can cut them:
    shutting a socket down wakes up the thread that is blocked reading it."""

    def __init__(self):
        self._lock = threading.Lock()
        self._socks: list = []
        self._closed = False

    def track(self, sock) -> None:
        with self._lock:
            if not self._closed:
                self._socks.append(sock)
                return
        _hang_up(sock)                  # the deadline has passed already

    def close(self) -> None:
        with self._lock:
            self._closed = True
            socks, self._socks = self._socks, []
        for sock in socks:
            _hang_up(sock)


def _hang_up(sock) -> None:
    # socket.socket.shutdown, not sock.shutdown: an SSLSocket's own
    # shutdown() also drops its TLS state under the feet of the other thread
    with contextlib.suppress(OSError, ValueError):
        socket.socket.shutdown(sock, socket.SHUT_RDWR)


class _Tracking:
    """Mixin for urllib's HTTP and HTTPS handlers: the connections they open
    register their sockets with a _Gate - right after the TCP connect (which
    covers a proxy's CONNECT and the TLS handshake) and again once connect()
    is done (the TLS socket is a new object)."""

    def __init__(self, gate: _Gate):
        super().__init__()
        self._gate = gate

    def do_open(self, http_class, req, **kw):
        gate = self._gate

        class Tracked(http_class):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                create = self._create_connection

                def create_and_track(*a, **k):
                    sock = create(*a, **k)
                    gate.track(sock)
                    return sock
                self._create_connection = create_and_track

            def connect(self):
                super().connect()
                gate.track(self.sock)

        return super().do_open(Tracked, req, **kw)


class _HTTP(_Tracking, urllib.request.HTTPHandler):
    pass


if hasattr(urllib.request, "HTTPSHandler"):     # Python built without ssl
    class _HTTPS(_Tracking, urllib.request.HTTPSHandler):
        pass


def _within_deadline(seconds: float, work):
    """work(gate) on a thread of its own, for at most `seconds` in all.
    Returns its value or raises what it raised. When the time is up the
    request's sockets are hung up (so the thread lets go of them at once)
    and TimeoutError is raised; the thread is a daemon, so even one that
    cannot wake up - stuck in a DNS lookup - does not outlive the process."""
    gate = _Gate()
    outcome: list = []

    def run():
        try:
            outcome.append((True, work(gate)))
        except BaseException as e:       # handed on to the caller below
            outcome.append((False, e))

    worker = threading.Thread(target=run, name=WORKER_NAME, daemon=True)
    worker.start()
    end = time.monotonic() + seconds
    try:
        while worker.is_alive():
            left = end - time.monotonic()
            if left <= 0:
                break
            worker.join(min(left, 0.25))  # short slices: Ctrl+C gets through
    finally:
        expired = worker.is_alive()
        if expired:
            gate.close()
            worker.join(1.0)              # a moment to let go of its socket
    if expired:
        raise TimeoutError
    ok, value = outcome[0]
    if ok:
        return value
    raise value


def _timed_out(cfg: dict) -> ModelError:
    return ModelError(
        f"model at {cfg['base_url']} timed out after "
        f"{_fmt_seconds(cfg['timeout'])} (raise it with --timeout / "
        f"PROMPTREGRESS_TIMEOUT)")


def _too_big(cfg: dict) -> ModelError:
    return ModelError(
        f"model at {cfg['base_url']} sent a reply larger than the "
        f"{cfg['max_reply_bytes']} bytes allowed (raise the limit with "
        f"--max-reply-bytes / PROMPTREGRESS_MAX_REPLY_BYTES if the model "
        f"really answers that much)")


def _read_capped(resp, cfg: dict) -> bytes:
    """The body of `resp`, at most cfg["max_reply_bytes"] of it: a server
    that never stops sending must not fill the memory. IncompleteRead when
    the body ends before the Content-Length it announced."""
    limit = cfg["max_reply_bytes"]
    try:
        announced = int(resp.headers.get("Content-Length", ""))
    except ValueError:
        announced = 0
    if announced > limit:
        raise _too_big(cfg)               # no need to read any of it
    raw = resp.read(limit + 1)
    if len(raw) > limit:
        raise _too_big(cfg)
    if resp.length:   # read(n) does not notice a body shorter than announced
        raise http.client.IncompleteRead(raw, resp.length)
    return raw


def complete(prompt: str, *, base_url=None, api_key=None, model=None,
             temperature: float = 0.0, timeout: float | None = None,
             max_reply_bytes: int | None = None) -> str:
    """One chat completion, within `timeout` seconds in all - from connecting
    to the last byte of the reply, however slowly the server sends it
    (default PROMPTREGRESS_TIMEOUT or 60): a hung endpoint fails one request,
    not the whole suite times 120 s. A reply of more than `max_reply_bytes`
    (default PROMPTREGRESS_MAX_REPLY_BYTES or 8 MiB) is an error; it is never
    read into memory.

    Returns the answer text. Everything else that can come back (an HTTP
    error, a cut-off or non-UTF-8 or non-JSON body, a reply with no text)
    raises ModelError with a plain message, and no message ever contains the
    API key."""
    cfg = _cfg(base_url, api_key, model, timeout, max_reply_bytes)
    check_key(cfg["api_key"])
    try:
        return _within_deadline(
            cfg["timeout"],
            lambda gate: _complete(prompt, cfg, temperature, gate))
    except TimeoutError:
        raise _timed_out(cfg) from None
    except ModelError as e:
        # whatever a server echoed back into its error text
        raise ModelError(_scrub(str(e), cfg["api_key"])) from None


def _complete(prompt: str, cfg: dict, temperature: float, gate: _Gate) -> str:
    timeout = cfg["timeout"]
    body = json.dumps({
        "model": cfg["model"],
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
    }).encode()
    req = urllib.request.Request(
        cfg["base_url"] + "/chat/completions", data=body,
        headers={"Content-Type": "application/json"})
    # Not in `headers=`: a header added there is copied onto a request that
    # follows a redirect; an "unredirected" one never is (belt and braces
    # next to _NoRedirects).
    req.add_unredirected_header("Authorization", f"Bearer {cfg['api_key']}")
    handlers = [_NoRedirects, _HTTP(gate)]
    if hasattr(urllib.request, "HTTPSHandler"):
        handlers.append(_HTTPS(gate))
    # Not build_opener(): its defaults would also serve file:, ftp: and data:
    opener = urllib.request.OpenerDirector()
    for handler in (urllib.request.ProxyHandler(),
                    urllib.request.UnknownHandler(),
                    urllib.request.HTTPDefaultErrorHandler(),
                    urllib.request.HTTPErrorProcessor(), *handlers):
        opener.add_handler(handler() if isinstance(handler, type)
                           else handler)
    try:
        with opener.open(req, timeout=timeout) as resp:
            raw = _read_capped(resp, cfg)
    except urllib.error.HTTPError as e:
        redirect = _redirect_message(e, cfg["base_url"], req.full_url)
        if redirect:
            with contextlib.suppress(Exception):
                e.close()
            raise ModelError(redirect)
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
    except TimeoutError:  # a read timeout is raised bare, not as URLError
        raise _timed_out(cfg)
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
