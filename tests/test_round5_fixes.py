"""Round-5 panel regressions (final verification panel, 2026-10-17).

Each class starts from what the panel reproduced: a suite that passes with
nothing in it, an API key that leaks, a suite that crashes with exit code 1
(the code for "a case failed") ... All tests run against fake servers on
127.0.0.1 and temp directories; none touches the network, HOME or the real
environment of the machine they run on.
"""

import contextlib
import io
import json
import os
import re
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import urllib.request
from contextlib import redirect_stderr, redirect_stdout
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest import mock

from promptregress import asserts as asserts_mod
from promptregress import cli as cli_mod
from promptregress import miniyaml as miniyaml_mod
from promptregress import model as model_mod
from promptregress import runner as runner_mod
from promptregress import suite as suite_mod
from promptregress.demo import start_fake_model

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROMPT = "Classify: {{input}}\n"
KEY = "sk-test-0000-not-a-real-key"


def make_project(cases_text, prompt=PROMPT, name="s", version="v1"):
    """A temp project with prompts/<name>.<version>.txt and the given
    cases file. Returns the root."""
    root = tempfile.mkdtemp(prefix="prg-r5-")
    d = os.path.join(root, "prompts")
    os.makedirs(d)
    with open(os.path.join(d, f"{name}.{version}.txt"), "w",
              encoding="utf-8", newline="\n") as f:
        f.write(prompt)
    with open(os.path.join(d, f"{name}.cases.yaml"), "w",
              encoding="utf-8", newline="\n") as f:
        f.write(cases_text)
    return root


@contextlib.contextmanager
def clean_env(**extra):
    """The process environment without any PROMPTREGRESS_* setting (a real
    one on the machine running the tests must not leak in), plus `extra`."""
    env = {k: v for k, v in os.environ.items()
           if not k.startswith("PROMPTREGRESS_")}
    env.update(extra)
    with mock.patch.dict(os.environ, env, clear=True):
        yield


def run_cli(argv, **env):
    """cli.main(argv) in this process: (exit code, stdout, stderr)."""
    out, err = io.StringIO(), io.StringIO()
    with clean_env(**env), redirect_stdout(out), redirect_stderr(err):
        code = cli_mod.main(argv)
    return code, out.getvalue(), err.getvalue()


class Recorder(BaseHTTPRequestHandler):
    """Chat-completions endpoint that records every request it gets."""
    answer = "bar"
    requests: list = []

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(n).decode("utf-8", "replace")
        type(self).requests.append({
            "path": self.path, "headers": dict(self.headers.items()),
            "body": json.loads(body) if body else None})
        payload = json.dumps({"choices": [{"message": {
            "role": "assistant", "content": type(self).answer}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass


def start_recorder(answer="bar"):
    """(server, url, requests): requests is the list the handler appends to."""
    handler = type("R", (Recorder,), {"answer": answer, "requests": []})
    server = HTTPServer(("127.0.0.1", 0), handler)
    serve(server)
    return server, f"http://127.0.0.1:{server.server_port}/v1", handler.requests


def serve(server):
    """Serve in a daemon thread; the short poll makes stop() quick."""
    threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.02},
                     daemon=True).start()


def stop(server):
    """Stop a test server and release its listening socket."""
    server.shutdown()
    server.server_close()


# ---------------------------------------------------------------------
# P1 - a suite with nothing in it must not pass

NO_CASES = {
    "typo'd key (case: instead of cases:)":
        "case:\n  - name: a\n    input: x\n    assert:\n      - contains: foo\n",
    "empty list": "cases: []\n",
    "key with nothing after it": "cases:\n",
    "empty file": "",
    "comments only": "# nothing here yet\n",
    "another top-level key only": "name: sentiment\n",
}


class TestEmptySuiteIsAnError(unittest.TestCase):
    """`case:` instead of `cases:` ran zero cases and printed `0/0 passed`
    with exit code 0 - a green CI job that tested nothing."""

    def test_loader_raises_naming_the_file(self):
        for label, text in NO_CASES.items():
            with self.subTest(label):
                root = make_project(text)
                path = os.path.join(root, "prompts", "s.cases.yaml")
                with self.assertRaises(ValueError) as ctx:
                    suite_mod.load_cases(root, "s")
                msg = str(ctx.exception)
                self.assertIn(path, msg)
                self.assertIn("cases", msg)

    def test_the_message_shows_what_the_file_has_instead(self):
        root = make_project(NO_CASES["typo'd key (case: instead of cases:)"])
        with self.assertRaises(ValueError) as ctx:
            suite_mod.load_cases(root, "s")
        self.assertIn("'case'", str(ctx.exception))

    def test_run_exits_2_and_never_prints_a_pass(self):
        server, url, seen = start_recorder()
        try:
            for label, text in NO_CASES.items():
                with self.subTest(label):
                    root = make_project(text)
                    code, out, err = run_cli(
                        ["--root", root, "--base-url", url, "run",
                         "--prompt", "s", "--version", "v1", "--no-cache"])
                    self.assertEqual(code, 2, (out, err))
                    self.assertNotIn("passed", out)
                    self.assertTrue(err.startswith("error: "), err)
                    self.assertIn("cases", err)
                    self.assertNotIn("Traceback", err)
        finally:
            stop(server)
        self.assertEqual(seen, [], "no model call for a suite with no cases")

    def test_json_run_and_diff_and_all_exit_2_too(self):
        root = make_project(NO_CASES["empty list"])
        with open(os.path.join(root, "prompts", "s.v2.txt"), "w") as f:
            f.write(PROMPT)
        for argv in (["run", "--prompt", "s", "--version", "v1",
                      "--format", "json"],
                     ["run", "--all"],
                     ["diff", "v1", "v2", "--prompt", "s"]):
            with self.subTest(argv):
                code, out, err = run_cli(
                    ["--root", root, "--base-url", "http://127.0.0.1:1/v1",
                     *argv])
                self.assertEqual(code, 2, (out, err))
                self.assertEqual(out, "")
                self.assertIn("cases", err)

    def test_list_reports_the_bad_file_instead_of_a_traceback(self):
        root = make_project(NO_CASES["typo'd key (case: instead of cases:)"])
        code, out, err = run_cli(["--root", root, "list"])
        self.assertEqual(code, 2)
        self.assertIn("s: versions v1", out)
        self.assertIn("cases", out + err)
        self.assertNotIn("Traceback", err)


# ---------------------------------------------------------------------
# P2 - the API key must never be shown, and never sent to another address

def start_redirector(location, code=302):
    """Answers every POST with `code` and a Location header; `location` is
    a string or a function of the server's own port."""
    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            self.send_response(code)
            where = location(self.server.server_port) if callable(
                location) else location
            self.send_header("Location", where)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), H)
    serve(server)
    return server, f"http://127.0.0.1:{server.server_port}/v1"


def bearer(requests):
    """The Authorization header of each recorded request."""
    return [r["headers"].get("Authorization") for r in requests]


class TestKeyWhitespaceAndShape(unittest.TestCase):
    """A key ending in CR or LF (copied from a file, `echo`ed into a
    secret) crashed http.client with `Invalid header value b'Bearer sk-...'`
    - and promptregress printed that, key included, on stderr."""

    def test_surrounding_whitespace_is_dropped_from_the_key(self):
        server, url, seen = start_recorder()
        try:
            for raw in (KEY + "\n", KEY + "\r\n", KEY + "\r", "  " + KEY + " \t",
                        "\n" + KEY):
                with self.subTest(repr(raw)):
                    seen.clear()
                    out = model_mod.complete("hi", base_url=url, api_key=raw,
                                             model="m", timeout=5)
                    self.assertEqual(out, "bar")
                    self.assertEqual(bearer(seen), ["Bearer " + KEY])
                    # the same key from the environment
                    with clean_env(PROMPTREGRESS_API_KEY=raw):
                        model_mod.complete("hi", base_url=url, model="m",
                                           timeout=5)
                    self.assertEqual(bearer(seen), ["Bearer " + KEY] * 2)
        finally:
            stop(server)

    def test_a_blank_key_means_no_key(self):
        server, url, seen = start_recorder()
        try:
            with clean_env():
                model_mod.complete("hi", base_url=url, api_key="  \n",
                                   model="m", timeout=5)
        finally:
            stop(server)
        self.assertEqual(bearer(seen), ["Bearer ollama"])

    def test_the_flag_beats_the_environment_and_is_what_gets_sent(self):
        server, url, seen = start_recorder()
        try:
            code, out, err = run_cli(
                ["--root", make_project(
                    "cases:\n  - name: a\n    input: x\n    assert:\n"
                    "      - contains: bar\n"),
                 "--base-url", url, "run", "--prompt", "s", "--version", "v1",
                 "--no-cache", "--api-key", KEY + "\n"],
                PROMPTREGRESS_API_KEY="sk-from-the-environment")
        finally:
            stop(server)
        self.assertEqual(code, 0, err)
        self.assertEqual(bearer(seen), ["Bearer " + KEY])

    def test_unusable_keys_are_refused_without_being_shown(self):
        # an embedded line break, a control character, a character that is
        # not ASCII (http.client would raise UnicodeEncodeError, which names
        # it): none of the key may reach a message, a log or stderr
        bad = ["sk-test-AAAA\nsk-test-BBBB", "sk-test-AAAA\x00BBBB",
               "sk-test-AAAA\tBBBB", "sk-test-AAAA\x7fBBBB",
               "sk-test-AAAA\u2019BBBB", "sk-test-AAAA\u00e9BBBB"]
        server, url, seen = start_recorder()
        try:
            for key in bad:
                with self.subTest(repr(key)):
                    with self.assertRaises(model_mod.ModelError) as ctx:
                        model_mod.complete("hi", base_url=url, api_key=key,
                                           model="m", timeout=5)
                    msg = str(ctx.exception)
                    for piece in ("AAAA", "BBBB", "sk-test"):
                        self.assertNotIn(piece, msg)
                    self.assertIn("API key", msg)
                    root = make_project(
                        "cases:\n  - name: a\n    input: x\n    assert:\n"
                        "      - contains: bar\n")
                    argv = ["--root", root, "--base-url", url, "run",
                            "--prompt", "s", "--version", "v1", "--no-cache"]
                    for how, args, env in (("flag", ["--api-key", key], {}),
                                           ("env", [], {
                                               "PROMPTREGRESS_API_KEY": key})):
                        if how == "env" and "\x00" in key:
                            continue  # an environment cannot hold a NUL
                        code, out, err = run_cli(argv + args, **env)
                        self.assertEqual(code, 2, (how, out, err))
                        self.assertEqual(out, "")
                        self.assertIn("API key", err)
                        for piece in ("AAAA", "BBBB", "sk-test"):
                            self.assertNotIn(piece, err, how)
                        self.assertNotIn("Traceback", err)
        finally:
            stop(server)
        self.assertEqual(seen, [], "an unusable key never reaches the network")

    def test_demo_does_not_care_about_a_broken_key_in_the_environment(self):
        root = tempfile.mkdtemp()
        code, out, err = run_cli(["--root", root, "--demo", "run", "--prompt",
                                  "sentiment", "--version", "v1"],
                                 PROMPTREGRESS_API_KEY="line\nbreak")
        self.assertEqual(code, 0, err)
        self.assertIn("3/3 passed", out)

    def test_a_server_that_echoes_the_key_does_not_get_it_printed(self):
        class Echo(BaseHTTPRequestHandler):
            def do_POST(self):
                self.rfile.read(int(self.headers.get("Content-Length", 0)))
                text = json.dumps({"error": {"message": "bad key: " + (
                    self.headers.get("Authorization") or "")}}).encode()
                self.send_response(401)
                self.send_header("Content-Length", str(len(text)))
                self.end_headers()
                self.wfile.write(text)

            def log_message(self, *args):
                pass

        server = HTTPServer(("127.0.0.1", 0), Echo)
        serve(server)
        try:
            with self.assertRaises(model_mod.ModelError) as ctx:
                model_mod.complete(
                    "hi", base_url=f"http://127.0.0.1:{server.server_port}/v1",
                    api_key=KEY, model="m", timeout=5)
        finally:
            stop(server)
        self.assertIn("HTTP 401", str(ctx.exception))
        self.assertNotIn(KEY, str(ctx.exception))


class TestKeyIsNotSentToAnotherAddress(unittest.TestCase):
    """urllib forwards the Authorization header when a server answers 302
    with a Location on another host: the second server logged the key."""

    def _other(self):
        return start_recorder()

    def test_a_redirect_to_another_origin_is_refused_and_nothing_is_sent(self):
        other, other_url, other_seen = self._other()
        try:
            for code in (301, 302, 303, 307, 308):
                with self.subTest(code=code):
                    first, first_url = start_redirector(other_url + "/chat", code)
                    try:
                        with self.assertRaises(model_mod.ModelError) as ctx:
                            model_mod.complete("hi", base_url=first_url,
                                               api_key=KEY, model="m",
                                               timeout=5)
                    finally:
                        stop(first)
                    msg = str(ctx.exception)
                    self.assertNotIn(KEY, msg)
                    self.assertIn("redirect", msg)
                    self.assertIn(other_url, msg)
                    self.assertIn(f"HTTP {code}", msg)
        finally:
            stop(other)
        self.assertEqual(other_seen, [], "the other server got a request")

    def test_a_redirect_on_the_same_server_is_refused_too(self):
        # the POST would turn into a GET and could not work anyway; the
        # message says where it points so the base URL can be fixed
        first, first_url = start_redirector("/v1/elsewhere", 301)
        try:
            with self.assertRaises(model_mod.ModelError) as ctx:
                model_mod.complete("hi", base_url=first_url, api_key=KEY,
                                   model="m", timeout=5)
        finally:
            stop(first)
        self.assertIn("/v1/elsewhere", str(ctx.exception))
        self.assertNotIn(KEY, str(ctx.exception))

    def test_a_redirect_to_plain_http_from_anywhere_sends_nothing(self):
        # an https -> http downgrade, or a made-up scheme: not even tried
        for target in ("http://127.0.0.1:1/x", "ftp://127.0.0.1/x",
                       "file:///etc/passwd"):
            with self.subTest(target):
                first, first_url = start_redirector(target)
                try:
                    with self.assertRaises(model_mod.ModelError):
                        model_mod.complete("hi", base_url=first_url,
                                           api_key=KEY, model="m", timeout=5)
                finally:
                    stop(first)

    def test_the_message_does_not_repeat_secrets_from_the_location(self):
        first, first_url = start_redirector(
            "http://user:hunter2@127.0.0.1:1/p?token=abc123#frag\x1b[31m")
        try:
            with self.assertRaises(model_mod.ModelError) as ctx:
                model_mod.complete("hi", base_url=first_url, api_key=KEY,
                                   model="m", timeout=5)
        finally:
            stop(first)
        msg = str(ctx.exception)
        for secret in ("hunter2", "abc123", "frag", "\x1b"):
            self.assertNotIn(secret, msg)
        self.assertIn("127.0.0.1:1/p", msg)

    def test_cli_exits_2_and_the_other_server_never_sees_the_key(self):
        other, other_url, other_seen = self._other()
        first, first_url = start_redirector(other_url + "/chat", 302)
        try:
            root = make_project(
                "cases:\n  - name: a\n    input: x\n    assert:\n"
                "      - contains: bar\n")
            code, out, err = run_cli(
                ["--root", root, "--base-url", first_url, "run", "--prompt",
                 "s", "--version", "v1", "--no-cache"],
                PROMPTREGRESS_API_KEY=KEY)
        finally:
            stop(first)
            stop(other)
        self.assertEqual(code, 2, (out, err))
        self.assertIn("redirect", err)
        self.assertNotIn(KEY, err + out)
        self.assertEqual(other_seen, [])


# ---------------------------------------------------------------------
# P5 - an apostrophe must not switch comment stripping off

COMMENT_CASES = (
    # (what follows `key: ` or `- `, the value it holds)
    ("It's fine   # TODO", "It's fine"),
    ("don't # c", "don't"),
    ("5 o'clock # c", "5 o'clock"),
    ("it's \"quoted\" # c", "it's \"quoted\""),
    ("\"it's # not a comment\"", "it's # not a comment"),
    ("\"it's # not a comment\" # a comment", "it's # not a comment"),
    ("'it''s # not a comment' # c", "it's # not a comment"),
    ("'a' # c", "a"),
    ('"say \\"hi\\" # kept" # c', 'say "hi" # kept'),
    ("plain#hash # c", "plain#hash"),
    ("a # b # c", "a"),
    ("http://x/y#frag # c", "http://x/y#frag"),
    ("4 o'clock\t# a tab before the comment", "4 o'clock"),
)


class TestCommentsAndQuotes(unittest.TestCase):
    """`input: It's fine   # TODO` sent the comment to the model: any quote
    character, an apostrophe inside a word included, switched comment
    stripping off for the rest of the line. A quote only opens a quoted
    string at the start of a scalar."""

    def test_a_comment_after_a_value_is_dropped_whatever_quotes_it_holds(self):
        for value, want in COMMENT_CASES:
            with self.subTest(value):
                self.assertEqual(miniyaml_mod.safe_load("k: " + value + "\n"),
                                 {"k": want})
                self.assertEqual(miniyaml_mod.safe_load("- " + value + "\n"),
                                 [want])
                self.assertEqual(
                    miniyaml_mod.safe_load("- k: " + value + "\n"),
                    [{"k": want}])

    def test_flow_lists(self):
        load = miniyaml_mod.safe_load
        self.assertEqual(load("k: [it's, ok] # c\n"), {"k": ["it's", "ok"]})
        self.assertEqual(load("k: [\"a # b\", 'c # d', e] # c\n"),
                         {"k": ["a # b", "c # d", "e"]})
        self.assertEqual(load("k: ['it''s', \"x, y\", 5 o'clock]\n"),
                         {"k": ["it's", "x, y", "5 o'clock"]})
        self.assertEqual(load("k: [a, b] # it's over\n"), {"k": ["a", "b"]})

    def test_only_a_comment_is_dropped(self):
        load = miniyaml_mod.safe_load
        self.assertEqual(load("# it's all comment\nk: 1  # it's one\n"),
                         {"k": 1})
        self.assertEqual(load("k:   # nothing here\n"), {"k": None})
        self.assertEqual(load("k: a#b\n"), {"k": "a#b"})

    def test_unterminated_quotes_are_still_errors(self):
        for text in ('k: "abc # c\n', "k: 'abc # c\n", 'k: [a, "b\n',
                     "k: [a, 'b]\n"):
            with self.subTest(text):
                with self.assertRaises(ValueError):
                    miniyaml_mod.safe_load(text)

    def test_the_model_is_not_sent_the_comment(self):
        root = make_project("cases:\n  - name: a\n"
                            "    input: It's fine   # TODO remove\n"
                            "    assert:\n      - contains: bar # it's ok\n")
        server, url, seen = start_recorder()
        try:
            code, out, err = run_cli(
                ["--root", root, "--base-url", url, "run", "--prompt", "s",
                 "--version", "v1", "--no-cache"])
        finally:
            stop(server)
        self.assertEqual(code, 0, (out, err))
        prompt = seen[0]["body"]["messages"][0]["content"]
        self.assertEqual(prompt, "Classify: It's fine\n")


# ---------------------------------------------------------------------
# P4 (support) - a plain number keeps the text it was written as

class TestPlainNumbersKeepTheirText(unittest.TestCase):
    """`contains: 3.10` searched for "3.1" and `starts_with: 007` for "7":
    the parser turned the scalar into a number and the number back into
    text."""

    def test_numbers_are_numbers_that_remember_how_they_were_written(self):
        data = miniyaml_mod.safe_load(
            "a: 3.10\nb: 007\nc: 1e5\nd: +5\ne: -0.50\nf: 12\ng: 5.\nh: .5\n")
        self.assertEqual(data, {"a": 3.1, "b": 7, "c": 100000.0, "d": 5,
                                "e": -0.5, "f": 12, "g": 5.0, "h": 0.5})
        self.assertEqual(
            {k: miniyaml_mod.scalar_text(v) for k, v in data.items()},
            {"a": "3.10", "b": "007", "c": "1e5", "d": "+5", "e": "-0.50",
             "f": "12", "g": "5.", "h": ".5"})
        self.assertIsInstance(data["b"], int)
        self.assertNotIsInstance(data["b"], bool)
        self.assertIsInstance(data["a"], float)

    def test_words_python_would_read_as_numbers_stay_text(self):
        data = miniyaml_mod.safe_load(
            "a: nan\nb: Infinity\nc: 1_000\nd: inf\ne: 0x1F\n"
            "f: \u0661\u0662\nh: 1e\ni: 1.2.3\nj: 10:30\n")
        for key, want in (("a", "nan"), ("b", "Infinity"), ("c", "1_000"),
                          ("d", "inf"), ("e", "0x1F"), ("f", "\u0661\u0662"),
                          ("h", "1e"), ("i", "1.2.3"), ("j", "10:30")):
            self.assertEqual(data[key], want, key)
            self.assertIsInstance(data[key], str, key)

    def test_scalar_text_only_speaks_for_text_and_numbers(self):
        text = miniyaml_mod.scalar_text
        self.assertEqual(text("x"), "x")
        self.assertEqual(text(5), "5")          # a number made by hand
        self.assertEqual(text(2.5), "2.5")
        for other in (True, False, None, ["a"], {"a": 1}):
            self.assertIsNone(text(other))

    def test_quoted_numbers_are_text_to_begin_with(self):
        data = miniyaml_mod.safe_load("a: \"3.10\"\nb: '007'\n")
        self.assertEqual(data, {"a": "3.10", "b": "007"})
        self.assertIsInstance(data["a"], str)


# ---------------------------------------------------------------------
# P3 (support) - nothing in the text can pass for the parser's own marker

class TestBlockScalarMarkerCannotBeForged(unittest.TestCase):
    """A block scalar is parked as `\x00block:N` and put back afterwards. A
    double-quoted "\u0000block:99" decodes to the same text: it crashed the
    parse with IndexError, and "\u0000block:0" swapped in somebody else's
    block."""

    def test_a_quoted_escape_is_just_text(self):
        text = ('a: |\n  real\nb: "\\u0000block:0"\nc: "\\u0000block:99"\n'
                'd:\n  - "\\u0000block:0"\n')
        data = miniyaml_mod.safe_load(text)
        self.assertEqual(data["a"], "real\n")
        self.assertEqual(data["b"], "\x00block:0")
        self.assertEqual(data["c"], "\x00block:99")
        self.assertEqual(data["d"], ["\x00block:0"])

    def test_a_raw_nul_is_refused_with_its_line(self):
        for text, line in (("a: 1\nb: x\x00y\n", 2),
                           ("a: |\n  one\n  t\x00wo\n", 3),
                           ("\x00block:0\n", 1)):
            with self.subTest(repr(text)):
                with self.assertRaisesRegex(
                        ValueError, rf"NUL.*\(line {line}\)"):
                    miniyaml_mod.safe_load(text)


# ---------------------------------------------------------------------
# P4 - verdicts that were silently wrong

def assertion(text):
    """One assertion as the cases file loader hands it over: the YAML list
    item `- <text>`."""
    return miniyaml_mod.safe_load("- " + text + "\n")[0]


def verdict(text, output):
    return asserts_mod.evaluate(assertion(text), output)[0]


class TestNumbersAreSearchedAsWritten(unittest.TestCase):
    def test_contains_3_10_does_not_match_python_3_12(self):
        self.assertFalse(verdict("contains: 3.10", "Python 3.12"))
        self.assertFalse(verdict("contains: 3.10", "Python 3.1"))
        self.assertTrue(verdict("contains: 3.10", "Python 3.10"))
        ok, detail = asserts_mod.evaluate(assertion("contains: 3.10"),
                                          "Python 3.12")
        self.assertEqual(detail, "missing '3.10'")

    def test_starts_with_007_is_not_7(self):
        self.assertTrue(verdict("starts_with: 007", "007 Bond"))
        self.assertFalse(verdict("starts_with: 007", "7 Bond"))
        self.assertTrue(verdict("ends_with: 1e5", "got 1e5"))
        self.assertFalse(verdict("ends_with: 1e5", "got 100000.0"))
        self.assertTrue(verdict("regex: 007", "agent 007"))
        self.assertFalse(verdict("regex: 007", "agent 7"))

    def test_not_contains_is_the_same_text(self):
        self.assertTrue(verdict("not_contains: 3.10", "Python 3.1"))
        self.assertFalse(verdict("not_contains: 3.10", "Python 3.10"))

    def test_plain_numbers_still_work(self):
        self.assertTrue(verdict("contains: 1", "got 1"))
        self.assertTrue(verdict("starts_with: 42", "42 is it"))
        self.assertTrue(verdict("contains: -0.5", "x = -0.5"))

    def test_a_cases_file_with_such_numbers_runs_as_written(self):
        root = make_project("cases:\n  - name: version\n    input: which\n"
                            "    assert:\n      - contains: 3.10\n"
                            "  - name: agent\n    input: who\n"
                            "    assert:\n      - starts_with: 007\n")
        outcomes = {}
        for answer in ("Python 3.12", "Python 3.10"):
            server, url, seen = start_recorder(answer)
            try:
                code, out, err = run_cli(
                    ["--root", root, "--base-url", url, "run", "--prompt",
                     "s", "--version", "v1", "--no-cache"])
            finally:
                stop(server)
            outcomes[answer] = (code, out)
        code, out = outcomes["Python 3.12"]
        self.assertEqual(code, 1, out)
        self.assertIn("missing '3.10'", out)
        code, out = outcomes["Python 3.10"]
        self.assertEqual(code, 1, out)       # the agent case never passes
        self.assertNotIn("missing '3.10'", out)
        self.assertIn("does not start with '007'", out)


class TestJsonValidMeansWhatItSays(unittest.TestCase):
    """`json_valid: false` passed a reply of {"a":1}: the value was never
    looked at."""

    def test_true_wants_json_and_false_wants_text_that_is_not(self):
        for text, output, want in (
                ("json_valid: true", '{"a": 1}', True),
                ("json_valid: true", "nope", False),
                ("json_valid: false", '{"a": 1}', False),
                ("json_valid: false", "nope", True),
                ("json_valid: false", "", True),
                ("json_valid: yes", "[1]", True),
                ("json_valid: no", "[1]", False)):
            with self.subTest(text, output=output):
                self.assertIs(verdict(text, output), want)

    def test_the_detail_says_which_way_it_went_wrong(self):
        ok, detail = asserts_mod.evaluate(assertion("json_valid: false"),
                                          '{"a": 1}')
        self.assertFalse(ok)
        self.assertIn("json_valid: false", detail)
        ok, detail = asserts_mod.evaluate(assertion("json_valid: true"), "x")
        self.assertIn("invalid JSON", detail)

    def test_anything_but_true_or_false_is_refused(self):
        for text in ("json_valid: maybe", "json_valid: 1", "json_valid: ~",
                     "json_valid: [a]", "json_valid: \"true\""):
            with self.subTest(text):
                with self.assertRaisesRegex(ValueError,
                                            r"json_valid: needs true or false"):
                    verdict(text, "{}")

    def test_a_reply_nested_too_deep_for_json_is_not_valid_json(self):
        deep = "[" * 100000 + "]" * 100000
        self.assertFalse(verdict("json_valid: true", deep))
        self.assertTrue(verdict("json_valid: false", deep))
        ok, detail = asserts_mod.evaluate({"schema": {"type": "array"}}, deep)
        self.assertFalse(ok)
        self.assertIn("JSON", detail)


class TestTokenGuardrails(unittest.TestCase):
    """`min_tokens: 1` passed an empty reply: an empty text counted as one
    token."""

    def test_an_empty_or_blank_reply_is_no_tokens(self):
        for blank in ("", " ", "\n", "  \t\r\n "):
            with self.subTest(repr(blank)):
                self.assertEqual(asserts_mod.estimate_tokens(blank), 0)
                self.assertFalse(verdict("min_tokens: 1", blank))
                self.assertTrue(verdict("max_tokens: 0", blank))

    def test_real_text_still_counts_from_one(self):
        self.assertEqual(asserts_mod.estimate_tokens("a"), 1)
        self.assertEqual(asserts_mod.estimate_tokens("hello world"), 2)
        self.assertEqual(asserts_mod.estimate_tokens("x" * 104), 26)
        self.assertTrue(verdict("min_tokens: 1", "hi"))
        self.assertTrue(verdict("max_tokens: 3", "hello world"))


class TestSchemaChecksWhatItSays(unittest.TestCase):
    """`schema: {type: str}` checked nothing: an unknown type name passed
    every reply. A schema that cannot be enforced is an error."""

    def test_unknown_type_names_are_refused_with_a_hint(self):
        for typo, hint in (("str", "string"), ("int", "integer"),
                           ("bool", "boolean"), ("float", "number"),
                           ("dict", "object"), ("list", "array"),
                           ("none", "null")):
            with self.subTest(typo):
                with self.assertRaises(ValueError) as ctx:
                    asserts_mod.evaluate(
                        {"schema": {"type": typo}}, '"x"')
                msg = str(ctx.exception)
                self.assertIn(f"unknown type '{typo}'", msg)
                self.assertIn(f"did you mean '{hint}'", msg)
                self.assertIn("one of object, array, string, number, "
                              "integer, boolean, null", msg)

    def test_an_unknown_type_is_refused_even_when_the_reply_is_not_json(self):
        with self.assertRaisesRegex(ValueError, "unknown type 'str'"):
            asserts_mod.evaluate({"schema": {"type": "str"}}, "plain text")

    def test_nested_schemas_are_checked_too(self):
        with self.assertRaisesRegex(
                ValueError, r"schema\.properties\.name\.type: unknown type "
                            r"'str'"):
            asserts_mod.evaluate(
                {"schema": {"type": "object", "properties": {
                    "name": {"type": "str"}}}}, '{"name": "x"}')
        with self.assertRaisesRegex(ValueError,
                                    r"schema\.items\.type: unknown type"):
            asserts_mod.evaluate(
                {"schema": {"type": "array", "items": {"type": "txt"}}}, "[]")

    def test_a_list_of_types_is_a_union(self):
        schema = {"type": ["integer", "null"]}
        for output, want in (("5", True), ("null", True), ('"x"', False),
                             ("1.5", False), ("true", False)):
            with self.subTest(output):
                self.assertIs(
                    asserts_mod.evaluate({"schema": schema}, output)[0], want)
        ok, detail = asserts_mod.evaluate({"schema": schema}, '"x"')
        self.assertIn("integer or null", detail)

    def test_a_constraint_the_validator_cannot_enforce_is_refused(self):
        for key, value in (("minimum", 5), ("maximum", 9), ("pattern", "^a"),
                           ("additionalProperties", False),
                           ("oneOf", [{"type": "string"}]),
                           ("minItems", 1), ("maxLength", 3), ("format", "uri"),
                           ("const", 1)):
            with self.subTest(key):
                with self.assertRaisesRegex(
                        ValueError, rf"schema: '{key}' is not supported"):
                    asserts_mod.evaluate(
                        {"schema": {"type": "object", key: value}}, "{}")

    def test_annotations_are_ignored_on_purpose(self):
        schema = {"$schema": "http://json-schema.org/draft-07/schema#",
                  "title": "Reply", "description": "d", "default": {},
                  "examples": [{}], "type": "object"}
        self.assertTrue(asserts_mod.evaluate({"schema": schema}, "{}")[0])

    def test_malformed_schemas_are_refused(self):
        for schema, where in (
                ("object", r"schema: must be a mapping"),
                ({"type": 5}, r"schema\.type: unknown type 5"),
                ({"type": []}, r"schema\.type: needs at least one type"),
                ({"enum": "red"}, r"schema\.enum: must be a list"),
                ({"required": "name"}, r"schema\.required: must be a list"),
                ({"required": [1]}, r"schema\.required: must be a list"),
                ({"properties": ["a"]}, r"schema\.properties: must be a map"),
                ({"properties": {"a": "string"}},
                 r"schema\.properties\.a: must be a mapping"),
                ({"items": [{"type": "string"}]},
                 r"schema\.items: must be a mapping")):
            with self.subTest(repr(schema)):
                with self.assertRaisesRegex(ValueError, where):
                    asserts_mod.evaluate({"schema": schema}, "{}")

    def test_required_properties_and_items_do_not_need_a_type_beside_them(self):
        self.assertFalse(asserts_mod.evaluate(
            {"schema": {"required": ["a"]}}, '{"b": 1}')[0])
        self.assertFalse(asserts_mod.evaluate(
            {"schema": {"properties": {"a": {"type": "string"}}}},
            '{"a": 5}')[0])
        self.assertFalse(asserts_mod.evaluate(
            {"schema": {"items": {"type": "string"}}}, '["a", 5]')[0])
        self.assertTrue(asserts_mod.evaluate(
            {"schema": {"required": ["a"]}}, '{"a": 1}')[0])

    def test_the_readmes_example_schema_still_works(self):
        schema = miniyaml_mod.safe_load(
            "type: object\nrequired: [sentiment]\nproperties:\n"
            "  sentiment:\n    type: string\n"
            "    enum: [positive, negative, neutral]\n")
        for output, want in (('{"sentiment": "positive"}', True),
                             ('{"sentiment": "angry"}', False),
                             ('{"other": 1}', False), ("[]", False)):
            self.assertIs(
                asserts_mod.evaluate({"schema": schema}, output)[0], want,
                output)


# ---------------------------------------------------------------------
# P3 - a suite that cannot run is a configuration error (exit 2)

def one_case(body):
    """A cases file with one case: the lines of `body`, under a `name: c`
    line unless `body` brings a name of its own."""
    lines = body.strip("\n").split("\n")
    if not body.startswith("name:"):
        lines.insert(0, "name: c")
    return "cases:\n  - " + lines[0] + "\n" + "".join(
        "    " + ln + "\n" for ln in lines[1:])


# (label, the lines of the one case, pieces the error must contain)
BAD_CASES = (
    ("input is a boolean", "input: no\nassert:\n  - contains: x",
     ("input", "YAML boolean")),
    ("input is empty", "input:\nassert:\n  - contains: x",
     ("input: is empty",)),
    ("input is a list", "input: [a, b]\nassert:\n  - contains: x",
     ("input: needs text, got list",)),
    ("input is missing", "assert:\n  - contains: x",
     ("needs 'input' and 'assert'",)),
    ("assert is missing", "input: hi", ("needs 'input' and 'assert'",)),
    ("assert is empty", "input: hi\nassert:", ("assert", "no assertions")),
    ("assert is an empty list", "input: hi\nassert: []",
     ("assert", "no assertions")),
    ("assert is a mapping", "input: hi\nassert:\n  contains: x",
     ("assert", "must be a list")),
    ("an assertion is a word", "input: hi\nassert:\n  - contains",
     ("assert[0]", "must be a mapping")),
    ("an assertion has two keys",
     "input: hi\nassert:\n  - contains: x\n    regex: y",
     ("assert[0]", "exactly one key")),
    ("unknown assertion", "input: hi\nassert:\n  - contain: x",
     ("assert[0]", "unknown assertion 'contain'",
      "did you mean 'contains'")),
    ("a boolean to search for", "input: hi\nassert:\n  - contains: yes",
     ("assert[0]", "contains", "YAML boolean")),
    ("nothing to search for", "input: hi\nassert:\n  - contains:",
     ("assert[0]", "contains: needs text, got null")),
    ("a regex that does not compile", "input: hi\nassert:\n  - regex: (",
     ("assert[0]", "regex: invalid regex")),
    ("a count that is not a number", "input: hi\nassert:\n  - max_tokens: lots",
     ("assert[0]", "max_tokens: needs a whole number")),
    ("a negative count", "input: hi\nassert:\n  - min_tokens: -1",
     ("assert[0]", "min_tokens: needs a whole number")),
    ("json_valid maybe", "input: hi\nassert:\n  - json_valid: maybe",
     ("assert[0]", "json_valid: needs true or false")),
    ("a schema type that is not JSON's",
     "input: hi\nassert:\n  - schema:\n      type: str",
     ("assert[0]", "schema.type: unknown type 'str'", "string")),
    ("a schema keyword that is not enforced",
     "input: hi\nassert:\n  - schema:\n      type: integer\n"
     "      minimum: 5",
     ("assert[0]", "schema: 'minimum' is not supported")),
    ("an unknown key in the case",
     "input: hi\nvarss:\n  tone: terse\nassert:\n  - contains: x",
     ("unknown key 'varss'", "did you mean 'vars'")),
    ("vars is a list", "input: hi\nvars: [a, b]\nassert:\n  - contains: x",
     ("vars", "must be a mapping")),
    ("a var is a boolean", "input: hi\nvars:\n  tone: yes\nassert:\n  - contains: x",
     ("vars.tone", "YAML boolean")),
    ("a var is empty", "input: hi\nvars:\n  tone:\nassert:\n  - contains: x",
     ("vars.tone", "null")),
    ("an empty judge", "input: hi\nassert:\n  - judge:",
     ("assert[0]", "judge")),
    ("a judge without a rubric",
     "input: hi\nassert:\n  - judge:\n      model: big",
     ("assert[0]", "judge", "rubric")),
    ("a judge rubric that is a boolean",
     "input: hi\nassert:\n  - judge:\n      rubric: yes",
     ("assert[0]", "judge.rubric", "YAML boolean")),
    ("a name that is a boolean", "name: yes\ninput: hi\nassert:\n  - contains: x",
     ("name", "YAML boolean")),
    ("an empty text to search for", 'input: hi\nassert:\n  - contains: ""',
     ("assert[0]", "contains: is empty", "match every reply")),
    ("an empty text that must be absent",
     'input: hi\nassert:\n  - not_contains: ""',
     ("assert[0]", "not_contains: is empty", "fail every reply")),
    ("an empty pattern", 'input: hi\nassert:\n  - regex: ""',
     ("assert[0]", "regex: is empty")),
    ("a count written the way int() takes it",
     'input: hi\nassert:\n  - max_tokens: "1_0"',
     ("assert[0]", "max_tokens: needs a whole number")),
    ("a case key with no look-alike",
     "input: hi\ntags: [a]\nassert:\n  - contains: x",
     ("unknown key 'tags'", "put notes in a # comment")),
    ("a judge with an empty rubric", 'input: hi\nassert:\n  - judge: ""',
     ("assert[0]", "judge: is empty")),
    ("a judge with a key of its own",
     "input: hi\nassert:\n  - judge:\n      rubric: be brief\n      model: big",
     ("assert[0]", "judge: unknown key 'model'")),
)


class TestUnusableSuitesAreConfigErrors(unittest.TestCase):
    """`input: no`, `input: 42`, an empty `assert:`, a schema type that is a
    list, a forged block marker ... each ended in a traceback and exit 1,
    the code for "a case failed"."""

    def test_the_loader_names_the_file_the_case_and_the_key(self):
        for label, body, pieces in BAD_CASES:
            with self.subTest(label):
                root = make_project(one_case(body))
                path = os.path.join(root, "prompts", "s.cases.yaml")
                with self.assertRaises(ValueError) as ctx:
                    suite_mod.load_cases(root, "s")
                msg = str(ctx.exception)
                self.assertIn(path, msg)
                self.assertIn("case 0", msg)
                for piece in pieces:
                    self.assertIn(piece, msg)
                self.assertNotIn("Traceback", msg)

    def test_a_case_that_is_not_a_mapping(self):
        root = make_project("cases:\n  - just some text\n")
        with self.assertRaisesRegex(ValueError,
                                    r"case 0 needs 'input' and 'assert'"):
            suite_mod.load_cases(root, "s")

    def test_run_exits_2_before_any_model_call(self):
        server, url, seen = start_recorder()
        try:
            for label, body, pieces in BAD_CASES:
                with self.subTest(label):
                    root = make_project(one_case(body))
                    code, out, err = run_cli(
                        ["--root", root, "--base-url", url, "run", "--prompt",
                         "s", "--version", "v1", "--no-cache"])
                    self.assertEqual(code, 2, (out, err))
                    self.assertEqual(out, "")
                    self.assertTrue(err.startswith("error: "), err)
                    self.assertNotIn("Traceback", err)
                    for piece in pieces:
                        self.assertIn(piece, err)
        finally:
            stop(server)
        self.assertEqual(seen, [], "an unusable suite never reaches the model")

    def test_every_problem_is_reported_in_one_go(self):
        root = make_project(
            "cases:\n"
            "  - name: first\n    input: no\n    assert:\n      - contains: x\n"
            "  - name: second\n    input: hi\n    assert:\n"
            "      - contains: yes\n      - regex: (\n"
            "  - name: third\n    input: hi\n    assert:\n")
        with self.assertRaises(ValueError) as ctx:
            suite_mod.load_cases(root, "s")
        lines = str(ctx.exception).split("\n")
        self.assertEqual(len(lines), 4, lines)
        for line, pieces in zip(lines, (("case 0 ('first')", "input"),
                                        ("case 1 ('second')", "assert[0]"),
                                        ("case 1 ('second')", "assert[1]"),
                                        ("case 2 ('third')", "assert"))):
            for piece in pieces:
                self.assertIn(piece, line)

    def test_a_file_that_is_wrong_everywhere_does_not_flood_the_log(self):
        root = make_project("cases:\n" + "".join(
            f"  - name: n{i}\n    input: no\n    assert:\n"
            f"      - contains: x\n" for i in range(50)))
        with self.assertRaises(ValueError) as ctx:
            suite_mod.load_cases(root, "s")
        lines = str(ctx.exception).split("\n")
        self.assertEqual(len(lines), suite_mod.MAX_PROBLEMS + 1)
        self.assertIn("and 30 more problem(s)", lines[-1])

    def test_a_cases_file_that_is_not_utf8_names_itself(self):
        root = make_project("")
        path = os.path.join(root, "prompts", "s.cases.yaml")
        with open(path, "wb") as f:
            f.write(b"cases:\n  - name: caf\xe9\n")
        with self.assertRaisesRegex(ValueError, "not UTF-8 text") as ctx:
            suite_mod.load_cases(root, "s")
        self.assertIn(path, str(ctx.exception))

    def test_run_all_reads_every_suite_before_the_first_model_call(self):
        root = make_project("cases:\n  - name: a\n    input: x\n"
                            "    assert:\n      - contains: bar\n",
                            name="a")
        with open(os.path.join(root, "prompts", "b.v1.txt"), "w") as f:
            f.write(PROMPT)
        with open(os.path.join(root, "prompts", "b.cases.yaml"), "w") as f:
            f.write("cases:\n  - name: broken\n    input: no\n"
                    "    assert:\n      - contains: x\n")
        server, url, seen = start_recorder()
        try:
            code, out, err = run_cli(["--root", root, "--base-url", url,
                                      "run", "--all", "--no-cache"])
        finally:
            stop(server)
        self.assertEqual(code, 2, (out, err))
        self.assertIn("b.cases.yaml", err)
        self.assertEqual(out, "")
        self.assertEqual(seen, [], "suite a was run before suite b was read")

    def test_list_marks_such_a_file_invalid(self):
        root = make_project(one_case("input: hi\nassert:\n  - contains: yes"))
        code, out, err = run_cli(["--root", root, "list"])
        self.assertEqual(code, 2)
        self.assertIn("invalid cases file", out)
        self.assertIn("YAML boolean", err)
        self.assertNotIn("Traceback", err)


class TestPlausibleSuitesLoad(unittest.TestCase):
    def test_numbers_are_text_as_written(self):
        root = make_project(
            "cases:\n  - name: 42\n    input: 3.10\n"
            "    vars:\n      version: 007\n      n: 1e3\n"
            "    assert:\n      - contains: 3.10\n      - starts_with: 007\n"
            "      - max_tokens: \"10\"\n      - regex: 1\n")
        case = suite_mod.load_cases(root, "s")[0]
        self.assertEqual(case, {
            "name": "42", "input": "3.10",
            "vars": {"version": "007", "n": "1e3"},
            "assert": [{"contains": "3.10"}, {"starts_with": "007"},
                       {"max_tokens": 10}, {"regex": "1"}]})
        for value in (case["name"], case["input"], *case["vars"].values()):
            self.assertIs(type(value), str)
        self.assertEqual(suite_mod.render("{{version}}/{{input}}", case["input"],
                                          case["vars"]), "007/3.10")

    def test_a_byte_order_mark_is_not_part_of_the_first_key(self):
        root = make_project("")
        with open(os.path.join(root, "prompts", "s.cases.yaml"), "wb") as f:
            f.write(b"\xef\xbb\xbfcases:\n  - input: hi\n    assert:\n"
                    b"      - contains: x\n")
        self.assertEqual(len(suite_mod.load_cases(root, "s")), 1)

    def test_defaults(self):
        root = make_project("cases:\n  - input: hi\n    assert:\n"
                            "      - contains: x\n  - name:\n    input: hi\n"
                            "    assert:\n      - contains: x\n")
        cases = suite_mod.load_cases(root, "s")
        self.assertEqual([c["name"] for c in cases], ["case-0", "case-1"])
        self.assertEqual([c["vars"] for c in cases], [{}, {}])

    def test_a_schema_with_a_union_type_runs(self):
        root = make_project(
            "cases:\n  - name: nullable\n    input: hi\n    assert:\n"
            "      - schema:\n          type: object\n"
            "          properties:\n            age:\n"
            "              type: [integer, \"null\"]\n")
        for answer, want in (('{"age": 5}', 0), ('{"age": null}', 0),
                             ('{"age": "old"}', 1)):
            server, url, seen = start_recorder(answer)
            try:
                code, out, err = run_cli(
                    ["--root", root, "--base-url", url, "run", "--prompt", "s",
                     "--version", "v1", "--no-cache"])
            finally:
                stop(server)
            self.assertEqual(code, want, (answer, out, err))
            self.assertNotIn("Traceback", err)

    def test_a_forged_block_marker_is_just_the_text_it_says(self):
        root = make_project(
            "cases:\n  - name: forged\n    input: \"\\u0000block:99\"\n"
            "    assert:\n      - contains: x\n"
            "  - name: real\n    input: |\n      line one\n      line two\n"
            "    assert:\n      - contains: x\n")
        cases = suite_mod.load_cases(root, "s")
        self.assertEqual(cases[0]["input"], "\x00block:99")
        self.assertEqual(cases[1]["input"], "line one\nline two\n")

    def test_judges_load_in_both_spellings(self):
        root = make_project(
            "cases:\n  - name: j\n    input: hi\n    assert:\n"
            "      - judge: Must be one word\n"
            "      - judge:\n          rubric: Must be polite\n")
        case = suite_mod.load_cases(root, "s")[0]
        self.assertEqual(case["assert"], [
            {"judge": "Must be one word"},
            {"judge": {"rubric": "Must be polite"}}])


class TestCheckIsIdempotent(unittest.TestCase):
    def test_checking_a_checked_assertion_changes_nothing(self):
        for text in ("contains: 3.10", "regex: 1", 'max_tokens: "10"',
                     "json_valid: false", "judge: Be brief",
                     "judge:\n    rubric: Be brief",
                     "schema:\n    type: [integer, \"null\"]"):
            with self.subTest(text):
                once = asserts_mod.check(assertion(text))
                self.assertEqual(asserts_mod.check(once), once)


class TestTheRunnerStillDefendsItself(unittest.TestCase):
    """Cases made by hand (the runner API) skip the loader."""

    def test_a_bad_assertion_in_a_hand_built_case_is_a_failed_case(self):
        server, url, seen = start_recorder()
        root = make_project("cases:\n  - name: a\n    input: x\n    assert:\n"
                            "      - contains: bar\n")
        try:
            prompt = suite_mod.load_prompt(root, "s", "v1")
            case = {"name": "by hand", "input": "x", "vars": {},
                    "assert": [{"contains": True}, {"regex": "("},
                               {"bogus": 1}, {"contains": "bar"}]}
            result = runner_mod.run_case(root, "s", "v1", prompt, case,
                                         {"base_url": url}, use_cache=False)
        finally:
            stop(server)
        self.assertFalse(result.passed)
        details = [a.detail for a in result.assertions]
        self.assertIn("bad assertion: contains: True is a YAML boolean",
                      details[0])
        self.assertIn("bad assertion: regex: invalid regex", details[1])
        self.assertIn("bad assertion: unknown assertion", details[2])
        self.assertTrue(result.assertions[3].passed)

    def test_a_bad_judge_in_a_hand_built_case_never_reaches_the_model(self):
        server, url, seen = start_recorder()
        root = make_project("cases:\n  - name: a\n    input: x\n    assert:\n"
                            "      - contains: bar\n")
        try:
            prompt = suite_mod.load_prompt(root, "s", "v1")
            case = {"name": "by hand", "input": "x", "vars": {},
                    "assert": [{"judge": {}}, {"judge": None},
                               {"judge": {"rubric": True}}, "contains"]}
            result = runner_mod.run_case(root, "s", "v1", prompt, case,
                                         {"base_url": url}, use_cache=False)
        finally:
            stop(server)
        self.assertFalse(result.passed)
        for a in result.assertions:
            self.assertFalse(a.passed)
            self.assertTrue(a.detail.startswith("bad assertion: "), a.detail)
        self.assertEqual(len(seen), 1, "the case itself, and no judge call")


# ---------------------------------------------------------------------
# P6 - --timeout is a deadline, and a reply has a size limit

def read_request(conn):
    """Read one HTTP request (head and body) from `conn`."""
    buf = b""
    while b"\r\n\r\n" not in buf:
        chunk = conn.recv(65536)
        if not chunk:
            return buf
        buf += chunk
    head, _, body = buf.partition(b"\r\n\r\n")
    need = 0
    for line in head.split(b"\r\n"):
        if line.lower().startswith(b"content-length:"):
            need = int(line.split(b":", 1)[1])
    while len(body) < need:
        chunk = conn.recv(65536)
        if not chunk:
            break
        body += chunk
    return buf


def start_scripted(script, requests=None):
    """A raw TCP endpoint. For every connection: read the request (appended
    to `requests`, if given), then call script(conn, stopped); `stopped` is
    set when the test is over. Returns (stop, url)."""
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(5)
    stopped = threading.Event()

    def run(conn):
        with conn:
            try:
                request = read_request(conn)
                if requests is not None:
                    requests.append(request)
                script(conn, stopped)
            except OSError:
                pass            # the client hung up: that is the point

    def accept():
        while True:
            try:
                conn, _ = srv.accept()
            except OSError:
                return
            threading.Thread(target=run, args=(conn,), daemon=True).start()

    threading.Thread(target=accept, daemon=True).start()

    def stop():
        stopped.set()
        srv.close()
    return stop, f"http://127.0.0.1:{srv.getsockname()[1]}/v1"


def reply_bytes(body, length="exact"):
    """An HTTP response carrying `body`. `length` is the Content-Length
    announced: "exact", a number, or None for none (the body then ends when
    the server hangs up)."""
    if length == "exact":
        length = len(body)
    head = "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
    if length is not None:
        head += f"Content-Length: {length}\r\n"
    return (head + "Connection: close\r\n\r\n").encode() + body


def drip(reply, pause, fast=0):
    """Script: the first `fast` bytes of `reply` at once, the rest one byte
    every `pause` seconds."""
    def script(conn, stopped):
        conn.sendall(reply[:fast])
        for i in range(fast, len(reply)):
            if stopped.wait(pause):
                return
            conn.sendall(reply[i:i + 1])
    return script


GOOD_BODY = b'{"choices":[{"message":{"content":"hi"}}]}'


def request_threads():
    return [t for t in threading.enumerate()
            if t.name == "promptregress-request"]


class TestTimeoutIsADeadline(unittest.TestCase):
    """`--timeout 3` was the limit for each socket read: a server that sent
    one byte every 0.4 s kept a request alive for as long as it liked (25 s
    and more, a minute and a half per case)."""

    def ask(self, url, **kw):
        """(ModelError text, seconds it took) for complete() against url."""
        started = time.monotonic()
        with self.assertRaises(model_mod.ModelError) as ctx:
            model_mod.complete("hi", base_url=url, model="m", **kw)
        return str(ctx.exception), time.monotonic() - started

    def test_a_body_that_drips_in_is_cut_off_at_the_deadline(self):
        reply = reply_bytes(GOOD_BODY)
        stop, url = start_scripted(
            drip(reply, 0.2, fast=len(reply) - len(GOOD_BODY)))
        try:
            msg, took = self.ask(url, timeout=1)
        finally:
            stop()
        self.assertIn("timed out after 1s", msg)
        self.assertIn("--timeout", msg)
        self.assertLess(took, 2.5)

    def test_so_is_a_head_that_drips_in(self):
        stop, url = start_scripted(drip(reply_bytes(GOOD_BODY), 0.2))
        try:
            msg, took = self.ask(url, timeout=1)
        finally:
            stop()
        self.assertIn("timed out after 1s", msg)
        self.assertLess(took, 2.5)

    def test_a_server_that_says_nothing_is_cut_off_too(self):
        stop, url = start_scripted(lambda conn, stopped: stopped.wait(30))
        try:
            msg, took = self.ask(url, timeout=1)
        finally:
            stop()
        self.assertIn("timed out after 1s", msg)
        self.assertLess(took, 2.5)

    def test_nothing_keeps_running_after_the_deadline(self):
        stop, url = start_scripted(drip(reply_bytes(GOOD_BODY), 0.2))
        try:
            self.ask(url, timeout=1)
            give_up = time.monotonic() + 3
            while request_threads() and time.monotonic() < give_up:
                time.sleep(0.05)
            self.assertEqual(request_threads(), [])
        finally:
            stop()

    def test_a_slow_reply_that_makes_it_in_time_is_not_disturbed(self):
        reply = reply_bytes(GOOD_BODY)
        stop, url = start_scripted(drip(reply, 0.005))
        try:
            self.assertEqual(model_mod.complete("hi", base_url=url, model="m",
                                                timeout=10), "hi")
        finally:
            stop()
        self.assertEqual(request_threads(), [])

    def test_the_cli_ends_the_run_at_the_deadline_with_exit_2(self):
        root = make_project(one_case("input: hi\nassert:\n  - contains: hi"))
        stop, url = start_scripted(drip(reply_bytes(GOOD_BODY), 0.2))
        started = time.monotonic()
        try:
            code, out, err = run_cli(
                ["--root", root, "--base-url", url, "run", "--prompt", "s",
                 "--version", "v1", "--no-cache", "--timeout", "1"])
        finally:
            stop()
        self.assertEqual(code, 2, (out, err))
        self.assertLess(time.monotonic() - started, 3.5)
        self.assertIn("timed out after 1s", err)
        self.assertIn("stopped at case 'c'", err)


class TestReplySizeIsLimited(unittest.TestCase):
    """A reply of 400 MB was read into memory (2 GB resident)."""

    def ask(self, url, **kw):
        with self.assertRaises(model_mod.ModelError) as ctx:
            model_mod.complete("hi", base_url=url, model="m", timeout=10, **kw)
        return str(ctx.exception)

    def test_default_limit_is_8_mib(self):
        self.assertEqual(model_mod.DEFAULT_MAX_REPLY_BYTES, 8 * 1024 * 1024)
        with clean_env():
            self.assertEqual(model_mod._cfg()["max_reply_bytes"],
                             8 * 1024 * 1024)

    def test_a_reply_over_the_limit_is_an_error(self):
        body = json.dumps({"choices": [{"message": {
            "content": "x" * 5000}}]}).encode()
        stop, url = start_scripted(
            lambda conn, stopped: conn.sendall(reply_bytes(body)))
        try:
            msg = self.ask(url, max_reply_bytes=1000)
        finally:
            stop()
        self.assertIn("larger than the 1000 bytes allowed", msg)
        self.assertIn("--max-reply-bytes", msg)
        self.assertIn("PROMPTREGRESS_MAX_REPLY_BYTES", msg)

    def test_the_limit_is_the_size_that_is_still_fine(self):
        stop, url = start_scripted(
            lambda conn, stopped: conn.sendall(reply_bytes(GOOD_BODY)))
        try:
            self.assertEqual(model_mod.complete(
                "hi", base_url=url, timeout=10,
                max_reply_bytes=len(GOOD_BODY)), "hi")
            msg = self.ask(url, max_reply_bytes=len(GOOD_BODY) - 1)
        finally:
            stop()
        self.assertIn(f"larger than the {len(GOOD_BODY) - 1} bytes", msg)

    def test_an_endless_reply_is_abandoned_after_the_limit(self):
        sent = []

        def endless(conn, stopped):
            conn.sendall(reply_bytes(b"", length=None))
            chunk = b"x" * 65536
            total = 0
            try:
                while total < 64 * 1024 * 1024:
                    conn.sendall(chunk)
                    total += len(chunk)
            finally:
                sent.append(total)

        stop, url = start_scripted(endless)
        started = time.monotonic()
        try:
            msg = self.ask(url, max_reply_bytes=100000)
            took = time.monotonic() - started
            give_up = time.monotonic() + 5
            while not sent and time.monotonic() < give_up:
                time.sleep(0.02)
        finally:
            stop()
        self.assertIn("larger than the 100000 bytes allowed", msg)
        self.assertLess(took, 3.0)
        self.assertTrue(sent, "the server never noticed the client left")
        self.assertLess(sent[0], 32 * 1024 * 1024)

    def test_an_announced_size_over_the_limit_is_refused_at_once(self):
        stop, url = start_scripted(lambda conn, stopped: (
            conn.sendall(reply_bytes(b"", length=10 ** 9)),
            stopped.wait(30)))
        started = time.monotonic()
        try:
            msg = self.ask(url, max_reply_bytes=1000)
        finally:
            stop()
        self.assertIn("larger than the 1000 bytes allowed", msg)
        self.assertLess(time.monotonic() - started, 3.0, "waited for the body")

    def test_a_big_reply_inside_the_default_limit_works(self):
        body = json.dumps({"choices": [{"message": {
            "content": "y" * 300000}}]}).encode()
        stop, url = start_scripted(
            lambda conn, stopped: conn.sendall(reply_bytes(body)))
        try:
            answer = model_mod.complete("hi", base_url=url, timeout=10)
        finally:
            stop()
        self.assertEqual(answer, "y" * 300000)

    def test_a_reply_shorter_than_announced_is_still_cut_off(self):
        stop, url = start_scripted(lambda conn, stopped: conn.sendall(
            reply_bytes(GOOD_BODY[:12], length=len(GOOD_BODY))))
        try:
            msg = self.ask(url)
        finally:
            stop()
        self.assertIn("hung up before the whole reply arrived (got 12 bytes)",
                      msg)

    def test_the_judge_has_the_same_limit(self):
        server, url, seen = start_recorder("PASS" + " " * 500)
        try:
            with self.assertRaisesRegex(model_mod.ModelError,
                                        "larger than the 200 bytes allowed"):
                model_mod.judge("out", "be brief", base_url=url,
                                max_reply_bytes=200)
        finally:
            stop(server)

    def test_the_cli_option_and_the_environment_reach_the_request(self):
        root = make_project(one_case("input: hi\nassert:\n  - contains: x"))
        server, url, seen = start_recorder("x" * 500)
        try:
            base = ["--root", root, "--base-url", url, "run", "--prompt", "s",
                    "--version", "v1", "--no-cache"]
            code, out, err = run_cli(base + ["--max-reply-bytes", "100"])
            self.assertEqual(code, 2, (out, err))
            self.assertIn("larger than the 100 bytes allowed", err)
            code, out, err = run_cli(base, PROMPTREGRESS_MAX_REPLY_BYTES="100")
            self.assertEqual(code, 2, (out, err))
            self.assertIn("larger than the 100 bytes allowed", err)
            code, out, err = run_cli(base + ["--max-reply-bytes", "100000"],
                                     PROMPTREGRESS_MAX_REPLY_BYTES="100")
            self.assertEqual(code, 0, (out, err))     # the flag wins
        finally:
            stop(server)


class TestTheProxyIsStillUsed(unittest.TestCase):
    def test_a_proxy_in_the_environment_gets_the_request(self):
        seen = []
        stop, url = start_scripted(
            lambda conn, stopped: conn.sendall(reply_bytes(GOOD_BODY)), seen)
        env = {k: v for k, v in os.environ.items()
               if not k.lower().endswith("_proxy")}
        env["http_proxy"] = url.rsplit("/v1", 1)[0]
        try:
            with mock.patch.dict(os.environ, env, clear=True):
                answer = model_mod.complete(
                    "hi", base_url="http://models.example.invalid/v1",
                    timeout=5)
        finally:
            stop()
        self.assertEqual(answer, "hi")
        self.assertTrue(seen[0].startswith(
            b"POST http://models.example.invalid/v1/chat/completions "), seen)


class TestTimeAndSizeSettings(unittest.TestCase):
    def test_seconds(self):
        for good, want in (("30", 30.0), ("2.5", 2.5), (3, 3.0), (" 7 ", 7.0),
                           ("86400", 86400.0), (0.01, 0.01)):
            with self.subTest(good):
                self.assertEqual(model_mod.parse_seconds(good), want)
        for bad in (0, "0", -1, "-0.5", "nan", "inf", "-inf", "abc", "",
                    None, [], 86401, "1e999", float("nan")):
            with self.subTest(repr(bad)):
                with self.assertRaisesRegex(ValueError,
                                            "number of seconds above 0"):
                    model_mod.parse_seconds(bad)

    def test_byte_limits(self):
        for good, want in (("1", 1), (8388608, 8388608), (" 100 ", 100),
                           (str(1 << 30), 1 << 30)):
            with self.subTest(good):
                self.assertEqual(model_mod.parse_byte_limit(good), want)
        for bad in (0, "0", -5, "-5", "1.5", 1.5, True, "1_000", "0x10", "",
                    None, (1 << 30) + 1, "\u0663"):
            with self.subTest(repr(bad)):
                with self.assertRaisesRegex(ValueError, "whole number of bytes"):
                    model_mod.parse_byte_limit(bad)

    def test_the_parser_refuses_bad_flags(self):
        for flag, bad in (("--timeout", "0"), ("--timeout", "-3"),
                          ("--timeout", "nan"), ("--timeout", "inf"),
                          ("--timeout", "abc"), ("--timeout", "90000"),
                          ("--max-reply-bytes", "0"),
                          ("--max-reply-bytes", "-1"),
                          ("--max-reply-bytes", "1.5"),
                          ("--max-reply-bytes", "lots")):
            with self.subTest(flag + " " + bad):
                err = io.StringIO()
                with clean_env(), redirect_stderr(err):
                    with self.assertRaises(SystemExit) as ctx:
                        cli_mod.main([flag, bad, "list"])
                self.assertEqual(ctx.exception.code, 2)
                self.assertIn(f"argument {flag}", err.getvalue())

    def test_a_broken_environment_setting_is_exit_2_and_names_itself(self):
        root = make_project(one_case("input: hi\nassert:\n  - contains: bar"))
        server, url, seen = start_recorder()
        try:
            base = ["--root", root, "--base-url", url, "run", "--prompt", "s",
                    "--version", "v1", "--no-cache"]
            for name, value in (("PROMPTREGRESS_TIMEOUT", "abc"),
                                ("PROMPTREGRESS_TIMEOUT", "0"),
                                ("PROMPTREGRESS_TIMEOUT", "nan"),
                                ("PROMPTREGRESS_MAX_REPLY_BYTES", "big"),
                                ("PROMPTREGRESS_MAX_REPLY_BYTES", "0")):
                with self.subTest(name + "=" + value):
                    code, out, err = run_cli(base, **{name: value})
                    self.assertEqual(code, 2, (out, err))
                    self.assertEqual(out, "")
                    self.assertTrue(err.startswith(f"error: {name} "), err)
                    self.assertIn(repr(value), err)
            self.assertEqual(seen, [])
            # empty is not set; the demo does not care what the environment says
            code, out, err = run_cli(base, PROMPTREGRESS_TIMEOUT="",
                                     PROMPTREGRESS_MAX_REPLY_BYTES=" ")
            self.assertEqual(code, 0, (out, err))
            code, out, err = run_cli(
                ["--root", root, "--demo", "run", "--prompt", "sentiment",
                 "--version", "v1", "--no-cache"],
                PROMPTREGRESS_TIMEOUT="abc")
            self.assertEqual(code, 0, (out, err))
        finally:
            stop(server)

    def test_the_environment_sets_the_defaults(self):
        with clean_env(PROMPTREGRESS_TIMEOUT="7",
                       PROMPTREGRESS_MAX_REPLY_BYTES="4096"):
            cfg = model_mod._cfg()
            self.assertEqual((cfg["timeout"], cfg["max_reply_bytes"]),
                             (7.0, 4096))
            cfg = model_mod._cfg(timeout=2, max_reply_bytes=10)
            self.assertEqual((cfg["timeout"], cfg["max_reply_bytes"]),
                             (2.0, 10))

    def test_complete_refuses_unusable_arguments_in_words(self):
        for kw, word in (({"timeout": 0}, "timeout"),
                         ({"timeout": float("inf")}, "timeout"),
                         ({"max_reply_bytes": 0}, "max_reply_bytes")):
            with self.subTest(kw):
                with self.assertRaisesRegex(model_mod.ModelError,
                                            f"^{word} needs "):
                    model_mod.complete("hi", base_url="http://127.0.0.1:1/v1",
                                       **kw)


# ---------------------------------------------------------------------
# P7 - promises the README makes that no test pinned. Each class names the
# promise and fails when the code stops keeping it.

JUDGED = ("input: hi\nassert:\n  - contains: PASS\n"
          "  - judge: The reply must say PASS")


class TestTheKeyGoesInOneHeader(unittest.TestCase):
    """README: "The API key goes out in one `Authorization: Bearer` header
    and nowhere else"."""

    def test_the_header_is_there_and_the_key_is_nowhere_else(self):
        server, url, seen = start_recorder()
        try:
            model_mod.complete("hi", base_url=url, api_key=KEY, model="m",
                               timeout=5)
        finally:
            stop(server)
        self.assertEqual(bearer(seen), ["Bearer " + KEY])
        request = seen[0]
        elsewhere = [request["path"], json.dumps(request["body"])] + [
            v for k, v in request["headers"].items()
            if k.lower() != "authorization"]
        for text in elsewhere:
            self.assertNotIn(KEY, text)

    def test_without_a_key_the_placeholder_is_sent(self):
        server, url, seen = start_recorder()
        try:
            with clean_env():
                model_mod.complete("hi", base_url=url, model="m", timeout=5)
        finally:
            stop(server)
        self.assertEqual(bearer(seen), ["Bearer ollama"])

    def test_api_key_reaches_every_request_of_run_and_diff(self):
        # run: the case and its judge; diff: both versions
        root = make_project(one_case(JUDGED))
        with open(os.path.join(root, "prompts", "s.v2.txt"), "w") as f:
            f.write("Again: {{input}}\n")
        server, url, seen = start_recorder("PASS")
        try:
            code, out, err = run_cli(
                ["--root", root, "--base-url", url, "run", "--prompt", "s",
                 "--version", "v1", "--no-cache", "--api-key", KEY])
            self.assertEqual(code, 0, (out, err))
            self.assertEqual(len(seen), 2)           # the case, the judge
            code, out, err = run_cli(
                ["--root", root, "--base-url", url, "--api-key", KEY, "diff",
                 "v1", "v2", "--prompt", "s", "--no-cache"])
            self.assertEqual(code, 0, (out, err))
            self.assertEqual(len(seen), 6)
            code, out, err = run_cli(
                ["--root", root, "--base-url", url, "run", "--all",
                 "--no-cache"], PROMPTREGRESS_API_KEY=KEY)
            self.assertEqual(code, 0, (out, err))
        finally:
            stop(server)
        self.assertEqual(bearer(seen), ["Bearer " + KEY] * len(seen))
        self.assertEqual(len(seen), 6 + 4)


class TestTemperatureIsZero(unittest.TestCase):
    """README: "Temperature is 0 for determinism"."""

    def test_complete_asks_for_temperature_0_unless_told_otherwise(self):
        server, url, seen = start_recorder()
        try:
            model_mod.complete("hi", base_url=url, model="m", timeout=5)
            model_mod.complete("hi", base_url=url, model="m", timeout=5,
                               temperature=0.7)
        finally:
            stop(server)
        self.assertEqual([r["body"]["temperature"] for r in seen], [0, 0.7])

    def test_every_request_of_a_run_asks_for_it_the_judge_too(self):
        root = make_project(one_case(JUDGED))
        server, url, seen = start_recorder("PASS")
        try:
            code, out, err = run_cli(
                ["--root", root, "--base-url", url, "run", "--prompt", "s",
                 "--version", "v1", "--no-cache"])
        finally:
            stop(server)
        self.assertEqual(code, 0, (out, err))
        self.assertEqual(len(seen), 2)
        for request in seen:
            self.assertEqual(request["body"]["temperature"], 0)


class TestTheCacheKnowsWhatTheReplyDependsOn(unittest.TestCase):
    """README: responses are "cached ... keyed by prompt text + case + model
    + base URL"."""

    BASE = dict(name="s", version="v1", prompt_text="Classify: {{input}}",
                case={"name": "c", "input": "x", "vars": {},
                      "assert": [{"contains": "y"}]},
                model="m", base_url="http://a.example/v1")

    def test_every_part_changes_the_key(self):
        key = suite_mod.cache_key(**self.BASE)
        other = {"name": "t", "version": "v2",
                 "prompt_text": "Classify, please: {{input}}",
                 "case": dict(self.BASE["case"], input="z"),
                 "model": "n", "base_url": "http://b.example/v1"}
        for part, value in other.items():
            with self.subTest(part):
                changed = dict(self.BASE, **{part: value})
                self.assertNotEqual(suite_mod.cache_key(**changed), key)
        self.assertEqual(suite_mod.cache_key(**dict(self.BASE)), key)

    def test_editing_the_prompt_text_means_a_fresh_call(self):
        root = make_project(one_case("input: hi\nassert:\n  - contains: bar"))
        prompt = os.path.join(root, "prompts", "s.v1.txt")
        server, url, seen = start_recorder()
        argv = ["--root", root, "--base-url", url, "run", "--prompt", "s",
                "--version", "v1"]
        try:
            self.assertEqual(run_cli(argv)[0], 0)
            self.assertEqual(len(seen), 1)
            self.assertEqual(run_cli(argv)[0], 0)
            self.assertEqual(len(seen), 1, "the second run is from the cache")
            with open(prompt, "w") as f:
                f.write("New wording: {{input}}\n")
            self.assertEqual(run_cli(argv)[0], 0)
            self.assertEqual(len(seen), 2, "a reworded prompt is a new call")
        finally:
            stop(server)


class TestNoCacheMeansNoCache(unittest.TestCase):
    """README: "`--no-cache` forces fresh calls and writes nothing"."""

    def run_twice(self, body, answer):
        root = make_project(one_case(body))
        server, url, seen = start_recorder(answer)
        try:
            for _ in range(2):
                code, out, err = run_cli(
                    ["--root", root, "--base-url", url, "run", "--prompt",
                     "s", "--version", "v1", "--no-cache"])
                self.assertEqual(code, 0, (out, err))
        finally:
            stop(server)
        return root, seen

    def test_fresh_calls_and_nothing_written(self):
        root, seen = self.run_twice("input: hi\nassert:\n  - contains: bar",
                                    "bar")
        self.assertEqual(len(seen), 2)
        self.assertFalse(os.path.exists(os.path.join(root, ".promptregress")))

    def test_judge_verdicts_are_not_written_either(self):
        root, seen = self.run_twice(JUDGED, "PASS")
        self.assertEqual(len(seen), 4)
        self.assertFalse(os.path.exists(os.path.join(root, ".promptregress")))

    def test_what_an_earlier_run_cached_is_not_used(self):
        root = make_project(one_case("input: hi\nassert:\n  - contains: bar"))
        server, url, seen = start_recorder()
        argv = ["--root", root, "--base-url", url, "run", "--prompt", "s",
                "--version", "v1"]
        try:
            self.assertEqual(run_cli(argv)[0], 0)
            self.assertEqual(len(seen), 1)
            self.assertEqual(run_cli(argv + ["--no-cache"])[0], 0)
            self.assertEqual(len(seen), 2, "--no-cache answered from cache")
        finally:
            stop(server)


# ---------------------------------------------------------------------
# P8 - `diff` reports; it does not judge. Its exit code is documented.

class TestDiffExitsZeroWhenVerdictsFlip(unittest.TestCase):
    """The README's own example, `diff v1 v2`, shows a case that regressed
    and exits 0. That is the design (a flip can be a fix as well as a
    regression, `run` is what gates CI); what was missing was saying so."""

    def project(self):
        root = make_project(one_case("input: hi\nassert:\n  - contains: bar"))
        with open(os.path.join(root, "prompts", "s.v2.txt"), "w") as f:
            f.write("Wording B: {{input}}\n")
        return root

    def test_a_regression_is_shown_and_still_exits_0(self):
        root = self.project()
        server, url = start_fake_model({"Wording B": "nope"}, default="bar")
        try:
            base = ["--root", root, "--base-url", url]
            code, out, err = run_cli(base + ["run", "--prompt", "s",
                                             "--version", "v2", "--no-cache"])
            self.assertEqual(code, 1, "run is what fails a build")
            code, out, err = run_cli(base + ["diff", "v1", "v2", "--prompt",
                                             "s", "--no-cache"])
            self.assertEqual(code, 0, (out, err))
            self.assertIn("CHANGED", out)
            self.assertIn("1/1 case(s) changed verdict v1 -> v2", out)
            code, out, err = run_cli(base + ["diff", "v1", "v2", "--prompt",
                                             "s", "--no-cache", "--format",
                                             "json"])
            self.assertEqual(code, 0, (out, err))
            row = json.loads(out)["cases"][0]
            self.assertEqual((row["v1"], row["v2"], row["changed"]),
                             ("pass", "FAIL", True))
        finally:
            server.shutdown()
            server.server_close()

    def test_diff_says_so_in_its_help_and_the_readme(self):
        out = io.StringIO()
        with clean_env(), redirect_stdout(out):
            with self.assertRaises(SystemExit) as ctx:
                cli_mod.main(["diff", "--help"])
        self.assertEqual(ctx.exception.code, 0)
        text = " ".join(out.getvalue().split())
        self.assertIn("Exit status: 0 whenever the comparison was made", text)
        self.assertIn("gate a build with `run`", text)
        with open(os.path.join(REPO, "README.md"), encoding="utf-8") as f:
            readme = " ".join(f.read().split())
        self.assertIn("`diff` reports, it does not judge", readme)


# ---------------------------------------------------------------------
# Release 0.1.1: the package and the command line name the same version

class TestVersionsAgree(unittest.TestCase):
    def test_the_package_and_the_cli_name_the_same_version(self):
        with open(os.path.join(REPO, "pyproject.toml"),
                  encoding="utf-8") as f:
            found = re.search(r'^version\s*=\s*"([^"]+)"', f.read(), re.M)
        self.assertIsNotNone(found)
        self.assertEqual(found.group(1), cli_mod.VERSION)

    def test_dash_dash_version_prints_it(self):
        out = io.StringIO()
        with redirect_stdout(out):
            with self.assertRaises(SystemExit) as ctx:
                cli_mod.main(["--version"])
        self.assertEqual(ctx.exception.code, 0)
        self.assertEqual(out.getvalue().strip(),
                         f"promptregress {cli_mod.VERSION}")


if __name__ == "__main__":
    unittest.main()
