"""Runner + CLI tests against the fake model server."""

import errno
import io
import json
import os
import socket
import stat
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import warnings
from contextlib import redirect_stderr, redirect_stdout
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest import mock

from promptregress import cli as cli_mod
from promptregress import demo as demo_mod
from promptregress import miniyaml as miniyaml_mod
from promptregress import model as model_mod
from promptregress import runner as runner_mod
from promptregress import suite as suite_mod

from promptregress.demo import start_fake_model


def stop_server(server):
    """Stop a fake server and release its listening socket: shutdown() ends
    serve_forever(), only server_close() closes the socket."""
    server.shutdown()
    server.server_close()


def start_error_server(status: int, body: str):
    """An endpoint that answers every POST with `status` and `body`."""
    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            payload = body.encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_port}/v1"


def make_suite(tmp: str) -> None:
    d = os.path.join(tmp, "prompts")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "cls.v1.txt"), "w") as f:
        f.write("Classify: {{input}}\nReply with one word.")
    with open(os.path.join(d, "cls.v2.txt"), "w") as f:
        f.write("Classify sentiment of: {{input}}\nOne word only.")
    with open(os.path.join(d, "cls.cases.yaml"), "w") as f:
        f.write("""cases:
  - name: neg
    input: "terrible"
    assert:
      - contains: negative
  - name: pos
    input: "great"
    assert:
      - contains: positive
""")


def make_judge_suite(tmp: str) -> None:
    d = os.path.join(tmp, "prompts")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "cls.v1.txt"), "w") as f:
        f.write("Judge me: {{input}}")
    with open(os.path.join(d, "cls.cases.yaml"), "w") as f:
        f.write("""cases:
  - name: j1
    input: "hello"
    assert:
      - judge: "must be a single word"
""")


class TestModel(unittest.TestCase):
    def test_complete(self):
        server, url = start_fake_model({"hello": "world"})
        try:
            out = model_mod.complete("say hello", base_url=url,
                                     api_key="x", model="fake")
            self.assertEqual(out, "world")
        finally:
            stop_server(server)

    def test_unreachable(self):
        with self.assertRaises(model_mod.ModelError):
            model_mod.complete("hi", base_url="http://127.0.0.1:1/v1",
                               timeout=2)

    def test_judge_pass_fail(self):
        server, url = start_fake_model({"rubric-word": "PASS",
                                        "other": "FAIL because wrong tone"})
        kw = {"base_url": url, "api_key": "x", "model": "fake"}
        try:
            ok, _ = model_mod.judge("some output", "rubric-word", **kw)
            self.assertTrue(ok)
            ok, detail = model_mod.judge("some output", "other", **kw)
            self.assertFalse(ok)
            self.assertIn("FAIL", detail)
        finally:
            stop_server(server)


class TestRunner(unittest.TestCase):
    def test_run_and_cache(self):
        tmp = tempfile.mkdtemp()
        make_suite(tmp)
        server, url = start_fake_model({"terrible": "negative",
                                        "great": "positive"})
        kw = {"base_url": url, "api_key": "x", "model": "fake"}
        try:
            results, _ = runner_mod.run_suite(tmp, "cls", "v1", model_kw=kw)
            self.assertEqual(len(results), 2)
            self.assertTrue(all(r.passed for r in results))
            self.assertFalse(any(r.cached for r in results))
            # second run hits cache — shut the server to prove it
            stop_server(server)
            results2, _ = runner_mod.run_suite(tmp, "cls", "v1", model_kw=kw)
            self.assertTrue(all(r.cached for r in results2))
            self.assertTrue(all(r.passed for r in results2))
        finally:
            try:
                stop_server(server)
            except Exception:
                pass

    def test_cache_scoped_to_base_url(self):
        # Same model name at a different provider address must not be
        # served from the first provider's cache.
        tmp = tempfile.mkdtemp()
        make_suite(tmp)
        server, url = start_fake_model({"terrible": "negative",
                                        "great": "positive"})
        try:
            runner_mod.run_suite(tmp, "cls", "v1", model_kw={
                "base_url": url, "api_key": "x", "model": "fake"})
        finally:
            stop_server(server)
        results, _ = runner_mod.run_suite(tmp, "cls", "v1", model_kw={
            "base_url": "http://127.0.0.1:1/v1", "api_key": "x",
            "model": "fake"})
        self.assertFalse(any(r.cached for r in results))
        self.assertFalse(any(r.passed for r in results))
        self.assertTrue(all(r.error for r in results))

    def test_judge_cache_scoped_to_judge_provider(self):
        tmp = tempfile.mkdtemp()
        make_judge_suite(tmp)
        s1, u1 = start_fake_model({"You are a strict test judge": "PASS"})
        s2, u2 = start_fake_model({"You are a strict test judge": "FAIL"})
        try:
            r1, _ = runner_mod.run_suite(tmp, "cls", "v1", model_kw={
                "base_url": u1, "model": "fake"})
            self.assertTrue(r1[0].passed)
            prompt = suite_mod.load_prompt(tmp, "cls", "v1")
            case = suite_mod.load_cases(tmp, "cls")[0]
            r2 = runner_mod.run_case(
                tmp, "cls", "v1", prompt, case,
                {"base_url": u1, "model": "fake"},
                judge_model_kw={"base_url": u2, "model": "fake"})
            self.assertFalse(r2.passed)  # judge 2's own verdict
        finally:
            stop_server(s1)
            stop_server(s2)

    def test_failure_detail(self):
        tmp = tempfile.mkdtemp()
        make_suite(tmp)
        server, url = start_fake_model({})
        kw = {"base_url": url, "api_key": "x", "model": "fake"}
        try:
            results, _ = runner_mod.run_suite(tmp, "cls", "v1", model_kw=kw)
            self.assertFalse(all(r.passed for r in results))
            self.assertTrue(results[0].assertions)
        finally:
            stop_server(server)


    def test_judge_verdict_cached(self):
        # Second identical run must not call the judge again.
        tmp = tempfile.mkdtemp()
        make_judge_suite(tmp)
        server, url = start_fake_model({"You are a strict test judge":
                                        "PASS"})
        kw = {"base_url": url, "api_key": "x", "model": "fake"}
        try:
            with mock.patch.object(model_mod, "judge",
                                   wraps=model_mod.judge) as spy:
                r1, _ = runner_mod.run_suite(tmp, "cls", "v1", model_kw=kw)
                self.assertTrue(all(r.passed for r in r1))
                self.assertEqual(spy.call_count, 1)
                r2, _ = runner_mod.run_suite(tmp, "cls", "v1", model_kw=kw)
                self.assertTrue(all(r.passed for r in r2))
                self.assertTrue(all(r.cached for r in r2))
                self.assertEqual(spy.call_count, 1)
        finally:
            stop_server(server)

    def test_judge_cache_files(self):
        # Verdicts land in the existing response-cache dir, namespaced.
        tmp = tempfile.mkdtemp()
        make_judge_suite(tmp)
        server, url = start_fake_model({"You are a strict test judge":
                                        "FAIL"})
        kw = {"base_url": url, "api_key": "x", "model": "fake"}
        try:
            runner_mod.run_suite(tmp, "cls", "v1", model_kw=kw)
        finally:
            stop_server(server)
        cached = os.listdir(os.path.join(tmp, ".promptregress", "cache"))
        judges = [f for f in cached if f.startswith("judge-")]
        self.assertEqual(len(judges), 1)


class TestSuite(unittest.TestCase):
    def test_list_and_render(self):
        tmp = tempfile.mkdtemp()
        make_suite(tmp)
        self.assertEqual(suite_mod.list_prompts(tmp), {"cls": ["v1", "v2"]})
        t = suite_mod.load_prompt(tmp, "cls", "v1")
        self.assertIn("terrible", suite_mod.render(t, "terrible"))
        cases = suite_mod.load_cases(tmp, "cls")
        self.assertEqual(len(cases), 2)
        self.assertEqual(cases[0]["name"], "neg")

    def test_templating_vars(self):
        out = suite_mod.render("Hi {{name}}, {{input}}!", "yo",
                               {"name": "Dan"})
        self.assertEqual(out, "Hi Dan, yo!")

    def test_malformed_cases_yaml(self):
        tmp = tempfile.mkdtemp()
        make_suite(tmp)
        p = os.path.join(tmp, "prompts", "cls.cases.yaml")
        with open(p, "w") as f:
            f.write('cases:\n  - name: broken\n      input: "oops"\n')
        with self.assertRaises(ValueError) as ctx:
            suite_mod.load_cases(tmp, "cls")
        self.assertIn(p, str(ctx.exception))
        self.assertIn("malformed", str(ctx.exception))

    def test_cases_yaml_top_level_not_mapping(self):
        tmp = tempfile.mkdtemp()
        make_suite(tmp)
        p = os.path.join(tmp, "prompts", "cls.cases.yaml")
        with open(p, "w") as f:
            f.write("- just\n- a\n- list\n")
        with self.assertRaises(ValueError):
            suite_mod.load_cases(tmp, "cls")

    def test_miniyaml_parses_init_cases(self):
        # The vendored parser must handle the exact scaffold fixture shape.
        data = miniyaml_mod.safe_load(cli_mod.INIT_CASES)
        self.assertEqual(len(data["cases"]), 3)
        self.assertEqual(data["cases"][0]["assert"],
                         [{"contains": "negative"}])
        self.assertEqual(data["cases"][2]["assert"][1],
                         {"max_tokens": 10})
        self.assertIn("positive", data["cases"][2]["assert"][0]["regex"])


class TestCLI(unittest.TestCase):
    def _run(self, tmp, *argv, env_model=None):
        if env_model:
            os.environ["PROMPTREGRESS_BASE_URL"] = env_model
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = cli_mod.main(["--root", tmp, *argv])
        if env_model:
            del os.environ["PROMPTREGRESS_BASE_URL"]
        return code, buf.getvalue()

    def test_run_pass(self):
        tmp = tempfile.mkdtemp()
        make_suite(tmp)
        server, url = start_fake_model({"terrible": "negative",
                                        "great": "positive"})
        try:
            code, out = self._run(tmp, "run", "--prompt", "cls",
                                  "--version", "v1", env_model=url)
            self.assertEqual(code, 0)
            self.assertIn("2/2 passed", out)
        finally:
            stop_server(server)

    def test_run_fail(self):
        tmp = tempfile.mkdtemp()
        make_suite(tmp)
        server, url = start_fake_model({})  # default "ok" fails contains
        try:
            code, out = self._run(tmp, "run", "--prompt", "cls",
                                  "--version", "v1", env_model=url)
            self.assertEqual(code, 1)
            self.assertIn("FAIL neg", out)
        finally:
            stop_server(server)

    def test_diff(self):
        tmp = tempfile.mkdtemp()
        make_suite(tmp)
        # v1 answers right, v2 answers wrong
        server, url = start_fake_model({"Classify:": "negative",
                                        "Classify sentiment": "positive"})
        try:
            code, out = self._run(tmp, "diff", "v1", "v2", "--prompt", "cls",
                                  env_model=url)
            self.assertEqual(code, 0)
            self.assertIn("CHANGED", out)
        finally:
            stop_server(server)

    def test_diff_json(self):
        tmp = tempfile.mkdtemp()
        make_suite(tmp)
        server, url = start_fake_model({"terrible": "negative",
                                        "great": "positive"})
        try:
            code, out = self._run(tmp, "diff", "v1", "v2", "--prompt", "cls",
                                  "--format", "json", env_model=url)
            data = json.loads(out)
            self.assertEqual(data["prompt"], "cls")
            self.assertEqual(len(data["cases"]), 2)
        finally:
            stop_server(server)

    def test_run_json_exit_code(self):
        # --format json must exit nonzero when cases fail (CI honesty).
        tmp = tempfile.mkdtemp()
        make_suite(tmp)
        server, url = start_fake_model({"terrible": "negative",
                                        "great": "positive"})
        try:
            code, out = self._run(tmp, "run", "--prompt", "cls",
                                  "--version", "v1", "--format", "json",
                                  env_model=url)
            self.assertEqual(code, 0)
            self.assertEqual(json.loads(out)["failed"], 0)
        finally:
            stop_server(server)
        tmp2 = tempfile.mkdtemp()
        make_suite(tmp2)
        server2, url2 = start_fake_model({})
        try:
            code, out = self._run(tmp2, "run", "--prompt", "cls",
                                  "--version", "v1", "--format", "json",
                                  env_model=url2)
            self.assertEqual(code, 1)
            self.assertEqual(json.loads(out)["failed"], 2)
        finally:
            stop_server(server2)

    def test_diff_shows_before_after(self):
        tmp = tempfile.mkdtemp()
        make_suite(tmp)
        server, url = start_fake_model({"Classify:": "negative",
                                        "Classify sentiment": "positive"})
        try:
            code, out = self._run(tmp, "diff", "v1", "v2", "--prompt", "cls",
                                  env_model=url)
            self.assertEqual(code, 0)
            self.assertIn("CHANGED", out)
            self.assertIn("'negative'", out)
            self.assertIn("'positive'", out)
        finally:
            stop_server(server)

    def test_diff_json_includes_outputs(self):
        tmp = tempfile.mkdtemp()
        make_suite(tmp)
        server, url = start_fake_model({"Classify:": "negative",
                                        "Classify sentiment": "positive"})
        try:
            code, out = self._run(tmp, "diff", "v1", "v2", "--prompt", "cls",
                                  "--format", "json", env_model=url)
            self.assertEqual(code, 0)
            data = json.loads(out)
            flipped = [c for c in data["cases"] if c["changed"]]
            self.assertEqual(len(flipped), 2)
            by_name = {c["name"]: c for c in data["cases"]}
            self.assertTrue(by_name["neg"]["changed"])
            self.assertEqual(by_name["neg"]["v1_output"], "negative")
            self.assertEqual(by_name["neg"]["v2_output"], "positive")
        finally:
            stop_server(server)

    def test_malformed_cases_yaml_cli(self):
        # Clean stderr message + exit 2, no traceback.
        tmp = tempfile.mkdtemp()
        make_suite(tmp)
        with open(os.path.join(tmp, "prompts", "cls.cases.yaml"),
                  "w") as f:
            f.write('cases:\n  - name: broken\n      input: "oops"\n')
        out_buf, err_buf = io.StringIO(), io.StringIO()
        with redirect_stdout(out_buf), redirect_stderr(err_buf):
            code = cli_mod.main(["--root", tmp, "run", "--prompt", "cls",
                                 "--version", "v1"])
        self.assertEqual(code, 2)
        err = err_buf.getvalue()
        self.assertIn("error:", err)
        self.assertIn("malformed", err)
        self.assertNotIn("Traceback", err)

    def test_demo_run_scaffolds_and_passes(self):
        # Zero-dependency trial: empty dir, no model server needed.
        tmp = tempfile.mkdtemp()
        os.environ.pop("PROMPTREGRESS_BASE_URL", None)
        code, out = self._run(tmp, "--demo", "run", "--prompt", "sentiment",
                              "--version", "v1")
        self.assertEqual(code, 0)
        self.assertIn("3/3 passed", out)

    def test_demo_uses_fake_model(self):
        # Existing prompts + --demo: answers come from the fake model,
        # so a missing Ollama is never contacted.
        tmp = tempfile.mkdtemp()
        make_suite(tmp)
        os.environ.pop("PROMPTREGRESS_BASE_URL", None)
        code, out = self._run(tmp, "--demo", "run", "--prompt", "cls",
                              "--version", "v1")
        self.assertEqual(code, 1)  # fake answers "ok" -> contains fails
        self.assertIn("FAIL", out)
        self.assertNotIn("unreachable", out)

    def test_demo_diff(self):
        # The demo's v2 chats on the neutral review (the README headline):
        # one verdict flips, with before/after outputs shown.
        tmp = tempfile.mkdtemp()
        os.environ.pop("PROMPTREGRESS_BASE_URL", None)
        code, out = self._run(tmp, "--demo", "diff", "v1", "v2",
                              "--prompt", "sentiment")
        self.assertEqual(code, 0)
        self.assertIn("1/3 case(s) changed", out)
        self.assertIn("neutral review    pass    FAIL  ← CHANGED", out)
        self.assertIn("v1 output: 'neutral'", out)
        self.assertIn(demo_mod.CHATTY_NEUTRAL, out)
        # and `run` on v2 fails that case for the documented reason
        code, out = self._run(tmp, "--demo", "run", "--prompt", "sentiment",
                              "--version", "v2")
        self.assertEqual(code, 1)
        self.assertIn("sentiment.v2: ..F", out)
        self.assertIn("{'max_tokens': 10} -> ~26 tokens (max 10)", out)
        self.assertIn("2/3 passed", out)
        code, out = self._run(tmp, "--demo", "run", "--prompt", "sentiment",
                              "--version", "v1")
        self.assertEqual(code, 0)
        self.assertIn("3/3 passed", out)

    def test_demo_run_never_feeds_real_run_cache(self):
        # HIGH regression: --demo wrote fake-model answers into the real
        # cache (key ignored base_url), so a later real run with no model
        # reachable passed from cache.
        tmp = tempfile.mkdtemp()
        cli_mod.scaffold_example(tmp)
        os.environ.pop("PROMPTREGRESS_BASE_URL", None)
        code, out = self._run(tmp, "--demo", "run", "--prompt", "sentiment",
                              "--version", "v1")
        self.assertEqual(code, 0)
        self.assertIn("3/3 passed", out)
        self.assertFalse(os.path.exists(
            os.path.join(tmp, ".promptregress", "cache")))
        out_buf, err_buf = io.StringIO(), io.StringIO()
        with redirect_stdout(out_buf), redirect_stderr(err_buf):
            code = cli_mod.main(["--root", tmp, "--base-url",
                                 "http://127.0.0.1:1/v1", "run", "--prompt",
                                 "sentiment", "--version", "v1"])
        self.assertEqual(code, 2)
        self.assertIn("model unreachable", err_buf.getvalue())
        self.assertNotIn("passed", out_buf.getvalue())

    def test_missing_prompt(self):
        tmp = tempfile.mkdtemp()
        code, _ = self._run(tmp, "run", "--prompt", "nope",
                            "--version", "v1")
        self.assertEqual(code, 2)

    def test_diff_unreachable_model(self):
        # Model endpoint dead: no misleading all-FAIL table,
        # clean stderr message + exit 2 so CI can gate on it.
        # Port 1 refuses instantly, but keep the URL unreachable-only;
        # no timeout plumbing needed since connect() fails fast.
        tmp = tempfile.mkdtemp()
        make_suite(tmp)
        out_buf, err_buf = io.StringIO(), io.StringIO()
        with redirect_stdout(out_buf), redirect_stderr(err_buf):
            code = cli_mod.main(["--root", tmp, "--base-url",
                                 "http://127.0.0.1:1",
                                 "diff", "v1", "v2", "--prompt", "cls",
                                 "--no-cache"])
        self.assertEqual(code, 2)
        err = err_buf.getvalue()
        self.assertIn("error: model unreachable", err)
        self.assertIn("PROMPTREGRESS_BASE_URL", err)
        # the all-FAIL table must not be emitted
        self.assertNotIn("CHANGED", out_buf.getvalue())
        self.assertNotIn("case(s) changed", out_buf.getvalue())

    def test_version_flag(self):
        out_buf = io.StringIO()
        with redirect_stdout(out_buf):
            with self.assertRaises(SystemExit) as ctx:
                cli_mod.main(["--version"])
        self.assertEqual(ctx.exception.code, 0)
        self.assertIn("promptregress", out_buf.getvalue())

    # ---- round-2 release-panel regressions ----

    def _run_capture(self, argv):
        out_buf, err_buf = io.StringIO(), io.StringIO()
        with redirect_stdout(out_buf), redirect_stderr(err_buf):
            code = cli_mod.main(argv)
        return code, out_buf.getvalue(), err_buf.getvalue()

    def test_demo_ignores_base_url_env(self):
        # PROMPTREGRESS_BASE_URL in the environment used to override
        # --demo, so the "zero-setup trial" silently hit a real (here:
        # dead) server. --demo must always use the fake model, and say
        # once that the variable is being ignored.
        tmp = tempfile.mkdtemp()
        old = os.environ.get("PROMPTREGRESS_BASE_URL")
        os.environ["PROMPTREGRESS_BASE_URL"] = "http://127.0.0.1:1/v1"
        try:
            code, out, err = self._run_capture(
                ["--root", tmp, "--demo", "run", "--prompt", "sentiment",
                 "--version", "v1"])
        finally:
            if old is None:
                del os.environ["PROMPTREGRESS_BASE_URL"]
            else:
                os.environ["PROMPTREGRESS_BASE_URL"] = old
        self.assertEqual(code, 0, err)
        self.assertIn("3/3 passed", out)
        self.assertNotIn("unreachable", err)
        self.assertEqual(err.count("ignoring PROMPTREGRESS_BASE_URL="), 1)
        # an explicit --base-url is ignored in --demo too, and said so
        code, out, err = self._run_capture(
            ["--root", tmp, "--demo", "--base-url", "http://127.0.0.1:1/v1",
             "run", "--prompt", "sentiment", "--version", "v1"])
        self.assertEqual(code, 0, err)
        self.assertIn("3/3 passed", out)
        self.assertIn("ignoring --base-url http://127.0.0.1:1/v1", err)

    def test_model_errors_show_status_and_message(self):
        # Every ModelError printed the same "model unreachable; set
        # PROMPTREGRESS_BASE_URL" line, so a bad API key (401), a wrong
        # path (404) and a dead host were indistinguishable.
        tmp = tempfile.mkdtemp()
        cli_mod.scaffold_example(tmp)
        argv = ["run", "--prompt", "sentiment", "--version", "v1",
                "--no-cache"]
        server, url = start_error_server(
            401, '{"error": {"message": "Invalid API key", "type": "auth"}}')
        try:
            code, out, err401 = self._run_capture(
                ["--root", tmp, "--base-url", url, *argv])
        finally:
            stop_server(server)
        self.assertEqual(code, 2)
        self.assertIn("error: model at http://127.0.0.1:", err401)
        self.assertIn("HTTP 401 Unauthorized: Invalid API key", err401)
        self.assertNotIn("unreachable", err401)
        self.assertEqual(out, "")  # no all-FAIL run printed
        # 404: the fake model only serves /v1/chat/completions
        server, url = start_fake_model({})
        try:
            code, out, err404 = self._run_capture(
                ["--root", tmp, "--base-url", url.replace("/v1", ""), *argv])
        finally:
            stop_server(server)
        self.assertEqual(code, 2)
        self.assertIn("HTTP 404 Not Found", err404)
        self.assertNotIn("<html", err404.lower())  # HTML page not dumped
        # connection refused keeps the "unreachable" wording and the hint
        code, out, errdead = self._run_capture(
            ["--root", tmp, "--base-url", "http://127.0.0.1:1/v1", *argv])
        self.assertEqual(code, 2)
        self.assertIn("error: model unreachable at http://127.0.0.1:1/v1",
                      errdead)
        self.assertIn("PROMPTREGRESS_BASE_URL", errdead)
        self.assertEqual(len({err401, err404, errdead}), 3)
        # diff reports the same real reason
        server, url = start_error_server(401, '{"error": "nope"}')
        try:
            code, out, err = self._run_capture(
                ["--root", tmp, "--base-url", url, "diff", "v1", "v2",
                 "--prompt", "sentiment", "--no-cache"])
        finally:
            stop_server(server)
        self.assertEqual(code, 2)
        self.assertIn("HTTP 401 Unauthorized: nope", err)
        self.assertNotIn("CHANGED", out)

    def test_demo_scaffold_dir_is_removed(self):
        # --demo scaffolded into tempfile.mkdtemp() and never cleaned up:
        # one promptregress-demo-* dir per trial run piled up in /tmp.
        # a private temp dir: another process (a parallel test run, another
        # promptregress) also creates and removes promptregress-demo-* dirs in
        # the shared one
        private = tempfile.mkdtemp()
        tmp = tempfile.mkdtemp()
        os.environ.pop("PROMPTREGRESS_BASE_URL", None)
        with mock.patch.object(tempfile, "tempdir", private):
            code, out, err = self._run_capture(
                ["--root", tmp, "--demo", "diff", "v1", "v2",
                 "--prompt", "sentiment"])
        self.assertEqual(code, 0, err)
        self.assertIn("scaffolded example prompts in", err)
        self.assertEqual(os.listdir(private), [])
        scaffolded = err.split("scaffolded example prompts in ", 1)[1]
        scaffolded = scaffolded.split(";", 1)[0]
        self.assertFalse(os.path.exists(scaffolded), scaffolded)

    def test_demo_json_output_is_pure_json(self):
        # the scaffold notice goes to stderr so --format json stays parseable
        tmp = tempfile.mkdtemp()
        os.environ.pop("PROMPTREGRESS_BASE_URL", None)
        code, out, err = self._run_capture(
            ["--root", tmp, "--demo", "run", "--prompt", "sentiment",
             "--version", "v1", "--format", "json"])
        self.assertEqual(code, 0, err)
        self.assertEqual(json.loads(out)["failed"], 0)

    def test_init_writes_cache_gitignore(self):
        tmp = tempfile.mkdtemp()
        code, out, _ = self._run_capture(["--root", tmp, "init"])
        self.assertEqual(code, 0)
        p = os.path.join(tmp, ".promptregress", ".gitignore")
        self.assertTrue(os.path.isfile(p))
        with open(p, encoding="utf-8") as f:
            self.assertIn("*\n!.gitignore\n", f.read())
        self.assertIn(".promptregress/.gitignore", out)
        # a run that fills the cache in a fresh clone creates it too
        tmp2 = tempfile.mkdtemp()
        make_suite(tmp2)
        server, url = start_fake_model({"terrible": "negative",
                                        "great": "positive"})
        try:
            runner_mod.run_suite(tmp2, "cls", "v1", model_kw={
                "base_url": url, "api_key": "x", "model": "fake"})
        finally:
            stop_server(server)
        self.assertTrue(os.path.isfile(
            os.path.join(tmp2, ".promptregress", ".gitignore")))


def start_hanging_server():
    """Accepts connections and never answers (a stuck model endpoint)."""
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(5)
    conns = []

    def accept():
        while True:
            try:
                c, _ = srv.accept()
            except OSError:
                return
            conns.append(c)

    threading.Thread(target=accept, daemon=True).start()

    def close():
        srv.close()
        for c in conns:
            c.close()
    return close, f"http://127.0.0.1:{srv.getsockname()[1]}/v1"


class TestRound3(unittest.TestCase):
    """Round-3 release-panel regressions."""

    def _run_capture(self, argv):
        out_buf, err_buf = io.StringIO(), io.StringIO()
        with redirect_stdout(out_buf), redirect_stderr(err_buf):
            code = cli_mod.main(argv)
        return code, out_buf.getvalue(), err_buf.getvalue()

    def test_cp1252_console_does_not_traceback(self):
        # A FAIL row prints ✗ and a CHANGED row prints ←: with a cp1252
        # (Windows pipe) stdout both died with UnicodeEncodeError.
        tmp = tempfile.mkdtemp()
        env = {**os.environ, "PYTHONIOENCODING": "cp1252:strict"}
        env.pop("PYTHONUTF8", None)
        env.pop("PROMPTREGRESS_BASE_URL", None)
        repo_root = os.path.dirname(os.path.dirname(
            os.path.abspath(cli_mod.__file__)))

        def run(*argv):
            return subprocess.run(
                [sys.executable, "-m", "promptregress", "--root", tmp, *argv],
                capture_output=True, cwd=repo_root, env=env, timeout=120)

        p = run("--demo", "run", "--prompt", "sentiment", "--version", "v2")
        self.assertEqual(p.returncode, 1, p.stderr)
        self.assertNotIn(b"Traceback", p.stderr)
        self.assertIn(b"FAIL neutral review", p.stdout)
        self.assertIn(b"? {'max_tokens': 10}", p.stdout)  # replaced mark
        self.assertIn(b"2/3 passed", p.stdout)
        p = run("--demo", "diff", "v1", "v2", "--prompt", "sentiment")
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertNotIn(b"Traceback", p.stderr)
        self.assertIn(b"? CHANGED", p.stdout)
        self.assertIn(b"1/3 case(s) changed", p.stdout)

    def test_raw_yaml_scalars_in_assertions_are_bad_assertions(self):
        # `contains: yes` (a YAML bool), `contains: ~` (null) and
        # `max_tokens: "10"` raised TypeError out of the evaluator;
        # `regex: (` raised re.error. A suite holding such assertions is
        # now refused when it is loaded - exit 2 with every problem named,
        # before any model is called - and what YAML hands over unambiguously
        # (the number 1, the text "10" for a count) still works.
        tmp = tempfile.mkdtemp()
        d = os.path.join(tmp, "prompts")
        os.makedirs(d)
        with open(os.path.join(d, "yn.v1.txt"), "w") as f:
            f.write("Answer: {{input}}")
        cases = os.path.join(d, "yn.cases.yaml")
        with open(cases, "w") as f:
            f.write("""cases:
  - name: bool
    input: "a"
    assert:
      - contains: yes
  - name: int
    input: "b"
    assert:
      - contains: 1
  - name: nothing
    input: "c"
    assert:
      - contains: ~
  - name: quoted-count
    input: "d"
    assert:
      - max_tokens: "10"
  - name: bad-regex
    input: "e"
    assert:
      - regex: (
""")
        server, url = start_fake_model({}, default="1 yes")
        argv = ["--root", tmp, "--base-url", url, "run", "--prompt", "yn",
                "--version", "v1", "--no-cache"]
        try:
            code, out, err = self._run_capture(argv)
            self.assertEqual(code, 2)
            self.assertNotIn("Traceback", err)
            self.assertEqual(out, "")          # nothing was judged
            self.assertIn("case 0 ('bool'): assert[0]: contains: True is a "
                          "YAML boolean", err)
            self.assertIn('quote the text you mean, e.g. contains: "yes"',
                          err)
            self.assertIn("case 2 ('nothing'): assert[0]: contains: needs "
                          "text, got null", err)
            self.assertIn("case 4 ('bad-regex'): assert[0]: regex: invalid "
                          "regex /(/", err)
            self.assertNotIn("case 1", err)    # the number 1 is the text "1"
            self.assertNotIn("case 3", err)    # "10" is a fine count
            with open(cases, "w") as f:
                f.write("cases:\n  - name: int\n    input: b\n    assert:\n"
                        "      - contains: 1\n  - name: quoted-count\n"
                        "    input: d\n    assert:\n"
                        "      - max_tokens: \"10\"\n")
            code, out, err = self._run_capture(argv)
        finally:
            stop_server(server)
        self.assertEqual(code, 0, err)
        self.assertIn("2/2 passed", out)

    def test_model_flags_accepted_after_the_subcommand(self):
        # --model/--base-url/--api-key/--demo were top-level only:
        # `run --prompt x --version v1 --base-url …` was "unrecognized
        # arguments". Both positions must work and mean the same.
        tmp = tempfile.mkdtemp()
        make_suite(tmp)
        server, url = start_fake_model({"terrible": "negative",
                                        "great": "positive"})
        try:
            for argv in (["--root", tmp, "--base-url", url, "--model", "m1",
                          "run", "--prompt", "cls", "--version", "v1",
                          "--no-cache"],
                         ["--root", tmp, "run", "--prompt", "cls",
                          "--version", "v1", "--no-cache", "--base-url", url,
                          "--model", "m1", "--api-key", "k", "--timeout", "5"],
                         ["--root", tmp, "diff", "v1", "v2", "--prompt", "cls",
                          "--no-cache", "--base-url", url]):
                code, out, err = self._run_capture(argv)
                self.assertEqual(code, 0, (argv, err))
                self.assertIn("passed" if "run" in argv else "case(s) changed",
                              out)
        finally:
            stop_server(server)
        # a top-level value survives an unset subcommand copy
        args = cli_mod.build_parser().parse_args(
            ["--base-url", "http://x/v1", "--demo", "run", "--prompt", "a",
             "--version", "v1"])
        self.assertEqual(args.base_url, "http://x/v1")
        self.assertTrue(args.demo)
        self.assertIsNone(args.timeout)
        args = cli_mod.build_parser().parse_args(
            ["run", "--prompt", "a", "--version", "v1", "--demo",
             "--timeout", "2.5"])
        self.assertTrue(args.demo)
        self.assertEqual(args.timeout, 2.5)
        # --demo after the subcommand really uses the fake model
        os.environ.pop("PROMPTREGRESS_BASE_URL", None)
        code, out, err = self._run_capture(
            ["--root", tempfile.mkdtemp(), "run", "--prompt", "sentiment",
             "--version", "v1", "--demo"])
        self.assertEqual(code, 0, err)
        self.assertIn("3/3 passed", out)

    def test_timeout_flag_and_fail_fast_on_hung_endpoint(self):
        # No --timeout and no fail-fast: a hung endpoint cost
        # cases x 120 s. Now the first unanswered case ends the run.
        tmp = tempfile.mkdtemp()
        make_suite(tmp)  # 2 cases
        close, url = start_hanging_server()
        try:
            t0 = time.monotonic()
            code, out, err = self._run_capture(
                ["--root", tmp, "run", "--prompt", "cls", "--version", "v1",
                 "--no-cache", "--base-url", url, "--timeout", "1"])
            elapsed = time.monotonic() - t0
        finally:
            close()
        self.assertEqual(code, 2)
        self.assertLess(elapsed, 4.0, "second case must not pay the timeout")
        self.assertIn("error: model at", err)
        self.assertIn("timed out after 1s", err)
        self.assertIn("--timeout", err)
        self.assertIn("stopped at case 'neg'; no further case was run", err)
        self.assertEqual(out, "")
        # the runner API: results stop at the failing case
        results, _ = runner_mod.run_suite(
            tmp, "cls", "v1", model_kw={"base_url": "http://127.0.0.1:1/v1"},
            use_cache=False)
        self.assertEqual(len(results), 1)
        self.assertTrue(results[0].error)
        results, _ = runner_mod.run_suite(
            tmp, "cls", "v1", model_kw={"base_url": "http://127.0.0.1:1/v1"},
            use_cache=False, fail_fast=False)
        self.assertEqual(len(results), 2)
        # env default and the built-in default
        self.assertEqual(model_mod._cfg()["timeout"], model_mod.DEFAULT_TIMEOUT)
        old = os.environ.get("PROMPTREGRESS_TIMEOUT")
        os.environ["PROMPTREGRESS_TIMEOUT"] = "7"
        try:
            self.assertEqual(model_mod._cfg()["timeout"], 7.0)
        finally:
            if old is None:
                del os.environ["PROMPTREGRESS_TIMEOUT"]
            else:
                os.environ["PROMPTREGRESS_TIMEOUT"] = old

    def test_run_all(self):
        # One command for CI: every prompt x every version, a summary
        # table, exit 1 when any case fails.
        tmp = tempfile.mkdtemp()
        make_suite(tmp)
        cli_mod.scaffold_example(tmp)  # sentiment v1/v2 too
        server, url = start_fake_model({
            "Classify sentiment": "positive",  # cls.v2 breaks the neg case
            "terrible": "negative", "great": "positive",
            **demo_mod.DEMO_ROUTES})
        try:
            code, out, err = self._run_capture(
                ["--root", tmp, "--base-url", url, "run", "--all",
                 "--no-cache"])
            self.assertEqual(code, 1, err)
            self.assertIn("cls.v1: ..", out)
            self.assertIn("cls.v2: F.", out)
            self.assertIn("sentiment.v1: ...", out)
            self.assertIn("sentiment.v2: ..F", out)
            lines = out.splitlines()
            self.assertIn("suite         passed", lines)
            self.assertIn("cls.v1        2/2", lines)
            self.assertIn("cls.v2        1/2  FAIL", lines)
            self.assertIn("sentiment.v2  2/3  FAIL", lines)
            self.assertIn("4 suite(s), 2 with failures; 8/10 cases passed", out)
            code, out, err = self._run_capture(
                ["--root", tmp, "--base-url", url, "run", "--all",
                 "--no-cache", "--format", "json"])
            self.assertEqual(code, 1)
            data = json.loads(out)
            self.assertEqual([(s["prompt"], s["version"]) for s in data["suites"]],
                             [("cls", "v1"), ("cls", "v2"),
                              ("sentiment", "v1"), ("sentiment", "v2")])
            self.assertEqual((data["passed"], data["failed"]), (8, 2))
        finally:
            stop_server(server)
        # all green -> exit 0
        tmp2 = tempfile.mkdtemp()
        make_suite(tmp2)
        server, url = start_fake_model({"terrible": "negative",
                                        "great": "positive"})
        try:
            code, out, err = self._run_capture(
                ["--root", tmp2, "--base-url", url, "run", "--all",
                 "--no-cache"])
        finally:
            stop_server(server)
        self.assertEqual(code, 0, err)
        self.assertIn("2 suite(s), 0 with failures; 4/4 cases passed", out)
        # a dead model stops the whole thing with the real reason
        code, out, err = self._run_capture(
            ["--root", tmp2, "--base-url", "http://127.0.0.1:1/v1", "run",
             "--all", "--no-cache"])
        self.assertEqual(code, 2)
        self.assertIn("error: model unreachable", err)
        self.assertEqual(out, "")
        # neither --all nor --prompt/--version: usage error, exit 2
        code, out, err = self._run_capture(["--root", tmp2, "run"])
        self.assertEqual(code, 2)
        self.assertIn("--all", err)
        # no prompts at all
        code, out, err = self._run_capture(
            ["--root", tempfile.mkdtemp(), "run", "--all"])
        self.assertEqual(code, 2)
        self.assertIn("no prompts found", err)

    def test_demo_run_all_scaffolds(self):
        tmp = tempfile.mkdtemp()
        os.environ.pop("PROMPTREGRESS_BASE_URL", None)
        code, out, err = self._run_capture(
            ["--root", tmp, "--demo", "run", "--all"])
        self.assertEqual(code, 1, err)  # the demo v2 regression
        self.assertIn("scaffolded example prompts", err)
        self.assertIn("sentiment.v1  3/3", out)
        self.assertIn("sentiment.v2  2/3  FAIL", out)

    def test_cache_helpers_are_shared(self):
        # response and judge-verdict entries go through one read/write pair
        tmp = tempfile.mkdtemp()
        runner_mod._cache_write(tmp, "k1", {"output": "o"})
        runner_mod._cache_write(tmp, "judge-k2", {"passed": True, "detail": "d"})
        self.assertEqual(runner_mod._cache_read(tmp, "k1"), {"output": "o"})
        self.assertEqual(runner_mod._cache_read(tmp, "judge-k2"),
                         {"passed": True, "detail": "d"})
        self.assertIsNone(runner_mod._cache_read(tmp, "missing"))
        with open(runner_mod._cache_path(tmp, "k1"), "w") as f:
            f.write("not json")
        self.assertIsNone(runner_mod._cache_read(tmp, "k1"))
        self.assertFalse(hasattr(runner_mod, "_judge_cache_get"))


def http_reply(body, status="200 OK", length=None):
    """A complete HTTP/1.1 response as bytes. `length` overrides the
    Content-Length header (to promise more body than is sent)."""
    n = len(body) if length is None else length
    return (f"HTTP/1.1 {status}\r\nContent-Type: application/json\r\n"
            f"Content-Length: {n}\r\nConnection: close\r\n\r\n"
            ).encode() + body


def start_raw_server(reply):
    """Reads one request, then writes `reply` (bytes) verbatim and hangs
    up: lets a test craft cut-off, garbled or non-UTF-8 replies exactly.
    Returns (close, url) like start_hanging_server."""
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(5)

    def serve():
        while True:
            try:
                conn, _ = srv.accept()
            except OSError:
                return
            with conn:
                buf = b""
                while b"\r\n\r\n" not in buf:
                    chunk = conn.recv(65536)
                    if not chunk:
                        break
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
                conn.sendall(reply)

    threading.Thread(target=serve, daemon=True).start()
    return srv.close, f"http://127.0.0.1:{srv.getsockname()[1]}/v1"


class TestRound4(unittest.TestCase):
    """Round-4 expert-review regressions."""

    # Python's own wording must never reach the user.
    JARGON = ("NoneType", "subscriptable", "indices", "KeyError",
              "IndexError", "TypeError", "AttributeError", "Traceback",
              "IncompleteRead", "codec")

    def _run_capture(self, argv):
        out_buf, err_buf = io.StringIO(), io.StringIO()
        with redirect_stdout(out_buf), redirect_stderr(err_buf):
            code = cli_mod.main(argv)
        return code, out_buf.getvalue(), err_buf.getvalue()

    def _model_error(self, reply):
        """The ModelError text complete() raises for the raw HTTP `reply`."""
        close, url = start_raw_server(reply)
        try:
            with self.assertRaises(model_mod.ModelError) as ctx:
                model_mod.complete("hi", base_url=url, timeout=5)
        finally:
            close()
        msg = str(ctx.exception)
        self.assertTrue(msg.startswith("model at http://127.0.0.1:"), msg)
        for word in self.JARGON:
            self.assertNotIn(word, msg)
        return msg

    def test_unexpected_reply_shapes_are_model_errors(self):
        # `"content": null` (an OpenAI refusal or tool-call reply), list /
        # number / object content, no choices, a non-object body ... used
        # to be returned as None / list / ... and crash later (TypeError in
        # an assertion, AttributeError in the judge) or surface as Python's
        # own wording ("list index out of range").
        shapes = [
            ('{"choices":[{"message":{"content":null}}]}',
             "answered without any text (content is null)"),
            ('{"choices":[{"message":{"role":"assistant","content":null,'
             '"refusal":"I cannot help with that."}}]}',
             "it refused: I cannot help with that."),
            ('{"choices":[{"message":{"content":null,'
             '"tool_calls":[{"id":"call_1"}]}}]}', "tool call"),
            ('{"choices":[{"message":{}}]}', "(content is missing)"),
            ('{"choices":[{"message":{"content":[]}}]}', "no text parts"),
            ('{"choices":[{"message":{"content":[{"type":"image_url",'
             '"image_url":{}}]}}]}', "no text parts"),
            ('{"choices":[{"message":{"content":5}}]}',
             "content is a number"),
            ('{"choices":[{"message":{"content":{"a":1}}}]}',
             "content is an object"),
            ('{"choices":[{"message":{"content":true}}]}',
             "content is a boolean"),
            ('{"choices":[]}', "no choices"),
            ('{"choices":null}', "no choices"),
            ('{"id":"x"}', "no choices"),
            ('{"choices":"abc"}', '"choices" is a string'),
            ('{"choices":[1]}', "no message"),
            ('{"choices":[{"message":"hi"}]}', "no message"),
            ('{"choices":[{"text":"legacy completions shape"}]}',
             "no message"),
            ('[1,2]', "got an array"),
            ('"hi"', "got a string"),
            ('7', "got a number"),
            ('null', "got null"),
            # an error envelope sent with HTTP 200 (some gateways do)
            ('{"error":{"message":"Provider returned error","code":502}}',
             "returned an error: Provider returned error"),
            ('{"error":"overloaded"}', "returned an error: overloaded"),
        ]
        for body, want in shapes:
            msg = self._model_error(http_reply(body.encode()))
            self.assertIn(want, msg, body)

    def test_legitimate_reply_shapes_still_work(self):
        ok = [
            ('{"choices":[{"message":{"content":"hello"}}]}', "hello"),
            # an empty answer is still an answer
            ('{"choices":[{"message":{"content":""}}]}', ""),
            ('{"id":"c1","choices":[{"index":0,"message":{"role":'
             '"assistant","content":"hi","tool_calls":[{"id":"1"}],'
             '"refusal":null},"finish_reason":"stop"}],'
             '"usage":{"total_tokens":3}}', "hi"),
            # content as a list of text parts: the text parts, joined
            ('{"choices":[{"message":{"content":[{"type":"text","text":'
             '"he"},{"type":"text","text":"llo"}]}}]}', "hello"),
            ('{"choices":[{"message":{"content":[{"type":"image_url"},'
             '{"type":"text","text":"x"},{"type":"refusal","refusal":'
             '"no"}]}}]}', "x"),
            # first choice wins
            ('{"choices":[{"message":{"content":"first"}},'
             '{"message":{"content":"second"}}]}', "first"),
            ('{"choices":[{"message":{"content":"caf\\u00e9 \\u2713"}}]}',
             "café ✓"),
        ]
        for body, want in ok:
            close, url = start_raw_server(http_reply(body.encode()))
            try:
                self.assertEqual(model_mod.complete("hi", base_url=url,
                                                    timeout=5), want, body)
            finally:
                close()
        # raw (not escaped) UTF-8 in the body is fine too
        close, url = start_raw_server(http_reply(
            '{"choices":[{"message":{"content":"café"}}]}'
            .encode("utf-8")))
        try:
            self.assertEqual(model_mod.complete("hi", base_url=url,
                                                timeout=5), "café")
        finally:
            close()

    def test_cut_off_and_garbled_replies_are_model_errors(self):
        # IncompleteRead (body shorter than Content-Length, or a chunked
        # body cut mid-chunk), a server that hangs up without answering,
        # a reply that is not HTTP, and a body that is not UTF-8 all
        # escaped complete() as raw exceptions.
        body = b'{"choices":[{"message":{"content":"hello"}}]}'
        cases = [
            (http_reply(body[:12], length=len(body)),
             "hung up before the whole reply arrived (got 12 bytes)"),
            (b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n"
             b"64\r\n" + body[:10],
             "hung up before the whole reply arrived"),
            (b"", "failed:"),
            (b"HELLO THERE\r\n\r\n", "failed: HELLO THERE"),
            (http_reply(b'{"choices":[{"message":{"content":"caf\xe9"}}]}'),
             "not UTF-8 text"),
            (http_reply('{"choices":[]}'.encode("utf-16")),
             "not UTF-8 text"),
            (http_reply(b"<html>login</html>"), "not valid JSON"),
            (http_reply(b""), "not valid JSON"),
        ]
        for reply, want in cases:
            self.assertIn(want, self._model_error(reply), reply[:60])

    def test_unusable_replies_stop_the_cli_with_exit_2(self):
        # Exit 2 and one plain "error:" line. A traceback exits 1, which
        # CI reads as "a case regressed".
        tmp = tempfile.mkdtemp()
        make_suite(tmp)
        replies = {
            "null content": http_reply(
                b'{"choices":[{"message":{"content":null}}]}'),
            "cut off": http_reply(b'{"choices":[', length=500),
            "not utf-8": http_reply(
                b'{"choices":[{"message":{"content":"caf\xe9"}}]}'),
        }
        for label, reply in replies.items():
            close, url = start_raw_server(reply)
            try:
                code, out, err = self._run_capture(
                    ["--root", tmp, "--base-url", url, "run", "--prompt",
                     "cls", "--version", "v1", "--no-cache"])
            finally:
                close()
            self.assertEqual(code, 2, (label, err))
            self.assertEqual(out, "", label)
            self.assertTrue(
                err.startswith("error: model at http://127.0.0.1:"),
                (label, err))
            self.assertIn("stopped at case 'neg'", err)
            for word in self.JARGON:
                self.assertNotIn(word, err, label)
        close, url = start_raw_server(
            replies["null content"])
        try:
            code, out, err = self._run_capture(
                ["--root", tmp, "--base-url", url, "diff", "v1", "v2",
                 "--prompt", "cls", "--no-cache"])
        finally:
            close()
        self.assertEqual(code, 2, err)
        self.assertIn("answered without any text", err)

    def test_cases_file_single_quoted_apostrophes(self):
        # 'don''t' in a cases file is don't, in the name, the input the
        # model is sent and the assertion text.
        tmp = tempfile.mkdtemp()
        make_suite(tmp)
        with open(os.path.join(tmp, "prompts", "cls.cases.yaml"), "w",
                  encoding="utf-8") as f:
            f.write("cases:\n"
                    "  - name: 'don''t panic'\n"
                    "    input: 'It''s fine, isn''t it?'\n"
                    "    assert:\n"
                    "      - contains: 'it''s'\n")
        case = suite_mod.load_cases(tmp, "cls")[0]
        self.assertEqual(case["name"], "don't panic")
        self.assertEqual(case["input"], "It's fine, isn't it?")
        self.assertEqual(case["assert"], [{"contains": "it's"}])
        server, url = start_fake_model({"isn't it?": "yes, it's fine"})
        try:
            results, _ = runner_mod.run_suite(
                tmp, "cls", "v1", use_cache=False,
                model_kw={"base_url": url, "model": "fake"})
        finally:
            stop_server(server)
        self.assertEqual([r.passed for r in results], [True])
        self.assertEqual(results[0].output, "yes, it's fine")

    def test_judge_reply_without_text_is_a_judge_outage_not_a_crash(self):
        # A judge model answering `content: null` crashed run_case with
        # AttributeError (None.strip()); like any judge outage it is now a
        # failed assertion with the reason, and is never cached.
        tmp = tempfile.mkdtemp()
        make_judge_suite(tmp)
        gen, gen_url = start_fake_model({})
        close, judge_url = start_raw_server(http_reply(
            b'{"choices":[{"message":{"content":null}}]}'))
        try:
            prompt = suite_mod.load_prompt(tmp, "cls", "v1")
            case = suite_mod.load_cases(tmp, "cls")[0]
            r = runner_mod.run_case(
                tmp, "cls", "v1", prompt, case,
                {"base_url": gen_url, "model": "fake"},
                judge_model_kw={"base_url": judge_url, "model": "fake"})
        finally:
            close()
            stop_server(gen)
        self.assertFalse(r.passed)
        self.assertIn("judge unavailable", r.assertions[0].detail)
        self.assertIn("answered without any text", r.assertions[0].detail)
        cached = os.listdir(os.path.join(tmp, ".promptregress", "cache"))
        self.assertEqual([f for f in cached if f.startswith("judge-")], [])


class ClosedPipe(io.StringIO):
    """A stdout whose reader has gone away. `fails` is the call that
    raises: "write" (unbuffered / write-through streams fail inside
    print()) or "flush" (buffered streams fail when flushed). `fd`, when
    given, is the descriptor the stream reports, so a test can look at
    where it points afterwards."""

    def __init__(self, fails="write", error=None, fd=None):
        super().__init__()
        self.fails = fails
        self.error = error or BrokenPipeError(errno.EPIPE, "Broken pipe")
        self.fd = fd

    def fileno(self):
        return super().fileno() if self.fd is None else self.fd

    def write(self, text):
        if self.fails == "write":
            raise self.error
        return super().write(text)

    def flush(self):
        raise self.error


# Run by TestClosedPipe in a child process. fd 1 is rebuilt the way Windows
# has it when nobody reads the pipe: the failing write is a plain
# OSError(EINVAL), not a BrokenPipeError. sys.stdout sits on top of it the
# way Python builds it ("buffered", or write-through like `python -u`), so
# the interpreter's own flush at exit meets the same failing write.
WINDOWS_CHILD = """
import errno, io, os, sys
from promptregress import cli


class WinPipe(io.RawIOBase):
    def writable(self):
        return True

    def fileno(self):
        return 1

    def write(self, data):
        try:
            return os.write(1, data)
        except BrokenPipeError:
            raise OSError(errno.EINVAL, 'Invalid argument') from None


cli._IS_WINDOWS = True
if sys.argv[1] == 'buffered':
    sys.stdout = io.TextIOWrapper(io.BufferedWriter(WinPipe()), 'utf-8')
else:
    sys.stdout = io.TextIOWrapper(WinPipe(), 'utf-8', write_through=True)
code = cli.main(sys.argv[2:])
try:
    os.write(1, b'x')
    print('fd1 redirected: yes', file=sys.stderr)
except OSError:
    print('fd1 redirected: no', file=sys.stderr)
sys.exit(code)
"""


class TestClosedPipe(unittest.TestCase):
    """`promptregress ... | head -1`: the reader quits while we still
    write. No traceback, no "Exception ignored" noise, and the exit code
    stays the command's own (for `run` it is the verdict)."""

    REPO = os.path.dirname(os.path.dirname(os.path.abspath(cli_mod.__file__)))

    def _into_closed_pipe(self, argv, unbuffered, both=False):
        """Run the CLI with stdout (and stderr when `both`) wired to a pipe
        whose reader is already gone. Returns (exit code, stderr text)."""
        env = {**os.environ}
        env.pop("PROMPTREGRESS_BASE_URL", None)
        env.pop("PYTHONUNBUFFERED", None)
        if unbuffered:  # write-through: the error is raised inside print()
            env["PYTHONUNBUFFERED"] = "1"
        r, w = os.pipe()
        os.close(r)
        try:
            p = subprocess.run(
                [sys.executable, "-m", "promptregress", *argv], stdout=w,
                stderr=w if both else subprocess.PIPE, cwd=self.REPO,
                env=env, timeout=120)
        finally:
            os.close(w)
        return p.returncode, (p.stderr or b"").decode("utf-8", "replace")

    def _assert_quiet(self, err, label):
        for noise in ("Traceback", "BrokenPipeError", "Exception ignored",
                      "Errno"):
            self.assertNotIn(noise, err, label)

    def test_closed_stdout_pipe_exits_quietly_with_the_commands_own_code(self):
        tmp = tempfile.mkdtemp()
        # (argv, the exit code the command computes) - never a false pass
        # for the failing run, never Python's 120 / 1 for the rest
        buffered = [
            (["--demo", "run", "--prompt", "sentiment", "--version", "v2"], 1),
            (["--demo", "run", "--prompt", "sentiment", "--version", "v1"], 0),
            (["--demo", "run", "--all", "--format", "json"], 1),
            (["--demo", "diff", "v1", "v2", "--prompt", "sentiment"], 0),
            (["list"], 0),
            (["--version"], 0),
            (["-h"], 0),
        ]
        unbuffered = buffered[:1] + buffered[4:5]
        for mode, cases in ((False, buffered), (True, unbuffered)):
            for argv, want in cases:
                code, err = self._into_closed_pipe(
                    ["--root", tmp, *argv], unbuffered=mode)
                label = (argv, "unbuffered" if mode else "buffered")
                self.assertEqual(code, want, (label, err))
                self._assert_quiet(err, label)

    def test_closed_pipe_on_both_streams_keeps_the_error_code(self):
        # `promptregress ... 2>&1 | head`: even the error line has nowhere
        # to go, and exit code 2 (model unusable) must survive
        tmp = tempfile.mkdtemp()
        make_suite(tmp)
        for unbuffered in (False, True):
            code, _ = self._into_closed_pipe(
                ["--root", tmp, "--base-url", "http://127.0.0.1:1/v1", "run",
                 "--prompt", "cls", "--version", "v1", "--no-cache"],
                unbuffered=unbuffered, both=True)
            self.assertEqual(code, 2, unbuffered)

    def test_in_process_closed_stdout_keeps_the_verdict(self):
        tmp = tempfile.mkdtemp()
        make_suite(tmp)
        server, url = start_fake_model({"terrible": "negative",
                                        "great": "positive"})
        server2, url2 = start_fake_model({})  # answers "ok": cases fail
        argv = ["--root", tmp, "run", "--prompt", "cls", "--version", "v1",
                "--no-cache", "--base-url"]
        try:
            for fails in ("write", "flush"):
                for base, want in ((url, 0), (url2, 1)):
                    code, err = self._run_main([*argv, base],
                                               ClosedPipe(fails))
                    self.assertEqual(code, want, (fails, base))
                    self.assertEqual(err, "", (fails, base))
        finally:
            stop_server(server)
            stop_server(server2)
        # help / version exit through SystemExit(0), not an error
        for flag in ("-h", "--version"):
            with redirect_stdout(ClosedPipe("write")):
                with self.assertRaises(SystemExit) as ctx:
                    cli_mod.main([flag])
            self.assertEqual(ctx.exception.code, 0)

    def test_stdout_is_restored_and_other_write_errors_still_raise(self):
        # only a closed pipe is tolerated: a full disk is still an error
        out = ClosedPipe("write", OSError(errno.ENOSPC, "No space left"))
        with redirect_stdout(out):
            with self.assertRaises(OSError) as ctx:
                cli_mod.main(["--root", tempfile.mkdtemp(), "list"])
        self.assertEqual(ctx.exception.errno, errno.ENOSPC)
        buf = io.StringIO()
        with redirect_stdout(buf):
            cli_mod.main(["--root", tempfile.mkdtemp(), "list"])
            self.assertIs(sys.stdout, buf)  # main() put the stream back
        self.assertIn("no prompts found", buf.getvalue())

    def test_closed_pipe_errno_by_platform(self):
        gone = cli_mod._is_closed_pipe

        class KeepsErrno(OSError):  # OSError(EPIPE, ...) itself comes back
            pass                    # as a BrokenPipeError; this does not

        pipe = (errno.EPIPE, "Broken pipe")
        invalid = OSError(errno.EINVAL, "Invalid argument")
        self.assertEqual(cli_mod._IS_WINDOWS, os.name == "nt")
        for windows in (False, True):
            with mock.patch.object(cli_mod, "_IS_WINDOWS", windows):
                self.assertTrue(gone(BrokenPipeError(*pipe)))
                self.assertTrue(gone(BrokenPipeError()))
                self.assertTrue(gone(KeepsErrno(*pipe)))
                self.assertFalse(gone(OSError(errno.ENOSPC, "No space left")))
                self.assertFalse(gone(ValueError("closed file")))
        # Windows reports a closed pipe as plain EINVAL; elsewhere EINVAL is
        # just an error
        with mock.patch.object(cli_mod, "_IS_WINDOWS", True):
            self.assertTrue(gone(invalid))
        with mock.patch.object(cli_mod, "_IS_WINDOWS", False):
            self.assertFalse(gone(invalid))

    # Windows: a pipe nobody reads is not an EPIPE. print() of a small
    # output works there (it only fills the buffer) and the flush then fails
    # with a plain OSError(EINVAL) - at interpreter exit as "Exception
    # ignored" with exit status 120, or under `python -u` out of print()
    # itself. The tests below reproduce that on any platform: the Windows
    # flag is switched on and main() gets a stdout that fails that way. The
    # real closed-pipe tests above are the proof on a real Windows.

    def _points_at_null_device(self, fd):
        """A closed pipe is a FIFO and refuses writes; the null device is a
        character device and takes them."""
        if not stat.S_ISCHR(os.fstat(fd).st_mode):
            return False
        try:
            os.write(fd, b"x")
        except OSError:
            return False
        return True

    def _dead_pipe_fd(self):
        """What fd 1 is in `cmd | head -1` once head has quit."""
        r, w = os.pipe()
        os.close(r)
        self.addCleanup(os.close, w)
        self.assertFalse(self._points_at_null_device(w))
        return w

    def _run_main(self, argv, stdout):
        """cli.main() with `stdout` in place: (exit code, stderr text). A
        SystemExit (-h, --version) counts as its code; anything else that
        escapes is a test error."""
        err = io.StringIO()
        with warnings.catch_warnings():
            # the garbage collector may report another test's unclosed
            # socket right here, into the redirected stderr
            warnings.simplefilter("ignore", ResourceWarning)
            with redirect_stdout(stdout), redirect_stderr(err):
                try:
                    code = cli_mod.main(argv)
                except SystemExit as e:
                    code = e.code
        return code, err.getvalue()

    def test_windows_closed_pipe_keeps_the_verdict(self):
        tmp = tempfile.mkdtemp()
        make_suite(tmp)
        server, url = start_fake_model({"terrible": "negative",
                                        "great": "positive"})
        server2, url2 = start_fake_model({})  # answers "ok": cases fail
        argv = ["--root", tmp, "run", "--prompt", "cls", "--version", "v1",
                "--no-cache", "--base-url"]
        try:
            for fails in ("flush", "write"):
                for base, want in ((url, 0), (url2, 1)):
                    fd = self._dead_pipe_fd()
                    out = ClosedPipe(
                        fails, OSError(errno.EINVAL, "Invalid argument"), fd)
                    with mock.patch.object(cli_mod, "_IS_WINDOWS", True):
                        code, err = self._run_main([*argv, base], out)
                    label = (fails, want)
                    self.assertEqual(code, want, label)  # never a false 0
                    self.assertEqual(err, "", label)
                    self.assertTrue(self._points_at_null_device(fd), label)
        finally:
            stop_server(server)
            stop_server(server2)

    def test_windows_closed_pipe_with_one_line_of_output(self):
        # `list` prints a single line: only the final flush can notice that
        # the pipe is gone, and it has to happen inside main(), not at exit
        for argv in (["list"], ["-h"], ["--version"]):
            for fails in ("flush", "write"):
                fd = self._dead_pipe_fd()
                out = ClosedPipe(
                    fails, OSError(errno.EINVAL, "Invalid argument"), fd)
                with mock.patch.object(cli_mod, "_IS_WINDOWS", True):
                    code, err = self._run_main(
                        ["--root", tempfile.mkdtemp(), *argv], out)
                label = (argv, fails)
                self.assertEqual(code, 0, label)
                self.assertEqual(err, "", label)
                self.assertTrue(self._points_at_null_device(fd), label)

    def test_einval_is_only_a_closed_pipe_on_windows(self):
        invalid = (errno.EINVAL, "Invalid argument")
        full = (errno.ENOSPC, "No space left on device")
        # (Windows?, error): none of these is a closed pipe, so none of
        # them may be swallowed
        for windows, error in ((False, invalid), (True, full),
                               (False, full)):
            for fails in ("flush", "write"):
                fd = self._dead_pipe_fd()
                out = ClosedPipe(fails, OSError(*error), fd)
                with mock.patch.object(cli_mod, "_IS_WINDOWS", windows):
                    with redirect_stdout(out):
                        with self.assertRaises(OSError) as ctx:
                            cli_mod.main(
                                ["--root", tempfile.mkdtemp(), "list"])
                label = (windows, error[0], fails)
                self.assertEqual(ctx.exception.errno, error[0], label)
                self.assertFalse(self._points_at_null_device(fd), label)

    def test_windows_exit_flush_is_clean(self):
        # The symptom itself, end to end: stdout rebuilt on a raw stream
        # whose failing write is OSError(EINVAL), so the interpreter's own
        # flush at exit would print "Exception ignored" and exit 120.
        tmp = tempfile.mkdtemp()
        env = {**os.environ}
        env.pop("PROMPTREGRESS_BASE_URL", None)
        cases = [
            (["--demo", "run", "--prompt", "sentiment", "--version", "v2"], 1),
            (["--demo", "run", "--prompt", "sentiment", "--version", "v1"], 0),
            (["list"], 0),
        ]
        for mode in ("buffered", "write-through"):
            for argv, want in cases:
                r, w = os.pipe()
                os.close(r)
                try:
                    p = subprocess.run(
                        [sys.executable, "-c", WINDOWS_CHILD, mode,
                         "--root", tmp, *argv],
                        stdout=w, stderr=subprocess.PIPE, cwd=self.REPO,
                        env=env, timeout=120)
                finally:
                    os.close(w)
                err = p.stderr.decode("utf-8", "replace")
                label = (mode, argv, err)
                self.assertEqual(p.returncode, want, label)
                self._assert_quiet(err, label)
                self.assertIn("fd1 redirected: yes", err, label)


if __name__ == "__main__":
    unittest.main()
