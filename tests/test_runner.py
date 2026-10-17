"""Runner + CLI tests against the fake model server."""

import glob
import io
import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
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
            server.shutdown()

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
            server.shutdown()


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
            server.shutdown()
            results2, _ = runner_mod.run_suite(tmp, "cls", "v1", model_kw=kw)
            self.assertTrue(all(r.cached for r in results2))
            self.assertTrue(all(r.passed for r in results2))
        finally:
            try:
                server.shutdown()
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
            server.shutdown()
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
            s1.shutdown()
            s2.shutdown()

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
            server.shutdown()


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
            server.shutdown()

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
            server.shutdown()
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
            server.shutdown()

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
            server.shutdown()

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
            server.shutdown()

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
            server.shutdown()

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
            server.shutdown()
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
            server2.shutdown()

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
            server.shutdown()

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
            server.shutdown()

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
            server.shutdown()
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
            server.shutdown()
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
            server.shutdown()
        self.assertEqual(code, 2)
        self.assertIn("HTTP 401 Unauthorized: nope", err)
        self.assertNotIn("CHANGED", out)

    def test_demo_scaffold_dir_is_removed(self):
        # --demo scaffolded into tempfile.mkdtemp() and never cleaned up:
        # one promptregress-demo-* dir per trial run piled up in /tmp.
        pattern = os.path.join(tempfile.gettempdir(), "promptregress-demo-*")
        before = set(glob.glob(pattern))
        tmp = tempfile.mkdtemp()
        os.environ.pop("PROMPTREGRESS_BASE_URL", None)
        code, out, err = self._run_capture(
            ["--root", tmp, "--demo", "diff", "v1", "v2",
             "--prompt", "sentiment"])
        self.assertEqual(code, 0, err)
        self.assertIn("scaffolded example prompts in", err)
        self.assertEqual(set(glob.glob(pattern)) - before, set())
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
            server.shutdown()
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
        # `regex: (` raised re.error. All must be reported per case.
        tmp = tempfile.mkdtemp()
        d = os.path.join(tmp, "prompts")
        os.makedirs(d)
        with open(os.path.join(d, "yn.v1.txt"), "w") as f:
            f.write("Answer: {{input}}")
        with open(os.path.join(d, "yn.cases.yaml"), "w") as f:
            f.write("""cases:
  - name: bool
    input: "a"
    assert:
      - contains: yes
  - name: int
    input: "b"
    assert:
      - contains: 1
  - name: null
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
        try:
            code, out, err = self._run_capture(
                ["--root", tmp, "--base-url", url, "run", "--prompt", "yn",
                 "--version", "v1", "--no-cache"])
        finally:
            server.shutdown()
        self.assertEqual(code, 1)
        self.assertNotIn("Traceback", err)
        self.assertIn("yn.v1: F.F.F", out)
        self.assertIn("bad assertion: contains: True is a YAML boolean", out)
        self.assertIn('quote the text you mean, e.g. contains: "yes"', out)
        self.assertIn("bad assertion: contains: needs text, got null", out)
        self.assertIn("bad assertion: regex: invalid regex /(/", out)
        self.assertIn("2/5 passed", out)  # int coerced, "10" coerced

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
            server.shutdown()
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
            server.shutdown()
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
            server.shutdown()
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


if __name__ == "__main__":
    unittest.main()
