"""Runner + CLI tests against the fake model server."""

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest import mock

from promptregress import cli as cli_mod
from promptregress import demo as demo_mod
from promptregress import miniyaml as miniyaml_mod
from promptregress import model as model_mod
from promptregress import runner as runner_mod
from promptregress import suite as suite_mod

from promptregress.demo import start_fake_model


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
        code, out = self._run(tmp, "--demo", "run", "--prompt", "summarize",
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
        tmp = tempfile.mkdtemp()
        os.environ.pop("PROMPTREGRESS_BASE_URL", None)
        code, out = self._run(tmp, "--demo", "diff", "v1", "v2",
                              "--prompt", "summarize")
        self.assertEqual(code, 0)
        self.assertIn("0/3 case(s) changed", out)

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


if __name__ == "__main__":
    unittest.main()
