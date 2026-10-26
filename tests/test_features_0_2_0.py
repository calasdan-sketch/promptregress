"""0.2.0: diff --fail-on-flip, --jobs, JUnit and SARIF reports, the pytest
plugin and the prompt-change guard."""
from __future__ import annotations

import importlib.util
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import xml.etree.ElementTree as ET
from contextlib import redirect_stderr, redirect_stdout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from promptregress import cli as cli_mod
from promptregress import guard as guard_mod
from promptregress import report as report_mod
from promptregress import runner as runner_mod
from promptregress import suite as suite_mod

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HAS_PYTEST = importlib.util.find_spec("pytest") is not None


def installed_plugin() -> bool:
    """Is promptregress installed with its pytest11 entry point?"""
    from importlib import metadata
    try:
        points = metadata.entry_points(group="pytest11")
    except TypeError:    # Python 3.9 style API; not reached on 3.10+
        return False
    return any(ep.value == "promptregress.pytest" for ep in points)


def run_main(*argv, env=None):
    out, err = io.StringIO(), io.StringIO()
    old = {}
    for k, v in (env or {}).items():
        old[k] = os.environ.get(k)
        os.environ[k] = v
    try:
        with redirect_stdout(out), redirect_stderr(err):
            code = cli_mod.main(list(argv))
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return code, out.getvalue(), err.getvalue()


def example_project() -> str:
    """A temp project holding the `promptregress init` example."""
    root = tempfile.mkdtemp()
    cli_mod.scaffold_example(root)
    return root


def write(root, rel, text):
    path = os.path.join(root, *rel.split("/"))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path


class SlowModel:
    """A fake model that answers after `delay` seconds, counts how many
    requests were open at once and can fail on chosen inputs."""

    def __init__(self, delay=0.15, fail_on=()):
        outer = self
        self.active = self.peak = 0
        self.lock = threading.Lock()
        self.fail_on = fail_on

        class H(BaseHTTPRequestHandler):
            def do_POST(self):
                n = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(n).decode()
                with outer.lock:
                    outer.active += 1
                    outer.peak = max(outer.peak, outer.active)
                time.sleep(delay)
                with outer.lock:
                    outer.active -= 1
                bad = any(f in body for f in outer.fail_on)
                payload = json.dumps(
                    {"choices": [{"message": {"content": "ok"}}]}).encode()
                self.send_response(500 if bad else 200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *a):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever,
                         daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_port}/v1"

    def close(self):
        self.server.shutdown()
        self.server.server_close()


def many_cases_project(n=6, name="p"):
    root = tempfile.mkdtemp()
    write(root, f"prompts/{name}.v1.txt", "Say ok to {{input}}\n")
    cases = "".join(
        f'  - name: case {i}\n    input: "item{i}"\n    assert:\n'
        f'      - contains: ok\n' for i in range(n))
    write(root, f"prompts/{name}.cases.yaml", "cases:\n" + cases)
    return root


class TestFailOnFlip(unittest.TestCase):
    def test_exit_1_when_a_pass_became_a_fail(self):
        code, out, err = run_main("--demo", "diff", "v1", "v2", "--prompt",
                                  "sentiment", "--fail-on-flip")
        self.assertEqual(code, 1, err)
        self.assertIn("1 case(s) went from pass to FAIL", out)

    def test_without_the_flag_diff_still_only_reports(self):
        code, out, _ = run_main("--demo", "diff", "v1", "v2", "--prompt",
                                "sentiment")
        self.assertEqual(code, 0)
        self.assertNotIn("--fail-on-flip", out)

    def test_a_fix_is_not_a_regression(self):
        code, out, _ = run_main("--demo", "diff", "v2", "v1", "--prompt",
                                "sentiment", "--fail-on-flip")
        self.assertEqual(code, 0)
        self.assertIn("1/3 case(s) changed verdict", out)
        self.assertIn("no case went from pass to FAIL", out)

    def test_no_change_no_failure(self):
        code, out, _ = run_main("--demo", "diff", "v1", "v1", "--prompt",
                                "sentiment", "--fail-on-flip")
        self.assertEqual(code, 0)

    def test_json_says_which_cases_regressed(self):
        code, out, _ = run_main("--demo", "diff", "v1", "v2", "--prompt",
                                "sentiment", "--fail-on-flip", "--format",
                                "json")
        data = json.loads(out)
        self.assertEqual(code, 1)
        self.assertEqual(data["regressions"], 1)
        self.assertEqual([r["name"] for r in data["cases"] if r["regressed"]],
                         ["neutral review"])

    def test_json_exit_code_is_0_without_the_flag(self):
        code, out, _ = run_main("--demo", "diff", "v1", "v2", "--prompt",
                                "sentiment", "--format", "json")
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out)["regressions"], 1)

    def test_a_usage_error_is_still_2(self):
        code, _, err = run_main("--demo", "diff", "v1", "v9", "--prompt",
                                "sentiment", "--fail-on-flip")
        self.assertEqual(code, 2)


class TestJobs(unittest.TestCase):
    def test_bad_values_are_usage_errors(self):
        for bad in ("0", "-1", "65", "many", "1.5"):
            with self.assertRaises(SystemExit) as ctx, \
                    redirect_stderr(io.StringIO()) as err:
                cli_mod.build_parser().parse_args(
                    ["run", "--all", "--jobs", bad])
            self.assertEqual(ctx.exception.code, 2, bad)
            self.assertIn("between 1 and 64", err.getvalue())

    def test_requests_really_run_at_the_same_time(self):
        model = SlowModel()
        try:
            root = many_cases_project(6)
            t0 = time.perf_counter()
            results, _ = runner_mod.run_suite(
                root, "p", "v1", model_kw={"base_url": model.url},
                use_cache=False, jobs=3)
            took = time.perf_counter() - t0      # noqa: F841 (not asserted: runners vary)
        finally:
            model.close()
        self.assertGreaterEqual(model.peak, 2)
        self.assertLessEqual(model.peak, 3)
        self.assertEqual(len(results), 6)
        # No wall-clock limit (a slow Windows runner took 1.2 s): requests that
        # were seen in flight together (peak >= 2 above) ran at the same time.

    def test_jobs_1_stays_one_at_a_time(self):
        model = SlowModel(delay=0.05)
        try:
            root = many_cases_project(4)
            runner_mod.run_suite(root, "p", "v1",
                                 model_kw={"base_url": model.url},
                                 use_cache=False, jobs=1)
        finally:
            model.close()
        self.assertEqual(model.peak, 1)

    def test_results_come_back_in_case_order(self):
        model = SlowModel(delay=0.02)
        try:
            root = many_cases_project(12)
            results, _ = runner_mod.run_suite(
                root, "p", "v1", model_kw={"base_url": model.url},
                use_cache=False, jobs=8)
        finally:
            model.close()
        self.assertEqual([r.name for r in results],
                         [f"case {i}" for i in range(12)])

    def test_report_is_the_same_with_any_number_of_jobs(self):
        outs = []
        for jobs in ("1", "3", "8"):
            code, out, _ = run_main("--demo", "run", "--all", "--format",
                                    "json", "--jobs", jobs)
            outs.append((code, out))
        self.assertEqual(outs[0], outs[1])
        self.assertEqual(outs[0], outs[2])
        self.assertEqual(outs[0][0], 1)

    def test_first_model_error_in_case_order_ends_the_list(self):
        model = SlowModel(delay=0.05, fail_on=("item2",))
        try:
            root = many_cases_project(6)
            results, _ = runner_mod.run_suite(
                root, "p", "v1", model_kw={"base_url": model.url},
                use_cache=False, jobs=6)
        finally:
            model.close()
        self.assertEqual([r.name for r in results],
                         ["case 0", "case 1", "case 2"])
        self.assertTrue(results[-1].error)
        self.assertFalse(any(r.error for r in results[:-1]))

    def test_cli_stops_on_the_model_error_with_exit_2(self):
        model = SlowModel(delay=0.02, fail_on=("item1",))
        try:
            root = many_cases_project(5)
            code, out, err = run_main(
                "--root", root, "run", "--prompt", "p", "--version", "v1",
                "--no-cache", "--jobs", "4", "--base-url", model.url,
                "--api-key", "k")
        finally:
            model.close()
        self.assertEqual(code, 2)
        self.assertIn("stopped at case 'case 1'", err)

    def test_a_cache_that_cannot_be_written_warns_once(self):
        model = SlowModel(delay=0.02)
        try:
            root = many_cases_project(8)
            open(os.path.join(root, ".promptregress"), "w").close()  # a file
            code, out, err = run_main(
                "--root", root, "run", "--prompt", "p", "--version", "v1",
                "--jobs", "8", "--base-url", model.url, "--api-key", "k")
        finally:
            model.close()
        self.assertEqual(code, 0, err)
        self.assertEqual(err.count("cache disabled"), 1, err)

    def test_the_cache_works_across_threads(self):
        model = SlowModel(delay=0.01)
        try:
            root = many_cases_project(6)
            kw = {"base_url": model.url}
            runner_mod.run_suite(root, "p", "v1", model_kw=kw, jobs=4)
            results, _ = runner_mod.run_suite(root, "p", "v1", model_kw=kw,
                                              jobs=4)
        finally:
            model.close()
        self.assertTrue(all(r.cached for r in results))

    def test_run_suite_refuses_a_silly_job_count(self):
        root = many_cases_project(2)
        for jobs in (0, 65):
            with self.assertRaises(ValueError):
                runner_mod.run_suite(root, "p", "v1", jobs=jobs)

    def test_diff_takes_jobs_too(self):
        code, out, _ = run_main("--demo", "diff", "v1", "v2", "--prompt",
                                "sentiment", "--jobs", "3")
        self.assertEqual(code, 0)
        self.assertIn("neutral review", out)


class TestJUnit(unittest.TestCase):
    def junit(self, *argv):
        code, out, err = run_main("--demo", "run", *argv, "--format", "junit")
        return code, ET.fromstring(out.encode("ascii")), out, err

    def test_failing_suite(self):
        code, root, out, _ = self.junit("--prompt", "sentiment",
                                        "--version", "v2")
        self.assertEqual(code, 1)
        self.assertEqual(root.tag, "testsuites")
        self.assertEqual((root.get("tests"), root.get("failures"),
                          root.get("errors")), ("3", "1", "0"))
        (suite,) = root.findall("testsuite")
        self.assertEqual(suite.get("name"), "sentiment.v2")
        names = [c.get("name") for c in suite.findall("testcase")]
        self.assertEqual(names, ["angry review", "happy review",
                                 "neutral review"])
        (fail,) = suite.iter("failure")
        self.assertIn("max_tokens", fail.get("message"))
        self.assertIn("output:", fail.text)

    def test_passing_suite_exits_0(self):
        code, root, *_ = self.junit("--prompt", "sentiment", "--version",
                                    "v1")
        self.assertEqual(code, 0)
        self.assertEqual(list(root.iter("failure")), [])

    def test_all_gives_one_testsuite_per_version(self):
        code, root, *_ = self.junit("--all")
        self.assertEqual([s.get("name") for s in root.findall("testsuite")],
                         ["sentiment.v1", "sentiment.v2"])
        self.assertEqual(root.get("tests"), "6")

    def test_output_is_ascii_and_has_a_declaration(self):
        _, _, out, _ = self.junit("--prompt", "sentiment", "--version", "v2")
        out.encode("ascii")
        self.assertTrue(out.startswith("<?xml"))

    def test_characters_xml_cannot_hold_are_replaced(self):
        r = runner_mod.CaseResult(
            name="bad \x00 name & <tag>", passed=False,
            output="x\x1b[31m\x00y\ud800z \u00e9 \U0001f600",
            assertions=[runner_mod.AssertionResult(
                {"contains": "a\x01"}, False, "missing \x0b")])
        xml = report_mod.junit_xml([("p", "v1", [r])])
        root = ET.fromstring(xml.encode("ascii"))     # must parse
        case = root.find("testsuite/testcase")
        self.assertEqual(case.get("name"), "bad \ufffd name & <tag>")
        self.assertIn("\u00e9", case.find("failure").text)
        self.assertIn("\U0001f600", case.find("failure").text)

    def test_a_model_error_is_an_error_not_a_failure(self):
        r = runner_mod.CaseResult("c", False, "", [], error="HTTP 500")
        root = ET.fromstring(report_mod.junit_xml([("p", "v1", [r])]).encode())
        self.assertEqual(root.get("errors"), "1")
        self.assertEqual(root.get("failures"), "0")
        self.assertEqual(root.find("testsuite/testcase/error").text,
                         "HTTP 500")

    def test_xml_text_helper(self):
        self.assertEqual(report_mod.xml_text("a\tb\nc\rd"), "a\tb\nc\rd")
        self.assertEqual(report_mod.xml_text("\x00\x08\x0c\x1f"),
                         "\ufffd" * 4)
        self.assertEqual(report_mod.xml_text("\ufffe\uffff"), "\ufffd" * 2)


class TestSarif(unittest.TestCase):
    def sarif(self, *argv):
        code, out, _ = run_main("--demo", "run", *argv, "--format", "sarif")
        return code, json.loads(out)

    def test_log_for_a_failing_run(self):
        code, log = self.sarif("--prompt", "sentiment", "--version", "v2")
        self.assertEqual(code, 1)
        self.assertEqual(log["version"], "2.1.0")
        (run,) = log["runs"]
        self.assertEqual(run["tool"]["driver"]["name"], "promptregress")
        self.assertEqual(run["tool"]["driver"]["version"], cli_mod.VERSION)
        (res,) = run["results"]
        self.assertEqual(res["ruleId"], "promptregress/max_tokens")
        self.assertEqual(res["level"], "error")
        self.assertIn("sentiment.v2", res["message"]["text"])
        self.assertIn("neutral review", res["message"]["text"])
        self.assertEqual(res["properties"]["case"], "neutral review")
        rules = run["tool"]["driver"]["rules"]
        self.assertEqual(rules[res["ruleIndex"]]["id"], res["ruleId"])
        loc = res["locations"][0]["physicalLocation"]
        self.assertEqual(loc["artifactLocation"]["uri"],
                         "prompts/sentiment.cases.yaml")

    def test_passing_run_has_no_results_and_exits_0(self):
        code, log = self.sarif("--prompt", "sentiment", "--version", "v1")
        self.assertEqual(code, 0)
        self.assertEqual(log["runs"][0]["results"], [])

    def test_location_is_the_line_of_the_case(self):
        root = example_project()
        path = os.path.join(root, "prompts", "sentiment.cases.yaml")
        with open(path, encoding="utf-8") as f:
            lines = f.read().splitlines()
        want = next(i for i, ln in enumerate(lines, 1)
                    if "neutral review" in ln)
        self.assertEqual(report_mod._case_line(root, "sentiment",
                                               "neutral review"), want)
        self.assertEqual(report_mod._case_line(root, "sentiment", "nope"), 1)
        self.assertEqual(report_mod._case_line(root, "missing", "x"), 1)

    def test_a_model_error_has_its_own_rule(self):
        r = runner_mod.CaseResult("c", False, "", [], error="HTTP 500")
        log = report_mod.sarif_log([("p", "v1", [r])], "0")
        (res,) = log["runs"][0]["results"]
        self.assertEqual(res["ruleId"], "promptregress/model-error")

    def test_fingerprints_are_stable_and_distinct(self):
        _, a = self.sarif("--all")
        _, b = self.sarif("--all")
        fa = [r["partialFingerprints"] for r in a["runs"][0]["results"]]
        fb = [r["partialFingerprints"] for r in b["runs"][0]["results"]]
        self.assertEqual(fa, fb)
        self.assertEqual(len({x["promptregress/v1"] for x in fa}), len(fa))

    def test_it_is_json_only_on_stdout(self):
        code, out, err = run_main("--demo", "run", "--all", "--format",
                                  "sarif")
        json.loads(out)


class TestGuardLogic(unittest.TestCase):
    def test_prompt_alone_is_flagged(self):
        (p,) = guard_mod.check(["prompts/a.v2.txt"])
        self.assertEqual(p["prompt_file"], "prompts/a.v2.txt")
        self.assertEqual(p["cases_file"], "prompts/a.cases.yaml")

    def test_prompt_with_its_cases_is_fine(self):
        self.assertEqual(guard_mod.check(
            ["prompts/a.v2.txt", "prompts/a.cases.yaml"]), [])

    def test_other_prompts_cases_do_not_count(self):
        self.assertEqual(len(guard_mod.check(
            ["prompts/a.v2.txt", "prompts/b.cases.yaml"])), 1)

    def test_cases_alone_and_unrelated_files_are_fine(self):
        self.assertEqual(guard_mod.check(
            ["prompts/a.cases.yaml", "README.md", "src/x.py",
             "prompts/notes.txt", "other/a.v1.txt", "prompts/sub/a.v1.txt"]),
            [])

    def test_names_with_dots_and_several_versions(self):
        out = guard_mod.check(["prompts/my.app.v10.txt",
                               "prompts/my.app.v2.txt"])
        self.assertEqual([(p["name"], p["version"]) for p in out],
                         [("my.app", "v10"), ("my.app", "v2")])

    def test_paths_are_normalised(self):
        self.assertEqual(guard_mod.split_changed(
            ["./prompts\\a.v1.txt, prompts/a.cases.yaml\n\nREADME.md,"]),
            ["prompts/a.v1.txt", "prompts/a.cases.yaml", "README.md"])
        self.assertEqual(guard_mod.check(guard_mod.split_changed(
            ["prompts\\a.v1.txt,prompts\\a.cases.yaml"])), [])

    def test_cases_exist_callback(self):
        (p,) = guard_mod.check(["prompts/a.v1.txt"], lambda name: False)
        self.assertIs(p["cases_exist"], False)


class TestGuardCli(unittest.TestCase):
    def test_exit_codes(self):
        root = example_project()
        code, out, _ = run_main("--root", root, "guard", "--changed-files",
                                "prompts/sentiment.v2.txt")
        self.assertEqual(code, 1)
        self.assertIn("prompts/sentiment.v2.txt changed, but "
                      "prompts/sentiment.cases.yaml did not", out)
        code, out, _ = run_main(
            "--root", root, "guard", "--changed-files",
            "prompts/sentiment.v2.txt,prompts/sentiment.cases.yaml")
        self.assertEqual(code, 0)
        self.assertIn("ok:", out)

    def test_missing_cases_file_is_said(self):
        root = tempfile.mkdtemp()
        code, out, _ = run_main("--root", root, "guard", "--changed-files",
                                "prompts/new.v1.txt")
        self.assertEqual(code, 1)
        self.assertIn("does not exist yet", out)

    def test_changed_files_can_repeat(self):
        root = example_project()
        code, *_ = run_main("--root", root, "guard", "--changed-files",
                            "prompts/sentiment.v1.txt", "--changed-files",
                            "prompts/sentiment.cases.yaml")
        self.assertEqual(code, 0)

    def test_json(self):
        root = example_project()
        code, out, _ = run_main("--root", root, "guard", "--changed-files",
                                "prompts/sentiment.v1.txt", "--format",
                                "json")
        data = json.loads(out)
        self.assertEqual(code, 1)
        self.assertFalse(data["ok"])
        self.assertEqual(data["problems"][0]["version"], "v1")

    def test_both_sources_is_a_usage_error(self):
        code, _, err = run_main("guard", "--changed-files", "a", "--since",
                                "HEAD")
        self.assertEqual(code, 2)
        self.assertIn("not both", err)

    def test_guard_needs_no_model(self):
        # no endpoint, no key: guard must not look at them
        code, *_ = run_main("--root", example_project(), "guard",
                            "--changed-files", "x", env={
                                "PROMPTREGRESS_API_KEY": "bad\nkey"})
        self.assertEqual(code, 0)


def git(repo, *args):
    env = dict(os.environ, GIT_CONFIG_GLOBAL=os.devnull,
               GIT_CONFIG_NOSYSTEM="1", GIT_TERMINAL_PROMPT="0")
    return subprocess.run(["git", "-C", repo, *args], check=True,
                          capture_output=True, env=env).stdout.decode()


def git_project():
    root = example_project()
    git(root, "init", "-q")
    git(root, "config", "user.email", "t@t.t")
    git(root, "config", "user.name", "t")
    git(root, "config", "commit.gpgsign", "false")
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "start")
    return root


class TestGuardWithGit(unittest.TestCase):
    def setUp(self):
        env = {"GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"}
        self._old = {k: os.environ.get(k) for k in env}
        os.environ.update(env)
        self.addCleanup(self._restore)

    def _restore(self):
        for k, v in self._old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_clean_tree_is_ok(self):
        code, out, _ = run_main("--root", git_project(), "guard")
        self.assertEqual(code, 0, out)

    def test_edited_prompt_alone_fails(self):
        root = git_project()
        write(root, "prompts/sentiment.v2.txt", "New wording {{input}}\n")
        code, out, _ = run_main("--root", root, "guard")
        self.assertEqual(code, 1)
        self.assertIn("sentiment.v2.txt", out)

    def test_edited_prompt_and_cases_pass(self):
        root = git_project()
        write(root, "prompts/sentiment.v2.txt", "New wording {{input}}\n")
        with open(os.path.join(root, "prompts", "sentiment.cases.yaml"),
                  "a", encoding="utf-8") as f:
            f.write('  - name: extra\n    input: "meh"\n    assert:\n'
                    '      - contains: neutral\n')
        code, *_ = run_main("--root", root, "guard")
        self.assertEqual(code, 0)

    def test_a_new_untracked_prompt_counts(self):
        root = git_project()
        write(root, "prompts/other.v1.txt", "x {{input}}\n")
        code, out, _ = run_main("--root", root, "guard")
        self.assertEqual(code, 1)
        self.assertIn("prompts/other.v1.txt", out)

    def test_a_deleted_prompt_is_not_a_change_to_guard(self):
        root = git_project()
        os.remove(os.path.join(root, "prompts", "sentiment.v1.txt"))
        code, *_ = run_main("--root", root, "guard")
        self.assertEqual(code, 0)

    def test_since_a_ref_looks_at_committed_changes(self):
        root = git_project()
        write(root, "prompts/sentiment.v2.txt", "New wording {{input}}\n")
        git(root, "commit", "-qam", "reword")
        self.assertEqual(run_main("--root", root, "guard")[0], 0)
        code, out, _ = run_main("--root", root, "guard", "--since", "HEAD~1")
        self.assertEqual(code, 1)

    def test_a_subfolder_root_sees_paths_relative_to_it(self):
        repo = tempfile.mkdtemp()
        project = os.path.join(repo, "svc")
        os.makedirs(project)
        cli_mod.scaffold_example(project)
        git(repo, "init", "-q")
        git(repo, "config", "user.email", "t@t.t")
        git(repo, "config", "user.name", "t")
        git(repo, "add", "-A")
        git(repo, "commit", "-q", "-m", "start")
        write(project, "prompts/sentiment.v1.txt", "changed {{input}}\n")
        code, out, _ = run_main("--root", project, "guard")
        self.assertEqual(code, 1)
        self.assertIn("prompts/sentiment.v1.txt", out)

    def test_not_a_repo_and_bad_ref_are_exit_2(self):
        code, _, err = run_main("--root", example_project(), "guard")
        self.assertEqual(code, 2)
        self.assertIn("error:", err)
        code, _, err = run_main("--root", git_project(), "guard", "--since",
                                "no-such-ref")
        self.assertEqual(code, 2)
        code, _, err = run_main("--root", git_project(), "guard",
                                "--since=--output=/tmp/x")
        self.assertEqual(code, 2)
        self.assertIn("not a git ref", err)

    def test_unusual_file_names_survive(self):
        root = git_project()
        write(root, "prompts/caf\u00e9 menu.v1.txt", "x {{input}}\n")
        code, out, _ = run_main("--root", root, "guard")
        self.assertEqual(code, 1)
        self.assertIn("caf\u00e9 menu.v1.txt", out)


@unittest.skipUnless(HAS_PYTEST, "needs pytest")
class TestPytestPlugin(unittest.TestCase):
    def pytest(self, root, *args):
        env = {k: v for k, v in os.environ.items()
               if not k.startswith("PROMPTREGRESS_")}
        env["PYTHONPATH"] = REPO + os.pathsep + env.get("PYTHONPATH", "")
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        # Installed, the package registers the plugin itself (entry point),
        # and loading it a second time with -p is an error in pytest.
        load = [] if installed_plugin() else ["-p", "promptregress.pytest"]
        p = subprocess.run(
            [sys.executable, "-m", "pytest", "-p", "no:cacheprovider",
             *load, *args], cwd=root, env=env,
            capture_output=True, text=True, timeout=300)
        return p.returncode, p.stdout + p.stderr

    def test_cases_become_tests(self):
        rc, out = self.pytest(example_project(), "--promptregress",
                              "--promptregress-demo", "-v",
                              "--promptregress-version", "v1")
        self.assertEqual(rc, 0, out)
        self.assertIn("prompts/sentiment.cases.yaml::v1::angry review "
                      "PASSED", out)
        self.assertIn("3 passed", out)

    def test_a_failing_case_fails_pytest_with_the_report(self):
        rc, out = self.pytest(example_project(), "--promptregress",
                              "--promptregress-demo",
                              "--promptregress-version", "v2")
        self.assertEqual(rc, 1, out)
        self.assertIn("1 failed, 2 passed", out)
        self.assertIn("max_tokens", out)
        self.assertIn("neutral review", out)

    def test_default_is_the_latest_version_and_all_is_all(self):
        root = example_project()
        rc, out = self.pytest(root, "--promptregress", "--promptregress-demo")
        self.assertIn("1 failed, 2 passed", out)      # v2 only
        rc, out = self.pytest(root, "--promptregress", "--promptregress-demo",
                              "--promptregress-version", "all")
        self.assertIn("1 failed, 5 passed", out)

    def test_nothing_runs_without_the_flag(self):
        rc, out = self.pytest(example_project())
        self.assertEqual(rc, 5, out)                   # no tests collected
        self.assertIn("no tests ran", out)

    def test_demo_runs_leave_no_cache_behind(self):
        root = example_project()
        self.pytest(root, "--promptregress", "--promptregress-demo")
        self.assertFalse(os.path.exists(os.path.join(root, ".promptregress")))

    def test_a_dead_model_skips_the_rest_after_the_first_error(self):
        root = example_project()
        rc, out = self.pytest(
            root, "--promptregress", "--promptregress-base-url",
            "http://127.0.0.1:9/v1", "--promptregress-timeout", "5",
            "--promptregress-version", "v1")
        self.assertEqual(rc, 1, out)
        self.assertIn("1 failed, 2 skipped", out)
        self.assertIn("the model could not answer", out)

    def test_a_broken_cases_file_is_a_collection_error(self):
        root = example_project()
        write(root, "prompts/sentiment.cases.yaml", "cases: []\n")
        rc, out = self.pytest(root, "--promptregress", "--promptregress-demo")
        self.assertEqual(rc, 2, out)           # "Interrupted: 1 error"
        self.assertIn("no cases", out)
        self.assertNotIn("Traceback", out)

    def test_other_yaml_files_are_ignored(self):
        root = example_project()
        write(root, "ci/other.cases.yaml", "not: cases\n")
        rc, out = self.pytest(root, "--promptregress", "--promptregress-demo",
                              "--promptregress-version", "v1")
        self.assertEqual(rc, 0, out)


class TestPytestOptionalImport(unittest.TestCase):
    def test_importing_the_plugin_without_pytest_says_what_to_install(self):
        code = ("import sys; sys.modules['pytest'] = None\n"
                "try:\n    import promptregress.pytest\n"
                "except ImportError as e:\n    print(e)\n")
        p = subprocess.run([sys.executable, "-c", code], cwd=REPO,
                           capture_output=True, text=True, timeout=60)
        self.assertIn("needs pytest", p.stdout)

    def test_the_rest_of_the_package_never_imports_pytest(self):
        code = ("import sys; sys.modules['pytest'] = None\n"
                "import promptregress.cli, promptregress.runner, "
                "promptregress.guard, promptregress.report\nprint('fine')\n")
        p = subprocess.run([sys.executable, "-c", code], cwd=REPO,
                           capture_output=True, text=True, timeout=60)
        self.assertEqual(p.stdout.strip(), "fine", p.stderr)

    def test_pyproject_declares_the_entry_point_and_an_optional_extra(self):
        with open(os.path.join(REPO, "pyproject.toml"), encoding="utf-8") as f:
            text = f.read()
        self.assertIn('[project.entry-points.pytest11]\n'
                      'promptregress = "promptregress.pytest"', text)
        self.assertRegex(text, r'(?m)^pytest = \["pytest>=7\.0"\]')
        deps = re.search(r"(?m)^dependencies\s*=", text)
        self.assertIsNone(deps)    # still stdlib-only


class TestPackagingAndDocs(unittest.TestCase):
    def read(self, name):
        with open(os.path.join(REPO, name), encoding="utf-8") as f:
            return f.read()

    def test_version_is_everywhere(self):
        self.assertEqual(cli_mod.VERSION, "0.2.0")
        self.assertIn('version = "0.2.0"', self.read("pyproject.toml"))
        self.assertIn("## 0.2.0", self.read("CHANGELOG.md"))

    def test_readme_documents_every_new_thing(self):
        text = self.read("README.md")
        for needle in ("--fail-on-flip", "--jobs", "--format junit",
                       "--format sarif", "pytest --promptregress",
                       "promptregress guard", "--changed-files"):
            self.assertIn(needle, text)
        self.assertNotIn("there is no parallel mode", text)

    def test_cli_help_lists_the_new_options(self):
        helps = {}
        for cmd in ("run", "diff", "guard"):
            with self.assertRaises(SystemExit), \
                    redirect_stdout(io.StringIO()) as out:
                cli_mod.build_parser().parse_args([cmd, "--help"])
            helps[cmd] = out.getvalue()
        self.assertIn("--jobs", helps["run"])
        self.assertIn("junit", helps["run"])
        self.assertIn("--fail-on-flip", helps["diff"])
        self.assertIn("--changed-files", helps["guard"])


if __name__ == "__main__":
    unittest.main()
