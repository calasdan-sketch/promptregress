"""Regression tests for 0.1.4: base URL checks, unwritable stderr, assertion
semantics, long YAML lines, packaging.

Fake servers on 127.0.0.1 and temporary directories only.
"""

import contextlib
import io
import json
import os
import re
import shutil
import sys
import tempfile
import threading
import unittest
from contextlib import redirect_stderr, redirect_stdout
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest import mock

import promptregress
from promptregress import asserts, miniyaml
from promptregress import cli as cli_mod
from promptregress import model as model_mod

try:
    import tomllib
except ImportError:  # Python 3.10
    tomllib = None

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read(*parts):
    with open(os.path.join(ROOT, *parts), encoding="utf-8") as f:
        return f.read()


@contextlib.contextmanager
def clean_env(**extra):
    env = {k: v for k, v in os.environ.items()
           if not k.startswith("PROMPTREGRESS_")}
    env.update(extra)
    with mock.patch.dict(os.environ, env, clear=True):
        yield


def run_cli(argv, **env):
    out, err = io.StringIO(), io.StringIO()
    with clean_env(**env), redirect_stdout(out), redirect_stderr(err):
        code = cli_mod.main(argv)
    return code, out.getvalue(), err.getvalue()


class _Answer(BaseHTTPRequestHandler):
    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        payload = json.dumps({"choices": [{"message": {
            "content": "neutral"}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *a):
        pass


class Project(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="prg-r8-")
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        d = os.path.join(self.root, "prompts")
        os.makedirs(d)
        with open(os.path.join(d, "s.v1.txt"), "w") as f:
            f.write("Classify: {{input}}\n")
        with open(os.path.join(d, "s.cases.yaml"), "w") as f:
            f.write('cases:\n  - name: one\n    input: "x"\n'
                    '    assert:\n      - contains: "neutral"\n')

    def run_suite(self, url, **env):
        return run_cli(["--root", self.root, "--base-url", url, "run",
                        "--prompt", "s", "--version", "v1", "--no-cache"],
                       **env)

    def model(self):
        server = HTTPServer(("127.0.0.1", 0), _Answer)
        threading.Thread(target=server.serve_forever,
                         kwargs={"poll_interval": 0.02}, daemon=True).start()

        def stop():
            server.shutdown()
            server.server_close()
        self.addCleanup(stop)
        return f"http://127.0.0.1:{server.server_port}/v1"


class TestBaseUrlIsHttpOnly(Project):
    def test_data_file_and_ftp_urls_are_refused_with_exit_2(self):
        for url in ("data:,", "file:///dev/shm", "ftp://127.0.0.1/x",
                    "localhost:11434/v1", "http://", "//host/v1", "x"):
            with self.subTest(url=url):
                code, out, err = self.run_suite(url)
                self.assertEqual(code, 2, (out, err))
                self.assertIn("must be an http:// or https:// address", err)
                self.assertNotIn("Traceback", err)

    def test_the_environment_variable_is_checked_too(self):
        code, out, err = self.run_suite(
            "", PROMPTREGRESS_BASE_URL="file:///etc")
        self.assertEqual(code, 2, (out, err))
        self.assertIn("must be an http:// or https:// address", err)

    def test_complete_raises_a_model_error_not_an_attribute_error(self):
        for url in ("data:,", "file:///dev/shm", "ftp://h/x"):
            with self.subTest(url=url), clean_env():
                with self.assertRaises(model_mod.ModelError):
                    model_mod.complete("hi", base_url=url)

    def test_the_opener_has_no_file_ftp_or_data_handler(self):
        seen = []
        real = model_mod.urllib.request.OpenerDirector.add_handler

        def spy(self, handler):
            seen.append(type(handler).__name__)
            return real(self, handler)
        server = self.model()
        with mock.patch.object(model_mod.urllib.request.OpenerDirector,
                               "add_handler", spy), clean_env():
            self.assertEqual(model_mod.complete("hi", base_url=server),
                             "neutral")
        for bad in ("FileHandler", "FTPHandler", "DataHandler"):
            self.assertNotIn(bad, seen)
        self.assertIn("_HTTP", seen)

    def test_http_and_https_addresses_still_work(self):
        url = self.model()
        code, out, err = self.run_suite(url)
        self.assertEqual(code, 0, (out, err))
        with clean_env():
            self.assertEqual(model_mod.check_base_url("https://api.x.y/v1"),
                             "https://api.x.y/v1")


class TestCredentialsInTheBaseUrl(Project):
    def test_a_url_with_a_password_is_refused_and_never_echoed(self):
        for url in ("http://user:SECRETPW@127.0.0.1:1/v1",
                    "http://SECRETPW@127.0.0.1:1/v1",
                    "http://:SECRETPW@127.0.0.1:1/v1"):
            with self.subTest(url=url):
                code, out, err = self.run_suite(url)
                self.assertEqual(code, 2, (out, err))
                self.assertIn("must not carry a user name or password", err)
                self.assertIn("PROMPTREGRESS_API_KEY", err)
                self.assertNotIn("SECRETPW", out + err)

    def test_a_malformed_url_with_a_password_does_not_leak_either(self):
        code, out, err = self.run_suite("ftp://u:SECRETPW@h/x")
        self.assertEqual(code, 2)
        self.assertNotIn("SECRETPW", out + err)

    def test_the_demo_banner_does_not_repeat_the_password(self):
        # demo mode ignores --base-url and says so
        code, out, err = run_cli(
            ["--demo", "--base-url", "http://u:SECRETPW@h/v1", "run",
             "--prompt", "sentiment", "--version", "v1"])
        self.assertNotIn("SECRETPW", out + err)


class _Full(io.StringIO):
    def write(self, text):
        raise OSError(28, "No space left on device")


class TestUnwritableStderr(Project):
    def test_an_unreachable_model_with_a_dead_stderr_exits_2_not_1(self):
        out = io.StringIO()
        with clean_env(), redirect_stdout(out), \
                mock.patch.object(sys, "stderr", _Full()):
            code = cli_mod.main(["--root", self.root, "--base-url",
                                 "http://127.0.0.1:1/v1", "run", "--all",
                                 "--no-cache"])
        self.assertEqual(code, 2)

    @unittest.skipUnless(os.path.exists("/dev/full"), "needs /dev/full")
    def test_a_real_dev_full_stderr_exits_2(self):
        import subprocess
        env = {k: v for k, v in os.environ.items()
               if not k.startswith("PROMPTREGRESS_")}
        env["PYTHONPATH"] = ROOT
        with open("/dev/full", "w") as full:
            p = subprocess.run(
                [sys.executable, "-m", "promptregress", "--root", self.root,
                 "--base-url", "http://127.0.0.1:1/v1", "run", "--all",
                 "--no-cache"], stderr=full, stdout=subprocess.PIPE,
                env=env, cwd=tempfile.gettempdir())
        self.assertEqual(p.returncode, 2)


class TestJsonAssertions(unittest.TestCase):
    def test_nan_and_infinity_are_not_json(self):
        for text in ("NaN", "Infinity", "-Infinity", '{"a": NaN}', "[1, Infinity]"):
            with self.subTest(text=text):
                ok, detail = asserts.evaluate({"json_valid": True}, text)
                self.assertFalse(ok, detail)
                ok, _ = asserts.evaluate({"json_valid": False}, text)
                self.assertTrue(ok)
                ok, _ = asserts.evaluate({"schema": {"type": "number"}}, text)
                self.assertFalse(ok)

    def test_ordinary_json_still_parses(self):
        for text in ("1", "1.5", '{"a": [1, 2e3, null, true]}', '"x"'):
            with self.subTest(text=text):
                self.assertTrue(
                    asserts.evaluate({"json_valid": True}, text)[0])

    def test_an_integer_valued_float_is_an_integer(self):
        schema = {"type": "integer"}
        for text, want in (("1.0", True), ("2", True), ("1e2", True),
                           ("1.5", False), ("true", False), ('"1"', False),
                           ("1e400", False)):
            with self.subTest(text=text):
                self.assertEqual(
                    asserts.evaluate({"schema": schema}, text)[0], want)

    def test_the_integer_property_in_an_object(self):
        schema = {"type": "object", "properties": {"n": {"type": "integer"}}}
        self.assertTrue(asserts.evaluate({"schema": schema}, '{"n": 3.0}')[0])
        self.assertFalse(asserts.evaluate({"schema": schema}, '{"n": 3.5}')[0])

    def test_the_readme_says_what_the_checks_are(self):
        readme = read("README.md")
        for word in ("NaN", "1.0", "no time", "`$`", "duplicate key"):
            self.assertIn(word, readme)


class TestLongYamlLines(unittest.TestCase):
    def test_a_huge_bad_line_is_cut_in_the_error(self):
        line = "input: [[[[" + "[" * 5000
        with self.assertRaises(ValueError) as cm:
            miniyaml.safe_load("cases:\n  - name: a\n    " + line + "\n")
        message = str(cm.exception)
        self.assertLess(len(message), 400, len(message))
        self.assertIn("characters in all", message)
        self.assertIn("(line 3)", message)

    def test_short_messages_are_unchanged(self):
        with self.assertRaises(ValueError) as cm:
            miniyaml.safe_load("a: &x 1\n")
        self.assertEqual(str(cm.exception),
                         "miniyaml: YAML anchors (&) are not supported "
                         "(line 1)")


class TestPackaging(unittest.TestCase):
    def test_version_agrees_everywhere(self):
        m = re.search(r'(?m)^version\s*=\s*"([^"]+)"', read("pyproject.toml"))
        self.assertEqual(m.group(1), cli_mod.VERSION)
        self.assertIn("## " + cli_mod.VERSION, read("CHANGELOG.md"))

    @unittest.skipIf(tomllib is None, "tomllib needs Python 3.11+")
    def test_metadata_has_authors_and_the_newest_python(self):
        with open(os.path.join(ROOT, "pyproject.toml"), "rb") as f:
            project = tomllib.load(f)["project"]
        self.assertEqual(project["authors"],
                         [{"name": "Calas Automations Inc."}])
        self.assertIn("Programming Language :: Python :: 3.14",
                      project["classifiers"])

    def test_workflows_are_there_and_least_privilege(self):
        ci = read(".github", "workflows", "ci.yml")
        self.assertRegex(ci, r"(?m)^permissions:\n  contents: read\n")
        self.assertEqual(ci.count("permissions:"), 1)
        for word in ("ubuntu-latest", "windows-latest", "macos-latest",
                     "twine check --strict", "unittest discover"):
            self.assertIn(word, ci)
        release = read(".github", "workflows", "release.yml")
        self.assertIn("pypa/gh-action-pypi-publish@release/v1", release)
        self.assertIn("environment: pypi", release)
        self.assertEqual(release.count("id-token: write"), 1)
        self.assertIn("tags: [\"v*\"]", release)
        # id-token is granted to the publish job only
        self.assertGreater(release.index("id-token: write"),
                           release.index("publish:"))

    def test_the_readme_installs_from_pypi(self):
        readme = read("README.md")
        self.assertIn("pip install promptregress\n", readme)
        self.assertIn("From source", readme)

    def test_the_manifest_ships_the_changelog(self):
        self.assertIn("include CHANGELOG.md", read("MANIFEST.in"))


if __name__ == "__main__":
    unittest.main()
