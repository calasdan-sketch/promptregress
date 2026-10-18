"""Round-6 panel regressions (second verification panel, 2026-10-18).

promptregress: a cache write or a scaffold write that failed (a read-only or
root-owned workspace, a file where a folder belongs) ended in a traceback with
exit code 1 - the code the README keeps for "a case failed". Every test here
uses fake servers on 127.0.0.1 and temp directories, nothing else.

The same panel round: a cases file in PyYAML's layout (a list at the indent
of its key) and a quoted key were refused without a word on what to change,
and a schema `enum` took `true` for the number 1.
"""

import builtins
import contextlib
import errno
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest import mock

from promptregress import asserts as asserts_mod
from promptregress import cli as cli_mod
from promptregress import miniyaml
from promptregress import runner as runner_mod
from promptregress import suite as suite_mod
from promptregress.demo import start_fake_model

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROMPT = "Classify: {{input}}\n"
ONE_CASE = ("cases:\n  - name: a\n    input: hi\n    assert:\n"
            "      - contains: neutral\n")
THREE_CASES = "cases:\n" + "".join(
    f"  - name: c{i}\n    input: hi {i}\n    assert:\n"
    f"      - contains: neutral\n" for i in range(3))
JUDGED = ("cases:\n  - name: j\n    input: hi\n    assert:\n"
          "      - contains: PASS\n      - judge: The reply must say PASS\n")


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


def refusing_open(*names, error=errno.EACCES, text="Permission denied"):
    """A replacement for open() that cannot open files called `names`."""
    real = builtins.open

    def fake(path, *args, **kw):
        if os.path.basename(str(path)) in names:
            raise OSError(error, text, str(path))
        return real(path, *args, **kw)
    return fake


def full_disk_open():
    """A replacement for open() whose writes of cache entries run out of
    space."""
    real = builtins.open

    def fake(path, mode="r", *args, **kw):
        if "w" in mode and str(path).endswith(".json"):
            raise OSError(errno.ENOSPC, "No space left on device", str(path))
        return real(path, mode, *args, **kw)
    return fake


class WorkDir(unittest.TestCase):
    """Temp projects that are removed again, with the model they talk to."""

    def project(self, cases=ONE_CASE, versions=("v1",), name="s"):
        root = tempfile.mkdtemp(prefix="prg-r6-")
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        d = os.path.join(root, "prompts")
        os.makedirs(d)
        for v in versions:
            self.write(os.path.join(d, f"{name}.{v}.txt"), PROMPT)
        self.write(os.path.join(d, f"{name}.cases.yaml"), cases)
        return root

    def bare_dir(self):
        root = tempfile.mkdtemp(prefix="prg-r6-")
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        return root

    @staticmethod
    def write(path, text=""):
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)

    def model(self, answer="neutral"):
        server, url = start_fake_model({}, default=answer)

        def stop():
            server.shutdown()
            server.server_close()
        self.addCleanup(stop)
        return url

    def block_cache_dir(self, root):
        """A FILE where .promptregress/cache belongs: creating the folder
        fails on every platform, for root as well."""
        os.makedirs(os.path.join(root, ".promptregress"))
        self.write(os.path.join(root, ".promptregress", "cache"))

    @staticmethod
    def tidy(text):
        return text.replace("\\", "/")


def lines_of(text):
    return [ln for ln in text.splitlines() if ln.strip()]


# ---------------------------------------------------------------------
# #1 - a cache is an optimisation: it must never fail or crash a run

class TestACacheThatCannotBeWrittenIsSkipped(WorkDir):
    """A file at `.promptregress/cache` (or a read-only or root-owned
    `.promptregress`) made `run` and `diff` end in a traceback with exit
    code 1, the code for "a case failed"."""

    def run_in(self, root, *argv):
        return run_cli(["--root", root, "--base-url", self.url, *argv])

    def setUp(self):
        self.url = self.model()

    def assert_one_warning(self, err, *mentions):
        lines = lines_of(err)
        self.assertEqual(len(lines), 1, f"want one warning line: {err!r}")
        self.assertTrue(lines[0].startswith("warning: cache disabled: "), err)
        self.assertNotIn("Traceback", err)
        for text in mentions:
            self.assertIn(text, self.tidy(lines[0]))

    def test_run_carries_on_and_warns_once(self):
        root = self.project(THREE_CASES)
        self.block_cache_dir(root)
        code, out, err = self.run_in(root, "run", "--prompt", "s",
                                     "--version", "v1")
        self.assertEqual(code, 0, (out, err))
        self.assertIn("3/3 passed", out)
        self.assert_one_warning(err, ".promptregress/cache")

    def test_the_answers_are_the_same_as_without_trouble(self):
        # a cache that is switched off changes what is paid, never the verdict
        clean = self.project(THREE_CASES)
        broken = self.project(THREE_CASES)
        self.block_cache_dir(broken)
        argv = ["run", "--prompt", "s", "--version", "v1", "--format", "json"]
        code1, out1, err1 = self.run_in(clean, *argv)
        code2, out2, err2 = self.run_in(broken, *argv)
        self.assertEqual((code1, code2), (0, 0))
        self.assertEqual(json.loads(out1), json.loads(out2))
        self.assertEqual(err1, "")

    def test_a_failing_run_still_exits_1_with_a_broken_cache(self):
        # the exit code is the verdict: 1 for a failed case, and only that
        root = self.project(ONE_CASE.replace("neutral", "positive"))
        self.block_cache_dir(root)
        code, out, err = self.run_in(root, "run", "--prompt", "s",
                                     "--version", "v1")
        self.assertEqual(code, 1, (out, err))
        self.assertIn("0/1 passed", out)
        self.assert_one_warning(err)

    def test_json_goes_to_stdout_alone_the_warning_to_stderr(self):
        root = self.project()
        self.block_cache_dir(root)
        code, out, err = self.run_in(root, "run", "--prompt", "s",
                                     "--version", "v1", "--format", "json")
        self.assertEqual(code, 0, (out, err))
        self.assertEqual(json.loads(out)["passed"], 1)
        self.assert_one_warning(err)

    def test_diff_carries_on_and_warns_once_for_both_versions(self):
        root = self.project(versions=("v1", "v2"))
        self.block_cache_dir(root)
        code, out, err = self.run_in(root, "diff", "v1", "v2",
                                     "--prompt", "s")
        self.assertEqual(code, 0, (out, err))
        self.assertIn("0/1 case(s) changed verdict v1 -> v2", out)
        self.assert_one_warning(err)

    def test_run_all_warns_once_for_every_suite(self):
        root = self.project(THREE_CASES, versions=("v1", "v2"))
        self.block_cache_dir(root)
        code, out, err = self.run_in(root, "run", "--all")
        self.assertEqual(code, 0, (out, err))
        self.assertIn("2 suite(s), 0 with failures; 6/6 cases passed", out)
        self.assert_one_warning(err)

    def test_a_judge_verdict_that_cannot_be_cached_is_still_a_verdict(self):
        self.url = self.model("PASS")
        root = self.project(JUDGED)
        self.block_cache_dir(root)
        code, out, err = self.run_in(root, "run", "--prompt", "s",
                                     "--version", "v1")
        self.assertEqual(code, 0, (out, err))
        self.assertIn("1/1 passed", out)
        self.assert_one_warning(err)

    def test_no_cache_never_touches_it_and_says_nothing(self):
        root = self.project()
        self.block_cache_dir(root)
        code, out, err = self.run_in(root, "run", "--prompt", "s",
                                     "--version", "v1", "--no-cache")
        self.assertEqual(code, 0, (out, err))
        self.assertEqual(err, "")

    def test_a_file_where_the_whole_cache_folder_belongs(self):
        root = self.project()
        self.write(os.path.join(root, ".promptregress"))     # a FILE
        code, out, err = self.run_in(root, "run", "--prompt", "s",
                                     "--version", "v1")
        self.assertEqual(code, 0, (out, err))
        self.assert_one_warning(err, ".promptregress")

    def test_a_write_that_is_refused_is_skipped(self):
        # the portable form of a read-only or root-owned .promptregress: the
        # function that creates the folder fails the way the system words it
        root = self.project()
        refuse = PermissionError(errno.EACCES, "Permission denied",
                                 os.path.join(root, ".promptregress"))
        with mock.patch.object(runner_mod.os, "makedirs", side_effect=refuse):
            code, out, err = self.run_in(root, "run", "--prompt", "s",
                                         "--version", "v1")
        self.assertEqual(code, 0, (out, err))
        self.assert_one_warning(err, "Permission denied", ".promptregress")

    def test_a_full_disk_is_skipped(self):
        root = self.project(THREE_CASES)
        with mock.patch.object(runner_mod, "open", full_disk_open(),
                               create=True):
            code, out, err = self.run_in(root, "run", "--prompt", "s",
                                         "--version", "v1")
        self.assertEqual(code, 0, (out, err))
        self.assertIn("3/3 passed", out)
        self.assert_one_warning(err, "No space left on device")

    def test_no_entry_is_cached_without_its_gitignore(self):
        # the .gitignore keeps model output out of commits; when it cannot
        # be written nothing is cached at all
        root = self.project(THREE_CASES)
        refuse = PermissionError(errno.EACCES, "Permission denied",
                                 os.path.join(root, ".promptregress",
                                              ".gitignore"))
        with mock.patch.object(runner_mod, "ensure_cache_ignored",
                               side_effect=refuse):
            code, out, err = self.run_in(root, "run", "--prompt", "s",
                                         "--version", "v1")
        self.assertEqual(code, 0, (out, err))
        self.assert_one_warning(err, "Permission denied", ".gitignore")
        found = [f for _, _, files in os.walk(root) for f in files
                 if f.endswith(".json")]
        self.assertEqual(found, [])

    @unittest.skipIf(os.name == "nt", "POSIX permission bits")
    @unittest.skipIf(hasattr(os, "geteuid") and os.geteuid() == 0,
                     "root ignores permission bits")
    def test_a_read_only_workspace(self):
        root = self.project()
        folder = os.path.join(root, ".promptregress")
        os.makedirs(folder)
        os.chmod(folder, 0o555)
        self.addCleanup(os.chmod, folder, 0o755)
        code, out, err = self.run_in(root, "run", "--prompt", "s",
                                     "--version", "v1")
        self.assertEqual(code, 0, (out, err))
        self.assert_one_warning(err, "Permission denied")

    def test_the_write_function_never_raises_and_warns_once(self):
        runner_mod.reset_cache_warnings()
        root = self.bare_dir()
        self.block_cache_dir(root)
        err = io.StringIO()
        with redirect_stderr(err):
            first = runner_mod._cache_write(root, "k1", {"output": "o"})
            second = runner_mod._cache_write(root, "k2", {"output": "o"})
        self.assertEqual((first, second), (False, False))
        self.assert_one_warning(err.getvalue())
        self.assertIsNone(runner_mod._cache_read(root, "k1"))

    def test_a_cache_that_works_is_still_written_and_read(self):
        runner_mod.reset_cache_warnings()
        root = self.bare_dir()
        err = io.StringIO()
        with redirect_stderr(err):
            self.assertTrue(runner_mod._cache_write(root, "k", {"output": "o"}))
        self.assertEqual(err.getvalue(), "")
        self.assertEqual(runner_mod._cache_read(root, "k"), {"output": "o"})

    def test_every_command_warns_again(self):
        # "once" means once per command, not once per process
        root = self.project()
        self.block_cache_dir(root)
        for _ in range(2):
            code, out, err = self.run_in(root, "run", "--prompt", "s",
                                         "--version", "v1")
            self.assertEqual(code, 0, (out, err))
            self.assert_one_warning(err)

    def test_the_real_command_line_does_the_same(self):
        # a real process, as the panel ran it, with output buffering as on
        # a user's machine
        root = self.project(THREE_CASES)
        self.block_cache_dir(root)
        env = {k: v for k, v in os.environ.items()
               if not k.startswith("PROMPTREGRESS_")
               and k != "PYTHONUNBUFFERED"}
        proc = subprocess.run(
            [sys.executable, "-m", "promptregress", "--root", root,
             "--base-url", self.url, "run", "--prompt", "s", "--version",
             "v1"], cwd=REPO, env=env, capture_output=True, timeout=120)
        out = proc.stdout.decode("utf-8", "replace")
        err = proc.stderr.decode("utf-8", "replace")
        self.assertEqual(proc.returncode, 0, (out, err))
        self.assertIn("3/3 passed", out)
        self.assert_one_warning(err)


class TestInitSaysWhatItCouldNotWrite(WorkDir):
    """`init` is asked to write files: when it cannot, that is a real
    failure - exit 2 and one `error:` line, never a traceback."""

    def one_error(self, err, *mentions):
        lines = lines_of(err)
        self.assertEqual(len(lines), 1, f"want one error line: {err!r}")
        self.assertTrue(lines[0].startswith("error: "), err)
        self.assertIn("init could not write its files: ", lines[0])
        self.assertNotIn("Traceback", err)
        for text in mentions:
            self.assertIn(text, self.tidy(lines[0]))

    def test_a_file_where_prompts_belongs(self):
        root = self.bare_dir()
        self.write(os.path.join(root, "prompts"))
        code, out, err = run_cli(["--root", root, "init"])
        self.assertEqual(code, 2, (out, err))
        self.one_error(err, "prompts")
        self.assertEqual(out, "")

    def test_a_file_where_the_cache_folder_belongs(self):
        root = self.bare_dir()
        self.write(os.path.join(root, ".promptregress"))
        code, out, err = run_cli(["--root", root, "init"])
        self.assertEqual(code, 2, (out, err))
        self.one_error(err, ".promptregress")
        # what could be written was written; a second init finishes the job
        self.assertIn("scaffolded prompts/", out)

    def test_a_refused_write(self):
        root = self.bare_dir()
        refuse = PermissionError(errno.EACCES, "Permission denied",
                                 os.path.join(root, "prompts"))
        with mock.patch.object(cli_mod.os, "makedirs", side_effect=refuse):
            code, out, err = run_cli(["--root", root, "init"])
        self.assertEqual(code, 2, (out, err))
        self.one_error(err, "Permission denied", "prompts")

    def test_a_root_under_a_file(self):
        root = self.bare_dir()
        self.write(os.path.join(root, "afile"))
        code, out, err = run_cli(["--root", os.path.join(root, "afile", "sub"),
                                  "init"])
        self.assertEqual(code, 2, (out, err))
        self.one_error(err)

    @unittest.skipIf(os.name == "nt", "POSIX permission bits")
    @unittest.skipIf(hasattr(os, "geteuid") and os.geteuid() == 0,
                     "root ignores permission bits")
    def test_a_read_only_folder(self):
        root = self.bare_dir()
        os.chmod(root, 0o555)
        self.addCleanup(os.chmod, root, 0o755)
        code, out, err = run_cli(["--root", root, "init"])
        self.assertEqual(code, 2, (out, err))
        self.one_error(err, "Permission denied")

    def test_init_in_a_folder_that_works_is_unchanged(self):
        root = self.bare_dir()
        code, out, err = run_cli(["--root", root, "init"])
        self.assertEqual((code, err), (0, ""))
        self.assertIn("scaffolded prompts/", out)
        self.assertTrue(os.path.isfile(os.path.join(root, ".promptregress",
                                                    ".gitignore")))
        code, out, err = run_cli(["--root", root, "init"])
        self.assertEqual((code, err), (0, ""))
        self.assertIn("prompts/ already scaffolded", out)


class TestOtherFileTroubleIsExit2(WorkDir):
    """Whatever else reaches the top as an OSError - a prompt or cases file
    that cannot be opened, a folder that cannot be listed - is one `error:`
    line and exit 2, never a traceback and never the 1 that means "a case
    failed"."""

    def setUp(self):
        self.url = self.model()

    def one_error(self, err, *mentions):
        lines = lines_of(err)
        self.assertEqual(len(lines), 1, f"want one error line: {err!r}")
        self.assertTrue(lines[0].startswith("error: "), err)
        self.assertNotIn("Traceback", err)
        for text in mentions:
            self.assertIn(text, self.tidy(lines[0]))

    def run_in(self, root, *argv):
        return run_cli(["--root", root, "--base-url", self.url, *argv])

    def test_a_prompt_file_that_cannot_be_opened(self):
        root = self.project()
        with mock.patch.object(suite_mod, "open",
                               refusing_open("s.v1.txt"), create=True):
            code, out, err = self.run_in(root, "run", "--prompt", "s",
                                         "--version", "v1", "--no-cache")
        self.assertEqual(code, 2, (out, err))
        self.one_error(err, "s.v1.txt", "Permission denied")
        self.assertEqual(out, "")

    def test_a_cases_file_that_cannot_be_opened(self):
        root = self.project()
        with mock.patch.object(suite_mod, "open",
                               refusing_open("s.cases.yaml"), create=True):
            code, out, err = self.run_in(root, "run", "--prompt", "s",
                                         "--version", "v1", "--no-cache")
        self.assertEqual(code, 2, (out, err))
        self.one_error(err, "s.cases.yaml", "Permission denied")

    def test_diff_with_a_prompt_file_that_cannot_be_opened(self):
        root = self.project(versions=("v1", "v2"))
        with mock.patch.object(suite_mod, "open",
                               refusing_open("s.v2.txt"), create=True):
            code, out, err = self.run_in(root, "diff", "v1", "v2",
                                         "--prompt", "s", "--no-cache")
        self.assertEqual(code, 2, (out, err))
        self.one_error(err, "s.v2.txt", "Permission denied")
        self.assertEqual(out, "")

    def test_every_unreadable_suite_of_run_all_is_listed(self):
        root = self.project()
        self.write(os.path.join(root, "prompts", "t.v1.txt"), PROMPT)
        self.write(os.path.join(root, "prompts", "t.cases.yaml"), ONE_CASE)
        with mock.patch.object(suite_mod, "open",
                               refusing_open("s.cases.yaml", "t.cases.yaml"),
                               create=True):
            code, out, err = self.run_in(root, "run", "--all", "--no-cache")
        self.assertEqual(code, 2, (out, err))
        lines = lines_of(err)
        self.assertEqual(len(lines), 2, err)
        self.assertTrue(all(ln.startswith("error: ") for ln in lines), err)
        self.assertNotIn("Traceback", err)

    def test_list_names_the_unreadable_cases_file_and_exits_2(self):
        root = self.project()
        with mock.patch.object(suite_mod, "open",
                               refusing_open("s.cases.yaml"), create=True):
            code, out, err = self.run_in(root, "list")
        self.assertEqual(code, 2, (out, err))
        self.assertIn("s: versions v1", out)
        self.one_error(err, "s.cases.yaml", "Permission denied")

    def test_a_prompts_folder_that_cannot_be_listed(self):
        root = self.project()
        refuse = PermissionError(errno.EACCES, "Permission denied",
                                 os.path.join(root, "prompts"))
        for argv in (["list"], ["run", "--all"], ["--demo", "run", "--all"],
                     ["--demo", "run", "--prompt", "sentiment",
                      "--version", "v1"]):
            with self.subTest(argv):
                with mock.patch.object(suite_mod.os, "listdir",
                                       side_effect=refuse):
                    code, out, err = run_cli(["--root", root, *argv])
                self.assertEqual(code, 2, (out, err))
                self.one_error(err, "prompts", "Permission denied")

    def test_a_cases_file_that_is_a_folder_was_and_stays_exit_2(self):
        root = self.project()
        os.remove(os.path.join(root, "prompts", "s.cases.yaml"))
        os.makedirs(os.path.join(root, "prompts", "s.cases.yaml"))
        code, out, err = self.run_in(root, "run", "--prompt", "s",
                                     "--version", "v1", "--no-cache")
        self.assertEqual(code, 2, (out, err))
        self.one_error(err, "no cases file")

    def test_a_prompt_that_is_a_folder_was_and_stays_exit_2(self):
        root = self.project()
        os.remove(os.path.join(root, "prompts", "s.v1.txt"))
        os.makedirs(os.path.join(root, "prompts", "s.v1.txt"))
        code, out, err = self.run_in(root, "run", "--prompt", "s",
                                     "--version", "v1", "--no-cache")
        self.assertEqual(code, 2, (out, err))
        self.one_error(err, "no such prompt")

    def test_the_net_under_main_catches_what_nothing_else_did(self):
        # whatever is not handled where it happens ends here, as exit 2
        with mock.patch.object(cli_mod, "cmd_list",
                               side_effect=OSError(errno.EIO,
                                                   "Input/output error",
                                                   "somewhere")):
            code, out, err = run_cli(["list"])
        self.assertEqual(code, 2, (out, err))
        self.one_error(err, "somewhere", "Input/output error")


class TestTheDocumentsSayIt(unittest.TestCase):
    """The help text and the README promise what the code now does."""

    def test_help_and_readme_name_the_rules(self):
        out = io.StringIO()
        with clean_env(), redirect_stdout(out):
            with self.assertRaises(SystemExit) as ctx:
                cli_mod.main(["--help"])
        self.assertEqual(ctx.exception.code, 0)
        text = " ".join(out.getvalue().split())
        self.assertIn("a file that cannot be read or written", text)
        self.assertIn("A response cache that cannot be written is no error",
                      text)
        with open(os.path.join(REPO, "README.md"), encoding="utf-8") as f:
            readme = " ".join(f.read().split())
        self.assertIn("`warning: cache disabled: ...`", readme)
        self.assertIn("A response cache that cannot be written is not an "
                      "error", readme)
        self.assertIn("a file that cannot be read or written", readme)


# ---------------------------------------------------------------------
# #2 - the YAML subset: PyYAML's layout is read, a quoted key says what to do

#: What yaml.dump writes: every list sits at the indent of its key.
PYYAML_CASES = (
    "cases:\n"
    "- name: pyyaml style\n"
    "  input: great\n"
    "  vars:\n"
    "    tone: terse\n"
    "  assert:\n"
    "  - contains: neutral\n"
    "  - schema:\n"
    "      type: object\n"
    "      required:\n"
    "      - sentiment\n"
    "      properties:\n"
    "        sentiment:\n"
    "          type: string\n"
    "          enum:\n"
    "          - positive\n"
    "          - neutral\n"
    "- name: second\n"
    "  input: |\n"
    "    two\n"
    "    lines\n"
    "  assert:\n"
    "  - not_contains: oops\n")
#: The same file with every list indented under its key.
INDENTED_CASES = (
    "cases:\n"
    "  - name: pyyaml style\n"
    "    input: great\n"
    "    vars:\n"
    "      tone: terse\n"
    "    assert:\n"
    "      - contains: neutral\n"
    "      - schema:\n"
    "          type: object\n"
    "          required:\n"
    "            - sentiment\n"
    "          properties:\n"
    "            sentiment:\n"
    "              type: string\n"
    "              enum:\n"
    "                - positive\n"
    "                - neutral\n"
    "  - name: second\n"
    "    input: |\n"
    "      two\n"
    "      lines\n"
    "    assert:\n"
    "      - not_contains: oops\n")


class TestAListMayStartAtItsKeysIndent(WorkDir):
    """`cases:` and then `- name: a` at the same indent is what PyYAML (and
    many editors) write. The parser said "list item inside mapping"."""

    def test_a_list_at_its_keys_indent_is_that_keys_value(self):
        self.assertEqual(miniyaml.safe_load("a:\n- 1\n- two\n"),
                         {"a": [1, "two"]})

    def test_it_reads_the_same_as_the_indented_layout(self):
        self.assertEqual(miniyaml.safe_load(PYYAML_CASES),
                         miniyaml.safe_load(INDENTED_CASES))

    def test_the_list_ends_with_the_indent_and_the_map_goes_on(self):
        self.assertEqual(
            miniyaml.safe_load("a:\n- 1\n- 2\nb: 3\nc:\n- x\nd: end\n"),
            {"a": [1, 2], "b": 3, "c": ["x"], "d": "end"})

    def test_lists_inside_the_maps_of_a_list(self):
        text = "a:\n- b:\n  - 1\n  - 2\n  c: 3\n- d: 4\n"
        self.assertEqual(miniyaml.safe_load(text),
                         {"a": [{"b": [1, 2], "c": 3}, {"d": 4}]})
        # and inside a list that has no key above it
        self.assertEqual(miniyaml.safe_load("- a:\n  - 1\n  - 2\n- b: 3\n"),
                         [{"a": [1, 2]}, {"b": 3}])

    def test_an_empty_key_stays_null_and_an_empty_item_too(self):
        self.assertEqual(miniyaml.safe_load("a:\nb:\n-\n- x\nc:\n"),
                         {"a": None, "b": [None, "x"], "c": None})
        # a null item next to a sibling list item of an outer list
        self.assertEqual(miniyaml.safe_load("- a:\n- b: 3\n"),
                         [{"a": None}, {"b": 3}])

    def test_block_scalars_and_flow_lists_as_items(self):
        text = ("a:\n- |\n  one\n  two\n- key: |\n    three\n  other: 4\n"
                "- [x, y]\nb: [1, 2]\n")
        self.assertEqual(miniyaml.safe_load(text), {
            "a": ["one\ntwo\n", {"key": "three\n", "other": 4}, ["x", "y"]],
            "b": [1, 2]})

    def test_a_comment_and_blank_lines_do_not_end_the_list(self):
        text = "a:\n- 1   # one\n\n# two\n- 2\nb: 3\n"
        self.assertEqual(miniyaml.safe_load(text), {"a": [1, 2], "b": 3})

    def test_a_dash_that_belongs_to_no_key_is_still_an_error(self):
        for text in ("a: 1\n- b\n", "a: [1]\n- b\n", "a: b\n- c: d\n"):
            with self.subTest(text):
                with self.assertRaisesRegex(
                        ValueError, r"list item inside mapping.*"
                                    r"`key:` line above.*\(line 2\)"):
                    miniyaml.safe_load(text)

    def test_a_dash_after_a_deeper_map_is_not_taken_for_the_outer_key(self):
        # `x:` has a value here (the map under it), so a dash back at the
        # indent of `x` is a stray one, not a second list for `x`
        with self.assertRaisesRegex(ValueError, "list item inside mapping"):
            miniyaml.safe_load("x:\n  y: 1\n- z\n")

    def test_the_depth_limit_still_holds(self):
        # shallow: fine; deep: the parser's own error, never RecursionError
        deep = "a:\n" + "".join(
            f"{' ' * (2 * i)}- b:\n{' ' * (2 * i)}  - c:\n"
            for i in range(300)) + "x\n"
        with self.assertRaisesRegex(ValueError, "maximum nesting depth"):
            miniyaml.safe_load(deep)
        shallow = "a:\n- b:\n  - c:\n    - 1\n"
        self.assertEqual(miniyaml.safe_load(shallow),
                         {"a": [{"b": [{"c": [1]}]}]})

    def test_load_cases_reads_the_pyyaml_layout(self):
        pyyaml = self.project(PYYAML_CASES)
        indented = self.project(INDENTED_CASES)
        got = suite_mod.load_cases(pyyaml, "s")
        self.assertEqual(got, suite_mod.load_cases(indented, "s"))
        self.assertEqual([c["name"] for c in got],
                         ["pyyaml style", "second"])
        self.assertEqual(got[0]["vars"], {"tone": "terse"})
        self.assertEqual(got[1]["input"], "two\nlines\n")

    def test_a_run_over_a_pyyaml_style_file(self):
        url = self.model()
        root = self.project(
            "cases:\n- name: a\n  input: hi\n  assert:\n"
            "  - contains: neutral\n- name: b\n  input: ho\n  assert:\n"
            "  - contains: neutral\n  - max_tokens: 10\n")
        code, out, err = run_cli(["--root", root, "--base-url", url, "run",
                                  "--prompt", "s", "--version", "v1"])
        self.assertEqual(code, 0, (out, err))
        self.assertIn("2/2 passed", out)

    def test_list_shows_the_cases_of_a_pyyaml_style_file(self):
        root = self.project(PYYAML_CASES)
        code, out, err = run_cli(["--root", root, "list"])
        self.assertEqual(code, 0, (out, err))
        self.assertIn("s: versions v1 (2 cases)", out)


class TestAQuotedKeyIsRefusedWithTheFix(WorkDir):
    """`"input": x` is valid YAML that this parser does not read. It says so
    and says what to write; a list item with one used to be reported as an
    "unterminated quoted string"."""

    def refused(self, text, line):
        with self.assertRaises(ValueError) as ctx:
            miniyaml.safe_load(text)
        message = str(ctx.exception)
        self.assertIn("quoted keys are not supported", message)
        self.assertIn("write the key without quotes", message)
        self.assertTrue(message.endswith(f"(line {line})"), message)
        return message

    def test_a_quoted_key_at_the_top_and_inside(self):
        self.refused('"cases":\n  - input: x\n', 1)
        self.refused("cases:\n  - name: a\n    'input': x\n", 3)
        self.refused('cases:\n  - name: a\n    "input" : x\n', 3)

    def test_a_quoted_key_of_a_list_item(self):
        self.refused('cases:\n  - "input": x\n', 2)
        self.refused("- 'input': x\n", 1)

    def test_a_quoted_key_with_a_block_scalar_or_without_a_value(self):
        self.refused('cases:\n  - name: a\n    "input": |\n      text\n', 3)
        self.refused('"cases":\n', 1)

    def test_the_message_shows_the_line(self):
        self.assertIn("\"input\": x",
                      self.refused('cases:\n  - "input": x\n', 2))

    def test_quoted_values_and_items_are_untouched(self):
        self.assertEqual(
            miniyaml.safe_load('a: "x: y"\nb: \'it\'\'s: ok\'\nc:\n'
                               '  - "q: r"\n  - \'s\'\n  - "t"\n'),
            {"a": "x: y", "b": "it's: ok", "c": ["q: r", "s", "t"]})
        self.assertEqual(miniyaml.safe_load("- a: b\n"), [{"a": "b"}])

    def test_a_cases_file_with_one_gives_one_error_line(self):
        root = self.project('cases:\n  - name: a\n    "input": hi\n'
                            "    assert:\n      - contains: neutral\n")
        for argv in (["run", "--prompt", "s", "--version", "v1"],
                     ["diff", "v1", "v1", "--prompt", "s"]):
            with self.subTest(argv):
                code, out, err = run_cli(["--root", root, *argv])
                self.assertEqual(code, 2, (out, err))
                self.assertEqual(len(lines_of(err)), 1, err)
                self.assertIn("quoted keys are not supported", err)
                self.assertIn("s.cases.yaml", self.tidy(err))
                self.assertIn("(line 3)", err)


class TestTheReadmeNamesTheYamlSubset(unittest.TestCase):
    def test_readme_and_parser_say_what_is_read_and_what_is_not(self):
        with open(os.path.join(REPO, "README.md"), encoding="utf-8") as f:
            readme = " ".join(f.read().split())
        for text in (
                "a list may start at the indent of its key",
                "A key is written without quotes",
                "a quoted key (`\"input\": ...`) is refused",
                "ASCII letters, digits, spaces and `_ . - /`",
                "anchors, aliases, tags, flow maps",
                "a nested `[...]`",
                "a key given twice"):
            self.assertIn(text, readme)
        self.assertIn("quoted keys", miniyaml.__doc__)
        self.assertIn("indent of its key", miniyaml.__doc__)


# ---------------------------------------------------------------------
# #3 - an enum tells true from 1

def verdict(enum, reply):
    return asserts_mod.evaluate({"schema": {"enum": enum}}, reply)


class TestEnumDoesNotTakeTrueForOne(unittest.TestCase):
    """`enum: [1]` accepted `true` and `enum: [0]` accepted `false` (Python
    says True == 1), while `type: integer` rightly refused a boolean."""

    def test_a_boolean_is_not_a_number_and_the_other_way_round(self):
        for enum, reply in (([1], "true"), ([0], "false"),
                            ([True], "1"), ([False], "0"),
                            ([1.0], "true"), ([True], "1.0"),
                            ([2, 1], "true"), ([True, 3], "1")):
            with self.subTest(enum=enum, reply=reply):
                ok, detail = verdict(enum, reply)
                self.assertFalse(ok, detail)
                self.assertIn("not in enum", detail)

    def test_what_is_the_same_still_matches(self):
        for enum, reply in (([1], "1"), ([0], "0"), ([True], "true"),
                            ([False], "false"), ([None], "null"),
                            (["a", 2, True], "true"), ([1, True], "1"),
                            (["1"], '"1"')):
            with self.subTest(enum=enum, reply=reply):
                ok, detail = verdict(enum, reply)
                self.assertTrue(ok, detail)

    def test_one_is_still_one_point_zero(self):
        # JSON has one kind of number
        for enum, reply in (([1], "1.0"), ([1.0], "1"), ([2.5], "2.5"),
                            ([0], "-0.0"), ([[1, 2.0]], "[1.0, 2]"),
                            ([100], "1e2")):
            with self.subTest(enum=enum, reply=reply):
                ok, detail = verdict(enum, reply)
                self.assertTrue(ok, detail)

    def test_text_and_numbers_stay_apart(self):
        for enum, reply in ((["1"], "1"), ([1], '"1"'), ([None], "0"),
                            ([None], "false"), ([0], "null"),
                            (["true"], "true")):
            with self.subTest(enum=enum, reply=reply):
                self.assertFalse(verdict(enum, reply)[0])

    def test_inside_lists(self):
        self.assertFalse(verdict([[1, 2]], "[true, 2]")[0])
        self.assertFalse(verdict([[True]], "[1]")[0])
        self.assertFalse(verdict([[0]], "[false]")[0])
        self.assertFalse(verdict([[[1]]], "[[true]]")[0])
        self.assertTrue(verdict([[1, 2]], "[1, 2]")[0])
        self.assertTrue(verdict([[True, 0]], "[true, 0]")[0])
        # length and order still matter
        self.assertFalse(verdict([[1, 2]], "[2, 1]")[0])
        self.assertFalse(verdict([[1, 2]], "[1, 2, 3]")[0])
        self.assertFalse(verdict([[1, 2]], "[1]")[0])

    def test_inside_objects(self):
        self.assertFalse(verdict([{"ok": 1}], '{"ok": true}')[0])
        self.assertFalse(verdict([{"ok": True}], '{"ok": 1}')[0])
        self.assertFalse(verdict([{"a": [1, {"b": 0}]}],
                                 '{"a": [1, {"b": false}]}')[0])
        self.assertTrue(verdict([{"a": [1, {"b": 0}]}],
                                '{"a": [1.0, {"b": 0}]}')[0])
        # key order does not matter, a missing or extra key does
        self.assertTrue(verdict([{"a": 1, "b": True}],
                                '{"b": true, "a": 1}')[0])
        self.assertFalse(verdict([{"a": 1}], '{"a": 1, "b": 2}')[0])
        self.assertFalse(verdict([{"a": 1, "b": 2}], '{"a": 1}')[0])
        self.assertFalse(verdict([{"a": 1}], '{"b": 1}')[0])
        # an object is not a list, and neither is a number
        self.assertFalse(verdict([{}], "[]")[0])
        self.assertFalse(verdict([[]], "{}")[0])
        self.assertFalse(verdict([[1]], "1")[0])

    def test_a_schema_with_a_type_beside_the_enum_is_unchanged(self):
        schema = {"type": "integer", "enum": [1]}
        ok, detail = asserts_mod.evaluate({"schema": schema}, "true")
        self.assertFalse(ok)
        self.assertIn("expected integer", detail)
        self.assertTrue(asserts_mod.evaluate({"schema": schema}, "1")[0])

    def test_the_comparison_runs_inside_properties_and_items(self):
        schema = {"type": "object", "properties": {
            "n": {"enum": [1, 2]},
            "xs": {"type": "array", "items": {"enum": [0, 1]}}}}
        good = '{"n": 2, "xs": [0, 1, 1.0]}'
        self.assertTrue(asserts_mod.evaluate({"schema": schema}, good)[0])
        for bad in ('{"n": true}', '{"xs": [0, false]}',
                    '{"xs": [true]}'):
            with self.subTest(bad):
                self.assertFalse(
                    asserts_mod.evaluate({"schema": schema}, bad)[0])


class TestEnumInACasesFile(WorkDir):
    def run_with(self, enum_line, answer):
        url = self.model(answer)
        root = self.project(
            "cases:\n  - name: e\n    input: hi\n    assert:\n"
            "      - schema:\n          enum: " + enum_line + "\n")
        return run_cli(["--root", root, "--base-url", url, "run",
                        "--prompt", "s", "--version", "v1", "--no-cache"])

    def test_true_does_not_pass_for_one(self):
        code, out, err = self.run_with("[1]", "true")
        self.assertEqual(code, 1, (out, err))
        self.assertIn("0/1 passed", out)
        self.assertIn("not in enum", out)

    def test_false_does_not_pass_for_zero(self):
        code, out, err = self.run_with("[0, 2]", "false")
        self.assertEqual(code, 1, (out, err))

    def test_yaml_booleans_in_an_enum_are_booleans(self):
        # `[yes, no]` is [true, false] in YAML 1.1; 1 and 0 are not those
        self.assertEqual(self.run_with("[yes, no]", "true")[0], 0)
        self.assertEqual(self.run_with("[yes, no]", "1")[0], 1)

    def test_the_right_kind_passes(self):
        self.assertEqual(self.run_with("[1]", "1")[0], 0)
        self.assertEqual(self.run_with("[1]", "1.0")[0], 0)
        self.assertEqual(self.run_with("[true]", "true")[0], 0)


class TestTheReadmeNamesTheEnumRule(unittest.TestCase):
    def test_readme_says_how_enum_compares(self):
        with open(os.path.join(REPO, "README.md"), encoding="utf-8") as f:
            readme = " ".join(f.read().split())
        self.assertIn("`enum` compares as JSON does: `true` is not the "
                      "number 1 (nor `false` the number 0), `1` equals "
                      "`1.0`", readme)


if __name__ == "__main__":
    unittest.main()
