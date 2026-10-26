"""Regression tests for 0.1.5: an unwritable stdout or stderr exits 2 with no
traceback. In 0.1.4 the last flush at the end of the run still failed outside
the guarded part: `... 2>/dev/full` and `> /dev/full` gave exit 120."""
from __future__ import annotations

import errno
import io
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest import mock

from promptregress import cli as cli_mod

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HAS_FULL = os.path.exists("/dev/full")


def _env():
    # buffered output is what a user has; PYTHONUNBUFFERED hides the defect
    env = {k: v for k, v in os.environ.items()
           if k not in ("PYTHONUNBUFFERED", "PROMPTREGRESS_BASE_URL")}
    return env


def _cli(*argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE):
    return subprocess.run(
        [sys.executable, "-m", "promptregress", *argv], cwd=REPO,
        stdout=stdout, stderr=stderr, env=_env(), timeout=300)


@unittest.skipUnless(HAS_FULL, "needs /dev/full")
class TestUnwritableOutput(unittest.TestCase):
    BAD = ("--demo", "run", "--prompt", "nonexist", "--version", "v1")
    GOOD = ("--demo", "run", "--prompt", "sentiment", "--version", "v1")

    def test_stderr_to_dev_full_exits_2_without_a_traceback(self):
        with open("/dev/full", "wb") as full:
            p = _cli(*self.BAD, stderr=full, stdout=subprocess.DEVNULL)
        self.assertEqual(p.returncode, 2)

    def test_stdout_to_dev_full_exits_2_with_one_error_line(self):
        with open("/dev/full", "wb") as full:
            p = _cli(*self.GOOD, stdout=full)
        err = p.stderr.decode()
        self.assertEqual(p.returncode, 2, err)
        self.assertNotIn("Traceback", err)
        self.assertNotIn("Exception ignored", err)
        errors = [ln for ln in err.splitlines() if ln.startswith("error:")]
        self.assertEqual(len(errors), 1, err)

    def test_a_failing_run_is_also_2_when_its_report_is_lost(self):
        # exit 1 would say "a case failed" about a report nobody can read
        with open("/dev/full", "wb") as full:
            p = _cli("--demo", "run", "--prompt", "sentiment", "--version",
                     "v2", stdout=full)
        self.assertEqual(p.returncode, 2, p.stderr)
        self.assertNotIn(b"Traceback", p.stderr)

    def test_list_to_dev_full_exits_2(self):
        with open("/dev/full", "wb") as full:
            p = _cli("--root", REPO, "list", stdout=full)
        self.assertEqual(p.returncode, 2, p.stderr)
        self.assertNotIn(b"Traceback", p.stderr)

    @unittest.skipIf(os.name == "nt", "pipe semantics differ on Windows")
    def test_a_closed_pipe_still_keeps_the_verdict(self):
        p = subprocess.Popen(
            [sys.executable, "-m", "promptregress", "--demo", "run",
             "--prompt", "sentiment", "--version", "v2"], cwd=REPO,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=_env())
        p.stdout.close()
        err = p.stderr.read()
        p.stderr.close()
        self.assertEqual(p.wait(timeout=300), 1, err)   # a case failed
        self.assertNotIn(b"Traceback", err)


class _Full(io.StringIO):
    def __init__(self, fails="flush"):
        super().__init__()
        self.fails = fails

    def flush(self):
        if self.fails == "flush":
            raise OSError(errno.ENOSPC, "No space left on device")


class TestStreamsInProcess(unittest.TestCase):
    def _main(self, out, err, *argv):
        with redirect_stdout(out), redirect_stderr(err):
            return cli_mod.main(["--root", tempfile.mkdtemp(), *argv])

    def test_a_failing_final_flush_of_stdout(self):
        err = io.StringIO()
        code = self._main(_Full(), err, "list")
        self.assertEqual(code, 2)
        lines = [ln for ln in err.getvalue().splitlines() if ln.strip()]
        self.assertTrue(any(ln.startswith("error:") and "No space" in ln
                            for ln in lines), lines)
        self.assertEqual(len([ln for ln in lines
                              if ln.startswith("error:")]), 1, lines)

    def test_a_failing_final_flush_of_stderr_is_silent_and_2(self):
        out = io.StringIO()
        err = _Full()   # stderr is the dead stream
        with redirect_stdout(out), redirect_stderr(err):
            err.write("warning: something\n")
            code = cli_mod.main(["--root", tempfile.mkdtemp(), "list"])
        self.assertEqual(code, 2)

    def test_the_streams_are_put_back(self):
        out, err = _Full(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            cli_mod.main(["--root", tempfile.mkdtemp(), "list"])
            self.assertIs(sys.stdout, out)
            self.assertIs(sys.stderr, err)

    def test_healthy_streams_keep_the_command_exit_code(self):
        out, err = io.StringIO(), io.StringIO()
        code = self._main(out, err, "list")
        self.assertEqual(code, 0, err.getvalue())
        self.assertIn("no prompts found", out.getvalue())
        self.assertEqual(err.getvalue(), "")

    def test_closed_pipe_is_not_a_write_error(self):
        class Gone(io.StringIO):
            def flush(self):
                raise BrokenPipeError(errno.EPIPE, "Broken pipe")
        err = io.StringIO()
        code = self._main(Gone(), err, "list")
        # nobody is reading: the command's own exit code stands
        self.assertEqual(code, 0, err.getvalue())
        self.assertEqual(err.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
