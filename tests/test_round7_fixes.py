"""Round-7 panel regressions (third verification panel, 2026-10-18).

promptregress: a YAML-only escape in a double-quoted string (\\xE9, \\U0001F600:
what PyYAML writes for every accented letter and every emoji) stayed backslash
text, so the model was asked 'Say: caf\\xE9' and a `not_contains` check passed
where it had to fail. The README promises that what the parser does not
understand is refused with an error, never ignored.

Every test uses fake servers on 127.0.0.1 and temp directories, nothing else.
"""

import ast
import contextlib
import io
import json
import os
import re
import shutil
import tempfile
import threading
import unittest
from contextlib import redirect_stderr, redirect_stdout
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest import mock

from promptregress import cli as cli_mod
from promptregress import miniyaml
from promptregress import suite as suite_mod


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


class _Recorder(BaseHTTPRequestHandler):
    """A model that answers every call with `answer` and keeps the prompts it
    was asked."""
    answer = "ok"
    prompts: list = []

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length).decode("utf-8"))
        type(self).prompts.append(body["messages"][0]["content"])
        payload = json.dumps({"choices": [{"message": {
            "role": "assistant", "content": type(self).answer}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass


class Project(unittest.TestCase):
    """Temp projects that are removed again, with the model they talk to."""

    def project(self, cases, prompt="Classify: {{input}}\n", versions=("v1",),
                name="s"):
        root = tempfile.mkdtemp(prefix="prg-r7-")
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        d = os.path.join(root, "prompts")
        os.makedirs(d)
        for v in versions:
            self.write(os.path.join(d, f"{name}.{v}.txt"), prompt)
        self.write(os.path.join(d, f"{name}.cases.yaml"), cases)
        return root

    @staticmethod
    def write(path, text=""):
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)

    def model(self, answer="neutral"):
        """The model's base URL; self.prompts lists what it was asked."""
        handler = type("R", (_Recorder,), {"answer": answer, "prompts": []})
        server = HTTPServer(("127.0.0.1", 0), handler)
        threading.Thread(target=server.serve_forever,
                         kwargs={"poll_interval": 0.02}, daemon=True).start()

        def stop():
            server.shutdown()
            server.server_close()
        self.addCleanup(stop)
        self.prompts = handler.prompts
        return f"http://127.0.0.1:{server.server_port}/v1"

    def run_suite(self, root, url, *more, name="s", version="v1"):
        return run_cli(["--root", root, "--base-url", url, "run", "--prompt",
                        name, "--version", version, "--no-cache", *more])

    @staticmethod
    def tidy(text):
        return text.replace("\\", "/")


# ---------------------------------------------------------------------
# #1 [MED] - a YAML-only escape in a double-quoted string

#: What follows the backslash -> the character it stands for: YAML 1.2's table
#: of escapes for a double-quoted scalar, the one PyYAML reads and writes.
YAML_ESCAPES = {
    "0": "\x00", "a": "\x07", "b": "\x08", "t": "\t", "n": "\n",
    "v": "\x0b", "f": "\x0c", "r": "\r", "e": "\x1b", " ": " ", '"': '"',
    "/": "/", "\\": "\\", "N": "\x85", "_": "\xa0", "L": "\u2028",
    "P": "\u2029",
    "x41": "A", "xE9": "\xe9", "xe9": "\xe9", "x00": "\x00", "xFF": "\xff",
    "u00E9": "\xe9", "u00e9": "\xe9", "u2603": "\u2603", "uFFFF": "\uffff",
    "U0001F600": "\U0001F600", "U000000e9": "\xe9", "U0010FFFF": "\U0010ffff",
}


class TestEveryYamlEscapeIsDecoded(unittest.TestCase):
    def test_the_whole_table(self):
        for escape, char in YAML_ESCAPES.items():
            with self.subTest(escape=escape):
                self.assertEqual(
                    miniyaml.safe_load('a: "[\\' + escape + ']"'),
                    {"a": "[" + char + "]"})

    def test_a_backslash_and_a_literal_tab_is_a_tab(self):
        self.assertEqual(miniyaml.safe_load('a: "x\\\ty"'), {"a": "x\ty"})

    def test_what_pyyaml_writes_reads_back(self):
        # yaml.safe_dump output for a cafe with its accent, an emoji, a bell,
        # an ESC, NEL, LS and a no-break space (PyYAML is not needed here)
        text = ('cases:\n'
                '- assert:\n'
                '  - not_contains: "caf\\xE9"\n'
                '  - contains: "x\\0y\\az"\n'
                '  input: "caf\\xE9 \\U0001F600\\nline2 \\N \\L \\_ end"\n'
                '  name: "caf\\xE9 \\U0001F600 tab\\there \\e[0m"\n')
        case = miniyaml.safe_load(text)["cases"][0]
        self.assertEqual(case["assert"], [{"not_contains": "caf\u00e9"},
                                          {"contains": "x\x00y\x07z"}])
        self.assertEqual(case["input"],
                         "caf\u00e9 \U0001F600\nline2 \x85 \u2028 \xa0 end")
        self.assertEqual(case["name"], "caf\u00e9 \U0001F600 tab\there \x1b[0m")

    def test_a_literal_tab_beside_an_escape(self):
        # JSON refuses a control character inside a string; YAML does not,
        # and the escapes beside it still mean what they say
        self.assertEqual(miniyaml.safe_load('a: "a\tb\\nc\\xE9"'),
                         {"a": "a\tb\nc\xe9"})

    def test_escapes_in_every_place_a_scalar_can_stand(self):
        load = miniyaml.safe_load
        self.assertEqual(load('- "caf\\xE9"'), ["caf\xe9"])
        self.assertEqual(load('a: ["caf\\xE9", \'x\', "\\U0001F600"]'),
                         {"a": ["caf\xe9", "x", "\U0001F600"]})
        self.assertEqual(load('- k: "caf\\xE9"\n  other: 1'),
                         [{"k": "caf\xe9", "other": 1}])
        self.assertEqual(load('a:\n  b:\n    - "\\x41\\x42"'),
                         {"a": {"b": ["AB"]}})

    def test_a_comment_after_an_escape_is_still_a_comment(self):
        self.assertEqual(miniyaml.safe_load('a: "caf\\xE9"  # the cafe'),
                         {"a": "caf\xe9"})
        self.assertEqual(miniyaml.safe_load('a: "# \\xE9 #"  # real'),
                         {"a": "# \xe9 #"})

    def test_a_json_surrogate_pair_is_one_character(self):
        # json.dumps writes an emoji this way; it was valid before and stays so
        self.assertEqual(miniyaml.safe_load('a: "\\ud83d\\ude00"'),
                         {"a": "\U0001F600"})
        self.assertEqual(miniyaml.safe_load('a: "\\uD83D\\uDE00 ok"'),
                         {"a": "\U0001F600 ok"})

    def test_text_that_takes_no_escapes_keeps_its_backslashes(self):
        load = miniyaml.safe_load
        # single quotes, plain scalars and block scalars are literal
        self.assertEqual(load("a: 'caf\\xE9'"), {"a": "caf\\xE9"})
        self.assertEqual(load("a: caf\\xE9"), {"a": "caf\\xE9"})
        self.assertEqual(load("a: |\n  caf\\xE9\\n\n"), {"a": "caf\\xE9\\n\n"})
        self.assertEqual(load("a: C:\\Users\\dan"), {"a": "C:\\Users\\dan"})
        self.assertEqual(load("a: 'C:\\Users\\dan'"), {"a": "C:\\Users\\dan"})

    def test_the_escapes_that_always_worked_still_do(self):
        load = miniyaml.safe_load
        self.assertEqual(load('a: "line1\\nline2"'), {"a": "line1\nline2"})
        self.assertEqual(load('a: "say \\"hi\\" \\\\ done"'),
                         {"a": 'say "hi" \\ done'})
        self.assertEqual(load('a: "caf\\u00e9"'), {"a": "caf\u00e9"})
        self.assertEqual(load('a: "\\u0000block:0"'), {"a": "\x00block:0"})
        self.assertEqual(load('a: "x\\\\"'), {"a": "x\\"})
        self.assertEqual(load('a: "\\\\\\""'), {"a": '\\"'})

    def test_an_escape_next_to_the_closing_quote(self):
        load = miniyaml.safe_load
        self.assertEqual(load('a: "\\""'), {"a": '"'})
        self.assertEqual(load('a: "\\xE9"'), {"a": "\xe9"})
        self.assertEqual(load('a: "\\\\"'), {"a": "\\"})


#: (what goes between the quotes, what the message has to say). The slip is
#: always on line 4 of the document built around it.
NOT_YAML = (
    ("x\\qy", r"unknown escape \\q"),
    ("x\\Zy", r"unknown escape \\Z"),
    ("x\\1y", r"unknown escape \\1"),
    ("x\\'y", r"unknown escape"),
    ("C:\\Users\\dan", r"\\U needs 8 hex digits"),
    ("x\\xZZy", r"\\x needs 2 hex digits"),
    ("x\\x4", r"\\x needs 2 hex digits"),
    ("x\\u12", r"\\u needs 4 hex digits"),
    ("x\\u12G4y", r"\\u needs 4 hex digits"),
    ("x\\U1234y", r"\\U needs 8 hex digits"),
    ("x\\u 123y", r"\\u needs 4 hex digits"),
    ("x\\u+123y", r"\\u needs 4 hex digits"),
    ("x\\u1_23y", r"\\u needs 4 hex digits"),
    ("x\\u\u0663\u0663\u0663\u0663y", r"\\u needs 4 hex digits"),
    ("x\\ud83dy", r"surrogate"),
    ("x\\ude00y", r"surrogate"),
    ("x\\ud83d\\u0041y", r"surrogate"),
    ("x\\ud83d\\ud83dy", r"surrogate"),
    ("x\\ude00\\ud83dy", r"surrogate"),
    ("x\\U0000D83Dy", r"surrogate"),
    ("x\\U0000DFFFy", r"surrogate"),
    ("x\\U00110000y", r"U\+10FFFF"),
    ("x\\UFFFFFFFFy", r"U\+10FFFF"),
)


class TestAnEscapeThatIsNotYamlIsRefused(unittest.TestCase):
    """Any other escape is a line-numbered error, never kept as text."""

    DOC = 'a: 1\nb:\n  - ok\n  - "{}"\nc: 2\n'

    def test_each_one_is_an_error_naming_the_line(self):
        for inside, message in NOT_YAML:
            with self.subTest(inside=inside):
                with self.assertRaisesRegex(
                        ValueError,
                        message + r".*\(line 4\)$") as ctx:
                    miniyaml.safe_load(self.DOC.format(inside))
                self.assertTrue(str(ctx.exception).startswith("miniyaml: "))

    def test_a_windows_path_written_the_two_ways_the_error_suggests(self):
        want = {"a": "C:\\Users\\dan"}
        self.assertEqual(miniyaml.safe_load('a: "C:\\\\Users\\\\dan"'), want)
        self.assertEqual(miniyaml.safe_load("a: 'C:\\Users\\dan'"), want)

    def test_the_error_says_what_to_write_instead(self):
        with self.assertRaises(ValueError) as ctx:
            miniyaml.safe_load('a: "C:\\Users\\dan"')
        text = str(ctx.exception)
        self.assertIn("\\\\", text)               # write \\ for a backslash
        self.assertIn("single quotes", text)

    def test_in_a_flow_list_and_in_a_key_value_and_after_a_dash(self):
        for doc, line in (('a: ["ok", "x\\qy"]', 1),
                          ('a: 1\nb: "x\\qy"', 2),
                          ('- fine\n- "x\\qy"', 2),
                          ('k:\n  - name: "x\\qy"\n    z: 1', 2)):
            with self.subTest(doc=doc):
                with self.assertRaisesRegex(
                        ValueError, rf"unknown escape.*\(line {line}\)"):
                    miniyaml.safe_load(doc)

    def test_a_backslash_before_the_closing_quote_is_no_closing_quote(self):
        # "abc\" never ends: the quote is escaped (it used to come back as abc\)
        for doc in ('a: "abc\\"', 'a: ["abc\\"]', '- "C:\\temp\\"'):
            with self.subTest(doc=doc):
                with self.assertRaisesRegex(ValueError,
                                            r"unterminated.*\(line 1\)"):
                    miniyaml.safe_load(doc)

    def test_the_other_errors_are_unchanged(self):
        with self.assertRaisesRegex(ValueError, r"unterminated quoted"):
            miniyaml.safe_load('a: "oops')
        with self.assertRaisesRegex(ValueError, r"quoted keys are not"):
            miniyaml.safe_load('"k": 1')


class TestYamlEscapesReachTheModel(Project):
    """The reviewer's case: a cases file as yaml.safe_dump writes it."""

    PYYAML_CASES = ('cases:\n'
                    '- assert:\n'
                    '  - not_contains: "caf\\xE9"\n'
                    '  input: "caf\\xE9"\n'
                    '  name: must not mention the brand\n')

    def test_the_model_is_asked_the_text_the_file_means(self):
        url = self.model("caf\u00e9")
        root = self.project(self.PYYAML_CASES, prompt="Say: {{input}}\n")
        code, out, err = self.run_suite(root, url)
        self.assertEqual(self.prompts, ["Say: caf\u00e9\n"])
        self.assertEqual(code, 1, (out, err))
        self.assertIn("0/1 passed", out)
        self.assertIn("must not mention the brand", out)

    def test_and_the_same_check_passes_when_the_model_is_silent_about_it(self):
        url = self.model("tea")
        root = self.project(self.PYYAML_CASES, prompt="Say: {{input}}\n")
        code, out, err = self.run_suite(root, url)
        self.assertEqual((code, self.prompts), (0, ["Say: caf\u00e9\n"]))

    def test_escapes_in_a_name_an_assertion_and_a_var(self):
        cases = ('cases:\n'
                 '- name: "sm\\xF6rg\\u00e5s"\n'
                 '  input: "x"\n'
                 '  vars:\n'
                 '    tone: "d\\xE9j\\xE0 vu"\n'
                 '  assert:\n'
                 '  - contains: "\\U0001F600"\n')
        url = self.model("ok \U0001F600")
        root = self.project(cases, prompt="{{tone}}: {{input}}\n")
        code, out, err = self.run_suite(root, url, "--format", "json")
        self.assertEqual(self.prompts, ["d\u00e9j\u00e0 vu: x\n"])
        data = json.loads(out)
        self.assertEqual(code, 0, (out, err))
        self.assertEqual(data["cases"][0]["name"], "sm\u00f6rg\u00e5s")

    def test_an_escape_that_is_not_yaml_stops_the_run_before_the_model(self):
        for inside in ("C:\\Users\\dan", "x\\qy", "x\\xZZ", "\\ud83d"):
            with self.subTest(inside=inside):
                cases = ('cases:\n'
                         '  - name: a\n'
                         f'    input: "{inside}"\n'
                         '    assert:\n'
                         '      - contains: x\n')
                url = self.model()
                root = self.project(cases)
                code, out, err = self.run_suite(root, url)
                self.assertEqual(code, 2, (out, err))
                self.assertEqual(self.prompts, [])      # nothing was sent
                self.assertEqual(out, "")
                line = err.strip()
                self.assertTrue(line.startswith("error: "), err)
                self.assertEqual(len(err.splitlines()), 1, err)
                self.assertIn("s.cases.yaml", self.tidy(line))
                self.assertIn("malformed YAML", line)
                self.assertIn("(line 3)", line)
                self.assertNotIn("Traceback", err)

    def test_list_reports_it_too(self):
        root = self.project('cases:\n  - name: a\n    input: "x\\qy"\n'
                            '    assert:\n      - contains: x\n')
        code, out, err = run_cli(["--root", root, "list"])
        self.assertEqual(code, 2, (out, err))
        self.assertIn("invalid cases file", out)
        self.assertIn("unknown escape", err)



# ---------------------------------------------------------------------
# #2 [LOW] - a prompt that never receives the case input

ONE_CASE = ("cases:\n  - name: a\n    input: hi\n    assert:\n"
            "      - contains: neutral\n")
WITH_VARS = ("cases:\n  - name: a\n    input: hi\n    vars:\n"
             "      tone: terse\n    assert:\n      - contains: neutral\n")


def warnings_in(err):
    return [ln for ln in err.splitlines() if ln.startswith("warning: ")]


class TestAPromptThatNeverReceivesTheInput(Project):
    """render() replaces the exact text {{input}} and nothing else. With
    {{ input }}, {{Input}}, {input} or no placeholder the case input went
    nowhere, silently, and the suite passed: a test of nothing."""

    def check(self, prompt, cases=ONE_CASE, *more):
        url = self.model()
        root = self.project(cases, prompt=prompt)
        return self.run_suite(root, url, *more)

    def test_a_prompt_without_the_placeholder_warns_and_runs_on(self):
        code, out, err = self.check("Say hi.\n")
        self.assertEqual(code, 0, (out, err))          # the verdict is as it was
        self.assertIn("1/1 passed", out)
        lines = warnings_in(err)
        self.assertEqual(len(lines), 1, err)
        self.assertEqual(len(err.splitlines()), 1, err)
        self.assertIn("s.v1.txt", self.tidy(lines[0]))
        self.assertIn("{{input}}", lines[0])
        self.assertIn("never sent", lines[0])
        self.assertEqual(self.prompts, ["Say hi.\n"])

    def test_the_near_misses_each_warn_and_are_sent_as_written(self):
        for prompt in ("Reply to: {{ input }}\n", "Reply to: {{Input}}\n",
                       "Reply to: {{INPUT}}\n", "Reply to: {input}\n",
                       "Reply to: {{input }}\n", "Reply to: $input\n",
                       "Reply to: <input>\n"):
            with self.subTest(prompt=prompt):
                code, out, err = self.check(prompt)
                self.assertEqual(code, 0, (out, err))
                self.assertGreaterEqual(len(warnings_in(err)), 1, err)
                self.assertIn("s.v1.txt", self.tidy(err))
                self.assertIn("{{input}}", err)
                self.assertEqual(self.prompts, [prompt])

    def test_a_leftover_name_is_called_out_with_its_spelling(self):
        for prompt, shown in (("{{  input  }}\n", "{{  input  }}"),
                              ("{{INPUT}}\n", "{{INPUT}}")):
            with self.subTest(prompt=prompt):
                code, out, err = self.check(prompt)
                self.assertTrue(any(shown in ln for ln in warnings_in(err)),
                                err)

    def test_the_right_placeholder_is_silent(self):
        for prompt in ("Say: {{input}}\n", "{{input}}", "A {{input}} B {{input}}\n",
                       "{{{input}}}\n"):
            with self.subTest(prompt=prompt):
                code, out, err = self.check(prompt)
                self.assertEqual((code, err), (0, ""), (out, err))

    def test_a_var_the_case_declares_is_replaced_and_silent(self):
        code, out, err = self.check("{{tone}}: {{input}}\n", WITH_VARS)
        self.assertEqual((code, err), (0, ""), (out, err))
        self.assertEqual(self.prompts, ["terse: hi\n"])

    def test_a_var_nobody_declares_is_a_leftover(self):
        code, out, err = self.check("{{tone}}: {{input}}\n")
        self.assertEqual(code, 0, (out, err))
        lines = warnings_in(err)
        self.assertEqual(len(lines), 1, err)
        self.assertIn("{{tone}}", lines[0])
        self.assertIn("s.v1.txt", self.tidy(lines[0]))
        self.assertNotIn("never sent", lines[0])       # the input does arrive
        self.assertEqual(self.prompts, ["{{tone}}: hi\n"])

    def test_a_var_only_some_cases_declare_is_a_leftover_for_the_others(self):
        cases = (ONE_CASE
                 + "  - name: b\n    input: yo\n    vars:\n      tone: terse\n"
                   "    assert:\n      - contains: neutral\n")
        code, out, err = self.check("{{tone}}: {{input}}\n", cases)
        self.assertEqual(code, 0, (out, err))
        self.assertEqual(len(warnings_in(err)), 1, err)
        self.assertIn("{{tone}}", err)

    def test_one_line_per_problem_however_many_cases_there_are(self):
        cases = "cases:\n" + "".join(
            f"  - name: c{i}\n    input: hi {i}\n    assert:\n"
            f"      - contains: neutral\n" for i in range(5))
        code, out, err = self.check("{{ input }} {{tone}}\n", cases)
        self.assertEqual(code, 0, (out, err))
        self.assertEqual(len(warnings_in(err)), 2, err)
        self.assertEqual(len(err.splitlines()), 2, err)

    def test_double_braces_that_are_not_names_are_left_alone(self):
        code, out, err = self.check('Answer as {{"label": "x"}} for {{input}}\n')
        self.assertEqual((code, err), (0, ""), (out, err))

    def test_braces_in_the_case_input_are_not_a_leftover(self):
        cases = ('cases:\n  - name: a\n    input: "use {{name}} and {{ x }}"\n'
                 '    assert:\n      - contains: neutral\n')
        code, out, err = self.check("{{input}}\n", cases)
        self.assertEqual((code, err), (0, ""), (out, err))
        self.assertEqual(self.prompts, ["use {{name}} and {{ x }}\n"])

    def test_empty_inputs_need_no_placeholder(self):
        cases = ('cases:\n  - name: a\n    input: ""\n    assert:\n'
                 '      - contains: neutral\n')
        code, out, err = self.check("Tell me a joke.\n", cases)
        self.assertEqual((code, err), (0, ""), (out, err))

    def test_json_output_stays_clean_and_the_warning_goes_to_stderr(self):
        code, out, err = self.check("Say hi.\n", ONE_CASE, "--format", "json")
        self.assertEqual(code, 0, (out, err))
        self.assertEqual(json.loads(out)["passed"], 1)
        self.assertEqual(len(warnings_in(err)), 1, err)

    def test_a_failing_run_keeps_its_exit_code(self):
        cases = ONE_CASE.replace("neutral", "positive")
        code, out, err = self.check("Say hi.\n", cases)
        self.assertEqual(code, 1, (out, err))
        self.assertEqual(len(warnings_in(err)), 1, err)

    def test_every_version_file_is_named_in_run_all_and_in_diff(self):
        url = self.model()
        root = self.project(ONE_CASE, prompt="Say hi.\n", versions=("v1", "v2"))
        self.write(os.path.join(root, "prompts", "s.v2.txt"), "Say: {{input}}\n")
        code, out, err = run_cli(["--root", root, "--base-url", url, "run",
                                  "--all", "--no-cache"])
        self.assertEqual(code, 0, (out, err))
        lines = warnings_in(err)
        self.assertEqual(len(lines), 1, err)           # only v1 is wrong
        self.assertIn("s.v1.txt", self.tidy(lines[0]))
        code, out, err = run_cli(["--root", root, "--base-url", url, "diff",
                                  "v1", "v2", "--prompt", "s", "--no-cache"])
        self.assertEqual(code, 0, (out, err))
        lines = warnings_in(err)
        self.assertEqual(len(lines), 1, err)
        self.assertIn("s.v1.txt", self.tidy(lines[0]))

    def test_a_cached_run_warns_too(self):
        # the cache saves the model call, not the fact that nothing is sent
        url = self.model()
        root = self.project(ONE_CASE, prompt="Say hi.\n")
        argv = ["--root", root, "--base-url", url, "run", "--prompt", "s",
                "--version", "v1"]
        run_cli(argv)
        code, out, err = run_cli(argv)
        self.assertIn("(1 cached)", out)
        self.assertEqual(len(warnings_in(err)), 1, err)


class TestTemplateProblems(unittest.TestCase):
    """suite.template_problems: the check itself, without a model."""

    def cases(self, *vars_):
        return [{"name": f"c{i}", "input": "text", "vars": dict(v),
                 "assert": []} for i, v in enumerate(vars_ or [{}])]

    def test_nothing_to_say_about_a_good_template(self):
        self.assertEqual(
            suite_mod.template_problems("a {{input}} b", self.cases()), [])
        self.assertEqual(suite_mod.template_problems(
            "{{x}} {{input}}", self.cases({"x": "1"})), [])

    def test_what_it_reports(self):
        none = suite_mod.template_problems("plain", self.cases())
        self.assertEqual(len(none), 1)
        self.assertIn("{{input}}", none[0])
        both = suite_mod.template_problems("{{ input }}", self.cases())
        self.assertEqual(len(both), 2)
        self.assertTrue(any("{{ input }}" in p for p in both))
        left = suite_mod.template_problems("{{input}} {{a}} {{b}}",
                                           self.cases({"a": "1"}))
        self.assertEqual(len(left), 1)
        self.assertIn("{{b}}", left[0])
        self.assertNotIn("{{a}}", left[0])

    def test_a_long_list_of_leftovers_is_cut_short(self):
        template = "{{input}} " + " ".join("{{v%d}}" % i for i in range(9))
        (problem,) = suite_mod.template_problems(template, self.cases())
        self.assertIn("{{v0}}", problem)
        self.assertNotIn("{{v8}}", problem)
        self.assertIn("more", problem)
        self.assertNotIn("\n", problem)

    def test_render_itself_is_unchanged(self):
        self.assertEqual(suite_mod.render("{{ input }} {{input}}", "X"),
                         "{{ input }} X")
        self.assertEqual(suite_mod.render("{{tone}} {{input}}", "X",
                                          {"tone": "t"}), "t X")


# ---------------------------------------------------------------------
# #3 [LOW] - a fake server that is stopped is also closed

def _statement_lists(tree):
    for node in ast.walk(tree):
        for field in ("body", "orelse", "finalbody"):
            stmts = getattr(node, field, None)
            if isinstance(stmts, list) and stmts and isinstance(
                    stmts[0], ast.stmt):
                yield stmts


def _calls(stmt, method):
    """What `<what>.method()` is called on in this statement (as a dump of
    the expression), or None when the statement is something else."""
    if (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call)
            and isinstance(stmt.value.func, ast.Attribute)
            and stmt.value.func.attr == method):
        return ast.dump(stmt.value.func.value)
    return None


class TestEveryStoppedServerIsClosed(unittest.TestCase):
    """shutdown() stops serve_forever(); only server_close() releases the
    listening socket. 32 tests of test_runner.py left it open: a ResourceWarning
    per test, and, when the garbage collector got to a socket at the wrong
    moment, a failure of an unrelated test that reads stderr."""

    def test_shutdown_is_followed_by_server_close_on_the_same_server(self):
        here = os.path.dirname(os.path.abspath(__file__))
        loose = []
        for name in sorted(os.listdir(here)):
            if not (name.startswith("test_") and name.endswith(".py")):
                continue
            with open(os.path.join(here, name), encoding="utf-8") as f:
                tree = ast.parse(f.read(), name)
            for stmts in _statement_lists(tree):
                for i, stmt in enumerate(stmts):
                    server = _calls(stmt, "shutdown")
                    if server is None:
                        continue
                    after = stmts[i + 1] if i + 1 < len(stmts) else None
                    if after is None or _calls(after, "server_close") != server:
                        loose.append(f"{name}:{stmt.lineno}")
        self.assertEqual(loose, [], "shutdown() without server_close() "
                         "right after it")

    def test_the_check_itself_sees_a_loose_shutdown(self):
        tree = ast.parse("def f(server):\n    try:\n        pass\n"
                         "    finally:\n        server.shutdown()\n")
        found = [s for stmts in _statement_lists(tree) for s in stmts
                 if _calls(s, "shutdown") is not None]
        self.assertEqual(len(found), 1)
        tree = ast.parse("server.shutdown()\nserver.server_close()\n")
        (first, second) = tree.body
        self.assertEqual(_calls(first, "shutdown"),
                         _calls(second, "server_close"))
        tree = ast.parse("a.shutdown()\nb.server_close()\n")
        (first, second) = tree.body
        self.assertNotEqual(_calls(first, "shutdown"),
                            _calls(second, "server_close"))


# ---------------------------------------------------------------------
# #5 [LOW] - the README on --root and on the tests that skip

class TestReadmeIsTrueAboutRootAndSkips(unittest.TestCase):
    @staticmethod
    def readme():
        path = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "README.md")
        with open(path, encoding="utf-8") as f:
            return " ".join(f.read().split())

    def test_root_is_documented_and_the_help_explains_it(self):
        self.assertTrue("`--root DIR`" in self.readme(),
                        "the README does not name --root DIR")
        help_text = " ".join(cli_mod.build_parser().format_help().split())
        self.assertRegex(help_text, r"--root ROOT +[A-Z]\w+")   # has a help text
        self.assertIn("prompts/", help_text)

    def test_root_really_is_a_option_before_the_subcommand(self):
        root = tempfile.mkdtemp(prefix="prg-r7-")
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        code, out, err = run_cli(["--root", root, "init"])
        self.assertEqual(code, 0, (out, err))
        self.assertTrue(os.path.isfile(
            os.path.join(root, "prompts", "sentiment.v1.txt")))
        with self.assertRaises(SystemExit):
            run_cli(["init", "--root", root])        # not accepted after it

    def test_the_readme_gives_no_count_of_skipped_tests(self):
        text = self.readme()
        section = text.split("## Tests")[1].split("## License")[0]
        counted = re.search(r"(?i)\b(one|two|three|\d+) tests? skips?\b", text)
        self.assertIsNone(counted, "the README counts skipped tests: "
                          + (counted.group(0) if counted else ""))
        self.assertTrue("tomllib" in section, "the reason for a skip is gone")
        self.assertTrue("root" in section, "running as root is not named")


if __name__ == "__main__":
    unittest.main()
