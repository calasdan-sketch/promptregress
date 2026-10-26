"""A pytest plugin: every golden case becomes a pytest test.

Needs pytest (`pip install promptregress pytest`); promptregress itself does
not depend on it. Switch it on with `--promptregress`:

    pytest --promptregress                     # latest version of each prompt
    pytest --promptregress --promptregress-version v2
    pytest --promptregress --promptregress-demo  # built-in fake model

pytest then collects prompts/<name>.cases.yaml and runs each case against a
version of its prompt, as a test called
`prompts/sentiment.cases.yaml::v2::angry review`. The model is the one the
promptregress command line uses (PROMPTREGRESS_BASE_URL, PROMPTREGRESS_MODEL,
PROMPTREGRESS_API_KEY, or the options below).

Nothing happens without `--promptregress`: the cases make real model
requests, which can cost money, so they never run by surprise.
"""

from __future__ import annotations

import contextlib
import os

try:
    import pytest
except ImportError as e:   # pragma: no cover - the message is the point
    raise ImportError("promptregress.pytest needs pytest: "
                      "pip install pytest") from e

from . import demo as demo_mod
from . import model as model_mod
from . import runner as runner_mod
from . import suite as suite_mod

_SUFFIX = ".cases.yaml"
_STATE = "_promptregress"
#: The hooks below use the pathlib hook arguments of pytest 7. pytest loads
#: this module by itself whenever promptregress is installed (an entry
#: point), so on an older pytest it must do nothing rather than make every
#: pytest run fail on a hook it does not know.
SUPPORTED = tuple(getattr(pytest, "version_tuple", (0,))[:2]) >= (7, 0)


def pytest_addoption(parser):
    if not SUPPORTED:
        return
    group = parser.getgroup("promptregress", "prompt regression tests")
    group.addoption("--promptregress", action="store_true", default=False,
                    help="Run the golden cases in prompts/*.cases.yaml as "
                         "tests (they call the model).")
    group.addoption("--promptregress-version", action="store", default=None,
                    metavar="vN",
                    help="Prompt version to test, or 'all' (default: the "
                         "latest version of each prompt).")
    group.addoption("--promptregress-demo", action="store_true",
                    default=False,
                    help="Use the built-in fake model (only knows the "
                         "`promptregress init` example).")
    group.addoption("--promptregress-no-cache", action="store_true",
                    default=False, help="Ask the model again, ignore the "
                                        "response cache.")
    group.addoption("--promptregress-model", action="store", default=None,
                    help="Model name (env PROMPTREGRESS_MODEL).")
    group.addoption("--promptregress-base-url", action="store", default=None,
                    help="OpenAI-compatible endpoint (env "
                         "PROMPTREGRESS_BASE_URL).")
    group.addoption("--promptregress-timeout", action="store", default=None,
                    type=float, metavar="SECONDS",
                    help="Seconds for one model request.")


class _Session:
    """What the whole pytest run shares."""

    def __init__(self, config):
        self.stack = contextlib.ExitStack()
        self.model_kw: dict = {}
        self.broken = ""        # why the model cannot be used, once known
        opts = config.option
        if opts.promptregress_demo:
            server, url = demo_mod.start_demo_model()
            self.stack.callback(server.server_close)
            self.stack.callback(server.shutdown)
            self.model_kw = {"base_url": url, "api_key": "demo",
                             "timeout": model_mod.DEFAULT_TIMEOUT,
                             "max_reply_bytes":
                                 model_mod.DEFAULT_MAX_REPLY_BYTES}
        else:
            for key, value in (("model", opts.promptregress_model),
                               ("base_url", opts.promptregress_base_url),
                               ("timeout", opts.promptregress_timeout)):
                if value is not None:
                    self.model_kw[key] = value
        # a demo answer must never be served to a later real run, or back
        self.use_cache = not (opts.promptregress_no_cache
                              or opts.promptregress_demo)


def pytest_configure(config):
    if SUPPORTED and config.getoption("promptregress", default=False):
        setattr(config, _STATE, _Session(config))


def pytest_unconfigure(config):
    session = getattr(config, _STATE, None)
    if session is not None:
        session.stack.close()
        delattr(config, _STATE)


def pytest_collect_file(file_path, parent):
    if getattr(parent.config, _STATE, None) is None:
        return None
    path = os.fspath(file_path)
    name = os.path.basename(path)
    if not name.endswith(_SUFFIX):
        return None
    if os.path.basename(os.path.dirname(path)) != suite_mod.PROMPTS_DIR:
        return None
    return CasesFile.from_parent(parent, path=file_path)


class PromptRegressFailure(Exception):
    """A case that did not pass; the message is the report."""


class CasesFile(pytest.File):
    def collect(self):
        path = os.fspath(self.path)
        root = os.path.dirname(os.path.dirname(path))
        prompt = os.path.basename(path)[:-len(_SUFFIX)]
        try:
            cases = suite_mod.load_cases(root, prompt)
        except (OSError, ValueError) as e:
            pytest.fail(runner_mod.error_text(e), pytrace=False)
        wanted = self.config.option.promptregress_version
        have = suite_mod.list_prompts(root).get(prompt, [])
        if not have:
            pytest.fail(f"{path}: no prompt file prompts/{prompt}.vN.txt "
                        "next to the cases", pytrace=False)
        if wanted == "all":
            versions = have
        elif wanted:
            if wanted not in have:
                return          # this prompt has no such version: nothing
            versions = [wanted]
        else:
            versions = have[-1:]
        for version in versions:
            text = suite_mod.load_prompt(root, prompt, version)
            for case in cases:
                yield CaseItem.from_parent(
                    self, name=f"{version}::{case['name']}", root=root,
                    prompt=prompt, version=version, prompt_text=text,
                    case=case)


class CaseItem(pytest.Item):
    def __init__(self, *, root, prompt, version, prompt_text, case, **kw):
        super().__init__(**kw)
        self.root, self.prompt, self.version = root, prompt, version
        self.prompt_text, self.case = prompt_text, case

    def runtest(self):
        session = getattr(self.config, _STATE)
        if session.broken:
            pytest.skip(f"the model could not be used earlier: "
                        f"{session.broken}")
        result = runner_mod.run_case(
            self.root, self.prompt, self.version, self.prompt_text,
            self.case, session.model_kw, session.use_cache)
        if result.error:
            session.broken = result.error
            raise PromptRegressFailure(f"the model could not answer: "
                                       f"{result.error}")
        if not result.passed:
            lines = [f"{a.assertion} -> {a.detail}"
                     for a in result.assertions if not a.passed]
            lines.append(f"output: {result.output[:300]!r}")
            raise PromptRegressFailure("\n".join(lines))

    def repr_failure(self, excinfo, style=None):
        if isinstance(excinfo.value, PromptRegressFailure):
            return (f"{self.prompt}.{self.version}, case "
                    f"{self.case['name']!r}:\n{excinfo.value}")
        return super().repr_failure(excinfo, style)

    def reportinfo(self):
        return self.path, 0, f"{self.prompt}.{self.version}: {self.name}"
