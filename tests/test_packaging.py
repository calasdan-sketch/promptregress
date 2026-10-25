"""Packaging and documentation claims: the Python versions the project
says it supports (pyproject, README, CI) must agree with each other, and
the metadata PyPI will show must be what the repo really is. Reads the
files as text - no build, no network."""

import importlib
import os
import re
import unittest

try:
    import tomllib
except ImportError:  # Python 3.10 has no tomllib
    tomllib = None

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_URL = "https://github.com/calasdan-sketch/promptregress"


def read(*parts):
    with open(os.path.join(ROOT, *parts), encoding="utf-8") as f:
        return f.read()


def versions(text):
    """{(3, 10), (3, 11), ...}: every 3.N in the text."""
    return {tuple(int(n) for n in v.split("."))
            for v in re.findall(r"\b3\.\d+\b", text)}


def show(vs):
    return ", ".join(f"{a}.{b}" for a, b in sorted(vs)) or "none"


class TestPythonVersionClaims(unittest.TestCase):
    def test_every_python_claim_agrees(self):
        # The code needs 3.10 (tempfile ignore_cleanup_errors) and nothing
        # newer; 3.9 crashes. So: requires-python >=3.10, a classifier per
        # tested version, CI running every classified version, and the
        # README saying the same.
        pyproject, readme = read("pyproject.toml"), read("README.md")
        ci = read(".github", "workflows", "ci.yml")
        m = re.search(r'requires-python\s*=\s*">=(3\.\d+)"', pyproject)
        self.assertTrue(m, 'pyproject.toml needs requires-python = ">=3.N"')
        floor = min(versions(m.group(1)))
        classified = versions(" ".join(re.findall(
            r'"Programming Language :: Python :: (3\.\d+)"', pyproject)))
        in_ci = versions(" ".join(
            re.findall(r"python:\s*\[([^\]]*)\]", ci)
            + re.findall(r'^\s*(?:-\s*)?python:\s*"(3\.\d+)"', ci, re.M)))
        self.assertEqual(min(classified), floor,
                         f"requires-python floor {show({floor})} vs "
                         f"classifiers {show(classified)}")
        # CI tests the oldest and the newest classified Python, and some
        # in between; every version it runs is classified
        self.assertLessEqual(in_ci, classified,
                             f"CI runs {show(in_ci)}, classifiers say "
                             f"{show(classified)}")
        self.assertIn(floor, in_ci, "CI must run the oldest Python")
        self.assertIn(max(classified), in_ci,
                      "CI must run the newest classified Python")
        m = re.search(r"Python (3\.\d+)\+", readme)
        self.assertTrue(m, 'README needs a "Python 3.N+" install claim')
        self.assertEqual(min(versions(m.group(1))), floor,
                         "README install claim vs requires-python")
        ci_text = next((p for p in readme.split("\n\n")
                        if "installs the package and runs the test suite"
                        in p), "")
        self.assertEqual(versions(ci_text), in_ci,
                         f"README's CI sentence {show(versions(ci_text))} "
                         f"vs ci.yml {show(in_ci)}")
        example = re.search(r'python-version:\s*"(3\.\d+)"',
                            read("examples", "promptregress-example.yml"))
        self.assertTrue(example)
        self.assertLessEqual(versions(example.group(1)), classified)


class TestPackagingMetadata(unittest.TestCase):
    def test_license_file_matches_the_declared_license(self):
        license_text = read("LICENSE")
        self.assertTrue(license_text.startswith("MIT License"))
        self.assertIn("Permission is hereby granted, free of charge",
                      license_text)
        self.assertRegex(read("pyproject.toml"), r'(?m)^license\s*=\s*"MIT"')

    @unittest.skipIf(tomllib is None, "tomllib needs Python 3.11+")
    def test_pyproject_metadata(self):
        with open(os.path.join(ROOT, "pyproject.toml"), "rb") as f:
            meta = tomllib.load(f)
        project = meta["project"]
        # SPDX licence expression + licence file (PEP 639, setuptools>=77)
        self.assertEqual(project["license"], "MIT")
        self.assertEqual(project["license-files"], ["LICENSE"])
        self.assertIn("setuptools>=77", meta["build-system"]["requires"])
        classifiers = set(project["classifiers"])
        licence_classifiers = [c for c in classifiers
                               if c.startswith("License ::")]
        self.assertFalse(licence_classifiers,
                         "a License classifier conflicts with the SPDX field")
        for want in ("Development Status :: 4 - Beta",
                     "Environment :: Console",
                     "Intended Audience :: Developers",
                     "Operating System :: OS Independent",
                     "Topic :: Software Development :: Testing"):
            self.assertIn(want, classifiers)
        self.assertTrue(project["keywords"])
        self.assertEqual(project["urls"]["Homepage"], REPO_URL)
        self.assertEqual(project["urls"]["Issues"], REPO_URL + "/issues")
        # the console script points at a real callable
        module, _, attr = project["scripts"]["promptregress"].partition(":")
        self.assertTrue(callable(getattr(importlib.import_module(module),
                                         attr)))


class TestSourceDistribution(unittest.TestCase):
    """The tests are part of the sdist, so they must find what they read
    there: setuptools ships the package, the README, the licence and
    tests/test*.py by itself, but not the CI workflow and the example
    workflow that TestPythonVersionClaims reads. MANIFEST.in names them."""

    #: files the test suite opens that are not in an sdist by default
    READ_BY_THE_TESTS = (".github/workflows/ci.yml",
                         "examples/promptregress-example.yml")

    def manifest(self):
        lines = (ln.strip() for ln in read("MANIFEST.in").splitlines())
        return [ln.split() for ln in lines if ln and not ln.startswith("#")]

    def test_manifest_ships_what_the_tests_read(self):
        included = {args[1] for args in self.manifest()
                    if args[0] == "include" and len(args) == 2}
        for path in self.READ_BY_THE_TESTS:
            with self.subTest(path=path):
                self.assertIn(path, included)
                self.assertTrue(os.path.isfile(os.path.join(ROOT, path)))

    def test_every_line_of_the_manifest_names_something_that_exists(self):
        # a misspelt path is skipped by setuptools with a warning nobody reads
        for args in self.manifest():
            with self.subTest(line=" ".join(args)):
                self.assertIn(args[0], ("include", "recursive-include"))
                if args[0] == "include":
                    self.assertTrue(os.path.isfile(
                        os.path.join(ROOT, *args[1].split("/"))))
                else:
                    self.assertTrue(os.path.isdir(
                        os.path.join(ROOT, *args[1].split("/"))))


if __name__ == "__main__":
    unittest.main()
