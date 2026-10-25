"""Machine-readable reports for `run`: JUnit XML and SARIF 2.1.0.

Both take the same input as the human report: a list of
(prompt name, version, [CaseResult]) in the order they ran.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import urllib.parse
import xml.etree.ElementTree as ET

from . import suite as suite_mod

SARIF_SCHEMA = "https://json.schemastore.org/sarif-2.1.0.json"
INFORMATION_URI = "https://github.com/calasdan-sketch/promptregress"
#: Longest piece of model output put into a report message.
OUTPUT_EXCERPT = 300

# XML 1.0 allows tab, newline, carriage return and the ranges below; a model
# can answer with any character (an escape code, a NUL), and one such
# character makes the whole file unreadable for the CI that parses it.
_XML_BAD = re.compile("[^\x09\x0a\x0d\x20-퟿-�"
                      "\U00010000-\U0010ffff]")


def xml_text(text: str) -> str:
    """`text` with every character XML 1.0 cannot hold replaced by U+FFFD."""
    return _XML_BAD.sub("�", text)


def _excerpt(text: str) -> str:
    return text[:OUTPUT_EXCERPT] + ("..." if len(text) > OUTPUT_EXCERPT
                                    else "")


def _failure_lines(result) -> "list[str]":
    lines = [f"✗ {a.assertion} -> {a.detail}"
             for a in result.assertions if not a.passed]
    if result.output:
        lines.append(f"output: {_excerpt(result.output)!r}")
    return lines


def junit_xml(suites, name: str = "promptregress") -> str:
    """JUnit XML (the dialect CI systems read): one <testsuite> per prompt
    version, one <testcase> per case, a <failure> per failing case and an
    <error> where the model could not answer. Pure ASCII (non-ASCII text is
    written as character references), so it survives any output encoding."""
    root = ET.Element("testsuites", name=name)
    all_tests = all_fail = all_err = 0
    all_time = 0.0
    for prompt, version, results in suites:
        label = f"{prompt}.{version}"
        failures = sum(1 for r in results if not r.passed and not r.error)
        errors = sum(1 for r in results if r.error)
        seconds = sum(r.seconds for r in results)
        node = ET.SubElement(
            root, "testsuite", name=xml_text(label), tests=str(len(results)),
            failures=str(failures), errors=str(errors), skipped="0",
            time=f"{seconds:.3f}")
        for r in results:
            case = ET.SubElement(node, "testcase", classname=xml_text(label),
                                 name=xml_text(r.name),
                                 time=f"{r.seconds:.3f}")
            if r.error:
                err = ET.SubElement(case, "error",
                                    message=xml_text(r.error[:200]),
                                    type="ModelError")
                err.text = xml_text(r.error)
            elif not r.passed:
                bad = [a for a in r.assertions if not a.passed]
                first = (f"{bad[0].assertion} -> {bad[0].detail}" if bad
                         else "case failed")
                fail = ET.SubElement(case, "failure",
                                     message=xml_text(first[:200]),
                                     type="AssertionFailed")
                fail.text = xml_text("\n".join(_failure_lines(r)))
        all_tests += len(results)
        all_fail += failures
        all_err += errors
        all_time += seconds
    root.set("tests", str(all_tests))
    root.set("failures", str(all_fail))
    root.set("errors", str(all_err))
    root.set("time", f"{all_time:.3f}")
    return ET.tostring(root, encoding="us-ascii",
                       xml_declaration=True).decode("ascii") + "\n"


# -- SARIF -------------------------------------------------------------------

def _assertion_kind(assertion) -> str:
    if isinstance(assertion, dict) and assertion:
        return str(next(iter(assertion)))
    return "assertion"


def _case_line(root: str, prompt: str, case_name: str) -> int:
    """Line of `name: <case>` in the cases file (1 when it cannot be told)."""
    path = os.path.join(root, suite_mod.PROMPTS_DIR, f"{prompt}.cases.yaml")
    try:
        with open(path, encoding="utf-8-sig") as f:
            for number, line in enumerate(f, 1):
                m = re.match(r"\s*(?:-\s+)?name\s*:\s*(.*?)\s*$", line)
                if m and m.group(1).strip("'\"") == case_name:
                    return number
    except (OSError, ValueError):
        pass
    return 1


def sarif_log(suites, version: str, root: str = ".") -> dict:
    """SARIF 2.1.0 for the failing cases: one result per failed assertion
    (or per case the model could not answer), located at the case in its
    cases file."""
    rules: "list[dict]" = []
    index: "dict[str, int]" = {}

    def rule(kind: str) -> int:
        rid = f"promptregress/{kind}"
        if rid not in index:
            index[rid] = len(rules)
            if kind == "model-error":
                text = "The model could not answer, so the case was not judged."
            else:
                text = f"An assertion of type '{kind}' did not hold."
            rules.append({
                "id": rid, "name": kind,
                "shortDescription": {"text": text},
                "help": {"text": "Compare the model output in the message "
                                 "with the assertion in the cases file."},
                "defaultConfiguration": {"level": "error"},
            })
        return index[rid]

    results = []
    for prompt, ver, cases in suites:
        uri = urllib.parse.quote(
            f"{suite_mod.PROMPTS_DIR}/{prompt}.cases.yaml", safe="/")
        for r in cases:
            if r.passed and not r.error:
                continue
            line = _case_line(root, prompt, r.name)
            found = ([("model-error", r.error)] if r.error else
                     [(_assertion_kind(a.assertion),
                       f"{a.assertion} -> {a.detail}")
                      for a in r.assertions if not a.passed])
            for kind, text in found:
                message = f"{prompt}.{ver}, case '{r.name}': {text}"
                if r.output and not r.error:
                    message += f" (output: {_excerpt(r.output)!r})"
                key = "\0".join((prompt, ver, r.name, kind, text))
                results.append({
                    "ruleId": f"promptregress/{kind}",
                    "ruleIndex": rule(kind),
                    "level": "error",
                    "message": {"text": message},
                    "locations": [{"physicalLocation": {
                        "artifactLocation": {"uri": uri,
                                             "uriBaseId": "%SRCROOT%"},
                        "region": {"startLine": line}}}],
                    "partialFingerprints": {"promptregress/v1": hashlib.sha1(
                        key.encode("utf-8")).hexdigest()},
                    "properties": {"prompt": prompt, "version": ver,
                                   "case": r.name},
                })
    return {
        "$schema": SARIF_SCHEMA,
        "version": "2.1.0",
        "runs": [{
            "tool": {"driver": {"name": "promptregress", "version": version,
                                "informationUri": INFORMATION_URI,
                                "rules": rules}},
            "originalUriBaseIds": {"%SRCROOT%": {"description": {
                "text": "The --root directory of the run."}}},
            "results": results,
            "invocations": [{"executionSuccessful": True}],
        }],
    }


def sarif_text(suites, version: str, root: str = ".") -> str:
    return json.dumps(sarif_log(suites, version, root), indent=2) + "\n"
