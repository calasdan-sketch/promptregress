"""promptregress: pytest for prompts, local-first.

Prompts are versioned files; golden cases live in YAML. Run them in CI,
diff versions A/B, and catch prompt drift before your users do.

  python3 -m promptregress init
  python3 -m promptregress --demo run --prompt summarize --version v1
  python3 -m promptregress diff v1 v2 --prompt summarize
  python3 -m promptregress list

(--demo trials the suite against a built-in fake model; after
`pip install .` the `promptregress` command is on your PATH.)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile

from . import demo as demo_mod
from . import runner as runner_mod
from . import suite as suite_mod

VERSION = "0.1.0"


def _model_kw(args) -> dict:
    kw: dict = {}
    if args.model:
        kw["model"] = args.model
    if args.base_url:
        kw["base_url"] = args.base_url
    if args.api_key:
        kw["api_key"] = args.api_key
    return kw


INIT_PROMPT_V1 = """Classify the sentiment of the following review.
Reply with exactly one word: positive, negative, or neutral.

Review: {{input}}
"""

INIT_PROMPT_V2 = """You are a sentiment classifier. Read the review and reply with
exactly one word: positive, negative, or neutral. No other text.

Review: {{input}}
"""

INIT_CASES = """cases:
  - name: angry review
    input: "This broke after two days. Terrible quality."
    assert:
      - contains: negative
  - name: happy review
    input: "Absolutely love it, works perfectly!"
    assert:
      - contains: positive
  - name: json shape
    input: "It's fine, nothing special."
    assert:
      - regex: (?i)\\bpositive\\b|\\bnegative\\b|\\bneutral\\b
      - max_tokens: 10
"""


def scaffold_example(root: str) -> list[str]:
    """Write the example prompts/ scaffold. Returns files written."""
    d = os.path.join(root, "prompts")
    os.makedirs(d, exist_ok=True)
    written = []
    for fn, content in [("summarize.v1.txt", INIT_PROMPT_V1),
                        ("summarize.v2.txt", INIT_PROMPT_V2),
                        ("summarize.cases.yaml", INIT_CASES)]:
        p = os.path.join(d, fn)
        if not os.path.exists(p):
            with open(p, "w") as f:
                f.write(content)
            written.append(fn)
    return written


def cmd_init(args) -> int:
    written = scaffold_example(args.root)
    if written:
        print(f"scaffolded prompts/: {', '.join(written)}")
    else:
        print("prompts/ already scaffolded")
    print("Point PROMPTREGRESS_BASE_URL at your model "
          "(default: Ollama at http://localhost:11434/v1).")
    return 0


def _demo_root(args) -> str:
    """Root for a --demo run. Uses the real prompts/ when the requested
    prompt+version exists there; otherwise scaffolds the example into a
    temp dir so the trial works with zero setup."""
    have = suite_mod.list_prompts(args.root).get(args.prompt, [])
    if args.cmd == "run" and args.version in have:
        return args.root
    if args.cmd == "diff" and args.v1 in have and args.v2 in have:
        return args.root
    tmp = tempfile.mkdtemp(prefix="promptregress-demo-")
    scaffold_example(tmp)
    print(f"(demo: scaffolded example prompts in {tmp})")
    return tmp


def cmd_list(args) -> int:
    prompts = suite_mod.list_prompts(args.root)
    if not prompts:
        print("no prompts found — run `promptregress init`")
        return 0
    for name, versions in prompts.items():
        try:
            n_cases = len(suite_mod.load_cases(args.root, name))
        except FileNotFoundError:
            n_cases = 0
        print(f"{name}: versions {', '.join(versions)} "
              f"({n_cases} cases)")
    return 0


def _print_run(name: str, version: str,
               results: list[runner_mod.CaseResult], fmt: str) -> int:
    if fmt == "json":
        print(json.dumps({
            "prompt": name, "version": version,
            "passed": sum(1 for r in results if r.passed),
            "failed": sum(1 for r in results if not r.passed),
            "cases": [r.as_dict() for r in results],
        }, indent=2))
        return 0 if all(r.passed for r in results) else 1
    # pytest-style dots
    line = "".join("." if r.passed else "F" for r in results)
    print(f"{name}.{version}: {line}")
    failed = [r for r in results if not r.passed]
    for r in failed:
        print(f"\n  FAIL {r.name}")
        if r.error:
            print(f"    error: {r.error}")
            continue
        for a in r.assertions:
            if not a.passed:
                print(f"    ✗ {a.assertion} -> {a.detail}")
        print(f"    output: {r.output[:300]!r}")
    n_pass = len(results) - len(failed)
    cached = sum(1 for r in results if r.cached)
    print(f"\n{n_pass}/{len(results)} passed"
          + (f" ({cached} cached)" if cached else ""))
    return 0 if not failed else 1


def cmd_run(args) -> int:
    root = _demo_root(args) if args.demo else args.root
    try:
        results, _ = runner_mod.run_suite(
            root, args.prompt, args.version,
            model_kw=_model_kw(args), use_cache=not args.no_cache)
    except (FileNotFoundError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    if any(r.error and not r.cached for r in results):
        # model unreachable — report, don't pretend
        print("error: model unreachable; "
              "set PROMPTREGRESS_BASE_URL / --base-url", file=sys.stderr)
        return 2
    return _print_run(args.prompt, args.version, results, args.format)


def cmd_diff(args) -> int:
    root = _demo_root(args) if args.demo else args.root
    kw = _model_kw(args)
    try:
        r1, _ = runner_mod.run_suite(root, args.prompt, args.v1,
                                     model_kw=kw,
                                     use_cache=not args.no_cache)
        r2, _ = runner_mod.run_suite(root, args.prompt, args.v2,
                                     model_kw=kw,
                                     use_cache=not args.no_cache)
    except (FileNotFoundError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    if any(r.error and not r.cached for rs in (r1, r2) for r in rs):
        # model unreachable — report, don't pretend with an all-FAIL table
        print("error: model unreachable; "
              "set PROMPTREGRESS_BASE_URL / --base-url", file=sys.stderr)
        return 2
    rows = []
    for a, b in zip(r1, r2):
        changed = a.passed != b.passed
        rows.append({
            "name": a.name,
            "v1": "pass" if a.passed else "FAIL",
            "v2": "pass" if b.passed else "FAIL",
            "changed": changed,
            # before/after output text on flips (truncated like run's)
            "v1_output": a.output[:300],
            "v2_output": b.output[:300],
        })
    if args.format == "json":
        print(json.dumps({
            "prompt": args.prompt, "v1": args.v1, "v2": args.v2,
            "cases": [{"name": r["name"], "v1": r["v1"], "v2": r["v2"],
                       "changed": r["changed"],
                       "v1_output": r["v1_output"],
                       "v2_output": r["v2_output"]} for r in rows],
        }, indent=2))
        return 0
    w = max([len(r["name"]) for r in rows] + [4])
    print(f"{'case':<{w}}  {args.v1:>6}  {args.v2:>6}")
    for r in rows:
        mark = "← CHANGED" if r["changed"] else " "
        print(f"{r['name']:<{w}}  {r['v1']:>6}  {r['v2']:>6}  {mark}")
    flipped = [r for r in rows if r["changed"]]
    for r in flipped:
        print(f"\nCHANGED {r['name']}:")
        print(f"  {args.v1} output: {r['v1_output']!r}")
        print(f"  {args.v2} output: {r['v2_output']!r}")
    print(f"\n{len(flipped)}/{len(rows)} case(s) changed verdict "
          f"{args.v1} -> {args.v2}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="promptregress", description=__doc__)
    p.add_argument("--version", action="version", version=f"%(prog)s {VERSION}")
    p.add_argument("--root", default=os.getcwd())
    p.add_argument("--model", default=None)
    p.add_argument("--base-url", default=None)
    p.add_argument("--api-key", default=None)
    p.add_argument("--demo", action="store_true",
                   help="Trial mode: run against the built-in in-process "
                        "fake model instead of a real endpoint. "
                        "Scaffolds the example prompts if the requested "
                        "prompt is missing.")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("init", help="Scaffold prompts/ with an example.")
    sub.add_parser("list", help="List prompts, versions, case counts.")

    r = sub.add_parser("run", help="Run golden cases against a version.")
    r.add_argument("--prompt", required=True)
    r.add_argument("--version", required=True)
    r.add_argument("--no-cache", action="store_true")
    r.add_argument("--format", choices=["human", "json"], default="human")

    d = sub.add_parser("diff", help="A/B two versions on the same cases.")
    d.add_argument("v1")
    d.add_argument("v2")
    d.add_argument("--prompt", required=True)
    d.add_argument("--no-cache", action="store_true")
    d.add_argument("--format", choices=["human", "json"], default="human")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    demo_server = None
    if args.demo and args.cmd in ("run", "diff"):
        # Zero-dependency trial: in-process fake model. An explicit
        # --base-url (or env) still wins — explicit intent beats the demo.
        demo_server, url = demo_mod.start_demo_model()
        if args.base_url is None and "PROMPTREGRESS_BASE_URL" not in os.environ:
            os.environ["PROMPTREGRESS_BASE_URL"] = url
    try:
        return {"init": cmd_init, "list": cmd_list,
                "run": cmd_run, "diff": cmd_diff}[args.cmd](args)
    finally:
        if demo_server is not None:
            demo_server.shutdown()


if __name__ == "__main__":
    sys.exit(main())
