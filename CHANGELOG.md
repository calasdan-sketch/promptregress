# Changelog

## 0.1.5

- A report or an error line that cannot be written (`> /dev/full`,
  `2>/dev/full`, a full disk) now really exits 2 with no traceback. In 0.1.4
  the last flush at the end of the run still failed and the exit code was 120.
  A closed pipe (`promptregress run | head -1`) still keeps the verdict as the
  exit code.
- The release workflow only publishes a tag that sits on `main`, and the build
  also runs the tests from the unpacked source distribution.

## 0.1.4

- `--base-url` (and `PROMPTREGRESS_BASE_URL`) must be an `http://` or
  `https://` address with a host. `file:`, `ftp:` and `data:` addresses were
  followed by urllib (a traceback or a read of the local disk); they now stop
  the run with exit 2, and the HTTP client only speaks HTTP and HTTPS.
- A base URL with a user name or password is refused (put the key in
  `PROMPTREGRESS_API_KEY`); it is never printed in an error.
- When stderr cannot be written (full disk) the exit code is 2, not 1.
- `json_valid` and `schema` read strict JSON: `NaN` and `Infinity` are not
  JSON. `type: integer` accepts `1.0`, as JSON Schema does. The README now
  says how `regex` behaves (`$` before a final newline, no time limit).
- A very long malformed line in a cases file is cut in the error message.
- CI on Linux, Windows and macOS with Python 3.10, 3.12 and 3.14, a package
  job (build, `twine check --strict`, install the wheel) and a release
  workflow that publishes to PyPI from a `v*` tag.
- `pip install promptregress`. Python 3.14 is a declared classifier.

## 0.1.3

- YAML-only escapes in quoted strings are decoded or refused, a prompt that
  never receives the case input warns, and the source distribution carries
  its tests.

## 0.1.2

- An unwritable cache is skipped with a warning, file errors exit 2,
  indentless YAML lists are read, `enum` compares as JSON does.

## 0.1.1 and earlier

- Empty suites fail, keys never leak, a whole-request timeout and a reply
  cap, assertions compare the text as written.
