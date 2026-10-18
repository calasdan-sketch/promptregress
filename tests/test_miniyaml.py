"""Regression tests for miniyaml hardening (anchors/aliases/tags, depth
limit, strict malformed-YAML rejection). Mirrored in pr-verify and
prompt-regress against their verbatim miniyaml copies."""

import unittest

from promptregress import miniyaml


def _deep(nesting: int) -> str:
    return (
        "".join(f"{'  ' * i}k{i}:\n" for i in range(nesting))
        + "  " * nesting
        + "v: 1\n"
    )


class TestAnchorsAliasesTags(unittest.TestCase):
    def test_anchor_value_rejected(self):
        with self.assertRaisesRegex(ValueError, r"anchors \(&\).*line 2"):
            miniyaml.safe_load("a:\n  b: &x")

    def test_anchor_on_nested_block_rejected(self):
        with self.assertRaisesRegex(ValueError, r"anchors \(&\)"):
            miniyaml.safe_load("a: &x\n  b: 1")

    def test_anchor_list_item_rejected(self):
        with self.assertRaisesRegex(ValueError, r"anchors \(&\)"):
            miniyaml.safe_load("- &x foo")

    def test_alias_value_rejected(self):
        with self.assertRaisesRegex(ValueError, r"aliases \(\*\)"):
            miniyaml.safe_load("a: *x")

    def test_alias_list_item_rejected(self):
        with self.assertRaisesRegex(ValueError, r"aliases \(\*\)"):
            miniyaml.safe_load("- *x")

    def test_tag_value_rejected(self):
        with self.assertRaisesRegex(ValueError, r"tags \(!\)"):
            miniyaml.safe_load("a: !tag foo")

    def test_tag_shorthand_rejected(self):
        with self.assertRaisesRegex(ValueError, r"tags \(!\)"):
            miniyaml.safe_load("a: !!str foo")

    def test_anchors_never_silently_dropped(self):
        # Before the fix, "a: &x" parsed as the string "&x".
        with self.assertRaises(ValueError):
            miniyaml.safe_load("a: &x")


class TestDepthLimit(unittest.TestCase):
    def test_600_deep_raises_value_error_not_recursion_error(self):
        try:
            miniyaml.safe_load(_deep(600))
        except ValueError as e:
            self.assertIn("nesting depth", str(e))
            self.assertIn("(line ", str(e))
        except RecursionError:
            self.fail("RecursionError leaked; expected ValueError")
        else:
            self.fail("expected ValueError for 600-deep nesting")

    def test_shallow_nesting_still_parses(self):
        doc = miniyaml.safe_load(_deep(40))
        d = doc
        for i in range(40):
            d = d[f"k{i}"]
        self.assertEqual(d, {"v": 1})


class TestStrictMalformed(unittest.TestCase):
    def test_unclosed_flow_list_rejected(self):
        with self.assertRaisesRegex(ValueError, r"unclosed flow list.*line 1"):
            miniyaml.safe_load("a: [x")

    def test_embedded_mapping_indicator_rejected(self):
        with self.assertRaisesRegex(ValueError, r"unexpected ':'.*line 2"):
            miniyaml.safe_load("x: 1\na: b: c")

    def test_flow_map_rejected(self):
        with self.assertRaisesRegex(ValueError, r"flow-style maps.*line 1"):
            miniyaml.safe_load("a: {x: y}")

    def test_block_scalar_indent_indicator_rejected(self):
        with self.assertRaisesRegex(ValueError,
                                    r"indentation indicators.*line 1"):
            miniyaml.safe_load("a: |2\n  x\n")

    def test_block_scalar_underindented_line_rejected(self):
        with self.assertRaisesRegex(ValueError, r"less indented.*line 3"):
            miniyaml.safe_load("a: |\n    x\n   y\n")

    def test_unterminated_quote_rejected(self):
        with self.assertRaisesRegex(ValueError, r"unterminated quoted"):
            miniyaml.safe_load('a: "oops')

    def test_duplicate_key_rejected(self):
        with self.assertRaisesRegex(ValueError, r"duplicate key.*line 2"):
            miniyaml.safe_load("a: 1\na: 2")

    def test_valid_lookalikes_still_parse(self):
        self.assertEqual(miniyaml.safe_load("u: http://x"), {"u": "http://x"})
        self.assertEqual(miniyaml.safe_load("t: 12:30"), {"t": "12:30"})
        self.assertEqual(miniyaml.safe_load("t: b:c"), {"t": "b:c"})
        self.assertEqual(miniyaml.safe_load("g: 'a: b'"), {"g": "a: b"})


class TestQuotedScalars(unittest.TestCase):
    # Double-quoted scalars take YAML's escapes (PyYAML agrees on these);
    # the whole table is in test_round7_fixes.py.
    def test_double_quoted_escapes(self):
        self.assertEqual(miniyaml.safe_load('a: "line1\\nline2"'),
                         {"a": "line1\nline2"})
        self.assertEqual(miniyaml.safe_load('a: "tab\\tsep"'), {"a": "tab\tsep"})
        self.assertEqual(miniyaml.safe_load('a: "say \\"hi\\" \\\\ done"'),
                         {"a": 'say "hi" \\ done'})
        self.assertEqual(miniyaml.safe_load('a: "caf\\u00e9"'), {"a": "caf\u00e9"})
        self.assertEqual(miniyaml.safe_load('- "x\\ny"'), ["x\ny"])
        self.assertEqual(miniyaml.safe_load('a: ["p\\nq", r]'), {"a": ["p\nq", "r"]})

    def test_yaml_only_escapes_and_single_quotes(self):
        # an escape JSON does not know but YAML does (\e is ESC) is decoded,
        # not kept as backslash text
        self.assertEqual(miniyaml.safe_load('a: "odd \\e esc \\"q\\""'),
                         {"a": 'odd \x1b esc "q"'})
        # single quotes are literal, as in YAML
        self.assertEqual(miniyaml.safe_load("a: 'no \\n escape'"),
                         {"a": "no \\n escape"})
        self.assertEqual(miniyaml.safe_load('a: "# not a comment"'),
                         {"a": "# not a comment"})


class TestSingleQuoteEscape(unittest.TestCase):
    # YAML: inside 'single quotes' a doubled apostrophe ('') is one ('),
    # so 'it''s' is it's. It used to come back as it''s. Expected values
    # match PyYAML 6. Nothing else about single quotes changes.
    def test_doubled_quote_is_one_apostrophe(self):
        load = miniyaml.safe_load
        self.assertEqual(load("a: 'it''s'"), {"a": "it's"})
        self.assertEqual(load("a: 'say ''hi'''"), {"a": "say 'hi'"})
        self.assertEqual(load("a: ''''"), {"a": "'"})
        self.assertEqual(load("a: '''x'"), {"a": "'x"})
        self.assertEqual(load("a: 'x'''"), {"a": "x'"})
        self.assertEqual(load("a: ''"), {"a": ""})

    def test_in_lists_and_inline_maps(self):
        load = miniyaml.safe_load
        self.assertEqual(load("- 'it''s'"), ["it's"])
        self.assertEqual(load("- key: 'it''s'\n  other: 1"),
                         [{"key": "it's", "other": 1}])
        self.assertEqual(load("a: ['it''s', 'b, c', d]"),
                         {"a": ["it's", "b, c", "d"]})

    def test_doubled_quote_does_not_end_the_scalar(self):
        # the '#' after '' is still inside the quotes; the one after the
        # closing quote starts a comment
        load = miniyaml.safe_load
        self.assertEqual(load("a: 'it''s # not a comment' # a comment"),
                         {"a": "it's # not a comment"})
        self.assertEqual(load("a: 'it''s: fine'"), {"a": "it's: fine"})

    def test_nothing_else_changes(self):
        load = miniyaml.safe_load
        # backslashes stay literal in single quotes
        self.assertEqual(load("a: 'no \\n escape'"), {"a": "no \\n escape"})
        # a double-quoted scalar keeps '' as two apostrophes
        self.assertEqual(load("a: \"it''s\""), {"a": "it''s"})
        # double quotes inside single quotes, and a single apostrophe
        # inside double quotes, are untouched
        self.assertEqual(load("a: 'say \"hi\"'"), {"a": 'say "hi"'})
        self.assertEqual(load("a: \"it's\""), {"a": "it's"})
        # a plain (unquoted) scalar is not single-quoted
        self.assertEqual(load("a: it''s"), {"a": "it''s"})


class TestBlockScalars(unittest.TestCase):
    # Expected values match PyYAML 6 on the same documents.
    def test_literal_clip(self):
        self.assertEqual(miniyaml.safe_load("a: |\n  line1\n  line2\n"),
                         {"a": "line1\nline2\n"})

    def test_literal_strip_and_keep(self):
        doc = "a: |-\n  x\n  y\n\n\nb: |+\n  x\n\n\nc: 1\n"
        self.assertEqual(miniyaml.safe_load(doc),
                         {"a": "x\ny", "b": "x\n\n\n", "c": 1})

    def test_literal_keeps_hashes_colons_blanks_indent(self):
        doc = ("a: |\n  # not a comment\n  x: y\n\n    nested\n"
               "  last\nb: 2\n")
        self.assertEqual(miniyaml.safe_load(doc), {
            "a": "# not a comment\nx: y\n\n  nested\nlast\n", "b": 2})

    def test_folded(self):
        doc = "a: >\n  one\n  two\n\n  three\n    indented\n  back\n"
        self.assertEqual(miniyaml.safe_load(doc),
                         {"a": "one two\nthree\n  indented\nback\n"})

    def test_folded_strip_and_keep(self):
        self.assertEqual(miniyaml.safe_load("a: >-\n  one\n  two\n"),
                         {"a": "one two"})
        self.assertEqual(miniyaml.safe_load("a: >+\n  one\n  two\n\n"),
                         {"a": "one two\n\n"})

    def test_list_items_and_inline_map_items(self):
        self.assertEqual(miniyaml.safe_load("- |\n  item\n  text\n- plain\n"),
                         ["item\ntext\n", "plain"])
        self.assertEqual(
            miniyaml.safe_load("- key: >\n    folded\n    text\n  other: 1\n"),
            [{"key": "folded text\n", "other": 1}])

    def test_empty_block_and_header_comment(self):
        self.assertEqual(miniyaml.safe_load("a: |\nb: 1\n"),
                         {"a": "", "b": 1})
        self.assertEqual(miniyaml.safe_load("a: |  # note\n  text\n"),
                         {"a": "text\n"})

    def test_crlf_input(self):
        self.assertEqual(miniyaml.safe_load("a: |\r\n  x\r\n  y\r\n"),
                         {"a": "x\ny\n"})

    def test_literal_keeps_trailing_spaces(self):
        # Body lines were rstrip()ped, dropping trailing spaces (and the
        # spaces of a whitespace-only line that is more indented than the
        # block). Expected values match PyYAML 6.
        self.assertEqual(miniyaml.safe_load("a: |\n  x  \n  y\n"),
                         {"a": "x  \ny\n"})
        self.assertEqual(miniyaml.safe_load("a: |-\n  x  \n"), {"a": "x  "})
        self.assertEqual(miniyaml.safe_load("- |\n  x \n  y\n- z\n"),
                         ["x \ny\n", "z"])
        self.assertEqual(miniyaml.safe_load("a: |\n  x\n    \n  y\n"),
                         {"a": "x\n  \ny\n"})
        self.assertEqual(miniyaml.safe_load("a: |\n  x\n    \n"),
                         {"a": "x\n  \n"})
        self.assertEqual(miniyaml.safe_load("a: >\n  one  \n  two\n"),
                         {"a": "one   two\n"})
        # a whitespace-only line at or below the parent indent is still an
        # empty line inside the block, not its end
        self.assertEqual(
            miniyaml.safe_load("k:\n  a: |\n    x\n  \n    y\n  b: 1\n"),
            {"k": {"a": "x\n\ny\n", "b": 1}})
        self.assertEqual(miniyaml.safe_load("a: |\n  x\n \n"), {"a": "x\n"})

    def test_multiline_case_input(self):
        doc = ("cases:\n  - name: long review\n    input: |\n"
               "      Line one.\n        Indented line.\n      Line three.\n"
               "    assert:\n      - contains: negative\n")
        case = miniyaml.safe_load(doc)["cases"][0]
        self.assertEqual(case["input"],
                         "Line one.\n  Indented line.\nLine three.\n")
        self.assertEqual(case["assert"], [{"contains": "negative"}])


if __name__ == "__main__":
    unittest.main()
