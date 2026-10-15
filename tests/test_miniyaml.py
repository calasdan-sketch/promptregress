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

    def test_block_scalar_rejected(self):
        with self.assertRaisesRegex(ValueError, r"multi-line.*line 1"):
            miniyaml.safe_load("a: |")

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


if __name__ == "__main__":
    unittest.main()
