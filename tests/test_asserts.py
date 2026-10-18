"""Assertion evaluator tests (no model needed)."""

import unittest

from promptregress import asserts as a


class TestBasic(unittest.TestCase):
    def test_contains(self):
        self.assertTrue(a.evaluate({"contains": "x"}, "axb")[0])
        self.assertFalse(a.evaluate({"contains": "z"}, "axb")[0])
        self.assertTrue(a.evaluate({"not_contains": "z"}, "axb")[0])
        self.assertFalse(a.evaluate({"not_contains": "x"}, "axb")[0])

    def test_regex(self):
        self.assertTrue(a.evaluate({"regex": r"\d+"}, "abc123")[0])
        self.assertFalse(a.evaluate({"regex": r"^\d+$"}, "abc123")[0])
        self.assertTrue(a.evaluate({"not_regex": r"^\d+$"}, "abc123")[0])

    def test_starts_ends(self):
        self.assertTrue(a.evaluate({"starts_with": "hi"}, "hi there")[0])
        self.assertTrue(a.evaluate({"ends_with": "there"}, "hi there\n")[0])
        self.assertFalse(a.evaluate({"starts_with": "bye"}, "hi there")[0])

    def test_unknown_raises(self):
        with self.assertRaises(ValueError):
            a.evaluate({"bogus": 1}, "x")

    def test_malformed_raises(self):
        with self.assertRaises(ValueError):
            a.evaluate({"contains": "a", "extra": "b"}, "x")


class TestJson(unittest.TestCase):
    def test_json_valid(self):
        self.assertTrue(a.evaluate({"json_valid": True}, '{"a": 1}')[0])
        self.assertFalse(a.evaluate({"json_valid": True}, "nope")[0])

    def test_schema_ok(self):
        out = '{"sentiment": "negative", "score": 0.2}'
        schema = {"type": "object",
                  "required": ["sentiment"],
                  "properties": {"sentiment": {"type": "string"},
                                 "score": {"type": "number"}}}
        self.assertTrue(a.evaluate({"schema": schema}, out)[0])

    def test_schema_missing_required(self):
        ok, detail = a.evaluate(
            {"schema": {"type": "object", "required": ["sentiment"]}},
            '{"other": 1}')
        self.assertFalse(ok)
        self.assertIn("sentiment", detail)

    def test_schema_wrong_type(self):
        ok, _ = a.evaluate({"schema": {"type": "array"}}, '{"a": 1}')
        self.assertFalse(ok)

    def test_schema_enum(self):
        schema = {"type": "string", "enum": ["a", "b"]}
        self.assertTrue(a.evaluate({"schema": schema}, '"a"')[0])
        self.assertFalse(a.evaluate({"schema": schema}, '"c"')[0])

    def test_schema_needs_json(self):
        ok, detail = a.evaluate({"schema": {"type": "object"}}, "plain")
        self.assertFalse(ok)
        self.assertIn("JSON", detail)


class TestTokens(unittest.TestCase):
    def test_max_tokens(self):
        self.assertTrue(a.evaluate({"max_tokens": 100}, "short")[0])
        self.assertFalse(a.evaluate({"max_tokens": 1}, "x" * 100)[0])

    def test_min_tokens(self):
        self.assertTrue(a.evaluate({"min_tokens": 1}, "hello world")[0])

    def test_estimate_is_chars_over_four(self):
        # documented as ~4 chars per token, not a whitespace count
        self.assertEqual(a.estimate_tokens("x" * 104), 26)
        self.assertEqual(a.estimate_tokens("a b c d e f g h"), 3)
        self.assertEqual(a.estimate_tokens("a"), 1)
        # an empty (or blank) reply has no tokens: min_tokens: 1 must fail it
        self.assertEqual(a.estimate_tokens(""), 0)


class TestRawYamlScalars(unittest.TestCase):
    """YAML hands the evaluator ints, bools and None where text or a
    count is meant. No TypeError/re.error may escape: coerce what is
    unambiguous, raise ValueError (-> "bad assertion") otherwise."""

    def test_number_operands_coerced_to_text(self):
        self.assertEqual(a.evaluate({"contains": 1}, "got 1"), (True, "contains '1'"))
        self.assertTrue(a.evaluate({"starts_with": 42}, "42 is it")[0])
        self.assertTrue(a.evaluate({"ends_with": 2.5}, "pi-ish 2.5")[0])
        self.assertTrue(a.evaluate({"not_contains": 7}, "none here")[0])

    def test_bool_and_null_operands_rejected_with_hint(self):
        with self.assertRaisesRegex(ValueError, r'YAML boolean.*contains: "yes"'):
            a.evaluate({"contains": True}, "yes")
        with self.assertRaisesRegex(ValueError, r"not_contains: .*YAML boolean"):
            a.evaluate({"not_contains": False}, "no")
        with self.assertRaisesRegex(ValueError, r"contains: needs text, got null"):
            a.evaluate({"contains": None}, "x")
        with self.assertRaisesRegex(ValueError, r"regex: needs text"):
            a.evaluate({"regex": None}, "x")
        with self.assertRaisesRegex(ValueError, r"needs text, got list"):
            a.evaluate({"contains": ["a"]}, "x")

    def test_token_counts_coerced_or_rejected(self):
        self.assertEqual(a.evaluate({"max_tokens": "10"}, "short"),
                         (True, "~1 tokens (max 10)"))
        self.assertTrue(a.evaluate({"min_tokens": 1.0}, "hello")[0])
        for bad in ("abc", "1.5", None, True, 2.5):
            with self.assertRaisesRegex(ValueError, r"max_tokens: needs a whole number"):
                a.evaluate({"max_tokens": bad}, "x")

    def test_invalid_regex_is_value_error(self):
        with self.assertRaisesRegex(ValueError, r"regex: invalid regex /\(/"):
            a.evaluate({"regex": "("}, "x")
        with self.assertRaisesRegex(ValueError, r"not_regex: invalid regex"):
            a.evaluate({"not_regex": "[unclosed"}, "x")
        # a numeric pattern still works (YAML int -> "1")
        self.assertTrue(a.evaluate({"regex": 1}, "a1b")[0])


if __name__ == "__main__":
    unittest.main()
