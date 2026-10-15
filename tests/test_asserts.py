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


if __name__ == "__main__":
    unittest.main()
