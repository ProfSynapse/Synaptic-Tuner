"""Tests for the shared answer-extraction abstraction (``shared.verifiers.extraction``)."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from shared.verifiers.extraction import ExtractedAnswer, extract
from shared.validation.parsing import parse_response


# ---------------------------------------------------------------------------
# output_regex priority
# ---------------------------------------------------------------------------

class TestOutputRegexPriority:

    def test_regex_wins_over_tool_call(self):
        # Both a parseable tool call AND a regex-matchable token are present.
        text = 'ANSWER=42 <tool_call>{"name": "foo", "arguments": {"a": 1}}</tool_call>'
        ea = extract(text, mode="tool_call", output_regex=r"ANSWER=(\d+)")
        assert ea.found is True
        # Regex result wins: answer_text comes from the regex, NOT the tool call.
        assert ea.answer_text == "42"
        assert ea.tool_name == ""
        assert ea.arguments == {}

    def test_named_group(self):
        ea = extract("the answer is <<7>>", output_regex=r"<<(?P<answer>\d+)>>")
        assert ea.found is True
        assert ea.answer_text == "7"

    def test_group_one_when_no_named_group(self):
        ea = extract("result: foo", output_regex=r"result: (\w+)")
        assert ea.found is True
        assert ea.answer_text == "foo"

    def test_group_zero_when_no_capture_group(self):
        ea = extract("hello world", output_regex=r"\w+")
        assert ea.found is True
        assert ea.answer_text == "hello"

    def test_no_match_returns_not_found(self):
        ea = extract("nothing here", output_regex=r"ZZZ(\d+)")
        assert ea.found is False
        assert ea.answer_text == ""
        assert ea.raw == "nothing here"


# ---------------------------------------------------------------------------
# tool_call parity with the canonical parser
# ---------------------------------------------------------------------------

class TestToolCallParity:

    QWEN = '<tool_call>{"name": "search", "arguments": {"q": "cats", "limit": 5}}</tool_call>'
    MISTRAL = '[TOOL_CALLS] [{"name": "search", "arguments": {"q": "cats", "limit": 5}}]'
    CHATML = 'tool_call: search\narguments: {"q": "cats", "limit": 5}'

    @pytest.mark.parametrize("text", [QWEN, MISTRAL, CHATML])
    def test_parity_with_parse_response(self, text):
        ea = extract(text, mode="tool_call")
        first = parse_response(text).first_tool_call
        assert first is not None
        assert ea.found is True
        assert ea.tool_name == first.name
        assert ea.arguments == first.arguments

    @pytest.mark.parametrize("text", [QWEN, MISTRAL, CHATML])
    def test_tool_call_args_mode_parity(self, text):
        ea = extract(text, mode="tool_call_args")
        first = parse_response(text).first_tool_call
        assert ea.found is True
        assert ea.arguments == first.arguments

    def test_no_tool_call_found_false(self):
        ea = extract("just some plain prose, no tool calls", mode="tool_call")
        assert ea.found is False
        assert ea.tool_name == ""
        assert ea.arguments == {}


# ---------------------------------------------------------------------------
# boxed
# ---------------------------------------------------------------------------

class TestBoxed:

    def test_simple(self):
        ea = extract(r"The result is \boxed{42}.", mode="boxed")
        assert ea.found is True
        assert ea.answer_text == "42"

    def test_last_box_wins(self):
        ea = extract(r"\boxed{1} then \boxed{2}", mode="boxed")
        assert ea.found is True
        assert ea.answer_text == "2"

    def test_nested_braces(self):
        ea = extract(r"\boxed{\frac{1}{2}}", mode="boxed")
        assert ea.found is True
        assert ea.answer_text == r"\frac{1}{2}"

    def test_empty_input(self):
        ea = extract("", mode="boxed")
        assert ea.found is False

    def test_no_box(self):
        ea = extract("no box at all", mode="boxed")
        assert ea.found is False


# ---------------------------------------------------------------------------
# last_line
# ---------------------------------------------------------------------------

class TestLastLine:

    def test_happy_path(self):
        ea = extract("first\nsecond\n\n  third  \n", mode="last_line")
        assert ea.found is True
        assert ea.answer_text == "third"

    def test_single_line(self):
        ea = extract("only", mode="last_line")
        assert ea.found is True
        assert ea.answer_text == "only"

    def test_empty_input(self):
        ea = extract("   \n  \n", mode="last_line")
        assert ea.found is False


# ---------------------------------------------------------------------------
# verbatim
# ---------------------------------------------------------------------------

class TestVerbatim:

    def test_happy_path(self):
        ea = extract("hello world", mode="verbatim")
        assert ea.found is True
        assert ea.answer_text == "hello world"

    def test_empty_input(self):
        ea = extract("", mode="verbatim")
        assert ea.found is False
        assert ea.answer_text == ""


# ---------------------------------------------------------------------------
# json_block
# ---------------------------------------------------------------------------

class TestJsonBlock:

    def test_fenced_block(self):
        text = 'prose\n```json\n{"a": 1, "b": [2, 3]}\n```\nmore'
        ea = extract(text, mode="json_block")
        assert ea.found is True
        assert ea.arguments == {"a": 1, "b": [2, 3]}

    def test_balanced_object_fallback(self):
        ea = extract('here: {"x": {"y": 1}} done', mode="json_block")
        assert ea.found is True
        assert ea.arguments == {"x": {"y": 1}}

    def test_empty_input(self):
        ea = extract("", mode="json_block")
        assert ea.found is False
        assert ea.arguments == {}

    def test_no_json(self):
        ea = extract("no braces here", mode="json_block")
        assert ea.found is False


# ---------------------------------------------------------------------------
# unknown mode
# ---------------------------------------------------------------------------

def test_unknown_mode_raises():
    with pytest.raises(ValueError, match="unknown extraction mode"):
        extract("x", mode="nope")
