# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from mlx_lm.server import ToolCallFormatter
from mlx_lm.tool_parsers import gemma4, qwen3_coder

try:
    from mlx_lm.tool_parsers import NotAToolCall, llama3
except ImportError:  # mlx-lm without eejd/py-mlx-lm#8
    NotAToolCall = llama3 = None

from dflash_mlx.server.tool_calls import (
    ToolCallParseError,
    apply_tool_choice,
    install_tool_parser,
    make_tool_parser,
    normalize_parallel_tool_calls,
)

needs_not_a_tool_call = pytest.mark.skipif(
    NotAToolCall is None,
    reason="needs mlx-lm with eejd/py-mlx-lm#8 (NotAToolCall, llama3 parser)",
)


def _format(formatter, texts):
    """(tool_calls, content) from either ToolCallFormatter return shape.

    mlx-lm up to 0.32.2-local03 returns the list of calls; with eejd/py-mlx-lm#8
    it returns (calls, content), content being the text a parser said is not a call.
    """
    result = formatter(texts)
    if isinstance(result, tuple):
        return result
    return result, ""


def _tools():
    return [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "limit": {"type": "integer"},
                    },
                },
            },
        }
    ]


def test_qwen_xml_tool_call_uses_upstream_parser():
    parser = make_tool_parser(qwen3_coder.parse_tool_call)

    calls = parser(
        "<function=lookup>\n"
        "<parameter=query>\nweather\n</parameter>\n"
        "<parameter=limit>\n3\n</parameter>\n"
        "</function>",
        _tools(),
    )

    assert calls == [
        {"id": None, "name": "lookup", "arguments": {"query": "weather", "limit": 3}}
    ]


def test_qwen_parser_accepts_json_tool_span_fallback():
    parser = make_tool_parser(qwen3_coder.parse_tool_call)

    calls = parser(
        json.dumps({"name": "lookup", "arguments": {"query": "weather"}}),
        _tools(),
    )

    assert calls == [
        {"id": None, "name": "lookup", "arguments": {"query": "weather"}}
    ]


def test_upstream_formatter_emits_openai_tool_call_from_json_fallback():
    parser = make_tool_parser(qwen3_coder.parse_tool_call)
    formatter = ToolCallFormatter(parser, _tools(), streaming=True)

    calls, content = _format(
        formatter, [json.dumps({"name": "lookup", "arguments": {"query": "x"}})]
    )

    assert content == ""
    assert calls[0]["type"] == "function"
    assert calls[0]["index"] == 0
    assert calls[0]["function"]["name"] == "lookup"
    assert json.loads(calls[0]["function"]["arguments"]) == {"query": "x"}


def test_upstream_formatter_does_not_swallow_strict_parse_errors():
    parser = make_tool_parser(qwen3_coder.parse_tool_call)
    formatter = ToolCallFormatter(parser, _tools())

    with pytest.raises(ToolCallParseError, match="failed to parse tool call"):
        formatter(["not json"])


@needs_not_a_tool_call
def test_not_a_tool_call_reaches_the_formatter_as_content():
    def upstream(text, tools):
        raise NotAToolCall("plain answer")

    formatter = ToolCallFormatter(make_tool_parser(upstream), _tools())

    # Not a ToolCallParseError, and not the JSON fallback: the formatter's
    # content channel gets the parser's text.
    assert formatter(["{}"]) == ([], "plain answer")


@needs_not_a_tool_call
def test_llama3_parser_through_the_dflash_wrapper():
    formatter = ToolCallFormatter(make_tool_parser(llama3.parse_tool_call), _tools())

    calls, content = formatter(
        ['<|python_tag|>{"name": "lookup", "parameters": {"query": "x"}}']
    )
    assert content == ""
    assert [c["function"]["name"] for c in calls] == ["lookup"]
    assert json.loads(calls[0]["function"]["arguments"]) == {"query": "x"}

    # Code behind the tag, and a call to an undeclared tool, are content.
    assert formatter(["<|python_tag|>print(1)"]) == ([], "print(1)")
    undeclared = '{"name": "other", "parameters": {}}'
    assert formatter([undeclared]) == ([], undeclared)


def test_openai_function_shape_accepts_string_arguments():
    parser = make_tool_parser(qwen3_coder.parse_tool_call)

    calls = parser(
        json.dumps(
            {
                "id": "call_123",
                "function": {
                    "name": "lookup",
                    "arguments": json.dumps({"query": "weather"}),
                },
            }
        ),
        _tools(),
    )

    assert calls == [
        {"id": "call_123", "name": "lookup", "arguments": {"query": "weather"}}
    ]


def test_gemma4_tool_call_uses_upstream_parser():
    parser = make_tool_parser(gemma4.parse_tool_call)

    calls = parser('call:lookup{query:<|"|>weather<|"|>}', _tools())

    assert calls == [
        {"id": None, "name": "lookup", "arguments": {"query": "weather"}}
    ]


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ("not json", "failed to parse tool call"),
        (json.dumps({"name": "missing", "arguments": {}}), "unknown tool call"),
        (json.dumps({"name": "lookup", "arguments": []}), "arguments must be"),
        (json.dumps([]), "did not contain any calls"),
    ],
)
def test_invalid_tool_call_spans_fail_fast(payload, message):
    parser = make_tool_parser(qwen3_coder.parse_tool_call)

    with pytest.raises(ToolCallParseError, match=message):
        parser(payload, _tools())


def test_tool_call_without_declared_tools_fails_fast():
    parser = make_tool_parser(qwen3_coder.parse_tool_call)

    with pytest.raises(ToolCallParseError, match="no declared tools"):
        parser(json.dumps({"name": "lookup", "arguments": {}}), None)


def test_install_tool_parser_refreshes_chat_template_parser():
    class FakeTokenizer:
        chat_template = "<tool_call>\n<function="

        def __init__(self):
            self._tool_parser = None

        @property
        def tool_parser(self):
            return self._tool_parser

        def encode(self, text, add_special_tokens=False):
            return [ord(ch) for ch in text]

    tokenizer = FakeTokenizer()

    install_tool_parser(tokenizer)

    assert tokenizer._tool_call_start == "<tool_call>"
    assert tokenizer._tool_call_end == "</tool_call>"
    assert tokenizer._tool_call_start_tokens == tuple(ord(ch) for ch in "<tool_call>")
    assert getattr(tokenizer._tool_parser, "_dflash_tool_parser") is True


def test_tool_choice_none_disables_declared_tools():
    request = SimpleNamespace(tools=_tools())

    apply_tool_choice(request, "none")

    assert request.tools is None


def test_tool_choice_auto_keeps_declared_tools():
    tools = _tools()
    request = SimpleNamespace(tools=tools)

    apply_tool_choice(request, {"type": "auto"})

    assert request.tools is tools


def test_function_specific_tool_choice_is_rejected():
    request = SimpleNamespace(tools=_tools())

    with pytest.raises(ValueError, match="function-specific"):
        apply_tool_choice(
            request,
            {"type": "function", "function": {"name": "lookup"}},
        )


def test_parallel_tool_calls_validation():
    assert normalize_parallel_tool_calls(None) is True
    assert normalize_parallel_tool_calls(True) is True
    assert normalize_parallel_tool_calls(False) is False
    with pytest.raises(ValueError, match="boolean"):
        normalize_parallel_tool_calls("false")
