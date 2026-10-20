"""Tool calls through the real `DFlashAPIHandler` (stream and non-stream).

Same harness as test_serve_tokenize_compat.py: a real handler and
`DFlashResponseGenerator` served over HTTP, running the installed mlx-lm's
`handle_completion`; only the model, the tokenizer and the DFlash decode loop are
fake. The fake decode loop answers the text the test gives it, and the tokenizer
carries a dflash-wrapped tool parser, as `install_tool_parser` leaves it.
"""

from __future__ import annotations

import json
import threading
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer

import mlx_lm.server as mlx_server
import pytest
from mlx_lm.tool_parsers import qwen3_coder

from dflash_mlx.serve import DFlashAPIHandler, DFlashResponseGenerator
from dflash_mlx.server.config import build_parser, normalize_cli_args
from dflash_mlx.server.protocol import (
    TEXT_STATE_MACHINE_API,
    build_generation_context,
    make_response,
)
from dflash_mlx.server.runtime import ServerRuntime
from dflash_mlx.server.tool_calls import make_tool_parser
from tests.test_serve_tokenize_compat import _FakeProvider, _FakeTokenizer

try:
    from mlx_lm.tool_parsers import NotAToolCall, llama3
except ImportError:  # mlx-lm without eejd/py-mlx-lm#8
    NotAToolCall = llama3 = None

needs_not_a_tool_call = pytest.mark.skipif(
    NotAToolCall is None, reason="needs mlx-lm with eejd/py-mlx-lm#8 (NotAToolCall)"
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
            },
        },
    }
]


class _ToolTokenizer(_FakeTokenizer):
    has_tool_calling = True
    tool_call_start = "<tool_call>"
    tool_call_end = "</tool_call>"

    def __init__(self, parser):
        self.tool_parser = parser


class _ScriptedRuntime(ServerRuntime):
    """DFlash runtime whose decode loop answers the scripted text."""

    answer = ""

    def serve_dflash_request(self, *, request_id, rqueue, request, args, prepared):
        rqueue.put(
            build_generation_context(
                self.model_provider.tokenizer,
                prepared.prompt,
                stop_words=args.stop_words,
                sequences=prepared.sequences,
                has_thinking=prepared.has_thinking,
                text_sm=prepared.text_state_machine,
                initial_state=prepared.initial_state,
            )
        )
        for text, finish_reason in ((self.answer, None), ("", "stop")):
            rqueue.put(
                make_response(
                    text=text,
                    token=3,
                    state=None,
                    match=None,
                    finish_reason=finish_reason,
                )
            )
        rqueue.put(None)


def _serve(parser, answer):
    """(port, close) of a server whose model answers `answer`."""
    args = normalize_cli_args(
        build_parser().parse_args(["--model", "fake-target", "--draft-model", "fake-draft"])
    )

    class Provider(_FakeProvider):
        def load(self, model_path, adapter_path=None, draft_model_path=None):
            super().load(model_path, adapter_path, draft_model_path)
            self.tokenizer = _ToolTokenizer(parser)
            return self.model, self.tokenizer

    provider = Provider(args)
    runtime = _ScriptedRuntime(
        host="127.0.0.1", port=0, model_provider=provider, version="test"
    )
    runtime.answer = answer
    generator = DFlashResponseGenerator(
        provider, mlx_server.LRUPromptCache(args.prompt_cache_size), runtime
    )
    httpd = ThreadingHTTPServer(
        ("127.0.0.1", 0), lambda *a, **kw: DFlashAPIHandler(generator, *a, **kw)
    )
    threading.Thread(target=httpd.serve_forever, daemon=True).start()

    def close():
        httpd.shutdown()
        httpd.server_close()
        generator.stop_and_join()

    return httpd.server_address[1], close


def _ask(port, stream):
    """(status, content, tool_calls, finish_reason) of one chat request with tools."""
    body = {
        "messages": [{"role": "user", "content": "hi"}],
        "tools": TOOLS,
        "max_tokens": 64,
        "stream": stream,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    conn = HTTPConnection("127.0.0.1", port, timeout=30)
    try:
        conn.request(
            "POST",
            "/v1/chat/completions",
            json.dumps(body),
            {"Content-Type": "application/json"},
        )
        response = conn.getresponse()
        raw = response.read().decode()
        status = response.status
    finally:
        conn.close()
    if not stream:
        payload = json.loads(raw)
        choice = payload["choices"][0]
        message = choice["message"]
        return (
            status,
            message.get("content") or "",
            message.get("tool_calls") or [],
            choice["finish_reason"],
        )
    content, calls, finish = "", [], None
    for line in raw.splitlines():
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        choice = json.loads(line[6:])["choices"][0]
        delta = choice.get("delta", {})
        content += delta.get("content") or ""
        calls += delta.get("tool_calls") or []
        finish = choice.get("finish_reason") or finish
    return status, content, calls, finish


@pytest.fixture
def serve():
    if not TEXT_STATE_MACHINE_API:
        pytest.skip("mlx-lm < 0.32 has no text state machine server API")
    closers = []

    def start(parser, answer):
        port, close = _serve(parser, answer)
        closers.append(close)
        return port

    yield start
    for close in closers:
        close()


@needs_not_a_tool_call
@pytest.mark.parametrize("stream", [False, True], ids=["non-stream", "stream"])
def test_not_a_tool_call_comes_back_as_message_text(serve, stream):
    def upstream(text, tools):
        raise NotAToolCall("plain answer")

    port = serve(make_tool_parser(upstream), "<tool_call>{}</tool_call>")

    status, content, calls, finish = _ask(port, stream)

    # Not an error response, not a cut-off stream: text, no call, finish "stop".
    assert status == 200
    assert content == "plain answer"
    assert calls == []
    assert finish == "stop"


@pytest.mark.parametrize("stream", [False, True], ids=["non-stream", "stream"])
@pytest.mark.parametrize(
    "parser_module, answer",
    [
        (
            qwen3_coder,
            "<function=lookup>\n<parameter=query>\nx\n</parameter>\n</function>",
        ),
        pytest.param(
            llama3,
            '{"name": "lookup", "parameters": {"query": "x"}}',
            marks=needs_not_a_tool_call,
        ),
    ],
    ids=["qwen3_coder", "llama3"],
)
def test_bare_call_without_start_marker_is_a_tool_call(
    serve, parser_module, answer, stream
):
    if not hasattr(mlx_server, "BareToolCallFilter"):
        pytest.skip("mlx-lm without bare-call filters")
    port = serve(make_tool_parser(parser_module.parse_tool_call), answer)

    status, content, calls, finish = _ask(port, stream)

    # The model wrote the call without <tool_call> / <|python_tag|>: mlx-lm's bare-call
    # filter of the parser's module must still see it through dflash's wrapper.
    assert status == 200
    assert content == ""
    assert [c["function"]["name"] for c in calls] == ["lookup"]
    assert json.loads(calls[0]["function"]["arguments"]) == {"query": "x"}
    assert finish == "tool_calls"
