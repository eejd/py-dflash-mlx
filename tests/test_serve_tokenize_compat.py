"""`dflash serve` against the installed mlx-lm's `_tokenize` result shape.

mlx-lm's `ResponseGenerator._tokenize` returns (prompt, segments,
segment_types, initial_state) up to 0.32.0 and appends `vision` from 0.32.1.
These tests POST to the real `DFlashAPIHandler` over HTTP, served by a real
`DFlashResponseGenerator` thread that calls the installed mlx-lm `_tokenize`;
only the model, the tokenizer and the DFlash decode loop are fake.
"""

from __future__ import annotations

import json
import threading
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from dflash_mlx.server import protocol
from dflash_mlx.server.config import build_parser, normalize_cli_args
from dflash_mlx.server.model_provider import DFlashModelProvider
from dflash_mlx.server.protocol import (
    TEXT_STATE_MACHINE_API,
    build_generation_context,
    make_response,
)
from dflash_mlx.server.runtime import ServerRuntime
from dflash_mlx.serve import DFlashAPIHandler, DFlashResponseGenerator

import mlx_lm.server as mlx_server

THINK_START = 7
THINK_END = 8


@pytest.mark.parametrize(
    "tokenized",
    [
        ([1, 2], [[1, 2]], ["assistant"], "reasoning"),
        ([1, 2], [[1, 2]], ["assistant"], "reasoning", None),
        ([1, 2], [[1, 2]], ["assistant"], "reasoning", None, "future"),
    ],
    ids=["4-tuple", "5-tuple", "6-tuple"],
)
def test_split_and_replace_tokenized_use_positions_for_any_arity(tokenized):
    assert protocol.split_tokenized(tokenized) == ([1, 2], "reasoning")
    replaced = protocol.replace_initial_state(tokenized, "normal")
    assert replaced[3] == "normal"
    assert replaced[:3] == tokenized[:3]
    assert replaced[4:] == tokenized[4:]
    assert len(replaced) == len(tokenized)


def test_split_and_replace_tokenized_bare_prompt_and_unknown_shapes():
    assert protocol.split_tokenized([1, 2]) == ([1, 2], "normal")
    assert protocol.replace_initial_state([1, 2], "normal") == [1, 2]
    with pytest.raises(ValueError, match="3 fields"):
        protocol.split_tokenized(([1], [[1]], ["assistant"]))
    short = ([1], [[1]], ["assistant"])
    assert protocol.replace_initial_state(short, "normal") is short


class _Detokenizer:
    last_segment = ""


class _FakeTokenizer:
    """Just enough tokenizer for mlx-lm's `_tokenize` and state machines."""

    has_chat_template = True
    has_tool_calling = False
    has_thinking = True
    think_start = "<think>"
    think_end = "</think>"
    tool_parser = None
    tool_call_start = None
    tool_call_end = None
    structural_markers = ()
    response_transitions = {}
    response_initial_state = None
    eos_token_ids = {2}
    detokenizer = _Detokenizer()

    def encode(self, text, add_special_tokens=True):
        return [10 + (ord(c) % 50) for c in text]

    def apply_chat_template(self, messages, add_generation_prompt=False, **kwargs):
        # The generation prompt of a thinking model ends inside <think>.
        tokens = [1]
        for message in messages:
            tokens += self.encode(message["content"])
        if add_generation_prompt:
            tokens.append(THINK_START)
        return tokens

    def _rfind(self, tokens, token, start=None, end=None):
        lo = max(start or 0, 0)
        hi = len(tokens) if end is None else end
        for i in range(hi - 1, lo - 1, -1):
            if tokens[i] == token:
                return i
        return -1

    def rfind_think_start(self, tokens, start=None, end=None):
        return self._rfind(tokens, THINK_START, start, end)

    def rfind_think_end(self, tokens, start=None, end=None):
        return self._rfind(tokens, THINK_END, start, end)


class _RecordingRuntime(ServerRuntime):
    """DFlash runtime whose decode loop answers "Hello" and records the request."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prepared = []

    def serve_dflash_request(self, *, request_id, rqueue, request, args, prepared):
        self.prepared.append(prepared)
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
        # The final stop token carries no text: the handler discards it.
        for text, finish_reason in (("Hello", None), ("", "stop")):
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


class _FakeProvider(DFlashModelProvider):
    """The real provider (cli args, defaults) with the model load replaced."""

    def load(self, model_path, adapter_path=None, draft_model_path=None):
        self.model = SimpleNamespace()
        self.tokenizer = _FakeTokenizer()
        self.draft_model = SimpleNamespace()
        self.model_key = ("fake-target", None, "fake-draft")
        return self.model, self.tokenizer


@pytest.fixture
def server():
    if not TEXT_STATE_MACHINE_API:
        pytest.skip("mlx-lm < 0.32 has no text state machine server API")
    args = normalize_cli_args(
        build_parser().parse_args(["--model", "fake-target", "--draft-model", "fake-draft"])
    )
    provider = _FakeProvider(args)
    runtime = _RecordingRuntime(
        host="127.0.0.1", port=0, model_provider=provider, version="test"
    )
    generator = DFlashResponseGenerator(
        provider, mlx_server.LRUPromptCache(args.prompt_cache_size), runtime
    )
    httpd = ThreadingHTTPServer(
        ("127.0.0.1", 0),
        lambda *a, **kw: DFlashAPIHandler(generator, *a, **kw),
    )
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield SimpleNamespace(
            port=httpd.server_address[1], runtime=runtime, generator=generator
        )
    finally:
        httpd.shutdown()
        httpd.server_close()
        generator.stop_and_join()


def _post(port, body, path="/v1/chat/completions"):
    conn = HTTPConnection("127.0.0.1", port, timeout=30)
    try:
        conn.request(
            "POST", path, json.dumps(body), {"Content-Type": "application/json"}
        )
        response = conn.getresponse()
        return response.status, json.loads(response.read().decode())
    finally:
        conn.close()


def test_chat_completion_post_runs_dflash_on_installed_mlx_lm_tokenize(server):
    status, payload = _post(
        server.port,
        {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 8},
    )

    assert status == 200, payload
    assert len(server.runtime.prepared) == 1
    prepared = server.runtime.prepared[0]
    assert prepared.prompt == [1, *_FakeTokenizer().encode("hi"), THINK_START]
    # The generation prompt ends inside <think>: the stream starts in reasoning.
    assert prepared.initial_state == "reasoning"
    message = payload["choices"][0]["message"]
    assert message["reasoning_content"] == "Hello"
    assert not message["content"]


def test_chat_completion_post_with_thinking_off_starts_in_normal_state(server):
    status, payload = _post(
        server.port,
        {
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 8,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )

    assert status == 200, payload
    prepared = server.runtime.prepared[0]
    assert prepared.initial_state == "normal"
    assert prepared.has_thinking is False
    message = payload["choices"][0]["message"]
    assert message["content"] == "Hello"
    assert not message.get("reasoning_content")


def test_text_completion_post_runs_dflash_on_installed_mlx_lm_tokenize(server):
    status, payload = _post(
        server.port,
        {"prompt": "hi", "max_tokens": 8},
        path="/v1/completions",
    )

    assert status == 200, payload
    assert server.runtime.prepared[0].prompt == _FakeTokenizer().encode("hi")
    assert payload["choices"][0]["text"] == "Hello"
