# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

from typing import Any, Optional

import mlx_lm.server as mlx_server

# mlx-lm server API generations this module adapts to:
#  - text state machine (mlx-lm >= 0.32): GenerationContext carries a
#    TextStateMachine (control sequences are stripped from decoded text by the
#    HTTP handler) and Response has no state/match; stop tokens are matched by
#    a StopSequences matcher on the producer side.
#  - stateful (mlx-lm 0.31): token-level SequenceStateMachine, Response.state
#    and Response.match.
#  - earlier releases: neither.
TEXT_STATE_MACHINE_API = "text_sm" in getattr(
    mlx_server.GenerationContext, "__annotations__", {}
)
STATEFUL_SERVER_API = "state" in getattr(mlx_server.Response, "__annotations__", {})

# `ResponseGenerator._tokenize` returns (prompt, segments, segment_types,
# initial_state) in mlx-lm <= 0.32.0 and appends `vision` (the images of the
# prompt, None for text) in mlx-lm >= 0.32.1. Callers read the fields they
# need by position and leave the rest alone, so one more trailing field in a
# later release does not break them. DFlash serves text targets (its
# ModelProvider never loads a processor), so `vision` is always None here.
_TOKENIZED_INITIAL_STATE = 3


def split_tokenized(tokenized: Any) -> tuple[Any, str]:
    """Return (prompt, initial_state) from the value of `_tokenize`.

    Accepts the bare prompt of the oldest servers and a tuple of at least
    four fields; any fields after `initial_state` are ignored.
    """
    if not isinstance(tokenized, tuple):
        return tokenized, "normal"
    if len(tokenized) <= _TOKENIZED_INITIAL_STATE:
        raise ValueError(
            "unsupported mlx-lm _tokenize result: expected (prompt, segments, "
            f"segment_types, initial_state, ...), got {len(tokenized)} fields"
        )
    return tokenized[0], tokenized[_TOKENIZED_INITIAL_STATE]


def replace_initial_state(tokenized: Any, initial_state: str) -> Any:
    """Return `tokenized` with its `initial_state` field replaced.

    Other fields keep their position, so a tuple longer than four fields keeps
    its trailing ones. Values that are not such a tuple are returned as is.
    """
    if not isinstance(tokenized, tuple) or len(tokenized) <= _TOKENIZED_INITIAL_STATE:
        return tokenized
    fields = list(tokenized)
    fields[_TOKENIZED_INITIAL_STATE] = initial_state
    return tuple(fields)


def thinking_enabled_for_request(cli_args: Any, request_args: Any = None) -> bool:
    chat_args = getattr(cli_args, "chat_template_args", None)
    enabled = True
    if isinstance(chat_args, dict) and "enable_thinking" in chat_args:
        enabled = bool(chat_args.get("enable_thinking", False))

    request_chat_args = getattr(request_args, "chat_template_kwargs", None)
    if isinstance(request_chat_args, dict) and "enable_thinking" in request_chat_args:
        enabled = bool(request_chat_args["enable_thinking"])
    return enabled

def build_generation_context(
    tokenizer,
    prompt,
    stop_words=None,
    sequences=None,
    has_thinking: Optional[bool] = None,
    text_sm=None,
    initial_state: str = "normal",
):
    thinking = tokenizer.has_thinking if has_thinking is None else bool(has_thinking)
    if TEXT_STATE_MACHINE_API:
        if text_sm is None:
            from mlx_lm.generate import make_text_state_machine

            text_sm = make_text_state_machine(tokenizer, stop_words)
        return mlx_server.GenerationContext(
            has_thinking=thinking,
            has_tool_calling=tokenizer.has_tool_calling,
            tool_parser=tokenizer.tool_parser,
            text_sm=text_sm,
            initial_state=initial_state,
            prompt=prompt,
        )
    if STATEFUL_SERVER_API:
        return mlx_server.GenerationContext(
            has_thinking=thinking,
            has_tool_calling=tokenizer.has_tool_calling,
            tool_parser=tokenizer.tool_parser,
            sequences=sequences or {},
            prompt=prompt,
        )
    return mlx_server.GenerationContext(
        has_tool_calling=tokenizer.has_tool_calling,
        tool_call_start=tokenizer.tool_call_start,
        tool_call_end=tokenizer.tool_call_end,
        tool_parser=tokenizer.tool_parser,
        has_thinking=thinking,
        think_start_id=tokenizer.think_start_id,
        think_end=tokenizer.think_end,
        think_end_id=tokenizer.think_end_id,
        eos_token_ids=tokenizer.eos_token_ids,
        stop_token_sequences=[
            tokenizer.encode(stop_word, add_special_tokens=False)
            for stop_word in (stop_words or [])
        ],
        prompt=prompt,
    )

class _WithoutThinking:
    """A tokenizer view that reports no reasoning markers."""

    has_thinking = False

    def __init__(self, tokenizer: Any):
        self._tokenizer = tokenizer

    def __getattr__(self, name: str) -> Any:
        return getattr(self._tokenizer, name)

def make_text_state_machines(
    *,
    tokenizer: Any,
    stop_words: list[str],
    include_thinking: bool = True,
):
    """(StopSequences, TextStateMachine) for the mlx-lm >= 0.32 server API.

    With reasoning disabled the machine has no reasoning transitions, the same
    rule make_state_machine applies to the token-level machine.
    """
    from mlx_lm.generate import make_stop_sequences, make_text_state_machine

    view = tokenizer if include_thinking else _WithoutThinking(tokenizer)
    return (
        make_stop_sequences(view, stop_words),
        make_text_state_machine(view, stop_words),
    )

def make_state_machine(
    *,
    tokenizer: Any,
    stop_words: list[str],
    initial_state: str = "normal",
    include_thinking: bool = True,
):
    from mlx_lm.generate import SequenceStateMachine

    if not include_thinking and initial_state == "reasoning":
        initial_state = "normal"

    transitions: dict[str, list[tuple[tuple[int, ...], Optional[str]]]] = {}
    sequences: dict[tuple[int, ...], str] = {}

    common_stops: list[tuple[tuple[int, ...], Optional[str]]] = []
    for token_id in tokenizer.eos_token_ids:
        token_tuple = (int(token_id),)
        sequences[token_tuple] = tokenizer.convert_ids_to_tokens(token_id)
        common_stops.append((token_tuple, None))
    for stop_word in stop_words:
        tokens = tuple(tokenizer.encode(stop_word, add_special_tokens=False))
        sequences[tokens] = stop_word
        common_stops.append((tokens, None))

    transitions["normal"] = list(common_stops)

    if include_thinking and tokenizer.has_thinking:
        think_start = tokenizer.think_start_tokens
        think_end = tokenizer.think_end_tokens
        transitions["normal"].append((think_start, "reasoning"))
        transitions["reasoning"] = [(think_end, "normal")]
        transitions["reasoning"].extend(common_stops)
        sequences[think_start] = tokenizer.think_start
        sequences[think_end] = tokenizer.think_end

    if tokenizer.has_tool_calling:
        tool_start = tokenizer.tool_call_start_tokens
        tool_end = tokenizer.tool_call_end_tokens
        transitions["normal"].append((tool_start, "tool"))
        transitions["tool"] = [(tool_end, "normal")]
        transitions["tool"].extend(common_stops)
        sequences[tool_start] = tokenizer.tool_call_start
        sequences[tool_end] = tokenizer.tool_call_end

    return SequenceStateMachine(transitions, initial=initial_state), sequences

def make_response(
    *,
    text: str,
    token: int,
    state: Optional[str],
    match: Optional[tuple[int, ...]],
    finish_reason: Optional[str],
):
    if STATEFUL_SERVER_API:
        return mlx_server.Response(
            text,
            token,
            state or "normal",
            match,
            0.0,
            finish_reason,
            (),
        )
    return mlx_server.Response(
        text,
        token,
        0.0,
        finish_reason,
        (),
    )

def match_stream_token(
    sm: Optional[Any],
    sm_state: Optional[Any],
    token: int,
) -> tuple[Optional[Any], Optional[tuple[int, ...]], Optional[str], bool]:
    if sm is None:
        return sm_state, None, "normal", False
    if TEXT_STATE_MACHINE_API:
        # `sm` is a StopSequences matcher (advances in place); control
        # sequences are stripped from the text by the HTTP handler.
        return sm_state, None, "normal", bool(sm.advance(token))
    if sm_state is None or sm_state[0] is None:
        return sm_state, None, None, True
    next_state, match_sequence, current_state = sm.match(sm_state, token)
    terminal = match_sequence is not None and current_state is None
    return next_state, match_sequence, current_state, terminal
