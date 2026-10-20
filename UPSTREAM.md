# Work carried on `local` (eejd/py-dflash-mlx)

Primary upstream: https://github.com/bstnxbt/dflash-mlx (`main` mirrors it, fast-forward only).
Rules: eejd/macports-ports-local `docs/development-model.md`, "Forks, upstreams and port tiers".

`local` = upstream `main` + the topic branches below. Releases are tagged
`v<base>-local<NN>` (`<base>` is the last upstream release `main` contains).

| Branch | What | Origin | Upstream PR |
|---|---|---|---|
| `topic/mlx-lm-0.32-compat` | `dflash serve`, the KV-cache snapshot codecs and tool-parser inference run on mlx-lm 0.32 (TextStateMachine server API, `KVCache.state` is a 3-tuple, `_infer_tool_parser(tokenizer)`). Still runs on 0.31.x. | authored here; the `keys_and_values()` serialization approach is the one in `jundot/dflash-mlx@959684cc` (secondary upstream, mirrored as `track/jundot/main`) | not submitted: owner decision, nothing goes to true upstreams from this repo. Candidate; status in eejd/py-dflash-mlx's "Upstream candidate" issue |
| `topic/mlx-lm-0.32.2-compat` (stacked on `topic/mlx-lm-0.32-compat`) | `dflash serve` works on mlx-lm 0.32.1 and later: `ResponseGenerator._tokenize` returns a fifth field (`vision`, always `None` for DFlash's text targets) and every POST answered HTTP 404 "too many values to unpack (expected 4, got 5)". The fields are read by position (`split_tokenized`, `replace_initial_state` in `server/protocol.py`), so mlx-lm 0.32.0 (4 fields), 0.32.1/0.32.2 (5) and a later extra field all work. `tests/test_serve_tokenize_compat.py` POSTs to the real handler. | authored here against `eejd/py-mlx-lm@v0.32.2-local03` (same `_tokenize` as ml-explore/mlx-lm v0.32.2) | not submitted: owner decision, nothing goes to true upstreams from this repo. Candidate; status in eejd/py-dflash-mlx's "Upstream candidate" issue (#1) |
| `topic/mlx-lm-tool-formatter-compat` | `dflash serve` and its tests work on mlx-lm local builds that carry eejd/py-mlx-lm#8, where `ToolCallFormatter.__call__` returns `(tool_calls, content)` instead of a list and a tool parser raises `NotAToolCall` (a `ValueError`) for text that is content, not a call (the `llama3` parser). `dflash_mlx/server/tool_calls.py` re-raises `NotAToolCall` instead of turning it into the JSON fallback and a `ToolCallParseError` (HTTP 400 / a cut-off stream), so the formatter returns the text as message content. mlx-lm without #8 never raises it and is unchanged (0.32.2-local03 and earlier: list result). dflash-mlx's runtime never calls the formatter itself (it goes through `super().handle_completion`); `tests/test_server_tool_calls.py` reads either return shape and has tests that need #8 (skipped without it). Full suite: 1260 passed, 16 skipped on mlx-lm 0.32.2-local03; 1262 passed, 14 skipped on `eejd/py-mlx-lm@cd5acb5` (`local` with #8). | authored here against `eejd/py-mlx-lm@16adc05` (the commit that adds `NotAToolCall` and the `(calls, content)` return) | not submitted: owner decision, nothing goes to true upstreams from this repo. Not a candidate: it adapts to a change that exists only in eejd/py-mlx-lm |

Not adopted from `track/jundot/main` (omlx-pinned fork, 22 commits ahead of v0.1.10): Muse Glimmer
targets, DFlash 2 local convolution, min_p and repetition-penalty sampling, gemma4_unified targets.

## Releases

| Tag | Contents | Tested against |
|---|---|---|
| `v0.1.10-local01` | `main` (60803233) + `topic/mlx-lm-0.32-compat` | mlx-lm 0.32.0; fails on 0.32.1/0.32.2 (HTTP 404 on every POST) |
| `v0.1.10-local02` | `v0.1.10-local01` + `topic/mlx-lm-0.32.2-compat` | full suite 1260 passed, 14 skipped on mlx-lm 0.32.2-local03; `tests/test_serve_tokenize_compat.py` and `tests/test_serve_state_machine.py` also pass on 0.32.0; 0.32.1 is covered by reading its `_tokenize` (same 5-tuple), not run |
