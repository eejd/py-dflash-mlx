# Work carried on `local` (eejd/py-dflash-mlx)

Primary upstream: https://github.com/bstnxbt/dflash-mlx (`main` mirrors it, fast-forward only).
Rules: eejd/macports-ports-local `docs/development-model.md`, "Forks, upstreams and port tiers".

`local` = upstream `main` + the topic branches below. Releases are tagged
`v<base>-local<NN>` (`<base>` is the last upstream release `main` contains).

| Branch | What | Origin | Upstream PR |
|---|---|---|---|
| `topic/mlx-lm-0.32-compat` | `dflash serve`, the KV-cache snapshot codecs and tool-parser inference run on mlx-lm 0.32 (TextStateMachine server API, `KVCache.state` is a 3-tuple, `_infer_tool_parser(tokenizer)`). Still runs on 0.31.x. | authored here; the `keys_and_values()` serialization approach is the one in `jundot/dflash-mlx@959684cc` (secondary upstream, mirrored as `track/jundot/main`) | not submitted: owner decision, nothing goes to true upstreams from this repo. Candidate; status in eejd/py-dflash-mlx's "Upstream candidate" issue |

Not adopted from `track/jundot/main` (omlx-pinned fork, 22 commits ahead of v0.1.10): Muse Glimmer
targets, DFlash 2 local convolution, min_p and repetition-penalty sampling, gemma4_unified targets.
