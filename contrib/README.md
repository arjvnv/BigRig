# contrib

Work prepared for upstream projects, kept here so it is versioned and can be run again.

Nothing in this directory is imported by the engine and nothing here ships in the package
(`pyproject.toml` packages only `bigrig_engine` and `bigrig_layer`). Each folder holds the text
of a report, and where there is something to prove, a reproducer that runs standalone.

| folder | upstream | what it is | code |
|---|---|---|---|
| `mlx-gather-qmm-accuracy/` | ml-explore/mlx | `sorted_indices` changes the answer, not just the speed. The `False` path is about 4.4x further from an exact reference. | `repro.py`, `mlx_lm_effect.py` |
| `mlx-metal-import-residency/` | ml-explore/mlx | An imported buffer is made resident whole, so importing a large mapped region wires all of it on first use. Docs request. | `repro.py` |
| `kv-bytes-per-token/` | ml-explore/mlx-lm, vllm-project/vllm-metal | KV cache bytes per token by architecture family, verified against mlx_lm's own cache objects on 21 families. | `kv_bytes.py`, `verify.py` |
| `mlx-lm-expert-streaming/` | ml-explore/mlx-lm #1438 | What expert streaming actually costs, and which two hooks would let it live outside the library. | text only |
| `vllm-moe-offload-rfc/` | vllm-project/vllm #38256 | Measured evidence against three choices in the expert offloading RFC, with the transfer caveat stated first. | text only |
| `vllm-metal-macos-benchmarking/` | vllm-project/vllm-metal #713 | Four macOS measurement traps not on their list, each from a number we got wrong first. | text only |

Every reproducer exits non zero if the effect does not reproduce on the machine running it, so a
green run is evidence and a red run is a reason not to file. `mlx_lm_effect.py` and `verify.py`
read models already on the machine and skip cleanly when there are none. Nothing downloads.

## Running them

```bash
python contrib/mlx-gather-qmm-accuracy/repro.py          # mlx and numpy only
python contrib/mlx-gather-qmm-accuracy/mlx_lm_effect.py  # skips without a local MoE model
python contrib/mlx-metal-import-residency/repro.py       # writes and deletes one temp file
python contrib/kv-bytes-per-token/verify.py models       # extra model dirs are optional
```

## Status

Nothing here has been posted upstream yet. Each folder is a draft awaiting a decision to file.
