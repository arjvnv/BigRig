# contrib

Work prepared for upstream projects, kept here so it is versioned and re-runnable.

Nothing in this directory is imported by the engine and nothing here ships in the package. Each
folder holds a reproducer that runs standalone, plus the text of the report it belongs to.

| folder | upstream | what it is |
|---|---|---|
| `mlx-gather-qmm-accuracy/` | ml-explore/mlx | `sorted_indices` changes the answer, and the `False` path is about 4x less accurate. `repro.py` needs only mlx and numpy. |
| `mlx-metal-import-residency/` | ml-explore/mlx | An imported buffer is made resident whole, so importing a large mapped region wires all of it on first use. Docs request. |

Every reproducer exits non zero if the effect does not reproduce on the machine running it, so a
green run is evidence and a red run is a reason not to file.
