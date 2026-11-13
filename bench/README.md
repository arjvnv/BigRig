# bench/

The harnesses behind BigRig's speed and exactness numbers. Each one answers a single question and fixes its workload, arms and decision rule in the file before it runs. Each writes its numbers as JSON to `$LAB_RESULTS`, or to the path given on the command line (`--out PATH`; `confirm_decode.py` takes the path as its first argument). The module docstring of each file says what it measures, why, and how the numbers are computed.

| File | Question | Engines compared |
|---|---|---|
| `climb_decode.py` | How fast does the flagship model write once a reply has started, on a 16 GB Mac's budget? | the checkout's engine, against a locked reference reply |
| `climb_turn.py` | How long does one realistic turn (a 1,771-token prompt plus three short ones) take on a 16 GB Mac, against released BigRig? | released engine (exported) vs the live checkout, uncommitted edits included |
| `confirm_turn.py` | Repeated paired rounds of `climb_turn.py`'s eval: a turn speed-up with its 95% confidence interval. | released engine vs `--candidate`, both exported |
| `confirm_decode.py` | Repeated rounds of `climb_decode.py` across three engines, with and without memory pressure. | three exported commits |
| `same_answers.py` | Are BigRig's replies token-identical at every memory budget (every expert plan)? | `--base` vs `--candidate`, both exported |
| `race_llamacpp.py` | BigRig against llama.cpp on the same simulated 16 GB Mac. | released engine (exported) vs llama.cpp |

## How they measure

- **Arms are git commits.** An engine under test is `bigrig_engine/` exported from git with `git archive`, so nothing else differs between arms. The exceptions are named in the table: `climb_decode.py` measures the checkout's own engine, and `climb_turn.py`'s candidate is the live checkout.
- **Each harness invocation gets a throwaway engine home.** An exported arm has no `pyproject.toml` beside it, so `bigrig_engine.home()` would fall back to `~/.bigrig` and read experts from the safetensors shards instead of the packed blob the CLI sets up: a slower path with other kernels. Using the checkout as the home is no better, because each worker's Session restores and saves conversations under `data/sessions` and deletes ones it did not write. So the harnesses run their workers under `climb_turn.scratch_home()`: the checkout's `data/blobs`, `data/results`, `data/corpora` and `data/traces` linked in, and an empty, read-only `data/sessions`, so no run restores or saves a conversation. It is removed afterwards, also when a run is refused (`confirm_decode.py` and the race also on a SIGTERM); removing it unlinks the links and never touches what they point to. Records the engine writes under `data/results` still land in the checkout's. Before any run, each arm's own `expert_source` is asked whether it will read the packed blob (`climb_turn.engine_reads_blob`); a missing, incomplete or outdated blob stops the harness. Run them from the checkout that holds the blobs, not from a fresh worktree.
  - `confirm_turn.py` is frozen as its card ran it: it still uses the checkout itself as the engine home, so it overwrites the checkout's saved conversations for the model it runs, and it only checks that the blob file exists. `climb_decode.py` runs in the checkout too.
- **A 16 GB Mac is simulated on a bigger one** with a balloon of random data that holds back page cache: a fixed 5 GB in `climb_turn.py`, `confirm_turn.py`, `confirm_decode.py`'s 16 GB regime, `same_answers.py`'s speed guard and the race; 8 GB in `climb_decode.py` run on its own. `confirm_decode.py`'s second regime runs without one.
- **Plans.** `climb_turn.py`, `confirm_turn.py` and `confirm_decode.py` pin the expert plan (capacity 8, no whole layers), because BigRig's reply can depend on its plan and the plan on what else is open. `same_answers.py` varies plans on purpose; the race and `climb_decode.py` use the planner's own plan.
- **A fresh process per run.** Arms are interleaved (base / candidate / candidate / base in `climb_turn.py` and `confirm_turn.py`, every order in turn in `confirm_decode.py`, alternating rounds in the race and in `same_answers.py`'s speed guard), so drift such as heat and page cache lands on every arm alike; `same_answers.py`'s exactness cells run base first, then the candidate twice in opposite orders. The harnesses that compare engines stop or wait on Mac state (AC power, memory pressure, swap growth); `climb_decode.py` run on its own does not.
- **Exactness.** `climb_turn.py` records whether every candidate reply matches the base reply in its pair (the climb's config gates on it, and `confirm_turn.py` stops and fails on a mismatch). `climb_decode.py` and `confirm_decode.py` compare with a locked reference reply. `same_answers.py` compares every candidate cell with its first condition. The race records exactness but does not gate on it.

## Tests

They need no model and run in `./run-tests.sh`.

- `tests/test_climb_turn.py`: `climb_turn.py`'s arithmetic, every gate fed the failure it exists to catch, and the real parent/worker plumbing on fake engines, including that every worker inherits the caller's engine home.
- `tests/test_confirm_turn.py`: the turn confirm's arithmetic and decision rule (speed, interval, decode, exactness and memory cases) and its plumbing.
- `tests/test_same_answers.py`: the invariance verdict, the stops, and an end-to-end job on fake engines.
- `tests/test_bench_home.py`: the throwaway engine home and the pre-run blob check, through the real `main()` of `climb_turn.py`, `confirm_decode.py`, the race and `same_answers.py`.

Not covered by tests: `climb_decode.py` itself, `confirm_decode.py`'s analysis and verdicts, the race's decision and health checks, and the time-to-first-token side of `same_answers.py`'s speed guard.
