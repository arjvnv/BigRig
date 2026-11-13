# bench/

The harnesses behind BigRig's speed and exactness numbers. Each one answers a single question, fixes its workload, arms and decision rule in the file before it runs, and writes its numbers as JSON (to `$LAB_RESULTS`, or `--out`). The module docstring of each file says what it measures, why, and how the numbers are computed.

| File | Question |
|---|---|
| `climb_decode.py` | How fast does the flagship model write once a reply has started, on a 16 GB Mac's budget? |
| `climb_turn.py` | How long does one realistic turn (a 1,771-token prompt plus three short ones) take on a 16 GB Mac, against released BigRig? |
| `confirm_turn.py` | Repeated paired rounds of `climb_turn.py`: a turn speed-up with its 95% confidence interval. |
| `confirm_decode.py` | Repeated paired rounds of `climb_decode.py` across three engines, at two memory levels. |
| `same_answers.py` | Are BigRig's replies token-identical at every memory budget (every expert plan)? |
| `race_llamacpp.py` | BigRig against llama.cpp on the same simulated 16 GB Mac. |

## How they measure

- **Arms are git commits.** Each engine under test is `bigrig_engine/` exported from git with `git archive`, so nothing else differs between arms. (`climb_decode.py` alone measures the checkout's own engine; the others run it through exported arms.)
- **The engine home is pinned to the checkout.** An exported arm has no `pyproject.toml` beside it, so `bigrig_engine.home()` would fall back to `~/.bigrig` and read experts from the safetensors shards, a slower path users never run. Every harness that runs exported arms sets `BIGRIG_HOME` to the checkout for its workers and stops if the packed blob in `data/blobs/` is missing. Run them from the checkout that holds the blob, not from a fresh worktree.
- **A 16 GB Mac is simulated on a bigger one** with a fixed 5 GB balloon of random data that holds back page cache, and a pinned expert plan (capacity 8, no whole layers), so base and candidate are comparable token for token.
- **A fresh process per run, arms interleaved** (base / candidate / candidate / base, or every order in turn), so drift such as heat and page cache lands on every arm alike. Runs stop or wait on Mac state: AC power, memory pressure, swap growth.
- **Exactness is a gate, not a metric.** A candidate reply that differs from the base reply in its pair fails the run.

## Tests

`tests/test_climb_turn.py`, `tests/test_confirm_turn.py`, `tests/test_same_answers.py` and `tests/test_bench_home.py` cover the arithmetic, every decision rule on both sides of each threshold, and the worker plumbing on fake engines. They need no model and run in `./run-tests.sh`.
