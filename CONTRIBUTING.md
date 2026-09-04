# Contributing

## Running the tests

```bash
./run-tests.sh
```

That is the whole thing. It runs every suite and prints one line per assertion.

**On a fresh clone it is green, and some of it is skipped.** Around a third of the assertions
drive a real model -- a packed blob, a live server, a tokenizer -- and no model ships with the
repository. Those sections print `SKIPPED` with the command that would enable them rather than
failing, so a red suite always means something is actually wrong. Prepare a model once:

```bash
bigrig prepare mlx-community/OLMoE-1B-7B-0125-4bit
```

and the skipped sections run. Two blob manifests are tracked as fixtures -- byte offsets only,
no weights -- so the pool-sizing and CLI tests work without downloading anything.

## How the tests are written, and why

They are plain Python scripts, not pytest. Each file prints one line per assertion and exits
non-zero if any fail. No fixtures, no collection, no plugins — you can run any of them directly:

```bash
.venv/bin/python tests/test_stream.py
```

Two rules the suite is held to:

**Assert the invariant, not your expectation.** Several tests here were wrong the first time
because they encoded what the author assumed rather than what must be true — "a low miss rate
means a low disk share" was false at this machine's bandwidth ratio, and "warm miss falls below
the compulsory floor" is only true for short traces. When a test fails, check which of the two
is actually wrong before changing anything.

**Test behaviour, not source text.** Greps for a phrase break when the phrase moves, wraps across
a line, or matches something unrelated — `open(` matches `urlopen(`. Assert what the code *does*.
Six failures in this suite's history came from this and none from a real defect: build a stub and
call the function, or compare a returned value. `_aggregate` is the pattern to copy — it takes a
state object, so a three-line fake exercises every branch.

Where a check really must read a file it cannot execute — `webui.html` is the only one — match
against the normalised copies rather than the raw text: `_uic` has all whitespace removed, for
code; `_uip` has runs collapsed to one space, for prose. Keep each user-facing sentence in a
single string literal in the source, so a phrase a user quotes back at you is findable.

Most tests pin a bug that genuinely happened, and the comment says which. That comment is the
most valuable part of the test; keep it when you touch one.

## Where things live

| | |
|---|---|
| `bigrig_engine/` | the engine. Shipped. |
| `bigrig_layer/` | the quality meter. Shipped, and usable without the engine. |
| `tests/` | every assertion |
| `docs/` | user-facing documentation |

## Measuring anything

If a change claims a speedup or a quality cost, it needs a number from a real run, not an
argument. Two things that have caused wrong numbers here before:

- **Peak memory is a high-water mark.** Measuring two configurations in one process reports the
  first one's peak for both.
- **RSS undercounts MLX memory 8–16× on Apple Silicon**, because GPU buffers are wired and largely
  absent from the resident set. Watch the larger of RSS and MLX device memory rather than `ps`;
  `phys_footprint` from `libproc` is the figure that matched reality here.

## Before opening a change

```bash
./run-tests.sh          # must be green
rig doctor               # sanity: the CLI still works
```

## Licence

Contributions are accepted under the [Apache License 2.0](LICENSE), the same terms the
project is released under.
