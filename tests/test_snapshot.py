"""The history-boundary snapshot: multi-turn cache hits on caches that cannot be rolled back.

The miss this fixes (measured on Qwen3.6: three turns, three misses) is described on
Session._snapshot_at_history. What must hold: a trimmable cache takes no snapshot and keeps
hitting as before; a non-trimmable one now hits on turn two with the whole history reused; the
snapshot is keyed on exactly the tokens the next turn begins with; a raw prompt, a continued
turn, or a state too large for its segment takes none; and the progress callback still counts
the whole prompt. The non-trimmable regime is exercised on the real code path two ways: OLMoE
with mlx_lm's trim check forced off (fast, available whenever OLMoE is), and the real thing on
Qwen3.6 or Nemotron when one is on disk. Every scenario is its own process: a second Session in
one process plans against memory the first has not yet given back (see _snapshot_child.py).
"""
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHILD = os.path.join(ROOT, "tests", "_snapshot_child.py")

FAIL = []


def check(n, c, d=""):
    (print(f"  PASS  {n}") if c else (FAIL.append(n), print(f"  FAIL  {n}  {d}")))


def child(mode, model):
    env = dict(os.environ, BIGRIG_MAX_GB=os.environ.get("BIGRIG_MAX_GB", "9"))
    p = subprocess.run([sys.executable, CHILD, mode, model], capture_output=True, text=True,
                       env=env, timeout=900)
    line = next((ln for ln in p.stdout.splitlines() if ln.startswith("RESULT ")), None)
    if line is None:
        print("    child failed:", (p.stdout + p.stderr)[-1500:])
        return None
    return json.loads(line[len("RESULT "):])


OLMOE = "OLMoE-1B-7B-0125-4bit"
if not os.path.isdir(os.path.join(ROOT, "models", OLMOE)):
    print("  SKIPPED - needs OLMoE locally")
else:
    print("=" * 84); print("1. A TRIMMABLE CACHE IS LEFT ALONE"); print("=" * 84)
    T = child("trimmable", OLMOE)
    base = T["turns"] if T else []
    check("turn two and three hit by trimming the longer entry, as before",
          T is not None and base[1]["matched"] > 0 and base[2]["matched"] > base[1]["matched"],
          str([b["matched"] for b in base]))
    check("no snapshot is taken on a trimmable cache", T is not None and base[2]["snapshots"] == 0)

    print("\n" + "=" * 84); print("2. THE NON-TRIMMABLE REGIME, ON THE REAL CODE PATH"); print("=" * 84)
    N = child("nontrim", OLMOE)
    if N:
        miss, got = N["without"], N["with"]
        check("WITHOUT the snapshot a non-trimmable cache misses turn two (the defect, reproduced)",
              miss[1]["matched"] == 0 and miss[1]["misses"] == 2, str(miss[1]))
        check("WITH it, turn one stores a snapshot", got[0]["snapshots"] == 1, str(got[0]))
        check("...turn two hits on the history boundary",
              got[1]["matched"] >= N["min_reuse"] and got[1]["hits"] == 1, str(got[1]))
        check("...turn three reuses the whole prior history, more than turn two did",
              got[2]["matched"] > got[1]["matched"] and got[2]["hits"] == 2, str([g["matched"] for g in got]))
        check("...one snapshot per turn", got[2]["snapshots"] == 3)
        check("the first snapshot is keyed on exactly the boundary tokens",
              N["boundary"] in got[0]["entries"] and N["boundary"] < N["prompt_len"],
              f"boundary {N['boundary']}, prompt {N['prompt_len']}, entries {got[0]['entries']}")
        check("the history-only rendering records no reasoning-start note, the real one does",
              N["note_untouched"] and N["note_set"])
        if base:
            check("replies read the same as the trimmable run (the model saw identical tokens)",
                  [g["reply"] for g in got] == [b["reply"] for b in base],
                  "\n      " + "\n      ".join(f"{g['reply'][:60]!r} vs {b['reply'][:60]!r}"
                                              for g, b in zip(got, base)))
    else:
        check("the non-trimmable scenario ran", False)

    print("\n" + "=" * 84); print("3. WHEN IT MUST NOT RUN"); print("=" * 84)
    G = child("guards", OLMOE)
    if G:
        check("a raw prompt (no messages) takes no snapshot", G["raw_prompt_snapshots"] == 0)
        check("a continued turn takes no snapshot", G["continued_snapshots"] == 0)
        check("a state larger than its segment is not copied (the copy is the only extra memory)",
              G["too_large_snapshots"] == 0, str(G["too_large_snapshots"]))
        check("a snapshot is taken again once it fits", G["after_snapshots"] == 1, str(G["after_snapshots"]))
        seen = [tuple(x) for x in G["progress"]]
        totals = {t for _, t in seen}
        check("the prefill progress callback reports one total for the whole prompt and climbs to it",
              len(totals) == 1 and all(seen[i][0] <= seen[i + 1][0] for i in range(len(seen) - 1))
              and seen and seen[-1][0] >= seen[-1][1] - 1, str(seen))
        check("on a segment too small for snapshot + full entry, the full entry yields (snapshot alone is kept)",
              G["tight_boundary_only"], str(G["tight_entries_after_turn1"]))
        check("...and the next turn still hits", G["tight_turn2"]["hits"] == 1 and G["tight_turn2"]["matched"] > 0,
              str(G["tight_turn2"]))
    else:
        check("the guard scenario ran", False)

    print("\n" + "=" * 84); print("4. THE REAL THING: A MODEL WITH RECURRENT STATE"); print("=" * 84)
    real = next((m for m in ("Qwen3.6-35B-A3B-4bit", "NVIDIA-Nemotron-3-Nano-30B-A3B-4bit")
                 if os.path.isdir(os.path.join(ROOT, "models", m))), None)
    if real is None:
        print("  SKIPPED - no Qwen3.6 / Nemotron locally")
    else:
        R = child("real", real)
        if R:
            got = R["turns"]
            print(f"      {real}: cache {R['cache_gb']} GB, probation {R['probation_mb']} MB")
            for i, g in enumerate(got):
                print(f"      turn {i + 1}: matched {g['matched']:>4} hits {g['hits']} misses {g['misses']} "
                      f"snapshots {g['snapshots']} entries {g['entries']} held {g['held_mb']} MB  "
                      f"reply {g['reply'][:60]!r}")
            check("this model's cache really cannot be trimmed", not R["trimmable"])
            check("turn two hits on the boundary and turn three on the whole prior history",
                  got[1]["matched"] >= R["min_reuse"] and got[2]["matched"] > got[1]["matched"]
                  and got[2]["hits"] == 2 and got[2]["misses"] == 1, str([g["matched"] for g in got]))
            check("one snapshot per turn", got[2]["snapshots"] == 3)
            check("every reply is real text, not empty", all(len(g["reply"].strip()) > 10 for g in got))
        else:
            check("the real-model scenario ran", False)

print()
print("=" * 84)
print("ALL TESTS PASSED" if not FAIL else f"{len(FAIL)} FAILURES: " + ", ".join(FAIL))
print("=" * 84)
sys.exit(1 if FAIL else 0)
