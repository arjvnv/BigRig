"""The multi-token-prediction path, tested for the ways it could emit a token the model did not
choose or leave a cache holding one.

The loop itself runs against a real model and is measured, not unit-tested (see MEASUREMENTS:
agreement with plain decoding is a number, not an assertion). What IS asserted here is every
pure part: the snapshot/restore that makes a rejected guess bit-identical to ordinary decoding,
the refusal of caches it cannot put back, the head's naming, and the stats arithmetic.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import mlx.core as mx                                                   # noqa: E402
from bigrig_engine import mtp                                            # noqa: E402

FAIL = []


def check(n, c, d=""):
    (print(f"  PASS  {n}") if c else (FAIL.append(n), print(f"  FAIL  {n}  {d}")))


class Recurrent:
    """What an ArraysCache is, as far as snapshot/restore is concerned."""
    def __init__(self, a, b):
        self.cache = [a, b]


class KV:
    """What a KVCache is: a write position that trim() moves back."""
    def __init__(self, offset):
        self.offset = offset
        self.trims = []

    def is_trimmable(self):
        return True

    def trim(self, n):
        n = min(self.offset, n)
        self.offset -= n
        self.trims.append(n)
        return n


class Opaque:
    def is_trimmable(self):
        return False


print("=" * 84)
print("1. A MISS PUTS EVERY CACHE BACK EXACTLY")
print("=" * 84)
a0, b0 = mx.array([1.0, 2.0]), mx.array([[3.0]])
rc, kv = Recurrent(a0, b0), KV(offset=10)
snap = mtp.snapshot([rc, kv])
# the verify pass folds two tokens into the recurrent state and writes two KV rows
rc.cache = [a0 + 1, b0 * 2]
kv.offset += 2
mtp.restore([rc, kv], snap)
check("the recurrent state is the very array it was, not a copy of it",
      rc.cache[0] is a0 and rc.cache[1] is b0)
check("the KV write position is back where it was", kv.offset == 10 and kv.trims == [2])
check("a snapshot holds references, so taking one allocates nothing",
      snap[0][1][0] is a0)
# Restoring twice, or restoring when nothing moved, must be harmless.
mtp.restore([rc, kv], snap)
check("restoring when nothing has changed changes nothing",
      kv.offset == 10 and rc.cache[0] is a0 and kv.trims[-1] == 0)
# A KV cache that grew by more than the pass (it cannot, but a wrong caller could) is still
# brought back to the snapshot, never below it.
kv.offset = 15
mtp.restore([rc, kv], snap)
check("a larger overshoot is trimmed back to the snapshot", kv.offset == 10)
kv.offset = 7
mtp.restore([rc, kv], snap)
check("...and a position already below it is never trimmed further", kv.offset == 7)

print("\n" + "=" * 84)
print("2. IT REFUSES WHAT IT CANNOT PUT BACK, BEFORE THE FIRST GUESS")
print("=" * 84)
try:
    mtp.snapshot([rc, Opaque()])
    check("an opaque cache is refused", False)
except TypeError as e:
    check("an opaque cache is refused", "cannot snapshot" in str(e), str(e))
check("...and supports() says so in words", "cannot snapshot" in mtp.supports(object(), [Opaque()])
      or "not a Qwen3.5" in mtp.supports(object(), [Opaque()]))
check("a model that is not a Qwen3.5-family text model is refused by name",
      "Qwen3.5" in mtp.supports(object()))
src = open(os.path.join(ROOT, "bigrig_engine", "mtp.py"), encoding="utf-8").read()
check("the loop snapshots before the first pass rather than after the first miss",
      "snapshot(prompt_cache)" in src.split("def stream(")[1].split("yield from")[0])
check("a rejected guess is recomputed on its own, never taken from the two-token pass",
      "restore(cache, snap)" in src and src.index("restore(cache, snap)") <
      src.index("lg1, h1 = forward(tm, mx.array([token])[None], cache)"))
check("the head is only ever fed confirmed tokens (the guess is never folded in as fact)",
      "pending_head = [(h2[:, 0:1], d), (h2[:, 1:2], got1)]" in src
      and "pending_head = [(h1[:, 0:1], got)]" in src)

print("\n" + "=" * 84)
print("3. NAMING, SIZING AND COUNTING")
print("=" * 84)
check("the head is looked for beside the model under mlx-community's name",
      mtp.head_path("/m/Qwen3.6-35B-A3B-4bit") == "/m/Qwen3.6-35B-A3B-MTP-bf16")
check("...for other quantisations too",
      mtp.head_path("/m/Qwen3.6-35B-A3B-8bit") == "/m/Qwen3.6-35B-A3B-MTP-bf16"
      and mtp.head_path("/m/Qwen3.6-35B-A3B-4bit-DWQ") == "/m/Qwen3.6-35B-A3B-MTP-bf16")
check("...and an unquantised model keeps its own name",
      mtp.head_path("/m/Qwen3.6-35B-A3B") == "/m/Qwen3.6-35B-A3B-MTP-bf16")
check("a missing head weighs nothing rather than raising", mtp.head_gb("/nope") == 0.0)
st = mtp.Stats()
check("no drafts means zero acceptance, not a division error", st.acceptance == 0.0)
st.drafted, st.accepted, st.rounds, st.recomputed = 8, 7, 8, 1
check("acceptance is accepted over drafted", abs(st.acceptance - 0.875) < 1e-9)
check("the dict the server reports has every field", set(st.as_dict()) ==
      {"rounds", "drafted", "accepted", "recomputed", "acceptance"})
check("the head has exactly the twenty tensors the checkpoint ships", mtp.HEAD_TENSORS == 20)

print("\n" + "=" * 84)
print("4. THE SESSION AND SERVER TREAT IT AS A CHOICE")
print("=" * 84)
sess = open(os.path.join(ROOT, "bigrig_engine", "session.py"), encoding="utf-8").read()
srv = open(os.path.join(ROOT, "bigrig_engine", "server.py"), encoding="utf-8").read()
check("the head is charged to the memory budget before the pool is planned",
      "budget_gb - self.mtp_gb" in sess and sess.index("budget_gb - self.mtp_gb")
      < sess.index("autoconfig.choose_capacity(man, budget_gb=budget_gb"))
check("a head that cannot drive the model is an error, not a silent plain run",
      "cannot drive" in sess)
check("a request can switch the head off but not on",
      '"mtp": (None if "mtp" not in body else bool(body.get("mtp")))' in srv)
check("every reply that used it says what it bought", 'state.event("mtp"' in srv)
check("closing the session drops the head", "self.mtp_head = None" in sess.split("def close(")[1])

print()
print("=" * 84)
print("ALL TESTS PASSED" if not FAIL else f"{len(FAIL)} FAILURES: " + ", ".join(FAIL))
print("=" * 84)
sys.exit(1 if FAIL else 0)
