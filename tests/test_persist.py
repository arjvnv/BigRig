"""Conversations that survive a restart: exact, bounded, and never restored into the wrong numbers.

What must hold. A saved state comes back bit-identical. A restart resumes a conversation with a
full-prefix hit and the SAME reply a fresh, non-persisted run gives. The disk set mirrors the
memory set: bounded by the cache budget, and an evicted entry's file goes at the next flush. A
state is restored only for the configuration that computed it -- a different KV precision
discards it. Corrupt files are dropped, not tripped over. `--no-persist` writes nothing. The
process-boundary checks need a local model and skip cleanly on a fresh clone; the pure ones run
everywhere.
"""
import json
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from bigrig_engine import persist                                       # noqa: E402

FAIL = []


def check(n, c, d=""):
    (print(f"  PASS  {n}") if c else (FAIL.append(n), print(f"  FAIL  {n}  {d}")))


PY = sys.executable
MODEL = next((os.path.join(ROOT, "models", m) for m in ("OLMoE-1B-7B-0125-4bit",)
              if os.path.isdir(os.path.join(ROOT, "models", m))), None)

print("=" * 84); print("1. PURE: NAMING, EMPTY STORE, CLEAR"); print("=" * 84)
check("the key of a token list is stable and short",
      persist._key([1, 2, 3]) == persist._key([1, 2, 3]) and len(persist._key([1, 2, 3])) == 24)
check("different token lists get different keys", persist._key([1, 2, 3]) != persist._key([1, 2, 4]))
check("a temp name still ends in .safetensors (mx.save_safetensors appends one otherwise)",
      persist.TMP_SUFFIX.endswith(persist.SUFFIX))
with tempfile.TemporaryDirectory() as td:
    _orig_root = persist.root
    persist.root = lambda: os.path.join(td, "sessions")
    try:
        check("an empty store summarises to nothing", persist.summary() == [])
        check("clearing an empty store is a no-op that reports zero",
              persist.clear() == {"conversations": 0, "bytes": 0, "models": 0})
    finally:
        persist.root = _orig_root

print("\n" + "=" * 84); print("2. THE SIDE TABLE THE FLUSH WALKS NEVER FORGETS A HELD ENTRY, NEVER GROWS UNBOUNDED"); print("=" * 84)
try:
    import mlx.core as mx
    from mlx_lm.models.cache import KVCache
    from bigrig_engine.session import _TwoStagePromptCache
    _pc = _TwoStagePromptCache(10 ** 9, max_size=4)

    def _entry():
        c = KVCache()
        c.update_and_fetch(mx.zeros((1, 1, 3, 8)), mx.zeros((1, 1, 3, 8)))
        return [c]
    for i in range(6):
        _pc.insert_cache("m", [1, i], _entry(), proven=True)      # the LRU keeps the newest 4
    for i in range(10):
        _pc.insert_cache("m", [2, i], _entry(), proven=False)     # a burst in the OTHER segment
    _prov = sorted(k for k, v in _pc.known.items() if v["proven"])
    _unpr = sorted(k for k, v in _pc.known.items() if not v["proven"])
    check("the table is bounded per segment (4 + 4), not by global recency",
          len(_prov) == 4 and len(_unpr) == 4, f"{len(_prov)} / {len(_unpr)}")
    check("a held protected entry survives a burst of unproven inserts", _prov == [(1, 2), (1, 3), (1, 4), (1, 5)], str(_prov))
    check("held() -- the public-lookup census -- agrees exactly with the table",
          sorted(_pc.held()) == sorted(_prov + _unpr), str(sorted(_pc.held())))
    _pc.trim_to(n_bytes=0)
    check("after a trim to zero nothing is held, and the table is marked for reconciling",
          _pc.held() == [] and _pc.dirty)
except ModuleNotFoundError:
    print("  SKIPPED - mlx not installed")

if MODEL is None:
    print("\n  SKIPPED - the rest needs a local model (OLMoE)")
else:
    # Everything below runs in child processes so a "restart" is a real one: a new process, a new
    # Session, nothing in memory from before. BIGRIG_HOME-independent: the store root is pointed
    # at a temp dir through an env var the children read.
    HELPER = os.path.join(tempfile.gettempdir(), "bigrig_persist_helper.py")
    with open(HELPER, "w") as f:
        f.write(r'''
import os, sys, json
os.environ.setdefault("BIGRIG_MAX_GB", "9")
sys.path.insert(0, os.environ["BIGRIG_TEST_ROOT"])
from bigrig_engine import persist
persist.root = lambda: os.environ["BIGRIG_TEST_STORE"]
from bigrig_engine.session import Session
import mlx.core as mx
M = os.path.join(os.environ["BIGRIG_TEST_ROOT"], "models", os.environ.get("BIGRIG_TEST_MODEL", "OLMoE-1B-7B-0125-4bit"))
LONG = ("You are a careful assistant. Context: the Pacific Ocean is the largest and deepest of Earth's five "
        "oceanic divisions. It extends from the Arctic Ocean in the north to the Southern Ocean in the south, "
        "bounded by Asia and Australia in the west and the Americas in the east. At 165 million square "
        "kilometres it covers about 46% of Earth's water surface and about a third of its total surface area. "
        "Question: which continents bound it to the west?")
def hist_after(reps):
    h = [{"role": "user", "content": LONG}]
    for i, r in enumerate(reps):
        h += [{"role": "assistant", "content": r}, {"role": "user", "content": f"Follow-up {i+1}: name one more fact from the context."}]
    return h
def turn(s, hist, n=40):
    r = "".join(c for c, _ in s.stream_text(hist, max_tokens=n, temperature=0.0))
    st = s.stats()
    return r, st["prompt_cache_matched"], st["prompt_cache_hits"], st["prompt_cache_misses"], st["context_used"]
def states(cache):
    out = []
    for c in cache:
        st = c.state
        out.extend([a for a in (st if isinstance(st, (list, tuple)) else [st]) if isinstance(a, mx.array)])
    return out
cmd, args = sys.argv[1], sys.argv[2:]
kw = json.loads(os.environ.get("BIGRIG_TEST_KW", "{}"))
res = {}
if cmd == "talk":            # N turns, flushing between each like the server's idle tick; exit cleanly
    n = int(args[0]); s = Session(M, **kw)
    res["resumed"] = s.resumed
    reps = []; flushes = []
    for i in range(n):
        r, *rest = turn(s, hist_after(reps)); reps.append(r); flushes.append(s.flush_sessions())
    res["reps"] = reps; res["flushes"] = flushes; res["nbytes"] = s._prompt_cache.nbytes if s._prompt_cache else 0
    res["max_bytes"] = s._prompt_cache.max_bytes if s._prompt_cache else 0
    res["fp"] = persist.fingerprint(s); res["stats_persist"] = s.stats()["persist"]
    s.close()
elif cmd == "resume":        # one more turn on top of the given replies, report the hit
    reps = json.loads(args[0]); s = Session(M, **kw)
    res["resumed"] = s.resumed
    r, matched, hits, misses, used = turn(s, hist_after(reps))
    res.update(rep=r, matched=matched, hits=hits, misses=misses, used=used, fp=persist.fingerprint(s),
               stats_resumed=s.stats()["resumed_conversations"])
    s.close()
elif cmd == "roundtrip":     # save one entry and load it back; compare every array
    from mlx_lm.models.cache import load_prompt_cache
    s = Session(M, **kw); turn(s, hist_after([])); s.flush_sessions()
    pc = s._prompt_cache; key = next(iter(pc.known)); cache, rest, _ = pc.fetch_nearest_cache(s._cache_key, list(key))
    d = os.path.join(persist.model_dir(s.name), persist.fingerprint(s)); path = persist._path(d, key)
    back, meta = load_prompt_cache(path, return_metadata=True)
    a, b = states(cache), states(back)
    res["n_arrays"] = len(a)
    res["same_count"] = len(a) == len(b)
    res["equal"] = all(bool(mx.array_equal(x, y)) and x.dtype == y.dtype and x.shape == y.shape for x, y in zip(a, b))
    res["meta_same"] = all(type(x) is type(y) and x.meta_state == y.meta_state for x, y in zip(cache, back))
    res["tokens_back"] = json.loads(meta["tokens"]) == list(key)
    s.close()
elif cmd == "evict":         # fill, flush, then empty the cache and flush again: the files must go
    s = Session(M, **kw); turn(s, hist_after([])); f1 = s.flush_sessions()
    d = os.path.join(persist.model_dir(s.name), persist.fingerprint(s))
    res["files_before"] = len([f for f in os.listdir(d) if f.endswith(".safetensors")])
    s.trim_prompt_cache(0); f2 = s.flush_sessions()
    res["files_after"] = len([f for f in os.listdir(d) if f.endswith(".safetensors")])
    res["f1"] = f1; res["f2"] = f2
    s.close()
print("RESULT " + json.dumps(res))
''')

    def child(cmd, *args, kw=None, store=None):
        env = dict(os.environ, BIGRIG_TEST_ROOT=ROOT, BIGRIG_TEST_STORE=store,
                   BIGRIG_TEST_KW=json.dumps(kw or {}), BIGRIG_MAX_GB=os.environ.get("BIGRIG_MAX_GB", "9"))
        p = subprocess.run([PY, HELPER, cmd, *args], capture_output=True, text=True, env=env, timeout=600)
        line = next((ln for ln in p.stdout.splitlines() if ln.startswith("RESULT ")), None)
        if line is None:
            print("    child failed:", p.stderr[-1500:])
            return None
        return json.loads(line[len("RESULT "):])

    def files_in(store):
        out = []
        for dp, _, fs in os.walk(store):
            out += [os.path.join(dp, f) for f in fs]
        return out

    with tempfile.TemporaryDirectory() as td:
        store = os.path.join(td, "sessions")

        print("\n" + "=" * 84); print("3. A REAL RESTART RESUMES THE CONVERSATION, WITH THE SAME REPLY"); print("=" * 84)
        A = child("talk", "2", store=store)
        check("process A: turn two reused the first turn (the cache works before anything is saved)",
              A is not None and A["flushes"] and A["reps"] and len(A["reps"]) == 2)
        if A:
            fl = A["flushes"]
            check("...the flush between turns wrote turn one, and after turn two exactly one file remains "
                  "(turn one is a prefix of turn two and was let go)",
                  fl[0]["written"] == 1 and fl[1]["written"] == 1 and fl[1]["on_disk"] == 1,
                  json.dumps(fl))
            check("...and the file on disk is under the fingerprint directory",
                  len(files_in(store)) == 1 and A["fp"] in files_in(store)[0], str(files_in(store)))
            check("...stats says persistence is on", A["stats_persist"] is True)
            disk = sum(os.path.getsize(f) for f in files_in(store))
            check("the disk set is bounded by the cache budget (and smaller than memory: no step padding)",
                  0 < disk <= A["max_bytes"] and disk <= A["nbytes"], f"{disk} vs mem {A['nbytes']} cap {A['max_bytes']}")
            B = child("resume", json.dumps(A["reps"]), store=store)
            C = child("resume", json.dumps(A["reps"]), kw={"persist": False}, store=os.path.join(td, "empty"))
            check("process B restored exactly one conversation", B is not None and B["resumed"]["restored"] == 1,
                  str(B and B["resumed"]))
            if B and C:
                check("...its FIRST request was a hit on the whole prior conversation, no miss",
                      B["hits"] == 1 and B["misses"] == 0 and B["matched"] > 100, f"{B['matched']} matched, {B['hits']}/{B['misses']}")
                check("...and the reply is identical to a fresh, never-persisted run of the same turns",
                      B["rep"] == C["rep"], f"{B['rep']!r} vs {C['rep']!r}")
                check("...with the same context_used", B["used"] == C["used"])
                check("...and stats reports the resumed count", B["stats_resumed"] == 1)
                check("the never-persisted run wrote nothing", not os.path.exists(os.path.join(td, "empty")))

        print("\n" + "=" * 84); print("4. THE STATE COMES BACK BIT-IDENTICAL"); print("=" * 84)
        R = child("roundtrip", store=os.path.join(td, "rt"))
        check("every array in the saved state equals the live one, dtype and shape included",
              R is not None and R["equal"] and R["same_count"] and R["n_arrays"] > 0, str(R))
        check("the cache classes and their meta state match", R is not None and R["meta_same"])
        check("the token ids come back exactly", R is not None and R["tokens_back"])

        print("\n" + "=" * 84); print("5. A DIFFERENT CONFIGURATION NEVER RESTORES INTO THESE NUMBERS"); print("=" * 84)
        store2 = os.path.join(td, "s2")
        A2 = child("talk", "1", store=store2)
        B2 = child("resume", json.dumps(A2["reps"]) if A2 else "[]", kw={"kv_bits": 8}, store=store2)
        check("a run at another KV precision restores nothing",
              B2 is not None and B2["resumed"]["restored"] == 0, str(B2 and B2["resumed"]))
        check("...and discards the other configuration's files rather than keeping them around",
              B2 is not None and B2["resumed"]["discarded"] >= 1 and
              all(B2["fp"] in f for f in files_in(store2)), str(files_in(store2)))
        check("...so its first request is a miss, as a fresh start would be", B2 is not None and B2["misses"] == 1)

        print("\n" + "=" * 84); print("6. EVICTION IS MIRRORED; CORRUPT FILES ARE DROPPED"); print("=" * 84)
        E = child("evict", store=os.path.join(td, "ev"))
        check("an entry the cache lets go loses its file at the next flush",
              E is not None and E["files_before"] == 1 and E["files_after"] == 0, str(E))
        store3 = os.path.join(td, "s3")
        A3 = child("talk", "1", store=store3)
        good = files_in(store3)
        if good:
            with open(good[0], "r+b") as f:
                f.seek(0); f.write(b"\x00" * 64)                # trample the header
            bad = os.path.join(os.path.dirname(good[0]), "deadbeef" * 3 + ".safetensors")
            with open(bad, "wb") as f:
                f.write(b"not a safetensors file")
            B3 = child("resume", json.dumps(A3["reps"]), store=store3)
            after = files_in(store3)
            check("a corrupt file does not stop startup and is removed",
                  B3 is not None and B3["resumed"]["restored"] == 0 and B3["resumed"]["discarded"] == 2
                  and not any("deadbeef" in f for f in after), str(B3 and B3["resumed"]) + " " + str(after))
            check("...and the store then holds only the turn it served afresh -- a longer conversation, a new key",
                  len(after) == 1 and os.path.basename(after[0]) != os.path.basename(good[0])
                  and os.path.getsize(after[0]) > 1_000_000, str(after))
            check("...and the session still answers (a plain miss)", B3 is not None and B3["misses"] == 1)

        print("\n" + "=" * 84); print("7. THE COMMAND SHOWS AND CLEARS WHAT IS KEPT"); print("=" * 84)
        _orig_root = persist.root
        persist.root = lambda: store
        try:
            rows = persist.summary()
            check("summary lists the model with its saved conversations and size",
                  len(rows) == 1 and rows[0]["model"] == "OLMoE-1B-7B-0125-4bit" and rows[0]["conversations"] == 1
                  and rows[0]["bytes"] > 0, str(rows))
            r = persist.clear("no-such-model")
            check("clearing a model with nothing saved removes nothing", r["conversations"] == 0 and persist.summary() == rows)
            r = persist.clear("OLMoE-1B-7B-0125-4bit")
            check("clearing the model removes its files and reports them",
                  r["conversations"] == 1 and r["bytes"] > 0 and persist.summary() == [] and not files_in(store), str(r))
        finally:
            persist.root = _orig_root
    try:
        os.remove(HELPER)
    except OSError:
        pass

print()
print("=" * 84)
print("ALL TESTS PASSED" if not FAIL else f"{len(FAIL)} FAILURES: " + ", ".join(FAIL))
print("=" * 84)
sys.exit(1 if FAIL else 0)
