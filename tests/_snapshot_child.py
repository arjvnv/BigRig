"""One session per process, for tests/test_snapshot.py. Prints one RESULT line of JSON.

A second Session in the same process sees the memory the first one has not yet given back and
plans a smaller cache (or refuses outright), so every scenario here gets a fresh process.
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("BIGRIG_MAX_GB", "9")

from bigrig_engine.session import Session, MIN_REUSE_TOKENS          # noqa: E402

LONG = ("You are a careful assistant. Context: the Pacific Ocean is the largest and deepest of Earth's "
        "five oceanic divisions. It extends from the Arctic Ocean in the north to the Southern Ocean in "
        "the south, bounded by Asia and Australia in the west and the Americas in the east. At 165 "
        "million square kilometres it covers about 46% of Earth's water surface. Question: which "
        "continents bound it to the west? Answer in one short sentence.")


def turns(s, n, max_tokens=40):
    hist = [{"role": "user", "content": LONG}]
    out = []
    for i in range(n):
        r = "".join(c for c, _ in s.stream_text(hist, max_tokens=max_tokens, temperature=0.0))
        st = s.stats()
        out.append({"reply": r, "matched": st["prompt_cache_matched"], "hits": st["prompt_cache_hits"],
                    "misses": st["prompt_cache_misses"], "snapshots": st["history_snapshots"],
                    "used": st["context_used"],
                    "entries": sorted(len(k) for k in (s._prompt_cache.held() if s._prompt_cache else [])),
                    "held_mb": round((s._prompt_cache.nbytes if s._prompt_cache else 0) / 1e6)})
        hist += [{"role": "assistant", "content": r},
                 {"role": "user", "content": f"Follow-up {i + 1}: one more fact, one sentence."}]
    return out, hist


def main():
    mode, model = sys.argv[1], os.path.join(ROOT, "models", sys.argv[2])
    res = {"min_reuse": MIN_REUSE_TOKENS}
    if mode == "trimmable":
        s = Session(model, persist=False)
        res["turns"], _ = turns(s, 3)
        s.close()
    elif mode == "nontrim":
        # OLMoE with mlx_lm's trim check forced off: the regime Qwen3.6 and Nemotron live in, on
        # the real code path. First the miss without the snapshot, then the fix.
        import mlx_lm.models.cache as _cache
        _cache.can_trim_prompt_cache = lambda c: False
        s = Session(model, persist=False)
        snap = s._snapshot_at_history
        s._snapshot_at_history = lambda pc, full_ids, prompt_in, *a: prompt_in
        res["without"], _ = turns(s, 2)
        s.trim_prompt_cache(0)
        s.prompt_cache_hits = s.prompt_cache_misses = 0
        s._snapshot_at_history = snap
        res["with"], hist = turns(s, 3)
        full = s.tokenizer.encode(s._prompt(hist[:1], "", think=True))
        ho = s.tokenizer.encode(s._prompt(hist[:1], "", think=True, history_only=True))
        b = 0
        while b < min(len(full), len(ho)) and full[b] == ho[b]:
            b += 1
        res["boundary"], res["prompt_len"] = b, len(full)
        s._starts_in_reasoning = "sentinel"
        s._prompt(hist[:1], "", think=True, history_only=True)
        res["note_untouched"] = s._starts_in_reasoning == "sentinel"
        s._prompt(hist[:1], "", think=True)
        res["note_set"] = isinstance(s._starts_in_reasoning, bool)
        s.close()
    elif mode == "guards":
        import mlx_lm.models.cache as _cache
        _cache.can_trim_prompt_cache = lambda c: False
        s = Session(model, persist=False)
        "".join(c for c, _ in s.stream_text(prompt="Once upon a time, in a land of", max_tokens=8, temperature=0.0))
        res["raw_prompt_snapshots"] = s.history_snapshots
        hist = [{"role": "user", "content": LONG}, {"role": "assistant", "content": "The continents are"}]
        "".join(c for c, _ in s.stream_text(hist, max_tokens=8, temperature=0.0, continue_last=True))
        res["continued_snapshots"] = s.history_snapshots
        pc = s._prompt_cache
        s.trim_prompt_cache(0)
        keep = pc.probation.max_bytes
        pc.probation.max_bytes = 16
        "".join(c for c, _ in s.stream_text([{"role": "user", "content": LONG}], max_tokens=8, temperature=0.0))
        res["too_large_snapshots"] = s.history_snapshots
        pc.probation.max_bytes = keep
        seen = []
        "".join(c for c, _ in s.stream_text([{"role": "user", "content": LONG + " Extra words here."}],
                                            max_tokens=4, temperature=0.0,
                                            on_prefill=lambda d, t: seen.append((d, t))))
        res["after_snapshots"] = s.history_snapshots
        res["progress"] = seen
        # A segment that holds the snapshot but not the pair: the full entry must yield, and the
        # next turn must still hit. Sized off the state this model actually produced.
        s.trim_prompt_cache(0)
        s.prompt_cache_hits = s.prompt_cache_misses = 0
        pc.probation.max_bytes = int(s._snapshot_bytes * 1.5) if s._snapshot_bytes else keep
        tight, _ = turns(s, 2, max_tokens=8)
        res["tight_entries_after_turn1"] = tight[0]["entries"]
        res["tight_turn2"] = {k: tight[1][k] for k in ("matched", "hits", "misses")}
        res["tight_boundary_only"] = len(tight[0]["entries"]) == 1
        pc.probation.max_bytes = keep
        s.close()
    elif mode == "real":
        from mlx_lm.models.cache import make_prompt_cache, can_trim_prompt_cache
        s = Session(model, persist=False)
        res["trimmable"] = bool(can_trim_prompt_cache(make_prompt_cache(s.model)))
        res["cache_gb"] = s.prompt_cache_gb
        res["probation_mb"] = round(s._prompt_cache.probation.max_bytes / 1e6)
        res["turns"], _ = turns(s, 3, max_tokens=30)
        s.close()
    print("RESULT " + json.dumps(res))


if __name__ == "__main__":
    main()
