"""One session per process, for tests/test_snapshot.py. Prints one RESULT line of JSON.

A second Session in the same process sees the memory the first one has not yet given back and
plans a smaller cache (or refuses outright), so every scenario here gets a fresh process.
"""
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("BIGRIG_MAX_GB", "9")

from bigrig_engine.session import Session, MIN_REUSE_TOKENS          # noqa: E402
import mlx.core as mx                                                  # noqa: E402

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
    elif mode == "lookahead_rollback":
        # After a rejected guess on a model with recurrent state, every layer must be exactly what a
        # plain single step leaves: the same 1-row pass is re-run, so the arrays are bit-identical.
        from mlx_lm.models.cache import make_prompt_cache, can_trim_prompt_cache
        from bigrig_engine import lookahead as LA, rollback as RB
        s = Session(model, persist=False)
        ids = s.tokenizer.encode(s._prompt([{"role": "user", "content": LONG}], "", think=True))
        head, last = ids[:-1], ids[-1]

        def prefilled():
            c = make_prompt_cache(s.model)
            for i in range(0, len(head), 64):
                s.model(mx.array(head[i:i + 64])[None], cache=c)
            mx.eval([x.state for x in c])
            return c

        def arrays(c):
            out = []
            for x in c:
                st = x.state
                out.extend([a for a in (st if isinstance(st, (list, tuple)) else [st]) if isinstance(a, mx.array)])
            return out

        def offsets(c):
            return [int(getattr(x, "offset", -1)) for x in c]
        res["trimmable"] = bool(can_trim_prompt_cache(make_prompt_cache(s.model)))
        # plain step
        c_plain = prefilled()
        lg = s.model(mx.array([last])[None], cache=c_plain)
        t_plain = int(mx.argmax(lg[0, -1])); mx.eval([x.state for x in c_plain])
        # a wholly wrong draft through verify (greedy)
        c_la = prefilled()
        wrong = [(t_plain + 7) % 1000 + 1000, (t_plain + 8) % 1000 + 1000]
        st = LA.Stats()
        got, _, _ = LA.verify(s.model, c_la, last, wrong, None, st)
        mx.eval([x.state for x in c_la])
        a, b = arrays(c_plain), arrays(c_la)
        res["rejected_token_same"] = got == [t_plain]
        res["rejected_offsets_same"] = offsets(c_plain) == offsets(c_la)
        res["rejected_arrays_same_count"] = len(a) == len(b)
        res["rejected_bit_identical"] = all(x.shape == y.shape and x.dtype == y.dtype and bool(mx.array_equal(x, y))
                                            for x, y in zip(a, b))
        res["rejected_stats"] = st.as_dict()
        # WITHOUT the rollback (mlx_lm trim alone): the state must differ -- the defect, reproduced live
        c_bug = prefilled()
        from mlx_lm.models.cache import trim_prompt_cache
        s.model(mx.array([last] + wrong)[None], cache=c_bug); trim_prompt_cache(c_bug, 2); mx.eval([x.state for x in c_bug])
        res["bug_offsets_differ"] = offsets(c_bug) != offsets(c_plain)
        res["bug_arrays_differ"] = not all(x.shape == y.shape and bool(mx.array_equal(x, y)) for x, y in zip(arrays(c_plain), arrays(c_bug)))
        # a fully accepted draft: state must equal two plain steps within numerical noise, offsets exactly
        c_two = prefilled()
        s.model(mx.array([last])[None], cache=c_two); lg2 = s.model(mx.array([t_plain])[None], cache=c_two); mx.eval([x.state for x in c_two])
        c_acc = prefilled(); st2 = LA.Stats()
        got2, _, _ = LA.verify(s.model, c_acc, last, [t_plain], None, st2); mx.eval([x.state for x in c_acc])
        res["accepted_got"] = got2 == [t_plain, int(mx.argmax(lg2[0, -1]))]
        res["accepted_offsets_same"] = offsets(c_two) == offsets(c_acc)
        rel = []
        for x, y in zip(arrays(c_two), arrays(c_acc)):
            if x.shape != y.shape:
                continue
            d = float(mx.max(mx.abs(x.astype(mx.float32) - y.astype(mx.float32))))
            scale = float(mx.max(mx.abs(x.astype(mx.float32)))) or 1.0
            rel.append((d, scale, d / scale, str(x.dtype), list(x.shape)))
        res["accepted_max_abs_diff"] = max(r[0] for r in rel) if rel else None
        res["accepted_max_rel_diff"] = max(r[2] for r in rel) if rel else None
        res["accepted_worst"] = sorted(rel, key=lambda r: -r[2])[:3]
        res["accepted_arrays_identical"] = sum(1 for r in rel if r[0] == 0.0)
        res["accepted_arrays_total"] = len(rel)
        res["accepted_stats"] = st2.as_dict()
        s.close()
    elif mode == "vision":
        # The whole road: image bytes -> tower -> embeddings -> multimodal positions -> a reply
        # that reads the screenshot; then a text-only turn on the same session, untouched.
        import base64
        s = Session(model, persist=False, vision=True)
        st = s.stats()
        res["vision_stats"] = st["vision"]; res["reserved_gb"] = st["reserved_gb"]
        png = open(os.path.join(ROOT, "tests", "fixtures", "vision", "screenshot.png"), "rb").read()
        url = "data:image/png;base64," + base64.b64encode(png).decode()
        msgs = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}},
                                              {"type": "text", "text": "Transcribe all the text in this image exactly, then describe the shapes."}]}]
        t0 = time.perf_counter(); first = None; out = []
        for c, _ in s.stream_text(msgs, max_tokens=160, temperature=0.0, think=False):
            if first is None:
                first = time.perf_counter() - t0
            out.append(c)
        res["reply"] = "".join(out); res["first_token_s"] = round(first or 0, 2); res["seconds"] = round(time.perf_counter() - t0, 2)
        res["prompt_tokens"] = s._this_prompt_full; res["disarmed"] = all(sw.positions is None for sw in s._rope_switches)
        res["tower_held_after"] = s.tower is not None
        res["capacity"] = s.stats().get("capacity")
        res["cache_entries_after_image"] = len(s._prompt_cache.held()) if s._prompt_cache else 0
        # two images in one message: the second is a plain colour so the answer is checkable
        from PIL import Image
        import io
        im = Image.new("RGB", (256, 256), (20, 40, 200)); buf = io.BytesIO(); im.save(buf, "PNG")
        url2 = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        msgs2 = [{"role": "user", "content": [{"type": "text", "text": "First image:"}, {"type": "image_url", "image_url": {"url": url}},
                                               {"type": "text", "text": "Second image:"}, {"type": "image_url", "image_url": {"url": url2}},
                                               {"type": "text", "text": "What colour is the second image? Answer in one word."}]}]
        res["two_image_reply"] = "".join(c for c, _ in s.stream_text(msgs2, max_tokens=12, temperature=0.0, think=False))
        res["text_after"] = "".join(c for c, _ in s.stream_text([{"role": "user", "content": "Say hello in five words."}], max_tokens=20, temperature=0.0, think=False))
        s.close()
    print("RESULT " + json.dumps(res))


if __name__ == "__main__":
    main()
