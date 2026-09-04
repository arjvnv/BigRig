"""End-to-end example: watch a real model's quality while it writes.

Runs the same model twice -- once healthy, once with the expert cache deliberately starved so
quality degrades -- and prints what the meter says about each. This is the demonstration that
the meter tracks something real, not a synthetic distribution.

    python bigrig_layer/example_mlx.py
"""
import os, sys
import numpy as np
import mlx.core as mx

# These two lines run only when this file is executed as a script. As a module-level import
# they put two directories on sys.path for anyone who merely imported the package -- one of
# them a research directory that is deliberately not shipped, so the insert was dead as well
# as wrong.
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        import memguard
        memguard.arm()
    except ImportError:
        pass
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
import gen_bench as GB
from bigrig_layer import QualityMeter
from bigrig_layer.controller import AutoTuner

PROMPT = "Explain in detail how virtual memory works in a modern operating system."


def generate(model, tok, ids, ntok, eos, watcher=None, temp=0.8, top_p=0.95, seed=7):
    mx.random.seed(seed)
    kv = make_prompt_cache(model)
    model(mx.array(ids[:-1])[None], cache=kv); mx.eval([c.state for c in kv])
    y = int(ids[-1]); out = []
    for _ in range(ntok):
        lg = model(mx.array([y])[None], cache=kv); mx.eval(lg)
        v = lg[0, -1].astype(mx.float32)
        p = mx.softmax(v, axis=-1)
        if watcher is not None:
            watcher.observe(np.array(p, copy=False))
        lp = mx.softmax(v / temp, axis=-1)
        so = mx.argsort(-lp); sp = mx.take(lp, so); cum = mx.cumsum(sp)
        keep = mx.concatenate([mx.array([True]), (cum < top_p)[:-1]])
        sp = mx.where(keep, sp, mx.zeros_like(sp)); sp = sp / sp.sum()
        y = int(so[int(mx.random.categorical(mx.log(sp + 1e-20)).item())].item())
        if y in eos: break
        out.append(y)
        if watcher is not None:
            (watcher.meter if hasattr(watcher, "meter") else watcher).observe_token(y)
    return out


def main():
    mdl = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "Ling-mini-2.0-3bit")
    model, tok = load(mdl)
    cfg = __import__("json").load(open(os.path.join(mdl, "config.json")))
    kind, E = cfg["model_type"], cfg["num_experts"]
    GB.register(model, kind)
    ratio = GB.calibrate_ratio(model, tok, kind)
    eos = set()
    for a in ("eos_token_ids", "eos_token_id"):
        vv = getattr(tok, a, None)
        if isinstance(vv, (list, tuple, set)): eos |= {int(x) for x in vv}
        elif isinstance(vv, int): eos.add(vv)
    ids = tok.encode(tok.apply_chat_template(
        [{"role": "user", "content": PROMPT}], add_generation_prompt=True, tokenize=False))

    for label, lam in [("HEALTHY  (no toll)", 0.0), ("DEGRADED (toll 0.4)", 0.4)]:
        GB.install(model, kind, E, max(1, int(E * 0.25)), lam * ratio)
        m = QualityMeter()
        toks = generate(model, tok, ids, 200, eos, watcher=m)
        txt = tok.decode(toks)
        print(f"\n{'='*84}\n{label}\n{'='*84}")
        print(f"  meter score : {m.score():+.3f}   repetition: {m.repetition():.3f}")
        print(f"  DEGRADED    : {m.is_degraded()}   reason: {m.reason()}")
        print(f"  text        : {txt[:220].replace(chr(10),' ')}...")

    # the closed loop
    print(f"\n{'='*84}\nAUTO-TUNER — the dial adjusts itself\n{'='*84}")
    GB.install(model, kind, E, max(1, int(E * 0.25)), 0.4 * ratio)
    t = AutoTuner(dial_max=0.4, start=0.4)
    generate(model, tok, ids, 200, eos, watcher=t)
    s = t.summary()
    print(f"  started at 0.40 -> ended at {s['final_dial']:.3f}")
    print(f"  back-offs {s['backoffs']}, degraded {s['degraded_frac']:.0%} of acted-on steps"
          f", last reason: {s['last_reason']}")
    print("\n  NOTE: the meter reports THAT quality is poor, not WHY. On an intrinsically hard")
    print("  prompt the loop backs off unnecessarily -- costing speed, not quality.")
    GB.uninstall(kind)


if __name__ == "__main__":
    main()
