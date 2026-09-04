"""Measure the meter's overhead on a REAL model. The README says it is free; that must be proven.

Design carried over from the benchmark harness that survived a hostile audit:
  - arms INTERLEAVED and reversed on alternate blocks, so thermal drift cancels
  - a NULL control (baseline run twice) to expose measurement bias
  - the MINIMUM across blocks, because interference can only ever ADD time
"""
import os, sys, time
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
from bigrig_layer import QualityMeter

NTOK, BLOCKS = 64, 24


def run(model, ids, meter, eos):
    kv = make_prompt_cache(model)
    if len(ids) > 1:
        model(mx.array(ids[:-1])[None], cache=kv); mx.eval([c.state for c in kv])
    y = int(ids[-1])
    mx.synchronize(); t0 = time.perf_counter()
    for _ in range(NTOK):
        lg = model(mx.array([y])[None], cache=kv); mx.eval(lg)
        v = lg[0, -1].astype(mx.float32)
        p = mx.softmax(v, axis=-1)
        if meter is not None:
            meter.observe(np.array(p, copy=False))     # the cost under test
        y = int(mx.argmax(v).item())
        if y in eos: break
    mx.synchronize()
    return time.perf_counter() - t0


def main():
    mdl = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "Ling-mini-2.0-3bit")
    model, tok = load(mdl); memguard.checkpoint("loaded")
    eos = set()
    for a in ("eos_token_ids", "eos_token_id"):
        v = getattr(tok, a, None)
        if isinstance(v, (list, tuple, set)): eos |= {int(x) for x in v}
        elif isinstance(v, int): eos.add(v)
    ids = tok.encode(tok.apply_chat_template(
        [{"role": "user", "content": "Explain how virtual memory works in a modern OS."}],
        add_generation_prompt=True, tokenize=False))
    ARMS = [("A_baseline", False), ("A2_control", False), ("M_meter", True)]
    run(model, ids, None, eos)                                   # discarded warm-up
    T = {n: [] for n, _ in ARMS}
    for b in range(BLOCKS):
        order = ARMS if b % 2 == 0 else list(reversed(ARMS))
        for name, use in order:
            m = QualityMeter() if use else None
            T[name].append(run(model, ids, m, eos))
        if b % 8 == 0:
            print(f"  block {b:>2}  " + "  ".join(
                f"{k} {min(v)/NTOK*1000:.3f}ms" for k, v in T.items() if v), flush=True)
    ms = {k: np.array(v) / NTOK * 1000 for k, v in T.items()}
    base = ms["A_baseline"].min()
    bias = ms["A2_control"].min() - base
    ovh = ms["M_meter"].min() - base
    print(f"\n  baseline        {base:.4f} ms/token")
    print(f"  NULL control    {bias:+.4f} ms/token   <- measurement bias; the floor of what")
    print(f"                                            this rig can resolve")
    print(f"  METER overhead  {ovh:+.4f} ms/token   ({ovh/base*100:+.2f}% of a token)")
    if abs(bias) >= abs(ovh) * 0.5:
        print(f"\n  VERDICT: UNRESOLVED — the overhead is within the measurement bias. All that")
        print(f"           can be claimed is that it is BELOW {max(abs(bias),abs(ovh)):.4f} ms/token.")
    elif ovh / base < 0.01:
        print(f"\n  VERDICT: under 1% of per-token time. The 'free' claim holds.")
    else:
        print(f"\n  VERDICT: {ovh/base*100:.2f}% — NOT free. The README must say so.")
    import json
    json.dump({"baseline_ms": float(base), "null_bias_ms": float(bias),
               "meter_overhead_ms": float(ovh), "pct": float(ovh / base * 100)},
              open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "results", "layer_overhead.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
