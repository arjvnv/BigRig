"""END TO END ON AN ENGINE WE DID NOT WRITE.

Runs the quality meter against a live ollama server -- which is llama.cpp underneath -- using
nothing but the `logprobs` its ordinary HTTP API already returns. No patching, no fork, no
in-process hooks. That is the whole claim of Phase 1: the layer attaches to what people already
run.

To show the meter DOING something rather than merely running, the same prompt is generated at
several temperatures. TWO THINGS were learned building this and are worth stating, because the
obvious version of this demo does not work:

  1. TEMPERATURE 2.5 IS NOT DAMAGE THE METER CAN SEE. It produced "reverse-phase freezing and
     melting of water" -- factually wrong, entirely fluent. The meter detects LOOPING and
     INCOHERENCE, not false statements, and this is exactly the documented blind spot. Only at
     much higher temperatures does the text become incoherent rather than merely wrong.
  2. THE BASELINE MUST BE SHARED. A meter that learns its baseline from the damaged run alone
     decides that damage is this model's normal. It is primed on the healthy run first, which is
     also what a real deployment does.

    python -m bigrig_layer.example_ollama --model qwen2.5:1.5b
"""
import argparse
import json
import sys
import urllib.request

from .adaptive import AdaptiveMeter
from .adapters import observe_ollama_entry

PROMPT = ("Explain how a heat pump moves warmth from cold outdoor air into a house. "
          "Be concrete and specific.")


def generate(host, model, temperature, ntok, top_logprobs=8, timeout=600):
    body = json.dumps({
        "model": model, "prompt": PROMPT, "stream": False,
        "options": {"num_predict": ntok, "temperature": temperature, "seed": 7},
        "logprobs": True, "top_logprobs": top_logprobs,
    }).encode()
    req = urllib.request.Request(f"{host}/api/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def watch(payload, vocab, window, m=None, learn_only=False):
    """Feed every token and report what the meter saw. `m` carries the learned baseline across
    runs, which is the deployment pattern; a fresh meter per run would let each run define its
    own normal."""
    if m is None:
        m = AdaptiveMeter(window=window, warmup=2, min_window=max(4, window // 4))
    m.reset(keep_baseline=True)
    lps = payload.get("logprobs") or []
    fed = skipped = flagged = 0
    first_flag = None
    reasons = {}
    for i, e in enumerate(lps):
        got = observe_ollama_entry(m, e, vocab)
        if got is None:
            skipped += 1
            continue
        fed += 1
        if learn_only:
            continue
        d = m.is_degraded()
        if d:
            flagged += 1
            if first_flag is None:
                first_flag = i + 1
            r = m.reason()
            reasons[r] = reasons.get(r, 0) + 1
    return {"tokens": len(lps), "fed": fed, "skipped": skipped, "flagged": flagged,
            "flag_rate": 100.0 * flagged / max(fed, 1), "first_flag": first_flag,
            "reasons": reasons, "meter": m}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--model", default="qwen2.5:1.5b")
    ap.add_argument("--vocab", type=int, default=151936)
    ap.add_argument("--ntok", type=int, default=200)
    ap.add_argument("--window", type=int, default=32)
    ap.add_argument("--top-logprobs", type=int, default=8)
    a = ap.parse_args()

    print(f"engine: ollama at {a.host}   model: {a.model}   (llama.cpp underneath)")
    print(f"the meter sees ONLY the top-{a.top_logprobs} logprobs the HTTP API already returns\n")
    m = AdaptiveMeter(window=a.window, warmup=2, min_window=max(4, a.window // 4))
    # prime the baseline on healthy output, twice, before judging anything
    for _ in range(2):
        try:
            watch(generate(a.host, a.model, 0.7, a.ntok, a.top_logprobs), a.vocab, a.window,
                  m=m, learn_only=True)
        except Exception as e:
            print(f"  could not reach ollama: {e}")
            print("  start it with `ollama serve`, and pull the model first.")
            return 2
    print(f"  baseline learned from healthy output: z-bar {m.z_bar():.2f}, "
          f"repetition bar {m.rep_threshold():.3f}\n")

    rows = []
    for label, temp in (("NORMAL    (temperature 0.7)", 0.7),
                        ("HOT       (temperature 2.5)", 2.5),
                        ("INCOHERENT(temperature 5.0)", 5.0)):
        p = generate(a.host, a.model, temp, a.ntok, a.top_logprobs)
        r = watch(p, a.vocab, a.window, m=m)
        txt = (p.get("response") or "").replace("\n", " ")
        print(f"{'='*88}\n{label}\n{'='*88}")
        print(f"  tokens {r['tokens']}   fed to meter {r['fed']}   skipped {r['skipped']}")
        print(f"  FLAGGED {r['flagged']}/{r['fed']} windows ({r['flag_rate']:.1f}%)"
              f"   first flag at token {r['first_flag']}")
        print(f"  reasons {r['reasons'] or '{}'}")
        print(f"  text    {txt[:200]}...")
        rows.append((label, r))
        print()

    ok = rows[-1][1]["flag_rate"] > rows[0][1]["flag_rate"]
    print("=" * 88)
    for lab, r in rows:
        print(f"  {lab} flagged {r['flag_rate']:5.1f}% of windows")
    print(f"  -> {'THE METER SEPARATES THEM' if ok else 'IT DID NOT SEPARATE THEM'}")
    if not ok:
        print("     Report this rather than tuning until it passes: on a small model at a short")
        print("     window the separation can genuinely fail, and that is information.")
    print("=" * 88)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
