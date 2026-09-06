"""The quality meter acts: a degrading reply is ended, with why and what to try.

What must hold. The rule fires on the measured thresholds (a run of QUALITY_STOP_RUN flagged
tokens, or QUALITY_STOP_SHARE of the reply after QUALITY_STOP_MIN_TOKENS) and never below them;
a request or the server can turn it off; the remedy names the likeliest cause first; and the
verdict reaches the client on both OpenAI paths. The engine's rule is driven with the meter's
verdict patched on a real OLMoE session (skipped without the model); the thresholds themselves
were chosen from sixteen healthy replies on four models (longest run 2, share <= 6.1%) against a
corrupted-state reply (a run of 65) -- see session.QUALITY_STOP_RUN.
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
os.environ.setdefault("BIGRIG_MAX_GB", "9")

from bigrig_engine import session as S                                  # noqa: E402

FAIL = []


def check(n, c, d=""):
    (print(f"  PASS  {n}") if c else (FAIL.append(n), print(f"  FAIL  {n}  {d}")))


print("=" * 84); print("1. THE THRESHOLDS SIT WELL ABOVE HEALTHY OUTPUT"); print("=" * 84)
check("a run of 16 is 8x the longest run measured on a healthy reply (2)", S.QUALITY_STOP_RUN == 16)
check("a share of 35% is 5.7x the largest measured on a healthy reply (6.1%, a table)", S.QUALITY_STOP_SHARE == 0.35)
check("the share rule waits for 64 tokens so a short reply is never judged on a handful", S.QUALITY_STOP_MIN_TOKENS == 64)
check("both sit far above the meter's own noise floor (FLAG_RUN 3, 2%)",
      S.QUALITY_STOP_RUN > 4 * S.FLAG_RUN and S.QUALITY_STOP_SHARE > 10 * S.FLAG_NOISE_SHARE)


class _Stub:
    _quality_remedy = S.Session._quality_remedy

    def __init__(self, mode="stream"):
        self.strategy = {"mode": mode}


print("\n" + "=" * 84); print("2. THE REMEDY NAMES THE LIKELIEST CAUSE FIRST"); print("=" * 84)
check("guess ahead, when it was asked for", "guess ahead" in _Stub()._quality_remedy(True, 0.6))
check("compressed weights, when the model runs compressed", "--exact" in _Stub("compress")._quality_remedy(False, 0.6))
check("creativity, when the sampler runs hot", "creativity" in _Stub()._quality_remedy(False, 1.3))
check("otherwise: ask again", "ask again" in _Stub()._quality_remedy(False, 0.6))
check("guess ahead outranks compression outranks creativity",
      "guess ahead" in _Stub("compress")._quality_remedy(True, 1.3) and "--exact" in _Stub("compress")._quality_remedy(False, 1.3))

print("\n" + "=" * 84); print("3. THE RULE, ON A REAL SESSION WITH THE METER'S VERDICT PATCHED"); print("=" * 84)
OLMOE = os.path.join(ROOT, "models", "OLMoE-1B-7B-0125-4bit")
if not os.path.isdir(OLMOE):
    print("  SKIPPED - needs OLMoE locally")
else:
    s = S.Session(OLMOE, persist=False)
    msgs = [{"role": "user", "content": "Write a long paragraph about rivers."}]

    def run(**kw):
        last, n = None, 0
        for c, info in s.stream_text(msgs, max_tokens=120, temperature=0.0, **kw):
            n += 1
            last = info
        return n, last
    n0, last0 = run()
    check("with the real meter a healthy reply is never stopped by it (it ends on its own terms)",
          last0.get("stopped_for") is None and s.stats()["quality_stops"] == 0, f"{last0.get('finish_reason')} {last0.get('stopped_for')}")
    _real = s._observe
    s._observe = lambda r: True                                     # every token flagged
    n1, last1 = run()
    check("every token flagged: the reply ends at exactly the run threshold",
          last1.get("stopped_for") == "quality" and last1.get("quality_run") == S.QUALITY_STOP_RUN
          and last1.get("generation_tokens") == S.QUALITY_STOP_RUN, str({k: last1.get(k) for k in ("stopped_for", "quality_run", "generation_tokens")}))
    check("...with finish_reason stop, a reason and a remedy",
          last1.get("finish_reason") == "stop" and last1.get("quality_reason") and "ask again" in (last1.get("remedy") or ""), str(last1.get("remedy")))
    check("...and the session counts it", s.stats()["quality_stops"] == 1)
    n2, last2 = run(quality_stop=False)
    check("a request that opts out gets every token, flags and all",
          last2.get("stopped_for") is None and last2["generation_tokens"] > S.QUALITY_STOP_RUN, str(last2.get("stopped_for")))
    s.quality_stop = False
    n3, last3 = run()
    check("the server-wide switch (--no-quality-stop) does the same", last3.get("stopped_for") is None)
    s.quality_stop = True
    # Just under the run threshold, forever: a run of 15 then one clean token, repeated. The run
    # rule must not fire; the share rule (15/16 flagged) must, once 64 tokens are in.
    state = {"i": 0}

    def alt(r):
        state["i"] += 1
        return state["i"] % 16 != 0
    s._observe = alt
    n4, last4 = run()
    check("a pattern that never completes the run but flags most tokens is caught by the share rule, after 64 tokens",
          last4.get("stopped_for") == "quality" and last4.get("generation_tokens") == S.QUALITY_STOP_MIN_TOKENS
          and (last4.get("quality_run") or 0) < S.QUALITY_STOP_RUN, str({k: last4.get(k) for k in ("stopped_for", "quality_run", "generation_tokens", "quality_flagged")}))
    # Under both thresholds: a run of 15 at the start and clean after. Must run to the limit.
    state["i"] = 0
    s._observe = lambda r: (state.__setitem__("i", state["i"] + 1) or state["i"] <= 15)
    n5, last5 = run()
    check("a run one short of the threshold, then a clean reply, is never stopped",
          last5.get("stopped_for") is None and last5["generation_tokens"] > S.QUALITY_STOP_RUN, str(last5.get("stopped_for")))
    s._observe = _real
    _la_remedy = None
    s._observe = lambda r: True
    n6, last6 = run(lookahead=True)
    check("when guess ahead was asked for, the remedy says to turn it off", "guess ahead" in (last6.get("remedy") or ""), str(last6.get("remedy")))
    s._observe = _real
    s.close()

print("\n" + "=" * 84); print("4. THE VERDICT REACHES THE CLIENT"); print("=" * 84)
from _fakeserver import fake_server, post, sse_events, FakeSession    # noqa: E402


class _Stopping(FakeSession):
    """A session whose reply the meter ends after two words."""

    def stream_text(self, messages=None, prompt="", max_tokens=512, **kw):
        self.calls.append({"messages": messages, "prompt": prompt, "max_tokens": max_tokens, **kw})
        for i, w in enumerate(["one", " two"]):
            last = i == 1
            info = {"token": i + 1, "finish_reason": "stop" if last else None, "tok_s": 50.0, "from_draft": False,
                    "reasoning_delta": "", "prompt_tokens": 3, "generation_tokens": i + 1, "degraded": True,
                    "reasoning_tokens": None, "thinking_cut": False}
            if last:
                info.update({"stopped_for": "quality", "quality_reason": "looping", "quality_run": 16,
                             "quality_flagged": 16, "remedy": "ask again, or shorten the request; the model lost the thread on this one"})
            yield w, info


with fake_server(_Stopping()) as (url, state, fs):
    st, b, _ = post(url, "/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 50})
    bg = b.get("bigrig", {})
    check("blocking: finish_reason stays a plain 'stop' (no client breaks on an unknown value)",
          st == 200 and b["choices"][0]["finish_reason"] == "stop")
    check("...and the bigrig block carries the verdict, the reason, the run and the remedy",
          bg.get("stopped_for") == "quality" and bg.get("quality_reason") == "looping" and bg.get("quality_run") == 16
          and "ask again" in (bg.get("remedy") or ""), str(bg))
    st, payload, _ = post(url, "/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 50, "stream": True}, raw=True)
    fin = [e for e in sse_events(payload) if isinstance(e, dict) and e.get("choices") and e["choices"][0].get("finish_reason")]
    check("streaming: the final frame carries the same verdict",
          fin and fin[-1].get("bigrig", {}).get("stopped_for") == "quality" and fin[-1]["bigrig"].get("remedy"), str(fin and fin[-1].get("bigrig")))
    post(url, "/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 5, "quality_stop": False})
    check("`quality_stop: false` reaches the engine", fs.calls and fs.calls[-1].get("quality_stop") is False, str(fs.calls and fs.calls[-1].get("quality_stop")))
    post(url, "/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 5})
    check("...and it is on when unsaid", fs.calls[-1].get("quality_stop") is True)
with fake_server() as (url, state, fs):
    st, b, _ = post(url, "/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 3})
    check("a reply the meter did not stop carries no stopped_for key at all", "stopped_for" not in b.get("bigrig", {}))

print()
print("=" * 84)
print("ALL TESTS PASSED" if not FAIL else f"{len(FAIL)} FAILURES: " + ", ".join(FAIL))
print("=" * 84)
sys.exit(1 if FAIL else 0)
