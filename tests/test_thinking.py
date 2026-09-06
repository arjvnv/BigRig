"""The thinking budget: a reasoning model is left alone until the cap, then made to answer.

What must hold: under budget the processor changes nothing; at the budget, while the block is
still open, it forces the close tag and then steps aside; a model that closes on its own is never
touched; a model that is already answering is never interrupted; the prompt is not counted. The
processor checks are driven with a real tokenizer and synthetic logits; the end-to-end behaviour
(a 400-token unbudgeted run that never answered, versus a 150-token budget that did) was measured
live and is recorded in the commit.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from bigrig_engine.thinking import ThinkingBudget, resolve_budget, THINK_CLOSE   # noqa: E402

FAIL = []


def check(n, c, d=""):
    (print(f"  PASS  {n}") if c else (FAIL.append(n), print(f"  FAIL  {n}  {d}")))


print("=" * 84); print("1. RESOLVING A BUDGET FROM WHAT A REQUEST SENDS"); print("=" * 84)
check("nothing asked -> no cap", resolve_budget(None) is None)
check("an integer is the cap", resolve_budget(150) == 150 and resolve_budget("200") == 200)
check("zero or negative means no cap, not a zero-token cap",
      resolve_budget(0) is None and resolve_budget(-5) is None)
check("OpenAI's reasoning_effort maps to a cap, low < medium < high",
      resolve_budget(None, "low") < resolve_budget(None, "medium") < resolve_budget(None, "high"))
check("an unknown effort word is no cap", resolve_budget(None, "extreme") is None)
check("an explicit budget wins over an effort word", resolve_budget(100, "high") == 100)
try:
    resolve_budget("lots"); check("a non-integer budget is refused", False)
except ValueError:
    check("a non-integer budget is refused with a sentence", True)

print("\n" + "=" * 84); print("2. THE PROCESSOR, WITH A REAL TOKENIZER"); print("=" * 84)
_dir = next((os.path.join(ROOT, "models", m) for m in
             ("Qwen3.6-35B-A3B-4bit", "GLM-4.7-Flash-4bit", "NVIDIA-Nemotron-3-Nano-30B-A3B-4bit",
              "OLMoE-1B-7B-0125-4bit")
             if os.path.exists(os.path.join(ROOT, "models", m, "tokenizer.json"))), None)
if _dir is None:
    print("  SKIPPED - no local tokenizer")
else:
    import mlx.core as mx
    from transformers import AutoTokenizer
    from mlx_lm.tokenizer_utils import TokenizerWrapper
    tok = TokenizerWrapper(AutoTokenizer.from_pretrained(_dir))
    V = len(tok._tokenizer)
    enc = lambda s: tok._tokenizer.encode(s, add_special_tokens=False)   # noqa: E731
    close_ids = enc(THINK_CLOSE)
    PROMPT = enc("Question: what is two plus two?")
    word = enc(" reasoning")[0]

    def run(proc, gen_ids, prefer):
        lg = mx.full((V,), -10.0).at[prefer].add(20.0)
        toks = mx.array(PROMPT + gen_ids, dtype=mx.int32)
        return int(mx.argmax(proc(toks, lg)))

    # Shape A: the prompt opened the block (Qwen3.6, GLM). Every token counts from the start.
    p = ThinkingBudget(tok, budget=5, starts_in_reasoning=True)
    run(p, [], word)                                    # first call anchors past the prompt
    check("the prompt is not counted as reasoning", p.reasoning_tokens == 0 and p.seen == len(PROMPT))
    gen = []
    untouched = True
    for i in range(5):
        got = run(p, gen, word)
        untouched &= (got == word)
        gen.append(got)
    check("under budget, the model's own choice passes through untouched, 5 of 5", untouched)
    got = run(p, gen, word)
    check("AT the budget, while still thinking, the close tag is forced instead",
          got == close_ids[0], f"got {tok.decode([got])!r}")
    gen.append(got)
    # the remaining close ids, if the tag is multi-token
    for cid in close_ids[1:]:
        got = run(p, gen, word); check("...the rest of a multi-token close tag follows", got == cid); gen.append(got)
    check("after the forced close the block is marked closed", p.closed)
    got = run(p, gen, word)
    check("...and the model is left alone again to answer", got == word)
    # But a confused model that wants to emit the tag AGAIN after the close is refused -- measured
    # once in four sampled runs, it leaked a literal '</think>' into the answer text.
    got = run(p, gen, close_ids[0])
    check("after the close, the close tag itself is no longer a legal answer token", got != close_ids[0],
          repr(tok.decode([got])))
    got = run(p, gen, enc("<think>")[0])
    check("...nor is a fresh opening tag", got != enc("<think>")[0])

    # Shape B: the reply opens its own block. Nothing is counted until <think> appears, and an
    # answer given without any block is never touched.
    p = ThinkingBudget(tok, budget=3, starts_in_reasoning=False)
    run(p, [], word)
    gen = []
    for i in range(6):                                  # six plain tokens, no block: an answer
        got = run(p, gen, word); gen.append(got)
    check("a model that answers without thinking is never interrupted",
          all(t == word for t in gen) and p.reasoning_tokens == 0 and not p.closed)
    # now the block opens; count from there
    for t in enc("<think>"):
        gen.append(t); run(p, gen, word)
    check("the opening tag switches counting on", p.in_reasoning)
    for i in range(3):
        got = run(p, gen, word); gen.append(got)
    got = run(p, gen, word)
    check("the budget then applies from the opening tag, not from token zero",
          got == close_ids[0], f"reasoning_tokens={p.reasoning_tokens}")

    # A model that closes on its own, under budget, is never forced.
    p = ThinkingBudget(tok, budget=50, starts_in_reasoning=True)
    run(p, [], word)
    gen = [word, word] + close_ids
    for i in range(len(gen)):
        run(p, gen[:i + 1], word)
    check("a model that closes on its own is marked closed without being forced",
          p.closed and not p.forcing and p.reasoning_tokens == 2)
    got = run(p, gen, word)
    check("...and everything after is passed through", got == word)

    # A processor with no close ids (a tokenizer that cannot encode the tag) does nothing.
    class _T:
        def encode(self, s, **k): return []
        def decode(self, ids): return ""
    p = ThinkingBudget(_T(), budget=1, starts_in_reasoning=True)
    p.seen = 0
    out = p(mx.array([1, 2, 3], dtype=mx.int32), mx.zeros((V,)))
    check("a tokenizer without a close tag leaves logits untouched", float(mx.sum(mx.abs(out))) == 0.0)

print()
print("=" * 84)
print("ALL TESTS PASSED" if not FAIL else f"{len(FAIL)} FAILURES: " + ", ".join(FAIL))
print("=" * 84)
sys.exit(1 if FAIL else 0)
