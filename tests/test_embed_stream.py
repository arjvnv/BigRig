"""Streaming the input embedding: bit-exact, and off exactly when it would be wrong.

The guarantee is that a streamed embedding returns the same rows the resident module would, and
that it is attempted ONLY when it is safe -- untied from the output head, quantised in a shape
the gather handles. The row-gather-and-dequantise check needs a real quantised embedding, so it
runs against a local model when one is present and skips cleanly otherwise; the safety gates are
pure and always run.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from bigrig_engine import embed_stream as es                           # noqa: E402

FAIL = []


def check(n, c, d=""):
    (print(f"  PASS  {n}") if c else (FAIL.append(n), print(f"  FAIL  {n}  {d}")))


print("=" * 84); print("1. THE SAFETY GATES (pure)"); print("=" * 84)
check("a missing model directory yields no saving, not an exception",
      es.streamable_gb("/nonexistent/model/xyz") == 0.0)
_tied_dir = os.path.join(ROOT, "models", "Qwen3-MOE-4x0.6B-2.4B-Writing-Thunder-V1.2-mlx-4Bit")
if os.path.isdir(_tied_dir):
    check("a tied model is refused (the output head needs the whole matrix)",
          es.streamable_gb(_tied_dir) == 0.0)
else:
    print("  SKIPPED - no tied model locally to check the tie refusal")

print("\n" + "=" * 84); print("2. BIT-EXACT ROW GATHER (needs a local quantised embedding)"); print("=" * 84)
_dir = next((os.path.join(ROOT, "models", m) for m in
             ("Qwen3.6-35B-A3B-4bit", "DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx",
              "GLM-4.7-Flash-4bit", "Qwen3-30B-A3B-3bit")
            if os.path.isdir(os.path.join(ROOT, "models", m))
            and es.streamable_gb(os.path.join(ROOT, "models", m)) > 0), None)
if _dir is None:
    print("  SKIPPED - no local untied quantised model to gather against")
else:
    import mlx.core as mx
    from mlx_lm import load
    model, _tok = load(_dir, lazy=True)
    found = es._find_embedding(model)
    check("the embedding module is found", found is not None)
    parent, leaf, emb = found
    V = int(emb.weight.shape[0])
    ids = mx.array([[0, 1, 7, 123, 4000, V - 1, V // 2, 42]])
    resident = emb(ids); mx.eval(resident)
    attached = es.attach(model, _dir)
    check("attach succeeds on an untied quantised model", attached)
    streamed = getattr(parent, leaf)(ids); mx.eval(streamed)
    check("the streamed rows are the shape the resident module returned",
          resident.shape == streamed.shape, f"{resident.shape} vs {streamed.shape}")
    check("every gathered row is bit-identical to resident, first/last/middle of vocab included",
          float(mx.max(mx.abs(resident - streamed))) == 0.0,
          f"max|d| {float(mx.max(mx.abs(resident - streamed))):.2e}")
    # A second call with different ids must also match -- the gather is stateless.
    ids2 = mx.array([[V - 2, 3, 3, 3, 999]])
    r2 = emb(ids2) if False else None
    # emb was left resident on the model copy we attached to; reload a clean one to compare.
    model2, _ = load(_dir, lazy=True)
    _, _, emb2 = es._find_embedding(model2)
    base = emb2(ids2); mx.eval(base)
    got = getattr(parent, leaf)(ids2); mx.eval(got)
    check("a second gather with repeated and edge ids is also bit-identical",
          float(mx.max(mx.abs(base - got))) == 0.0)
    check("streamable_gb reports the three arrays' packed size, > 0",
          es.streamable_gb(_dir) > 0.0)

print()
print("=" * 84)
print("ALL TESTS PASSED" if not FAIL else f"{len(FAIL)} FAILURES: " + ", ".join(FAIL))
print("=" * 84)
sys.exit(1 if FAIL else 0)
