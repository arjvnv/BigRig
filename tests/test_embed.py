"""The embeddings endpoint: an encoder that matches its reference, served the way OpenAI's is.

What must hold. The MLX encoder reproduces sentence-transformers (PyTorch) on the same texts --
the reference vectors were computed once with torch and are kept as fixtures, so this re-checks
without torch installed. Both pooling shapes (CLS for bge, mean for MiniLM-style) match. The wire
shape is OpenAI's, including base64 and `dimensions`. Bad input is a 400 with a sentence; a server
started without `--embeddings` says how to enable it; the encoder shares the model thread's queue,
so a chat reply and an embedding never touch the device at once. The encoder checks need the
default model downloaded (`bigrig serve <model> --embeddings` fetches it) and skip cleanly
otherwise; the wire-shape checks run everywhere.
"""
import base64
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

from bigrig_engine import embed                                         # noqa: E402

FAIL = []


def check(n, c, d=""):
    (print(f"  PASS  {n}") if c else (FAIL.append(n), print(f"  FAIL  {n}  {d}")))


print("=" * 84); print("1. THE REQUEST, PARSED STRICTLY"); print("=" * 84)
check("a string is one input", embed.parse_input({"input": "hello"}) == ["hello"])
check("a list of strings is passed through", embed.parse_input({"input": ["a", "b"]}) == ["a", "b"])
for bad, why in (({"input": ""}, "empty"), ({"input": []}, "empty list"), ({"input": ["a", ""]}, "an empty item"),
                 ({"input": [[1, 2, 3]]}, "token ids"), ({"input": 5}, "a number"),
                 ({"input": ["x"] * (embed.MAX_ITEMS + 1)}, "too many"),
                 ({"input": "x" * (embed.MAX_CHARS + 1)}, "too long"),
                 ({"input": "a", "encoding_format": "hex"}, "unknown format"),
                 ({"input": "a", "dimensions": 0}, "zero dimensions"),
                 ({"input": "a", "dimensions": "384"}, "dimensions as a string")):
    try:
        embed.parse_input(bad)
        check(f"refuses {why}", False)
    except ValueError as e:
        check(f"refuses {why} with a sentence", len(str(e)) > 20 and "`" in str(e), str(e))

print("\n" + "=" * 84); print("2. THE RESPONSE, IN OPENAI'S SHAPE"); print("=" * 84)
V = np.array([[3.0, 4.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
V = V / np.linalg.norm(V, axis=1, keepdims=True)
r = embed.response(V, "enc", 7, 0)
check("object, data, model and usage are where clients look",
      r["object"] == "list" and r["model"] == "enc" and r["usage"] == {"prompt_tokens": 7, "total_tokens": 7}
      and [d["index"] for d in r["data"]] == [0, 1] and all(d["object"] == "embedding" for d in r["data"]))
check("floats come back as plain lists", r["data"][0]["embedding"][:2] == [0.6000000238418579, 0.800000011920929]
      or np.allclose(r["data"][0]["embedding"], V[0]))
check("no truncation, no note", "bigrig" not in r)
rb = embed.response(V, "enc", 7, 0, encoding_format="base64")
back = np.frombuffer(base64.b64decode(rb["data"][0]["embedding"]), dtype="<f4")
check("base64 is little-endian float32 that decodes to the same vector", np.allclose(back, V[0]) and len(back) == 4)
rd = embed.response(V, "enc", 7, 0, dimensions=2)
d0 = np.array(rd["data"][0]["embedding"])
check("`dimensions` cuts the vector and re-normalises it, as the OpenAI API does",
      len(d0) == 2 and abs(np.linalg.norm(d0) - 1.0) < 1e-6 and abs(d0[0] - 0.6) < 1e-6, str(d0))
check("...and asking for more than there are leaves it whole",
      len(embed.response(V, "enc", 7, 0, dimensions=99)["data"][0]["embedding"]) == 4)
rt = embed.response(V, "enc", 7, 2)
check("truncated inputs are said out loud, not hidden in a smaller usage number",
      rt.get("bigrig", {}).get("truncated_inputs") == 2 and "cut" in rt["bigrig"]["note"])

print("\n" + "=" * 84); print("3. THE ENCODER AGAINST ITS PYTORCH REFERENCE"); print("=" * 84)
DIR = embed.local_dir(embed.DEFAULT_REPO)
FIX = os.path.join(ROOT, "tests", "fixtures")
E = None
if not embed.is_local(embed.DEFAULT_REPO):
    print(f"  SKIPPED - {embed.DEFAULT_REPO} is not downloaded (`bigrig serve <model> --embeddings` fetches it)")
else:
    E = embed.Embedder(DIR)
    check("the default encoder loads with the documented shape",
          E.dimensions == 384 and E.max_tokens == 512 and E.pooling == "cls" and E.normalize
          and 0.12 < E.gb < 0.14, f"{E.dimensions} {E.max_tokens} {E.pooling} {E.normalize} {E.gb:.3f}")
    texts = json.load(open(os.path.join(FIX, "embed_texts.json")))
    ref = np.load(os.path.join(FIX, "embed_ref_bge_cls.npy"))
    ours, n_tok, cut = E.embed(texts, batch_size=8)
    cos = (ours * ref).sum(1)
    diff = np.abs(ours - ref).max()
    check(f"CLS pooling: all {len(texts)} vectors match sentence-transformers (cosine >= 0.99999)",
          bool((cos >= 0.99999).all()), f"min cosine {cos.min():.6f}")
    check("...to float32 noise, not to a tolerance argument", diff < 1e-5, f"max abs diff {diff:.2e}")
    check("...every vector unit length", np.allclose(np.linalg.norm(ours, axis=1), 1.0, atol=1e-5))
    check("the two inputs past the 512-token window are cut and counted", cut == 2, str(cut))
    check("tokens counted are the tokens read, so usage is honest about the cut",
          n_tok == sum(min(E.count_tokens(t), 512) for t in texts), str(n_tok))
    E.pooling = "mean"
    ref_m = np.load(os.path.join(FIX, "embed_ref_bge_mean.npy"))
    ours_m, _, _ = E.embed(texts, batch_size=8)
    cos_m = (ours_m * ref_m).sum(1)
    check("mean pooling (the MiniLM shape) matches its reference too",
          bool((cos_m >= 0.99999).all()) and np.abs(ours_m - ref_m).max() < 1e-5,
          f"min cosine {cos_m.min():.6f} max diff {np.abs(ours_m - ref_m).max():.2e}")
    E.pooling = "cls"
    one, _, _ = E.embed([texts[2]])
    check("a text embeds the same alone as inside a padded batch", np.abs(one[0] - ours[2]).max() < 1e-5)
    a, _, _ = E.embed(["The cat sits on the mat.", "A cat is sitting on a mat.", "Quarterly revenue grew twelve percent."])
    check("the semantics are right way round: paraphrase close, unrelated far",
          a[0] @ a[1] > 0.9 and a[0] @ a[2] < 0.5, f"{a[0] @ a[1]:.3f} vs {a[0] @ a[2]:.3f}")

print("\n" + "=" * 84); print("4. SERVED, THROUGH THE REAL SERVER"); print("=" * 84)
from _fakeserver import fake_server, get, post, FakeSession           # noqa: E402


class _FakeEmbedder:
    name, dimensions, max_tokens, gb = "fake-encoder", 4, 8, 0.001

    def embed(self, texts):
        out = np.zeros((len(texts), 4), dtype=np.float32)
        for i, t in enumerate(texts):
            out[i, len(t) % 4] = 1.0
        return out, sum(len(t.split()) for t in texts), sum(1 for t in texts if len(t.split()) > 8)


with fake_server() as (url, state, fs):
    st, b, _ = post(url, "/v1/embeddings", {"input": "hello"})
    check("without --embeddings the route says how to enable it, as a 404 not a crash",
          st == 404 and "--embeddings" in json.dumps(b), f"{st} {b}")
    _, m, _ = get(url, "/v1/models")
    check("...and /v1/models lists only the chat model", [x["id"] for x in m["data"]] == ["fake-model"])
    _, h, _ = get(url, "/health")
    check("...and /health says embeddings are off", h.get("embeddings") is None)

enc = E if E is not None else _FakeEmbedder()
with fake_server(embedder=enc) as (url, state, fs):
    st, b, _ = post(url, "/v1/embeddings", {"input": ["The cat sits on the mat.", "A cat is sitting on a mat.", "Tax law."]})
    check("a request returns one embedding per input, in order",
          st == 200 and [d["index"] for d in b["data"]] == [0, 1, 2]
          and all(len(d["embedding"]) == enc.dimensions for d in b["data"]), f"{st} {str(b)[:200]}")
    check("...with usage and the encoder's name", b["usage"]["prompt_tokens"] > 0 and b["model"] == enc.name)
    if E is not None:
        v = np.array([d["embedding"] for d in b["data"]])
        check("...and the vectors are the encoder's (paraphrase close, unrelated far)",
              v[0] @ v[1] > 0.9 and v[0] @ v[2] < 0.5, f"{v[0] @ v[1]:.3f} / {v[0] @ v[2]:.3f}")
    st, b64, _ = post(url, "/v1/embeddings", {"input": "hello", "encoding_format": "base64"})
    vec = np.frombuffer(base64.b64decode(b64["data"][0]["embedding"]), dtype="<f4")
    check("base64, which the OpenAI SDK asks for by default, decodes to the right width",
          st == 200 and len(vec) == enc.dimensions)
    st, bad, _ = post(url, "/v1/embeddings", {"input": [[1, 2, 3]]})
    check("token ids are refused with a sentence, as a 400", st == 400 and "another model" in json.dumps(bad), f"{st} {bad}")
    st, bad, _ = post(url, "/v1/embeddings", {"input": ""})
    check("an empty input is a 400", st == 400)
    _, m, _ = get(url, "/v1/models")
    ids = [x["id"] for x in m["data"]]
    check("/v1/models lists the encoder beside the chat model, marked as an embedding model",
          ids == ["fake-model", enc.name] and m["data"][1].get("bigrig", {}).get("kind") == "embedding", str(ids))
    _, h, _ = get(url, "/health")
    check("/health reports the encoder, its width, window and memory",
          (h.get("embeddings") or {}).get("dimensions") == enc.dimensions
          and h["embeddings"]["max_tokens"] == enc.max_tokens and "gb" in h["embeddings"], str(h.get("embeddings")))
    st, c, _ = post(url, "/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 3})
    _, h2, _ = get(url, "/health")
    check("a chat reply still flows through the same queue afterwards, and both count as served",
          st == 200 and h2["requests_served"] == 3, str(h2.get("requests_served")))

print()
print("=" * 84)
print("ALL TESTS PASSED" if not FAIL else f"{len(FAIL)} FAILURES: " + ", ".join(FAIL))
print("=" * 84)
sys.exit(1 if FAIL else 0)
