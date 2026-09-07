"""`bigrig doctor` with no model named: the ranked list of what fits.

What must hold. The verdict per model is the doctor's own (remote_shape -> verdict -> speed_tier,
the same functions, not a copy). Rows rank runs-and-fast first, then runs, then does-not-fit,
then unreadable; within a class by size. `--for` filters by tag. A repo the hub cannot serve is a
row that says so, never a failed report. The rendering claims nothing about quality, says nothing
was downloaded, and ends with what to do next. The hub is patched out here so the test is
deterministic and offline; the live path is exercised by hand (`bigrig doctor`, ~25 s).
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from bigrig_engine import recommend, preflight                          # noqa: E402

FAIL = []


def check(n, c, d=""):
    (print(f"  PASS  {n}") if c else (FAIL.append(n), print(f"  FAIL  {n}  {d}")))


print("=" * 84); print("1. THE LIST ITSELF"); print("=" * 84)
check("every entry is an mlx-community repo with tags and a note",
      all(r.startswith("mlx-community/") and tags and note for r, tags, note in recommend.CURATED))
check("every tag is one `--for` accepts",
      all(t in ("chat", "coding", "reasoning", "vision") for _, tags, _ in recommend.CURATED for t in tags))
check("the flagship measured here is on it, and so is the small test model",
      any("Qwen3.6-35B-A3B-4bit" in r for r, _, _ in recommend.CURATED) and any("OLMoE" in r for r, _, _ in recommend.CURATED))
check("no note makes a quality claim the engine has not measured",
      not any(w in note.lower() for _, _, note in recommend.CURATED for w in ("best model", "smartest", "state of the art", "beats")))

print("\n" + "=" * 84); print("2. RANKING, WITH THE HUB PATCHED OUT"); print("=" * 84)
# Synthetic shapes: (n_experts, top_k, per-expert bytes, layers, non-expert GB) chosen so that at a
# 9 GB budget one fits entirely, two stream at different tiers, one does not fit, one is unreadable.
man_path = os.path.join(ROOT, "data", "blobs", "Qwen3-30B-A3B-3bit.experts.manifest.json")
MAN = json.load(open(man_path)) if os.path.exists(man_path) else None


def shape(n_experts, top_k, per_expert, n_layers, non_expert_gb, download_gb):
    layers = {str(i): {"n_experts": n_experts, "bytes_per_expert": per_expert, "experts": {}} for i in range(n_layers)}
    return {"manifest": {"layers": layers, "total_bytes": n_experts * per_expert * n_layers},
            "n_layers": n_layers, "n_experts": n_experts, "top_k": top_k, "bytes_per_expert": per_expert,
            "expert_gb": n_experts * per_expert * n_layers / 1e9, "non_expert_gb": non_expert_gb,
            "download_gb": download_gb, "quantized": True, "dtype": "4bit", "arch": "test_moe"}


SHAPES = {
    "mlx-community/tiny": shape(8, 2, 5_000_000, 16, 0.4, 1.0),                 # fits entirely
    "mlx-community/mid": shape(128, 8, 2_000_000, 48, 0.7, 17.0),               # streams
    "mlx-community/big": shape(512, 10, 4_000_000, 48, 1.5, 44.0),              # streams, slower
    "mlx-community/huge": shape(128, 8, 200_000_000, 94, 6.0, 132.0),           # does not fit
}
REPOS = [("mlx-community/huge", ("chat",), "h"), ("mlx-community/tiny", ("chat",), "t"),
         ("mlx-community/gone", ("coding",), "g"), ("mlx-community/big", ("chat", "coding"), "b"),
         ("mlx-community/mid", ("coding",), "m")]


def fake_remote_shape(repo):
    if repo == "mlx-community/gone":
        raise RuntimeError("404 Client Error: repository not found")
    return SHAPES[repo]


_real = preflight.remote_shape
preflight.remote_shape = fake_remote_shape
try:
    seen = []
    rows = recommend.rank(9.0, 3.3, progress=lambda repo, i, n: seen.append((repo, i, n)), repos=REPOS)
    order = [r["repo"].split("/")[1] for r in rows]
    check("progress is reported once per repo, in list order, with the count",
          [s[0] for s in seen] == [r for r, _, _ in REPOS] and all(s[2] == 5 for s in seen))
    check("an unreadable repo is a row that says why, ranked last, and does not fail the report",
          order[-1] == "gone" and "404" in rows[-1]["error"], str(order))
    check("a model that does not fit ranks after every model that runs", order[-2] == "huge" and not rows[-2]["fits"], str(order))
    fits = [r for r in rows if not r["error"] and r["fits"]]
    check("the ones that run are ranked by tier, then by size", order[:3] == ["tiny", "mid", "big"] or
          [r["tier"] for r in fits] == sorted((r["tier"] for r in fits), key=lambda t: recommend.TIER_ORDER[t]), str([(r["repo"], r.get("tier")) for r in fits]))
    check("the verdict per row is the doctor's own: it agrees with calling verdict() directly",
          all(r["fits"] == preflight.verdict(SHAPES[r["repo"]], 9.0, 3.3, search=False)["fits_now"] for r in rows if not r["error"]))
    only = recommend.rank(9.0, 3.3, want="coding", repos=REPOS)
    check("`--for coding` keeps only coding-tagged models", all("coding" in r["tags"] for r in only) and len(only) == 3)
    text = recommend.render(rows, 9.0)
    check("the rendering says nothing was downloaded and that it is not a quality ranking",
          "nothing was downloaded" in text and "Not a quality ranking" in text)
    check("...names the smallest download and what to do next",
          "Smallest download:    tiny" in text and "bigrig prepare <model>" in text and "bigrig doctor <model>" in text)
    check("...and shows the unreadable row as such", "could not read" in text)
    text2 = recommend.render([r for r in rows if r["repo"].endswith("huge")], 9.0)
    check("when nothing runs it says so and what to change", "Nothing on this list runs" in text2 and "BIGRIG_MAX_GB" in text2)
finally:
    preflight.remote_shape = _real

print("\n" + "=" * 84); print("3. THE COMMAND"); print("=" * 84)
import subprocess                                                        # noqa: E402
h = subprocess.run([sys.executable, "-m", "bigrig_engine.cli", "doctor", "--help"], capture_output=True, text=True, cwd=ROOT).stdout
check("doctor takes no model, --for, and --no-recommend", "[model]" in h and "--for" in h and "--no-recommend" in h)

print()
print("=" * 84)
print("ALL TESTS PASSED" if not FAIL else f"{len(FAIL)} FAILURES: " + ", ".join(FAIL))
print("=" * 84)
sys.exit(1 if FAIL else 0)
