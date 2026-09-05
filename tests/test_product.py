"""Adversarial tests for the parts a user actually touches: sizing, the CLI, and the server.

The engine's correctness is pinned by test_stream.py. What is pinned here is everything that
decides whether a first-time user gets a working model or a crashed laptop:

  * sizing that consumes every free byte, so the first KV cache growth lands in swap
  * a "memory under pressure" warning driven by the compressor's CUMULATIVE occupancy, which sits
    above any static threshold on an idle machine after a few heavy runs
  * a server that dies on a malformed request instead of returning 400
  * a queue counter that leaks on the error path and reports a backlog that does not exist
"""
import inspect
import json
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bigrig_engine import autoconfig, cli, server

FAIL = []
def check(n, c, d=""):
    (print(f"  PASS  {n}") if c else (FAIL.append(n), print(f"  FAIL  {n}  {d}")))

BLOBS = os.path.join(ROOT, "data", "blobs")


def manifest(name):
    return json.load(open(os.path.join(BLOBS, f"{name}.experts.manifest.json")))


print("=" * 80); print("1. SIZING THE POOL FOR THIS MACHINE"); print("=" * 80)
QW = manifest("Qwen3-30B-A3B-3bit")
sh = autoconfig.model_shape(QW)
check("model shape is read from the manifest, not guessed",
      sh["n_layers"] == 48 and sh["n_experts"] == 128, str(sh))
check("bytes_per_expert x experts x layers equals the blob size",
      abs(sh["bytes_per_expert"] * sh["n_experts"] * sh["n_layers"] - sh["expert_bytes"]) < 1e-6,
      f"{sh['bytes_per_expert']*sh['n_experts']*sh['n_layers']} vs {sh['expert_bytes']}")

# ---------------------------------------------------------------- one reserve, not two
# doctor computed the reserve as runtime + working memory and a live Session computed it as
# runtime + working memory + prompt cache, so doctor promised a pool 0.5 GB larger than serve
# would build. Right at the boundary that is a "yes" followed by a refusal. Both now call
# session.serving_reserve_gb, and these checks exist so a term added to one reaches both.
from bigrig_engine.session import (serving_reserve_gb, OS_AND_RUNTIME_GB,       # noqa: E402
                                   WORKING_MEMORY_GB, PROMPT_CACHE_GB)
check("the reserve counts the prompt cache, which is what serve actually holds back",
      abs(serving_reserve_gb() - (OS_AND_RUNTIME_GB + WORKING_MEMORY_GB + PROMPT_CACHE_GB)) < 1e-9,
      f"{serving_reserve_gb()}")
check("...and it is strictly larger than the runtime-plus-working-memory sum doctor once used",
      serving_reserve_gb() > OS_AND_RUNTIME_GB + WORKING_MEMORY_GB)
check("a caller that switches the prompt cache off is not charged for it",
      abs(serving_reserve_gb(prompt_cache_gb=0) - (OS_AND_RUNTIME_GB + WORKING_MEMORY_GB)) < 1e-9)
check("a nonsensical negative prompt cache cannot shrink the reserve",
      serving_reserve_gb(prompt_cache_gb=-99) == serving_reserve_gb(prompt_cache_gb=0))
check("a per-model working memory flows through rather than being ignored",
      serving_reserve_gb(1.0) < serving_reserve_gb(3.0))
_cli_src = open(os.path.join(ROOT, "bigrig_engine", "cli.py")).read()
check("no path computes the reserve by hand any more, which is how the two drifted",
      "OS_AND_RUNTIME_GB + WORKING_MEMORY_GB" not in _cli_src)
check("...and every doctor site asks the one function for it",
      _cli_src.count("serving_reserve_gb()") >= 3, str(_cli_src.count("serving_reserve_gb()")))
# ---------------------------------------------------------------- resident costs less to run
# A model that keeps every expert in RAM never services a miss, so it never allocates the
# admission buffer that is the largest transient in a streamed step. Measured at 0.52-0.95 GB
# across 1.1 to 9.1 GB of resident weights, against 0.68-2.90 GB streamed. Charging a resident
# model the streamed reserve pushed models into streaming -- or into being COMPRESSED, which
# alters the weights -- when they had room to run whole and untouched.
from bigrig_engine.session import RESIDENT_WORKING_MEMORY_GB                    # noqa: E402
check("a resident model reserves less than a streamed one",
      serving_reserve_gb(streamed=False) < serving_reserve_gb(),
      f"{serving_reserve_gb(streamed=False)} vs {serving_reserve_gb()}")
check("...by exactly the difference between the two working-memory measurements",
      abs((serving_reserve_gb() - serving_reserve_gb(streamed=False))
          - (WORKING_MEMORY_GB - RESIDENT_WORKING_MEMORY_GB)) < 1e-9)
check("the resident figure still clears the worst resident run measured (0.95 GB)",
      RESIDENT_WORKING_MEMORY_GB >= 0.95 * 1.5, f"{RESIDENT_WORKING_MEMORY_GB}")
check("...and is not so large it stops being a saving",
      RESIDENT_WORKING_MEMORY_GB < WORKING_MEMORY_GB)
# The behaviour that matters: a model at the boundary now runs whole instead of being altered.
_ds = manifest("OLMoE-1B-7B-0125-4bit")
_kw = dict(top_k=8, non_expert_gb=0.27)
# A budget where the model's experts fit under the resident reserve but not under the streamed
# one -- found by search rather than hardcoded, so it survives a change to either constant.
def _mode(b, resident):
    """The mode at budget b, or None where the planner refuses the model outright."""
    try:
        kw = dict(_kw, budget_gb=b, reserve_gb=serving_reserve_gb())
        if resident:
            kw["resident_reserve_gb"] = serving_reserve_gb(streamed=False)
        return autoconfig.choose_strategy(_ds, **kw)["mode"]
    except MemoryError:
        return None

_bnd = next(b / 10 for b in range(40, 200)
            if _mode(b / 10, False) not in (None, "native") and _mode(b / 10, True) == "native")
_old = autoconfig.choose_strategy(_ds, budget_gb=_bnd, reserve_gb=serving_reserve_gb(), **_kw)
_new = autoconfig.choose_strategy(_ds, budget_gb=_bnd, reserve_gb=serving_reserve_gb(),
                                  resident_reserve_gb=serving_reserve_gb(streamed=False), **_kw)
check("at the boundary the resident reserve runs a model whole rather than altering it",
      _old["mode"] in ("compress", "stream") and _new["mode"] == "native",
      f'{_bnd} GB: {_old["mode"]} -> {_new["mode"]}')
check("...and a caller that does not ask for it gets exactly the old answer",
      autoconfig.choose_strategy(_ds, budget_gb=_bnd, reserve_gb=serving_reserve_gb(),
                                 **_kw)["mode"] == _old["mode"])
# It must never make a STREAMED plan larger: the streamed reserve is untouched by any of this.
for _b in (8.0, 9.0, 10.0, 11.0):
    _a = autoconfig.choose_capacity(QW, budget_gb=_b, reserve_gb=serving_reserve_gb())
    _c = autoconfig.choose_capacity(QW, budget_gb=_b, reserve_gb=serving_reserve_gb(),
                                    resident_reserve_gb=serving_reserve_gb(streamed=False))
    check(f"a streamed pool at {_b:.0f} GB is unchanged by the resident reserve",
          _a["capacity"] == _c["capacity"] and abs(_a["pool_gb"] - _c["pool_gb"]) < 1e-9,
          f'{_a["capacity"]} vs {_c["capacity"]}')

# The invariant that matters to a user: for one budget, doctor and serve size the same pool.
for _m, _name in ((QW, "Qwen3-30B"), (manifest("OLMoE-1B-7B-0125-4bit"), "OLMoE")):
    _sr = serving_reserve_gb()
    _a = autoconfig.choose_capacity(_m, budget_gb=12.0, reserve_gb=_sr)
    _b = autoconfig.choose_capacity(_m, budget_gb=12.0, reserve_gb=_sr)
    check(f"{_name}: doctor and serve size the same pool from the same budget",
          abs(_a["pool_gb"] - _b["pool_gb"]) < 1e-9 and _a["reserve_gb"] == _sr,
          f"{_a['pool_gb']:.3f} vs {_b['pool_gb']:.3f}")

c = autoconfig.choose_capacity(QW, budget_gb=12.0)
check("the chosen pool actually fits the stated budget",
      c["pool_gb"] + c["reserve_gb"] <= 12.0, f"pool {c['pool_gb']:.2f} + {c['reserve_gb']}")
check("sizing leaves real headroom rather than consuming every byte",
      c["headroom_gb"] >= autoconfig.MIN_HEADROOM_GB - 1e-6, f"{c['headroom_gb']:.3f} GB")
check("capacity never exceeds the expert count", c["capacity"] <= c["n_experts"])
prev = -1
for b in (8.0, 10.0, 12.0, 16.0, 24.0):
    cc = autoconfig.choose_capacity(QW, budget_gb=b)
    check(f"more memory never means fewer experts ({b:.0f} GB -> C={cc['capacity']})",
          cc["capacity"] >= prev, f"{cc['capacity']} < {prev}")
    prev = cc["capacity"]
big = autoconfig.choose_capacity(QW, budget_gb=64.0)
check("a machine with room for everything is told it does not need streaming",
      big["fits_entirely"] and "would only slow it down" in autoconfig.describe(big))
try:
    autoconfig.choose_capacity(QW, budget_gb=3.5)
    check("a machine that genuinely cannot run the model is told so", False)
except MemoryError as e:
    check("a machine that genuinely cannot run the model is told so",
          "cannot run" in str(e) or "held back" in str(e))
try:
    autoconfig.choose_capacity(QW, budget_gb=0.5)
    check("a budget below the reserve raises rather than returning a negative pool", False)
except MemoryError:
    check("a budget below the reserve raises rather than returning a negative pool", True)
small = autoconfig.choose_capacity(manifest("OLMoE-1B-7B-0125-4bit"), budget_gb=24.0)
check("a small model on a big machine is not streamed", small["fits_entirely"])

print("\n" + "=" * 80)
print("1b. HOW TO RUN IT, NOT JUST HOW MUCH TO KEEP")
print("=" * 80)
# Preference order is native -> compress -> stream, which is also increasing order of what it
# costs the user: nothing, then accuracy, then speed. Reaching for the engine on a model that
# already fits, or streaming when compressing would have done, is the engine making things worse.
seen = []
for budget in (64.0, 24.0, 12.0, 9.0, 8.0, 7.0, 6.0, 5.0):
    try:
        st = autoconfig.choose_strategy(QW, budget_gb=budget, non_expert_gb=0.67)
        seen.append((budget, st["mode"]))
    except MemoryError:
        seen.append((budget, "refused"))
rank = {"native": 0, "compress": 1, "stream": 2, "refused": 3}
idx = [rank[m] for _, m in seen]
check("shrinking memory only ever degrades the strategy, never improves it",
      all(a <= b for a, b in zip(idx, idx[1:])), str(seen))
check("a machine with room runs the model untouched", seen[0][1] == "native")
nat = autoconfig.choose_strategy(QW, budget_gb=64.0, non_expert_gb=0.67)
check("...and the native path puts no engine in front of it", "capacity" not in nat)
# EVERY MODE'S ONE-LINE DESCRIPTION MUST ACTUALLY RENDER.
#     `describe_strategy` is the line `doctor` and `prepare` print for each model, and its
#     streaming branch calls the speed word. When that word's signature changed, this call site
#     was missed and `bigrig doctor <local streamed model>` died with a TypeError -- a core
#     command, on the happy path, and nothing here noticed. Rendering all three modes catches it.
for _b in (64.0, 12.0, 9.0):
    try:
        _st = autoconfig.choose_strategy(QW, budget_gb=_b, non_expert_gb=0.67)
    except MemoryError:
        continue
    _line = autoconfig.describe_strategy(_st)
    check(f"the {_st['mode']} strategy renders a line", isinstance(_line, str) and len(_line) > 20
          and "None" not in _line, repr(_line))
    check(f"...and passing this Mac's measured disk renders too ({_st['mode']})",
          isinstance(autoconfig.describe_strategy(_st, 5.34), str))
_stream_st = autoconfig.choose_strategy(QW, budget_gb=9.0, non_expert_gb=0.67)
check("a streamed model's line carries the four-word speed verdict",
      any(w in autoconfig.describe_strategy(_stream_st, 5.34)
          for w in ("FAST", "GOOD", "USABLE", "SLOW")),
      autoconfig.describe_strategy(_stream_st, 5.34))
check("...and the shape the speed word needs travels with the strategy",
      all(k in _stream_st for k in ("n_layers", "top_k", "bytes_per_expert")), str(sorted(_stream_st)))

check("every verdict explains itself in words a user can act on",
      all("reason" in autoconfig.choose_strategy(QW, budget_gb=b, non_expert_gb=0.67)
          for b, m in seen if m != "refused"))
# 2-bit g128 is MLX's floor. A model whose 2-bit size still does not fit cannot be compressed
# into range, and saying so beats compressing anyway and shipping garbage.
sh = autoconfig.model_shape(QW)
q = QW["layers"][str(sh["layer_keys"][0])]["quant"]
params = sh["expert_bytes"] / ((q["bits"] + 32.0 / q["group_size"]) / 8.0)
floor_gb = params * (2 + 32.0 / 128) / 8.0 / 1e9
check("the 2-bit floor for this model is computed, not guessed",
      abs(floor_gb - 8.15) < 0.2, f"{floor_gb:.2f} GB")
tight = autoconfig.choose_strategy(QW, budget_gb=12.0, non_expert_gb=0.67)
check("a budget below the floor streams rather than compressing into garbage",
      tight["mode"] == "stream", tight["mode"])
check("...and it reports the floor so the user knows why", "floor_gb" in tight)
check("min_bits lets a user refuse the trade entirely",
      autoconfig.choose_strategy(QW, budget_gb=64.0, non_expert_gb=0.67,
                                 min_bits=99)["mode"] == "native")
# MEASURED, and it is why min_bits exists at all.
print("        measured on a 4x0.6B model: compressed to 2-bit it loaded, ran at 34.9 tok/s")
print("        and emitted '.\\n1\\n1' -- fast, resident, and useless. Speed without a quality")
print("        floor is not a feature.")

print("\n" + "=" * 80); print("2. MEMORY PRESSURE MUST MEAN NOW, NOT EVER"); print("=" * 80)
from bigrig_engine.calibrate import _vm, PAGE, under_pressure
comp = _vm("Pages occupied by compressor") * PAGE / 1e9
up = under_pressure()
print(f"        compressor currently holds {comp:.2f} GB; under_pressure() -> {up}")
check("an idle machine with free memory is not reported as under pressure",
      not (up and autoconfig.available_gb() > 8.0),
      f"available {autoconfig.available_gb():.1f} GB but pressure={up}")
check("the check is a growth measurement, not a static threshold",
      "grow_mb" in under_pressure.__code__.co_varnames)

print("\n" + "=" * 80); print("3. REQUEST VALIDATION"); print("=" * 80)
H = server.make_handler(server._State(None))
S = H._sampling
check("defaults are sane when nothing is supplied",
      S({}) == {"max_tokens": 512, "temperature": 0.7, "top_p": 0.95,
                "think": True, "continue_last": False,
                "lookahead": False, "lookahead_tokens": server.LOOKAHEAD_TOKENS,
                "mtp": None,                     # "whatever the server was started with"
                "tools": None,
                "_rid": ""}, str(S({})))
# Tools are absent unless sent, and a request that sends them gets them forwarded rather than
# quietly dropped -- which is what both endpoints did before, returning HTTP 200 and an essay
# about the file the caller asked to have read.
check("...and no tools are assumed", S({})["tools"] is None)
_T = [{"type": "function", "function": {"name": "f"}}]
check("a request that sends tools has them forwarded", S({"tools": _T})["tools"] == _T)
for _bad in ("nope", [1, 2], {"a": 1}):
    try:
        S({"tools": _bad})
        _rejected = False
    except ValueError:
        _rejected = True
    check(f"...and a malformed `tools` is rejected, not ignored: {_bad!r}", _rejected)
# Guessing ahead is OFF unless a request asks for it, and this is the check that keeps it that
# way. It is a trade with a real downside -- 1.48x on text being reproduced verbatim, 0.86x on
# original prose -- so a server-wide default would be right for one caller and wrong for the
# next. A change that flipped this on for everyone would show up here first.
check("...and guessing ahead is not one of them", S({})["lookahead"] is False)
check("a request may ask for it", S({"lookahead": True})["lookahead"] is True)
check("...and say how far ahead to guess", S({"lookahead": True, "lookahead_tokens": 3})
      ["lookahead_tokens"] == 3)
for bad_k, want in ((0, 1), (99, 32), (-4, 1), ("many", server.LOOKAHEAD_TOKENS),
                    (None, server.LOOKAHEAD_TOKENS)):
    got = S({"lookahead": True, "lookahead_tokens": bad_k})["lookahead_tokens"]
    check(f"...clamped rather than trusted: lookahead_tokens={bad_k!r} -> {want}", got == want,
          f"got {got}")
check("explicit nulls fall back to defaults rather than crashing",
      S({"temperature": None, "max_tokens": None})["temperature"] == 0.7)
for bad, why in (({"max_tokens": -5}, "negative"), ({"max_tokens": 10 ** 9}, "absurd"),
                 ({"max_tokens": "many"}, "not a number"), ({"temperature": "hot"}, "not a number"),
                 ({"temperature": 99}, "out of range"), ({"top_p": -1}, "out of range"),
                 ({"top_p": 5}, "out of range")):
    try:
        S(bad)
        check(f"rejects {why}: {bad}", False, "accepted")
    except ValueError:
        check(f"rejects {why}: {bad}", True)
check("temperature 0 is allowed -- it is how you get deterministic output",
      S({"temperature": 0})["temperature"] == 0.0)
check("the boundary values are inclusive",
      S({"max_tokens": 1})["max_tokens"] == 1 and S({"top_p": 1})["top_p"] == 1.0)

print("\n" + "=" * 80); print("4. THE CLI'S CONTRACT"); print("=" * 80)
p = cli.build_parser()
for cmd in ("doctor", "list", "prepare", "run", "serve"):
    check(f"`bigrig {cmd}` parses",
          getattr(p.parse_args([cmd] + ([] if cmd in ("doctor", "list") else ["m"])), "fn", None)
          is not None)
a = p.parse_args(["serve", "m", "--port", "9999", "--residency", "0.4", "--no-monitor"])
check("serve accepts port, residency and monitor flags",
      a.port == 9999 and a.residency == 0.4 and a.no_monitor)
check("run defaults to monitoring ON -- it is the differentiator",
      not p.parse_args(["run", "m"]).no_monitor)
check("no subcommand prints help and exits non-zero", cli.main([]) == 1)
try:
    cli.resolve_model("definitely-not-a-real-model", allow_download=False)
    check("an unknown local name raises with an actionable message", False)
except FileNotFoundError as e:
    check("an unknown local name raises with an actionable message",
          "Hugging Face repo id" in str(e) or "not downloaded" in str(e))
check("a bad model name exits 1 rather than traceback-ing",
      cli.main(["prepare", "no-such-model-xyz"]) == 1)
rows = cli._prepared()
if os.path.isdir(os.path.join(ROOT, "models")) and os.listdir(os.path.join(ROOT, "models")):
    check("prepared models are discovered from the blob directory", len(rows) >= 1)
else:
    print("  SKIPPED - no models on this machine, so there is nothing to discover.")
check("every prepared model is verified against its manifest size",
      all(r["complete"] for r in rows),
      str([r["name"] for r in rows if not r["complete"]]))
# The download filter is the first code a brand-new user hits, and it silently dropped the file
# that carries the chat template on newer checkpoints.
import inspect as _insp
_src = _insp.getsource(cli.resolve_model)
check("the download filter keeps chat templates (*.jinja)", '"*.jinja"' in _src)
check("...and weights, config, tokenizer and merges",
      all(x in _src for x in ('"*.safetensors"', '"*.json"', '"*.txt"', '"*.model"')))
check("executable model code is NOT downloaded unless explicitly asked for",
      '"*.py"' in _src and "trust_remote_code" in _src)
_pa = cli.build_parser().parse_args(["run", "m", "--trust-remote-code"])
check("--trust-remote-code is exposed on the commands that download", _pa.trust_remote_code)
check("a missing chat template warns instead of silently degrading",
      "no usable chat template" in _insp.getsource(
          __import__("bigrig_engine.session", fromlist=["Session"]).Session._prompt))
check("the console entry point is registered",
      "bigrig = " in open(os.path.join(ROOT, "pyproject.toml")).read())
# Only meaningful once someone has run `pip install -e .` into a .venv here. A fresh clone has
# not, and the entry point being *declared* is asserted just above, which is the part that can
# actually be wrong in the source.
if os.path.isdir(os.path.join(ROOT, ".venv")):
    check("the installed `bigrig` command exists",
          os.path.exists(os.path.join(ROOT, ".venv/bin/bigrig")))
else:
    print("  SKIPPED - no .venv here; `pip install -e .` first to check the installed command.")

print("\n" + "=" * 80)
print("4b. THE DOWNLOAD PATH AND LENIENT LOADING  (item 1: what a new user hits first)")
print("=" * 80)
from bigrig_engine import stream as _st
THUNDER = os.path.expanduser(
    os.path.join(ROOT, "models") + "/Qwen3-MOE-4x0.6B-2.4B-Writing-Thunder-V1.2-mlx-4Bit")
if os.path.isdir(THUNDER):
    check("a downloaded model brings its chat template with it",
          os.path.exists(os.path.join(THUNDER, "chat_template.jinja")))
    check("...and its tokenizer and config",
          all(os.path.exists(os.path.join(THUNDER, f))
              for f in ("config.json", "tokenizer.json", "merges.txt", "vocab.json")))
    check("executable model code was NOT downloaded",
          not [f for f in os.listdir(THUNDER) if f.endswith(".py")])
    # This checkpoint declares tie_word_embeddings but still ships lm_head.scales, so mlx_lm's
    # strict load refuses it outright. Tolerating leftovers is what makes it usable.
    from mlx_lm import load as _strict
    strict_failed = False
    try:
        _strict(THUNDER, lazy=True)
    except ValueError as e:
        strict_failed = "not in model" in str(e)
    check("mlx_lm's strict load rejects this checkpoint (so the leniency is load-bearing)",
          strict_failed)
    _m, _t = _st.load_lenient(THUNDER, lazy=True)
    check("load_lenient accepts it", _m is not None and _t is not None)
    _L = _st.find_moe_layers(_m)
    check("a 4-expert top-2 architecture is recognised",
          len(_L) == 28 and _st._num_experts(_L[0][1]) == 4, f"{len(_L)} layers")
    check("its top-k is read as 2, not assumed to be 8", _st.model_top_k(THUNDER) == 2)
    del _m, _t
    import mlx.core as _mx
    _mx.clear_cache()
else:
    # Normal for a fresh clone, and not a defect in anything: the section needs real weights.
    print("  SKIPPED - the test model is not downloaded; run `bigrig prepare` on it to exercise")
    print("  the tokenizer and chat-template checks above.")

# The safety property that matters more than the leniency: MISSING weights must never be
# tolerated. MLX reports extras BEFORE missing, so a blind strict=False retry would accept a
# model with absent weights -- which loads happily and generates fluent nonsense.
import inspect as _i2
_src = _i2.getsource(_st.load_lenient)
check("only an EXTRA-parameters error triggers the lenient retry", '"not in model" not in str(e)' in _src)
check("...and the lenient load is re-checked for missing parameters afterwards",
      "missing" in _src and "Refusing to load" in _src)
check("a non-extras ValueError is re-raised untouched", "raise " in _src)

# MEASURED on that model, recorded so the claim is not repeated without evidence.
print("        measured: E=4 / top-k=2 / 28 layers, bit-identical at C=4, C=3 and C=2,")
print("                  and C=1 (below top-k) is refused rather than run")

# The expert bank is found by its SHAPE, not by one architecture's spelling of its name.
# THE BUG: this matched `layer.mlp.switch_mlp` only. gpt-oss nests the same object at
# `layer.mlp.experts` and Llama-4 at `layer.feed_forward.experts`, so the engine found zero MoE
# blocks and streamed nothing -- without raising, because "no MoE layers" is a legitimate answer.
import mlx.nn as _nn
from mlx_lm.models.switch_layers import SwitchGLU as _SG
def _fake(battr, sattr, n=3):
    layers = []
    for _ in range(n):
        blk = _nn.Module(); setattr(blk, sattr, _SG(input_dims=8, hidden_dims=16, num_experts=4))
        lay = _nn.Module(); setattr(lay, battr, blk); layers.append(lay)
    inner = _nn.Module(); inner.layers = layers
    top = _nn.Module(); top.model = inner
    return top
for _label, _b, _a in (("Qwen3/Mixtral/DeepSeek", "mlp", "switch_mlp"),
                       ("gpt-oss", "mlp", "experts"),
                       ("Llama-4", "feed_forward", "experts")):
    _m2 = _fake(_b, _a)
    check(f"the expert bank is found on {_label}", len(_st.find_moe_layers(_m2)) == 3)
    check(f"...and {_label} reports where to put the replacement back",
          [a for _i, _bk, a, _sg in _st.find_moe_sites(_m2)] == [_a] * 3)
_dense = _nn.Module(); _di = _nn.Module()
_dl = _nn.Module(); _dl.mlp = _nn.Linear(8, 8); _di.layers = [_dl]; _dense.model = _di
check("a dense model still matches nothing, rather than something by accident",
      _st.find_moe_layers(_dense) == [])

print("\n" + "=" * 80)
print("4c. THE DISCLOSURE MUST REACH THE USER, ALWAYS")
print("=" * 80)
import inspect as _i3
from bigrig_engine import consent as _cn, server as _sv, session as _ss
# Every run must name the precision being served. The old summary said "100% of experts kept in
# RAM" for a COMPRESSED model -- true, and it never mentioned those experts were no longer the
# ones downloaded.
# Checked on the RENDERED line, not on the source text. An earlier version grepped the source
# and failed because the sentence was split across two lines -- testing the code's spelling
# rather than its behaviour.
class _FakeSession:
    describe_serving = _ss.Session.describe_serving

def _render(strategy, source_bits=4, source_precision=None):
    f = _FakeSession()
    f.strategy, f.source_bits = strategy, source_bits
    # The real Session always sets this; an unquantised model has no bit count and naming one
    # produced "running EXACT at 0-bit" for the model whose whole point is full precision.
    f.source_precision = source_precision or f"{source_bits}-bit"
    return f.describe_serving()

_comp = _render({"mode": "compress", "bits": 3, "group_size": 128})
_strm = _render({"mode": "stream", "residency": 0.48})
_full = _render({"mode": "stream", "residency": 1.0})
_nat = _render({"mode": "native"})

# AN UNQUANTISED MODEL MUST NOT BE DESCRIBED AS "0-bit". `quant` is correctly absent for weights
# that were never quantised, so source_bits read 0 and the one line that always names what is
# being served said "running EXACT at 0-bit". Found by running Qwen3-30B-A3B-bf16 end to end.
_bf = _render({"mode": "stream", "residency": 0.07}, source_bits=16,
              source_precision="bfloat16")
check("an unquantised model is named by its dtype, not by a bit count",
      "bfloat16" in _bf and "0-bit" not in _bf, _bf)
check("...and still says plainly that it is exact",
      "EXACT" in _bf and "bit-identical" in _bf)
_bf_full = _render({"mode": "stream", "residency": 1.0}, source_bits=16,
                   source_precision="bfloat16")
check("...at full residency too", "bfloat16" in _bf_full and "0-bit" not in _bf_full)
_bf_nat = _render({"mode": "native"}, source_bits=32, source_precision="float32")
check("...and when it is not streamed at all", "float32" in _bf_nat)
check("every mode names the precision it is serving",
      all("4-bit" in x or "3-bit" in x for x in (_comp, _strm, _full, _nat)),
      f"{_comp!r} {_strm!r} {_nat!r}")
check("the compressed line says outright that the output differs",
      "differs from the original" in _comp and "COMPRESSED" in _comp, _comp)
check("...and names both the new precision and the old one",
      "3-bit" in _comp and "was 4-bit" in _comp, _comp)
check("the exact lines promise bit-identity",
      "bit-identical" in _strm and "identical" in _full, f"{_strm!r} | {_full!r}")
check("a fully-resident model never claims to be streaming too",
      "streamed from disk" not in _full, _full)
check("...while a partly-resident one does say what it streams",
      "streamed from disk" in _strm, _strm)
check("the untouched case says untouched", "untouched" in _nat, _nat)
# Piped to a file, Python block-buffers stdout. An unflushed disclosure is lost if the process
# is killed -- measured: the banner was absent from a killed server's log until this was fixed.
check("the server flushes its disclosure", "flush=True" in _i3.getsource(_sv.serve))
check("the terminal path flushes its disclosure too",
      "flush=True" in _i3.getsource(cli.cmd_run))
# A decision the user has to make is not a crash.
_m = _i3.getsource(cli.main)
check("a needed decision exits cleanly instead of raising a traceback",
      "ConsentRequired" in _m and "return 2" in _m)
check("...with an exit code distinct from a real error", "return 1" in _m and "return 2" in _m)
_cs = _i3.getsource(cli._session)
check("--exact and --compress cannot both be given", "contradict each other" in _cs)
check("--forget-choice is wired to the consent store", "forget_choice" in _cs)
check("consent is requested with interactive=True from the CLI", "interactive=True" in _cs)
check("stats expose whether the weights were altered",
      "weights_altered" in _i3.getsource(_ss.Session.stats))

print("\n" + "=" * 80); print("5. THE SERVER, AGAINST A LIVE MODEL"); print("=" * 80)
# As the heading says: a live model. Without one there is no server to bring up, and the forty
# checks below would all fail on a machine that is perfectly healthy -- which is every fresh
# clone. Report the reason and stop here; sections 1-4 above have already run.
if not os.path.isdir(os.path.join(ROOT, "models", "OLMoE-1B-7B-0125-4bit")):
    print("  SKIPPED - models/OLMoE-1B-7B-0125-4bit is not on this machine. Run")
    print("  `bigrig prepare mlx-community/OLMoE-1B-7B-0125-4bit` once to exercise the server.")
    print("\n" + "=" * 80)
    print("ALL TESTS PASSED" if not FAIL else f"{len(FAIL)} FAILURES: " + ", ".join(FAIL))
    print("=" * 80)
    sys.exit(1 if FAIL else 0)
PORT = 8231
LOG = os.path.join(ROOT, "data/results/test_server.log")
# THE PORT MUST BE OURS, OR THE TEST IS TESTING SOMEBODY ELSE'S SERVER.
#     A leftover server from an earlier run held this port. Every run after that failed to bind,
#     said so only in a log nobody reads, and then happily sent its requests to the STALE process
#     -- which was running old code and crashed on them. The suite reported a failure in code
#     that was already fixed, and it cost a long detour to find. Bind it first and refuse to run.
_probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
_probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
try:
    _probe.bind(("127.0.0.1", PORT))
except OSError:
    print(f"\n  port {PORT} is already in use, so this test would talk to whatever is on it "
          f"instead of\n  the server it is supposed to start. Stop it and run again:"
          f"\n\n      lsof -ti tcp:{PORT} | xargs kill\n")
    sys.exit(1)
finally:
    _probe.close()

proc = subprocess.Popen(
    [os.path.join(ROOT, ".venv/bin/python"), "-m", "bigrig_engine.cli", "serve",
     "OLMoE-1B-7B-0125-4bit", "--force-stream", "--residency", "0.5", "--port", str(PORT)],
    stdout=open(LOG, "w"), stderr=subprocess.STDOUT, cwd=ROOT)


def get(path, timeout=10):
    with urllib.request.urlopen(f"http://127.0.0.1:{PORT}{path}", timeout=timeout) as r:
        return r.status, json.loads(r.read())


def post(path, body, timeout=180):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}",
                                 data=json.dumps(body).encode() if isinstance(body, dict)
                                 else body,
                                 headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


up = False
for _ in range(90):
    try:
        get("/health", timeout=2); up = True; break
    except Exception:
        time.sleep(1)
check("the server comes up", up, open(LOG).read()[-400:] if os.path.exists(LOG) else "")

if up:
    _, h = get("/health")
    check("/health reports live residency and miss rate",
          "residency" in h and "miss_rate" in h and h["streamed"], str(h)[:120])
    check("/health says whether the monitor is on", h["monitor"] is True)
    _, m = get("/v1/models")
    check("/v1/models returns the loaded model", m["data"][0]["id"] == "OLMoE-1B-7B-0125-4bit")

    st, d = post("/v1/chat/completions",
                 {"messages": [{"role": "user", "content": "Say hello."}],
                  "max_tokens": 16, "temperature": 0.0})
    check("a chat completion returns 200 with content",
          st == 200 and len(d["choices"][0]["message"]["content"]) > 0, str(d)[:140])
    check("usage counts are present and consistent",
          d["usage"]["total_tokens"] == d["usage"]["prompt_tokens"] + d["usage"]["completion_tokens"])
    check("the response carries quality and residency, which OpenAI's does not",
          "bigrig" in d and "degraded_share" in d["bigrig"] and "miss_rate" in d["bigrig"])
    check("the object type matches the endpoint", d["object"] == "chat.completion")

    st2, d2 = post("/v1/completions", {"prompt": "1 2 3", "max_tokens": 8, "temperature": 0.0})
    check("the legacy completions endpoint works",
          st2 == 200 and d2["object"] == "text_completion" and "text" in d2["choices"][0])

    # Temperature 0 must be reproducible, or "bit-identical" means nothing at the API level.
    _, r1 = post("/v1/chat/completions", {"messages": [{"role": "user", "content": "Say hello."}],
                                          "max_tokens": 16, "temperature": 0.0})
    check("temperature 0 is deterministic across requests",
          r1["choices"][0]["message"]["content"] == d["choices"][0]["message"]["content"],
          f"{r1['choices'][0]['message']['content'][:40]!r} vs "
          f"{d['choices'][0]['message']['content'][:40]!r}")

    for body, why in (({}, "no messages"), ({"messages": []}, "empty messages"),
                      ({"messages": [{"role": "user", "content": "x"}], "max_tokens": -1}, "bad max_tokens"),
                      ({"messages": [{"role": "user", "content": "x"}], "temperature": "hot"}, "bad temperature"),
                      ({"messages": [{"role": "user"}]}, "message with no content"),
                      ({"messages": "hello"}, "messages not a list")):
        st3, e = post("/v1/chat/completions", body)
        check(f"400 rather than a crash for {why}", st3 == 400 and "error" in e, f"{st3} {e}")
    st4, _ = post("/v1/chat/completions", b"{not json")
    check("400 for a body that is not JSON", st4 == 400)
    try:
        get("/nope"); check("404 for an unknown route", False)
    except urllib.error.HTTPError as e:
        check("404 for an unknown route", e.code == 404)

    # The queue counter must return to zero after concurrent load, including error paths.
    errs = []
    def hammer(i):
        try:
            post("/v1/chat/completions", {"messages": [{"role": "user", "content": f"hi {i}"}],
                                          "max_tokens": 8})
        except Exception as ex:
            errs.append(ex)
    ts = [threading.Thread(target=hammer, args=(i,)) for i in range(4)]
    [t.start() for t in ts]; [t.join() for t in ts]
    check("concurrent requests all complete", not errs, str(errs[:2]))
    _, h2 = get("/health")
    check("the queue drains back to zero", h2["queue_depth"] == 0, str(h2["queue_depth"]))
    check("every request is counted as served", h2["requests_served"] >= 7,
          str(h2["requests_served"]))
    check("the server survived every malformed request", proc.poll() is None)

    # THE SHIP-BLOCKER. A client that hangs up mid-stream used to kill the server permanently:
    # the HTTP thread stopped draining the queue, the ONE thread allowed to touch MLX kept
    # generating, and after 256 chunks its put() blocked forever. /health still answered, so it
    # looked alive. Any user pressing Ctrl-C in a chat client would trigger it.
    import socket

    def hang_up_midstream(n_tokens=400, read_bytes=200):
        sk = socket.create_connection(("127.0.0.1", PORT), timeout=10)
        body = json.dumps({"messages": [{"role": "user", "content": "Write a long essay."}],
                           "max_tokens": n_tokens, "stream": True}).encode()
        sk.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\n"
                   b"Content-Type: application/json\r\nContent-Length: "
                   + str(len(body)).encode() + b"\r\n\r\n" + body)
        try:
            sk.settimeout(8)
            sk.recv(read_bytes)
        except OSError:
            pass
        sk.close()                                   # gone, mid-generation

    for _i in range(3):
        hang_up_midstream()
        time.sleep(1.5)
    st5, d5 = post("/v1/chat/completions",
                   {"messages": [{"role": "user", "content": "say hi"}], "max_tokens": 6},
                   timeout=90)
    check("the server still generates after three mid-stream disconnects",
          st5 == 200 and bool(d5["choices"][0]["message"]["content"]), str(st5))
    _, h3 = get("/health")
    check("abandoned requests do not leak queue depth", h3["queue_depth"] == 0,
          str(h3["queue_depth"]))
    check("the generator thread is alive, not deadlocked", proc.poll() is None)

    # HTTP/1.1 keep-alive: rejecting an oversized body without consuming it desyncs the socket,
    # and the unread bytes then get parsed as the next request line.
    sk = socket.create_connection(("127.0.0.1", PORT), timeout=10)
    huge_len = server.MAX_BODY + 1024
    sk.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1:" + str(PORT).encode()
               + b"\r\nContent-Type: application/json\r\nContent-Length: "
               + str(huge_len).encode() + b"\r\n\r\n")
    # Deliberately send the HEADERS ONLY. The server rejects on Content-Length before reading a
    # byte, so pushing 4 MB first just races the close and the test reads nothing -- which is the
    # server behaving correctly and the test failing anyway.
    resp = b""
    try:
        sk.settimeout(8)
        resp = sk.recv(4096)
    except OSError:
        pass
    sk.close()
    check("an oversized body is refused rather than accepted", b"400" in resp, repr(resp[:80]))
    check("...and the client is told the connection is closing, so it will not reuse the socket",
          b"connection: close" in resp.lower(), repr(resp[:200]))
    st6, _ = post("/v1/chat/completions",
                  {"messages": [{"role": "user", "content": "still here?"}], "max_tokens": 5},
                  timeout=90)
    check("the server is healthy after the oversized body", st6 == 200)

    # ---------------------------------------------------------------- who may drive this server
    # This server has no authentication, so "who is asking" is the whole defence. A wildcard
    # CORS header let any page the user happened to have open POST here and READ the reply.
    # Removing that header is not enough on its own: a POST with a non-JSON content type is a
    # SIMPLE request, so a browser sends it with no preflight and the write lands anyway. All
    # three checks are asserted here because removing any one of them reopens the hole.
    def raw(req: bytes, timeout=10) -> bytes:
        sk = socket.create_connection(("127.0.0.1", PORT), timeout=timeout)
        try:
            sk.sendall(req)
            return sk.recv(4096)
        except OSError:
            return b""
        finally:
            sk.close()

    HOST = f"127.0.0.1:{PORT}".encode()
    BODY = b'{"messages":[{"role":"user","content":"hi"}],"max_tokens":4}'

    def req(*, origin=None, ctype=b"application/json", host=HOST, body=BODY):
        h = [b"POST /v1/chat/completions HTTP/1.1", b"Host: " + host]
        if ctype:
            h.append(b"Content-Type: " + ctype)
        if origin:
            h.append(b"Origin: " + origin)
        h.append(b"Content-Length: " + str(len(body)).encode())
        return b"\r\n".join(h) + b"\r\n\r\n" + body

    r = raw(req(origin=b"https://evil.example"), timeout=20)
    check("a page on another origin is refused outright",
          b" 403 " in r, repr(r[:60]))
    check("...and is not handed a CORS header it could read the reply through",
          b"access-control-allow-origin" not in r.lower(), repr(r[:200]))

    r = raw(req(ctype=b"text/plain"), timeout=20)
    check("a POST that dodges the preflight by sending text/plain is refused",
          b" 415 " in r, repr(r[:60]))

    r = raw(req(host=b"attacker.example"), timeout=20)
    check("a request carrying someone else's Host is refused (DNS rebinding)",
          b" 421 " in r, repr(r[:60]))

    for label, r in (("403", raw(req(origin=b"https://evil.example"), timeout=20)),
                     ("415", raw(req(ctype=b"text/plain"), timeout=20)),
                     ("421", raw(req(host=b"attacker.example"), timeout=20))):
        check(f"the {label} refusal closes the socket rather than leaving the body unread",
              b"connection: close" in r.lower(), repr(r[:200]))

    st7, _ = post("/v1/chat/completions",
                  {"messages": [{"role": "user", "content": "ordinary client"}],
                   "max_tokens": 5}, timeout=90)
    check("an ordinary client, which sends no Origin at all, still works", st7 == 200)

    own = f"http://127.0.0.1:{PORT}".encode()
    r = raw(req(origin=own), timeout=20)
    check("the server's own page IS allowed, so the bundled UI keeps working",
          b" 200 " in r, repr(r[:60]))
    check("...and its origin is echoed exactly, never as a wildcard",
          b"access-control-allow-origin: " + own in r.lower() and b"origin: *" not in r.lower(),
          repr(r[:300]))

proc.terminate()
try:
    proc.wait(timeout=15)
except subprocess.TimeoutExpired:
    proc.kill()

print("\n" + "=" * 80); print("THE FIRST-RUN PATH"); print("=" * 80)
import os as _os
from bigrig_engine import cli as _cli, session as _sess
_doc = inspect.getsource(_cli.cmd_doctor)

# THE BUG: doctor called non_expert_gb without a manifest, so for a model that was never packed
# it fell back to reading one from a blob path of "" -- and printed
#   ".manifest.json not found -- run pack_experts() before streaming"
# to stderr, naming a Python function no user has ever called, in the middle of a machine report.
check("doctor passes the manifest it already has rather than re-reading one",
      'non_expert_gb(md, manifest=r["manifest"])' in _doc)
check("...and a model it cannot read is reported, not raised through the report",
      "could not be read" in _doc)
check("a model name can be given, because a user will try it",
      'd.add_argument("model", nargs="?"' in inspect.getsource(_cli))
check("...and a name that matches nothing says so and points at `list`",
      "no model here matching" in _doc and "bigrig list" in _doc)

# THE BUG: doctor planned against whatever was free and reported 13 of 128 experts for gpt-oss,
# while serve planned against BIGRIG_MEM_GB and ran 4 of 128. Two commands, one machine, two
# answers -- and the first one a user reads is the one they believe.
check("both commands resolve the budget in one shared place",
      "resolve_budget" in _doc and "resolve_budget" in inspect.getsource(_sess.Session.__init__))
# THIS CHECK IS WHY THE BUG SURVIVED. It asserted that doctor's source contained the string
# "OS_AND_RUNTIME_GB + WORKING_MEMORY_GB" and called that "the same reserve serve uses" -- but
# serve's reserve had a third term, the prompt cache, so the string was present and the claim
# was false. The text matched; the invariant did not. Assert the VALUES agree instead, which
# is the thing a user experiences, and which no amount of moving code around can fake.
_doc_reserve = _sess.serving_reserve_gb()
_serve_reserve = round(_sess.OS_AND_RUNTIME_GB + _sess.WORKING_MEMORY_GB
                       + _sess.PROMPT_CACHE_GB, 2)
check("...and doctor plans with the same reserve serve uses",
      abs(_doc_reserve - _serve_reserve) < 1e-9, f"doctor {_doc_reserve} vs serve {_serve_reserve}")
check("...and doctor asks for it rather than adding the terms up itself",
      "serving_reserve_gb" in _doc and "OS_AND_RUNTIME_GB + WORKING_MEMORY_GB" not in _doc)
check("...and says so when the budget is not simply what is free",
      "budget for this run" in _doc)
_env = _os.environ.get("BIGRIG_MEM_GB")
try:
    _os.environ["BIGRIG_MEM_GB"] = "7.5"
    check("an explicit argument beats the environment", _sess.resolve_budget(4.0) == 4.0)
    check("...and the environment beats what happens to be free", _sess.resolve_budget() == 7.5)
    _os.environ["BIGRIG_MEM_GB"] = "not a number"
    check("a malformed environment value falls back rather than crashing the command",
          _sess.resolve_budget() > 0)
finally:
    _os.environ.pop("BIGRIG_MEM_GB", None)
    if _env is not None:
        _os.environ["BIGRIG_MEM_GB"] = _env

print("\n" + "=" * 80); print("WHAT `pip install bigrig` GIVES YOU"); print("=" * 80)
_pp = open(os.path.join(ROOT, "pyproject.toml")).read()

# THE BUG: MLX was an opt-in extra, so `pip install bigrig` then `bigrig doctor` -- the first
# command in this tool's own help -- ended in a traceback whose last line was
# "ModuleNotFoundError: No module named 'mlx'". Every single command failed that way.
check("the engine is installed by default on the machines it runs on",
      "sys_platform == 'darwin' and platform_machine == 'arm64'" in _pp)
check("...for both halves of it",
      _pp.count("sys_platform == 'darwin' and platform_machine == 'arm64'") == 2)
check("...but is still not an unconditional requirement, which would fail on Linux",
      'dependencies = [\n  "numpy' in _pp)
# The invariant is that the extra still EXISTS and still names the engine, so `pip install
# 'bigrig[engine]'` from an older set of instructions resolves. Pinning the assertion to a
# particular floor made raising the floors a test failure, which is the "assert the source text
# rather than the behaviour" mistake this suite is meant to avoid.
_ext = next((l for l in _pp.splitlines() if l.startswith("engine = [")), "")
check("the older extra keeps working for anyone following older docs",
      "mlx" in _ext and "mlx-lm" in _ext and "huggingface_hub" in _ext, _ext)
check("the page the engine serves is shipped in the wheel",
      'bigrig_engine = ["webui.html"]' in _pp)
check("both command names are declared", 'bigrig = "bigrig_engine.cli:main"' in _pp
      and 'rig = "bigrig_engine.cli:main"' in _pp)
# Case-insensitively: GitHub resolves either spelling, and the repository is `BigRig` while
# the package is `bigrig`. What must hold is the OWNER -- pointing a published package at
# someone else's repository is the failure worth catching.
check("the project points at the repository it actually lives in",
      "github.com/arjvnv/bigrig" in _pp.lower() and "github.com/bigrig/bigrig" not in _pp.lower())
check("research scripts and tests are not shipped",
      'packages = ["bigrig_layer", "bigrig_engine", "bigrig_engine.policies"]' in _pp)

# A missing engine dependency is a setup problem with a one-line fix, not a crash.
_main = _cli.main
def _boom(name):
    def fn(_a):
        raise ModuleNotFoundError(f"No module named {name!r}", name=name)
    return fn
import argparse as _ap
class _A:
    pass
def _run_with(fn):
    a = _A(); a.fn = fn
    real = _cli.build_parser
    _cli.build_parser = lambda: type("P", (), {"parse_args": lambda _s, _v: a,
                                               "print_help": lambda _s: None})()
    try:
        return _main([])
    finally:
        _cli.build_parser = real
check("a missing engine dependency exits cleanly instead of printing a traceback",
      _run_with(_boom("mlx")) == 1)
check("...for every engine dependency", all(_run_with(_boom(m)) == 1
                                            for m in ("mlx", "mlx_lm", "huggingface_hub")))
check("...and for a submodule of one", _run_with(_boom("mlx.core")) == 1)
try:
    _run_with(_boom("some_unrelated_module"))
    check("an unrelated import error is NOT swallowed", False)
except ModuleNotFoundError:
    check("an unrelated import error is NOT swallowed", True)

print("\n" + "=" * 80)
print(f"{'ALL TESTS PASSED' if not FAIL else str(len(FAIL))+' FAILURES: '+', '.join(FAIL)}")
print("=" * 80)
sys.exit(1 if FAIL else 0)
