"""Answer "will this model run here?" BEFORE it is downloaded.

WHY THIS EXISTS
    `bigrig doctor` could already answer the question exactly. It just answered it too late --
    after the weights were on disk. Someone downloaded 58 GB of Qwen3-30B-A3B-bf16, waited, and
    was then told it needs an 11.5 GB ceiling against the 9.0 in force. Everything in that
    sentence was computable beforehand from two files totalling a few kilobytes.

    "Download 58 GB, wait forty minutes, be told it does not fit" is the worst first experience
    this product can give, and it is entirely avoidable.

WHY IT DOES NOT DO ITS OWN ARITHMETIC
    The required ceiling was worked out by hand three times during one afternoon and was wrong
    three times -- 9.7, then 10.5, then 11.5 -- because each attempt reconstructed the sum from
    remembered constants and dropped a term. First the 0.30 GB OS reserve, then MIN_HEADROOM_GB,
    the 1.00 GB of slack that exists so the first KV-cache growth does not land in swap.

    So this module computes NOTHING about memory. It builds a manifest shaped exactly like the
    one a downloaded model produces and hands it to `autoconfig.choose_capacity`, the same
    function that decides for real, and searches for the smallest ceiling that function accepts.
    There is one arithmetic in this engine and it is not here. If the planner changes, this
    follows it for free; if it were reimplemented here, the two would drift and the drift would
    show up as a machine that swaps.
"""
from __future__ import annotations

import json
import re
import math

# Bytes per element, by the dtype names safetensors uses.
_WIDTH = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4, "U32": 4,
          "I16": 2, "U16": 2, "I8": 1, "U8": 1, "BOOL": 1, "F8_E4M3": 1, "F8_E5M2": 1}

# The names a stacked expert bank goes by. Same list the engine attaches on, for the same
# reason: matching one spelling is how gpt-oss and Llama-4 were missed.
# WHERE A MODEL KEEPS ITS STACKED EXPERTS, WHICH IS NOT A SETTLED CONVENTION.
#     A stacked expert tensor is (n_experts, out, in) and every family agrees on that shape while
#     disagreeing about the path. Three were known; `.ffn.switch_mlp.` was found by pointing
#     `bigrig doctor` at DeepSeek-V4-Flash -- 16,353 downloads in a month -- and being told it
#     "is not a mixture-of-experts model". It is: 256 experts, correctly stacked, at
#     `model.layers.N.ffn.switch_mlp.gate_proj.weight` with shape [256, 2048, 512]. One missing
#     string, and a model nobody could run.
#
#     NOT a general prefix match. `shared_experts` also lives under `.ffn.` and is NOT routed --
#     it runs for every token, so streaming it would fetch it every time and gain nothing. The
#     infixes are exact for that reason.
# The expert layout the streaming kernel actually computes. See the check in remote_shape.
# Both expert shapes this engine streams: the gated trio almost everything uses, and the
# two-projection pair Nemotron ships. A model carrying anything else is refused by name rather
# than streamed on a guess.
_STREAMABLE_PROJECTIONS = {"gate_proj", "up_proj", "down_proj", "fc1", "fc2"}
# Where different architectures hang their fused expert bank. `.block_sparse_moe.` is Mixtral's
# name, kept by MiniMax and Phi-MoE, whose experts are otherwise the same three projections
# everything else uses -- without it both were refused as "not a mixture-of-experts model".
_EXPERT_INFIXES = (".mlp.switch_mlp.", ".mlp.experts.", ".feed_forward.experts.",
                   ".ffn.switch_mlp.", ".ffn.experts.", ".mixer.switch_mlp.",
                   ".block_sparse_moe.switch_mlp.", ".block_sparse_moe.experts.")


def top_k_from_config(cfg: dict, default: int = 8) -> int:
    """How many experts a token routes to, from a model's config.json.

    Tolerant on purpose, and shared with `stream.model_top_k` so the two cannot answer
    differently. Architectures spell it four ways and Hunyuan gives it as a LIST -- which made
    `int()` raise a bare TypeError out of `doctor`, on a model whose layout this engine streams
    perfectly well.
    """
    for key in ("num_experts_per_tok", "moe_topk", "num_experts_per_token",
                "num_selected_experts", "top_k"):
        v = cfg.get(key)
        if v is None:
            v = (cfg.get("text_config") or {}).get(key)
        if isinstance(v, (list, tuple)):
            v = v[0] if v else None
        if isinstance(v, bool):
            continue
        if isinstance(v, (int, float)) and int(v) > 0:
            return int(v)
    return int(default)


def _nbytes(shape, dtype: str) -> int:
    n = 1
    for d in shape:
        n *= int(d)
    return n * _WIDTH.get(str(dtype).upper(), 2)


def remote_shape(repo_id: str, token: str | None = None) -> dict:
    """What a model is, read from the hub without downloading a single weight.

    Returns the pieces `choose_capacity` needs, plus enough to explain the answer. Raises
    ValueError with a plain sentence when the model is not something this engine can stream.
    """
    from huggingface_hub import get_safetensors_metadata, hf_hub_download
    # No progress bars in a verdict. Reading four headers takes a second and the bar left a
    # "Parse safetensors files: 100%" line in the middle of the doctor's report.
    try:
        from huggingface_hub.utils import disable_progress_bars
        disable_progress_bars()
    except Exception:                            # noqa: BLE001 -- cosmetic
        pass

    meta = get_safetensors_metadata(repo_id, token=token)
    cfg_path = hf_hub_download(repo_id, "config.json", token=token)
    with open(cfg_path) as fh:
        cfg = json.load(fh)
    cfg = {**cfg.get("text_config", {}), **cfg}

    # Every tensor, with its size, so expert and non-expert bytes can be told apart.
    sizes: dict[str, int] = {}
    for fname, fmeta in meta.files_metadata.items():
        for name, t in fmeta.tensors.items():
            sizes[name] = _nbytes(t.shape, t.dtype)
    if not sizes:
        raise ValueError(f"{repo_id} publishes no safetensors this engine can read")

    # Group the expert tensors by layer, exactly as the local manifest builder does.
    per_layer: dict[int, int] = {}
    n_experts = 0
    expert_dtype = None
    projections: set = set()
    for name, nb in sizes.items():
        infix = next((inf for inf in _EXPERT_INFIXES if inf in name), None)
        if infix is None:
            continue
        # The projection this tensor belongs to: `...switch_mlp.gate_proj.weight` -> gate_proj.
        tail = name.split(infix, 1)[1]
        if "." in tail:
            projections.add(tail.rsplit(".", 1)[0])
        # `.layers.N.` wherever it sits. `model.layers.` alone missed `backbone.layers.` (Nemotron)
        # and `language_model.model.layers.` (multimodal Qwen), and a miss here reads as "not an
        # MoE model" to the user, which is the wrong thing to tell someone about a 30B-A3B.
        m_li = re.search(r"\.layers\.(\d+)\.", name)
        if not m_li:
            continue
        li = int(m_li.group(1))
        per_layer[li] = per_layer.get(li, 0) + nb
        for fmeta in meta.files_metadata.values():
            if name in fmeta.tensors:
                t = fmeta.tensors[name]
                if t.shape:
                    n_experts = max(n_experts, int(t.shape[0]))
                # The WEIGHT's dtype, not whichever expert tensor came first. A 3-bit model
                # stores packed weights as U32 beside BF16 scales, and reporting the scales
                # would call a quantised model "BF16" -- the precise confusion this check
                # exists to prevent.
                if name.endswith(".weight"):
                    expert_dtype = str(t.dtype)
                break
    if not per_layer:
        raise ValueError(
            f"{repo_id} has no stacked expert tensors -- it is not a mixture-of-experts model, "
            f"or its experts are named in a way this engine does not recognise. Nothing to "
            f"stream; it either fits in memory or it does not.")
    # THE EXPERTS WERE FOUND. CAN THEY BE STREAMED? Not the same question. The streaming kernel
    # computes a SwiGLU expert -- down(silu(gate(x)) * up(x)) -- and is written against those
    # three projections. Nemotron's experts are a two-projection MLP (fc1, fc2) with different
    # arithmetic, so finding them is not the same as being able to run them. Without this check
    # doctor said RUNS, GOOD for a model `bigrig run` would have refused to load.
    unknown = projections - _STREAMABLE_PROJECTIONS
    if unknown:
        raise ValueError(
            f"{repo_id} is a mixture-of-experts model, but its experts use a "
            f"{'/'.join(sorted(projections))} layout that this engine cannot stream yet -- it "
            f"streams the gate/up/down (SwiGLU) layout used by Qwen, DeepSeek, GLM, Kimi and "
            f"Mixtral. It would have to fit in memory whole: "
            f"{sum(sizes.values()) / 1e9:.1f} GB. Support for this layout is on the list.")

    n_layers = len(per_layer)
    expert_bytes = sum(per_layer.values())
    # One expert at ONE layer. The whole bank at a layer divided by how many experts it holds.
    bytes_per_expert = max(per_layer.values()) // max(n_experts, 1)
    non_expert_bytes = sum(nb for name, nb in sizes.items()
                           if not any(inf in name for inf in _EXPERT_INFIXES))
    top_k = top_k_from_config(cfg)

    # Shaped exactly like a downloaded model's manifest, so the real planner accepts it.
    manifest = {
        "layers": {str(li): {"n_experts": n_experts,
                             "bytes_per_expert": per_layer[li] // max(n_experts, 1),
                             "spec": {}}
                   for li in sorted(per_layer)},
        "total_bytes": expert_bytes,
    }
    return {"repo_id": repo_id, "manifest": manifest,
            "n_layers": n_layers, "n_experts": n_experts, "top_k": top_k,
            "bytes_per_expert": bytes_per_expert,
            "expert_gb": expert_bytes / 1e9,
            "non_expert_gb": non_expert_bytes / 1e9,
            "download_gb": sum(sizes.values()) / 1e9,
            "dtype": expert_dtype or "?",
            "quantized": bool(cfg.get("quantization") or cfg.get("quantization_config")),
            "context_length": int(cfg.get("max_position_embeddings") or 0),
            "arch": cfg.get("model_type", "?")}


def smallest_ceiling(shape: dict, reserve_gb: float, hi: float = 1024.0) -> tuple:  # noqa: C901
    """(the smallest budget `choose_capacity` accepts, the plan it gives there), or (None, None).

    Found by ASKING the planner, never by adding up its constants. It is monotone -- more memory
    never makes a plan refuse -- so a bisection is exact, and it stays exact if the planner's
    terms ever change.
    """
    from . import autoconfig

    def accepts(gb):
        try:
            return autoconfig.choose_capacity(
                shape["manifest"], budget_gb=gb, top_k=shape["top_k"],
                reserve_gb=reserve_gb, non_expert_gb=shape["non_expert_gb"],
                headroom_gb=autoconfig.scaled_headroom(gb))
        except MemoryError:
            return None

    # BRACKET TIGHTLY FIRST, BECAUSE EACH PROBE IS EXPENSIVE.
    #     `choose_capacity` samples memory pressure on every successful call, and that sampling
    #     SLEEPS for 0.4 s -- it has to, to tell a compressor that is growing from one that
    #     merely has history. Sixty blind halvings over [0, 1024] therefore took most of a
    #     minute for a question a user asks before deciding whether to download. A budget that
    #     covers the reserve, the resident weights and every expert at once cannot fail, so it
    #     is an exact upper bound, and from there twenty halvings resolve past a megabyte.
    upper = reserve_gb + shape["non_expert_gb"] + shape["expert_gb"] + 1.0
    hi = min(hi, upper)
    if accepts(hi) is None:
        return None, None                     # not runnable at any sane budget
    lo = 0.0
    for _ in range(20):                       # 20 halvings of a bracketed range: sub-megabyte
        mid = (lo + hi) / 2
        if accepts(mid) is None:
            lo = mid
        else:
            hi = mid
    gb = math.ceil(hi * 10) / 10              # a tenth of a gigabyte, rounded up, never under
    plan = accepts(gb)
    while plan is None and gb < 1024:         # rounding must never land back in the refusal
        gb = round(gb + 0.1, 1)
        plan = accepts(gb)
    return gb, plan


def verdict(shape: dict, budget_gb: float, reserve_gb: float, search: bool = True) -> dict:
    """Whether it runs at `budget_gb`, and if `search`, what the smallest workable ceiling is.

    The search is optional because it is not free: every probe goes through `choose_capacity`,
    which samples memory pressure over a 0.4 s window on each success. A caller that only wants
    "does this fit" should not pay twenty of those.
    """
    from . import autoconfig

    try:
        plan = autoconfig.choose_capacity(
            shape["manifest"], budget_gb=budget_gb, top_k=shape["top_k"],
            reserve_gb=reserve_gb, non_expert_gb=shape["non_expert_gb"],
            headroom_gb=autoconfig.scaled_headroom(budget_gb))
        fits_now, why = True, ""
    except MemoryError as e:
        plan, fits_now, why = None, False, str(e)
    need_gb, need_plan = smallest_ceiling(shape, reserve_gb) if search else (None, None)
    return {"fits_now": fits_now, "plan": plan, "why_not": why,
            "needs_gb": need_gb, "plan_at_needs": need_plan}


# WHAT "RUNS" MEANS FOR SPEED, IN WORDS A PERSON CAN ACT ON.
#     A verdict of RUNS with no speed attached let a 37.7 GB model through the door at 1.6 tok/s,
#     which nobody would have downloaded had they known. The first version of this pinned four
#     words to residency buckets from three runs on the copy path. The zero-copy path and whole
#     layers made those numbers stale: measured 2026-09-02, 5% residency ran at 10.5 tok/s on a
#     cold page cache and 21 on a warm one, where the old table said 1-2.
#
#     So the prediction is physical now. A streamed token's cost is the expert bytes it moves:
#
#         bytes/token = miss rate x top_k x streamed layers x bytes per expert
#
#     and the rate it moves them at is the disk's when the page cache is cold and roughly
#     6.4 GB/s of effective traffic when it is warm (21 tok/s x 306 MB, Qwen3.6-35B-A3B-4bit).
#     The miss rate is taken as 0.6: measured 0.53-0.61 across 5-30% residency on two models,
#     because routing is skewed and whole layers take the slots that would otherwise raise it.
#     Cold reads through the zero-copy path run at about 0.65 of the disk's parallel bandwidth,
#     because the GPU faults pages in one at a time (measured 3.2-3.9 GB/s effective against a
#     5.34 GB/s disk). Every constant here is a measurement and says so; the words are
#     TOK_S_TIERS, the same four the console uses once it has measured.
MISS_RATE_ASSUMED = 0.6
WARM_GBS = 6.4
COLD_EFFICIENCY = 0.65
DEFAULT_DISK_GBS = 3.0            # a conservative SSD, when this Mac has not been calibrated


def expert_bytes_per_token(shape: dict, plan: dict) -> int:
    """Expert bytes one streamed token moves, given the plan's whole-layer split."""
    from .autoconfig import plan_full_layers
    n_layers, n_experts = int(shape["n_layers"]), int(shape["n_experts"])
    top_k = int(shape.get("top_k") or plan.get("top_k") or 1)
    cap = int(plan.get("capacity") or 0)
    if cap >= n_experts or plan.get("fits_entirely"):
        return 0
    nf, _rest = plan_full_layers(n_layers, n_experts, top_k, cap)
    return int(MISS_RATE_ASSUMED * top_k * (n_layers - nf) * int(shape["bytes_per_expert"]))


def predict_tok_s(shape: dict, plan: dict, disk_gbs: float | None = None) -> tuple:
    """(cold, warm) tokens/s a streamed plan should roughly reach: cold-disk to warm page cache.
    (inf, inf) for a model that fits whole. A prediction, said as one; never a measurement."""
    b = expert_bytes_per_token(shape, plan)
    if b <= 0:
        return float("inf"), float("inf")
    disk = float(disk_gbs or DEFAULT_DISK_GBS)
    return (disk * COLD_EFFICIENCY * 1e9 / b, WARM_GBS * 1e9 / b)


# THE SAME FOUR WORDS, PINNED TO WHAT A PERSON IS DOING WITH THE MODEL.
#     A coding agent issues many short turns and waits on each one; a person chatting reads at
#     about six or seven tokens a second. Used for a measured median on the console and for the
#     cold-cache prediction before a download, so the two never disagree about what a number means.
TOK_S_TIERS = (
    (15.0, "FAST",   "fast enough for a coding agent; replies arrive faster than most people read"),
    (8.0,  "GOOD",   "comfortable for chat, and workable for a coding agent"),
    (3.0,  "USABLE", "fine for chat, slow for a coding agent"),
    (0.0,  "SLOW",   "it runs, but you will be waiting on it"),
)


def tier_for_tok_s(tok_s: float) -> tuple:
    """(label, sentence) for a decode rate, measured or predicted."""
    for floor, label, why in TOK_S_TIERS:
        if tok_s >= floor:
            return label, why
    return TOK_S_TIERS[-1][1], TOK_S_TIERS[-1][2]


def speed_tier(shape: dict | None, plan: dict | None, disk_gbs: float | None = None,
               fits_entirely: bool = False) -> tuple:
    """(label, sentence) for a plan before anything has been measured. The label is the COLD
    prediction -- the floor a person sees on a cache that has not warmed -- and the sentence
    carries the range with the reason, never a bare number pretending to be a measurement."""
    if fits_entirely or shape is None or plan is None:
        return "FAST", "the whole model fits in memory; nothing is streamed"
    cold, warm = predict_tok_s(shape, plan, disk_gbs)
    if cold == float("inf"):
        return "FAST", "the whole model fits in memory; nothing is streamed"
    label, why = tier_for_tok_s(cold)
    return label, (f"expect roughly {cold:.0f}-{warm:.0f} tokens/s, cold disk to warm page cache "
                   f"({expert_bytes_per_token(shape, plan) / 1e6:.0f} MB of expert reads a token); "
                   f"{why}")


def suggest_repos(name: str, limit: int = 3, token: str | None = None) -> list:
    """Closest public MLX repos to a name that did not resolve, most-downloaded first.

    A typo or a half-remembered name is the first thing a new user does, and "check the name" is
    not an answer to it. Failure here is silent -- suggestions are a courtesy, and a hub outage
    must not turn one error into two.
    """
    try:
        from huggingface_hub import HfApi
        base = name.split("/", 1)[-1]
        # Strip quant suffixes and dots so "Qwen3.6-35b-4bit" finds "Qwen3.6-35B-A3B-4bit".
        stem = re.sub(r"[-_.](\d+bit|mxfp\d|q\d.*|bf16|fp16|instruct)$", "", base, flags=re.I)
        stem = re.split(r"[-_]", stem)[0] if len(stem) > 24 else stem
        api = HfApi(token=token)
        found = api.list_models(search=stem, author="mlx-community", limit=40)
        ranked = sorted(found, key=lambda m: -(getattr(m, "downloads", 0) or 0))
        return [m.id for m in ranked[:limit]]
    except Exception:                          # noqa: BLE001 -- a courtesy, never a second error
        return []


# THE CEILING IS A DEFAULT, AND ON A MAC WITH ROOM IT IS WORTH SAYING SO.
#     Measured on Qwen3.6-35B-A3B-4bit with an identical plan -- 38 of 256 experts, four whole
#     layers, a 2.64 GB pool -- raising the ceiling from 9.0 to 9.7 GB took a clean run from
#     5.4 to 11.2 tok/s. The pool did not grow; the headroom did, and the pool stopped being
#     compressed underneath a plan that looked fine. A person reading USABLE or SLOW deserves to
#     know that the number above is at the safe default, and what one variable did.
# Packed vs the model's own shards, same plan, same Mac (Qwen3.6-35B-A3B-4bit, 9.7 GB). The
# packed copy is what lets the GPU read an expert in place; the shards force a copy. With a warm
# page cache that was 5.4 -> 21.0 tok/s; when the disk is the bottleneck (cold cache, 2026-09-02)
# both paths wait on the same reads and it was 9.9 -> 10.6, 1.07x. Both measured here.
PACK_FROM, PACK_TO = 5.4, 21.0
PACK_COLD_RATIO = 1.07


def ceiling_hint(budget_gb: float, total_gb: float, tier: str, repo_id: str,
                 shape: dict | None = None, reserve_gb: float | None = None,
                 disk_gbs: float | None = None) -> str:
    """One sentence, or nothing. Only for verdicts a person might want to improve, only when the
    Mac actually has room, and only when the plan at the higher ceiling would actually move
    fewer bytes a token -- computed for THIS model, never quoted from another run. (The example
    this used to quote, 9.0 -> 9.7 GB taking one model from 5.4 to 11.2 tok/s, did not hold when
    re-measured: at both ceilings the knee chose the same 38 experts, 10.5 tok/s either way.)"""
    if tier not in ("USABLE", "SLOW") or not total_gb or budget_gb <= 0 or shape is None:
        return ""
    share = budget_gb / total_gb
    if share >= 0.45:                        # already raised well past the default; nothing to add
        return ""
    raised = round(min(total_gb * 0.42, budget_gb + 1.0), 1)
    if raised <= budget_gb + 0.2:
        return ""
    from .autoconfig import choose_capacity
    kw = dict(top_k=shape["top_k"], reserve_gb=reserve_gb, non_expert_gb=shape["non_expert_gb"])
    try:
        now = choose_capacity(shape["manifest"], budget_gb=budget_gb, **kw)
        up = choose_capacity(shape["manifest"], budget_gb=raised, **kw)
    except MemoryError:
        return ""
    b_now, b_up = expert_bytes_per_token(shape, now), expert_bytes_per_token(shape, up)
    if b_up <= 0 or b_now <= 0 or b_up > b_now * 0.87:      # less than ~15% fewer bytes: not worth a line
        return ""
    c_now, _ = predict_tok_s(shape, now, disk_gbs)
    c_up, _ = predict_tok_s(shape, up, disk_gbs)
    return (f"{budget_gb:.1f} GB is the safe default ({share:.0%} of this Mac's {total_gb:.1f} GB). "
            f"At {raised} GB the plan keeps {up['capacity']} of {shape['n_experts']} experts a layer "
            f"instead of {now['capacity']}, moving {b_up / 1e6:.0f} MB a token instead of "
            f"{b_now / 1e6:.0f}: roughly {c_now:.0f} -> {c_up:.0f} tokens/s on a cold cache.\n"
            f"to try {raised} GB:  BIGRIG_MAX_GB={raised} bigrig run {repo_id}")
