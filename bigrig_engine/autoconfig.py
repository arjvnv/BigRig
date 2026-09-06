"""Pick a residency for THIS machine, right now, without asking the user anything.

The single number a streaming engine has to get right is how many experts to keep in RAM. Too
many and the machine swaps, which on macOS means the compressor starts working and everything
including the user's editor stutters. Too few and every token waits on disk.

Nothing here is a guess: available memory comes from the host's own vm_stat, expert sizes from
the packed manifest, and the reserve from what a decode step was measured to actually need.
"""
from __future__ import annotations

import os

from .calibrate import available_gb, under_pressure

# Held back from the pool, in GB. Not padding -- each term was hit during development:
#   OS + other apps    the guard fired at 0.90 GB free with a healthy machine underneath
#   MLX runtime        Metal buffers, the command queue, the allocator's own cache
#   KV cache           grows with context; 8k tokens of a 48-layer model is ~1 GB
#   activations        transient, but peak matters because peak is what triggers swap
RESERVE_GB = 3.0
MIN_HEADROOM_GB = 1.0
# The absolute least slack to leave above everything the planner has accounted for, on any
# machine. Below this a transient MLX allocation lands in swap. Headroom scales DOWN toward this
# on a tight budget and never below it -- see scaled_headroom.
HEADROOM_FLOOR_GB = 0.5


def scaled_headroom(budget_gb: float) -> float:
    """Slack to leave free, scaled to the budget.

    A flat 1.0 GB is right at 9 GB and wrong at 3.5, where it is a third of everything the user
    has. The KV cache and the scratch peak are both accounted for elsewhere (the context limit
    and the working-memory reserve), so this is pure margin -- and margin should be a share of
    the budget, floored so it never vanishes. 12% of the budget, capped at the flat 1.0 it
    replaces and floored at HEADROOM_FLOOR_GB: unchanged at 8.3 GB and up, gently smaller below.
    """
    return round(max(HEADROOM_FLOOR_GB, min(MIN_HEADROOM_GB, float(budget_gb) * 0.12)), 2)


def model_shape(manifest: dict) -> dict:
    layers = manifest["layers"]
    keys = sorted(layers, key=int)
    first = layers[keys[0]]
    return {"n_layers": len(keys), "n_experts": first["n_experts"],
            "bytes_per_expert": first["bytes_per_expert"],
            "expert_bytes": manifest["total_bytes"],
            "layer_keys": [int(k) for k in keys]}


def plan_full_layers(n_layers: int, n_experts: int, top_k: int, capacity: int,
                     floor_over_topk: int = 1) -> tuple:
    """(how many layers to hold whole, what capacity the rest get) at the SAME total memory.

    WHY A LAYER HELD WHOLE IS FASTER, AND WHY IT ONLY BECAME TRUE RECENTLY
        Every streamed layer reads its router's output back to the host to decide what to fetch,
        and that read drains the GPU queue. Profiled over 42 tokens, the single np.array() each
        layer does accounted for 4.488 s of a 6.1 s run -- 2.23 ms a layer, 107 ms of a 127 ms
        token. It is the largest cost in the engine by a wide margin.

        A layer holding every expert cannot miss, so it translates expert ids to slots on device
        and never pays it. It costs `n_experts` slots, which must come out of the other layers.
        That was a losing trade for the whole life of this project, because a miss cost 0.665 ms
        and starving the other layers bought more misses than the saved syncs were worth. Reading
        an expert as a view of the file now costs 0.014 ms, so the cost side has collapsed.

        Measured on Qwen3-30B-A3B-3bit, 27 samples each, three interleaved rounds:

            uniform 36 slots           median 10.49 tok/s   mean 10.52   sd 1.86   3.57 GB
            10 whole + 11 elsewhere    median 12.90 tok/s   mean 12.95   sd 1.68   3.51 GB

        1.23x, at slightly LESS memory, with a sample from the second beating one from the first
        86% of the time. Byte-identical output, 6 of 6 replies -- the weights and the arithmetic
        are untouched, only which experts sit where.

    WHY IT TAKES AS MANY AS IT CAN
        Each whole layer saves one round-trip and costs the rest some residency. Since a miss is
        now nearly free, more is better until the floor binds. Measured 0 -> 6 -> 10 whole layers:
        12.78, 13.78, 14.48 tok/s, monotone.

    THE FLOOR IS NOT NEGOTIABLE. A layer routing to top_k experts needs at least that many slots
    or `_chunks` splits every single step, which is correct and ruinous. One above top_k leaves
    the pool somewhere to put anything at all between steps.
    """
    n_layers, n_experts = int(n_layers), int(n_experts)
    top_k, capacity = int(top_k), int(capacity)
    floor = max(1, top_k + int(floor_over_topk))
    if capacity >= n_experts or n_layers < 2 or floor >= n_experts:
        return 0, capacity                       # already whole, or nothing to redistribute
    total = capacity * n_layers                  # the slot budget, however it is spread
    if total < n_layers * floor:
        return 0, capacity                       # cannot even afford the floor everywhere
    # L whole layers plus (n_layers - L) at the floor must fit the same budget.
    room = n_experts - floor
    n_full = (total - n_layers * floor) // room if room > 0 else 0
    n_full = max(0, min(n_layers - 1, int(n_full)))
    if n_full == 0:
        return 0, capacity
    rest = (total - n_experts * n_full) // (n_layers - n_full)
    rest = max(floor, min(n_experts, int(rest)))
    # Never spend more than the caller had. Rounding down `rest` can leave a little over, which
    # is fine; going over is not, and is the failure this whole engine exists to avoid.
    while n_full > 0 and n_experts * n_full + rest * (n_layers - n_full) > total:
        n_full -= 1
        if n_full:
            rest = max(floor, (total - n_experts * n_full) // (n_layers - n_full))
    if n_full == 0:
        return 0, capacity
    return n_full, rest


def choose_capacity(manifest: dict, budget_gb: float | None = None, top_k: int = 8,
                    reserve_gb: float = RESERVE_GB, non_expert_gb: float = 0.0,
                    resident_reserve_gb: float | None = None,
                    headroom_gb: float | None = None) -> dict:
    """How many experts per layer fit, and what that implies.

    Returns `capacity`, plus `fits_entirely` when the whole model would fit anyway -- in which
    case the honest answer is to not stream at all. An engine that inserts itself into a model
    that already fits is pure overhead, and saying so is worth more than a sale.
    """
    sh = model_shape(manifest)
    avail = available_gb() if budget_gb is None else float(budget_gb)
    headroom = MIN_HEADROOM_GB if headroom_gb is None else float(headroom_gb)
    # MIN_HEADROOM is subtracted as well, not just declared. Sizing the pool to consume every
    # last byte leaves 0.0 GB of slack, and the first KV cache growth then lands in swap.
    # Attention, embeddings and the lm_head are resident whatever the pool does, so they come
    # out of the budget before the pool is sized. They are 0.67 GB on Qwen3-30B, which is why
    # leaving them out was survivable there, and 2.39 GB on gpt-oss-120b, which is why it was
    # not: the pool was planned as though a fifth of the budget were still free.
    pool_gb = avail - reserve_gb - headroom - max(0.0, float(non_expert_gb))
    per_layer_all = sh["bytes_per_expert"] * sh["n_experts"] / 1e9
    total_gb = sh["expert_bytes"] / 1e9

    if pool_gb <= 0:
        raise MemoryError(
            f"only {avail:.1f} GB is available, {reserve_gb:.1f} GB must be held back for the "
            f"runtime and the KV cache, and {non_expert_gb:.1f} GB of this model is resident "
            f"whatever the pool does. Close something, or use a smaller model.")

    per_expert_all_layers = sh["bytes_per_expert"] * sh["n_layers"] / 1e9
    cap = int(pool_gb / per_expert_all_layers)
    cap = max(0, min(sh["n_experts"], cap))
    # A model that fits entirely is resident, so it is judged against the resident reserve --
    # the same reasoning as choose_strategy. Falls back to `reserve_gb` when the caller has not
    # said, which keeps every existing caller's answer unchanged.
    fits = total_gb + (reserve_gb if resident_reserve_gb is None
                       else float(resident_reserve_gb)) <= avail

    if cap < top_k:
        raise MemoryError(
            f"this model needs at least {top_k} experts per layer resident "
            f"({top_k * per_expert_all_layers:.1f} GB) and only {max(pool_gb,0):.1f} GB is free for the "
            f"pool. It cannot run on this machine right now.")

    return {"capacity": cap, "n_experts": sh["n_experts"], "residency": cap / sh["n_experts"],
            "pool_gb": cap * per_expert_all_layers, "available_gb": avail,
            "reserve_gb": reserve_gb, "model_expert_gb": total_gb, "fits_entirely": fits,
            "n_layers": sh["n_layers"], "under_pressure": under_pressure(),
            "headroom_gb": avail - reserve_gb - cap * per_expert_all_layers,
            "bytes_per_expert": sh["bytes_per_expert"], "top_k": top_k}


def describe(choice: dict) -> str:
    if choice["fits_entirely"]:
        return (f"This model fits in RAM on its own ({choice['model_expert_gb']:.1f} GB of experts, "
                f"{choice['available_gb']:.1f} GB available). Streaming would only slow it down.")
    return (f"Keeping {choice['capacity']} of {choice['n_experts']} experts resident per layer "
            f"({choice['residency']*100:.0f}%), a {choice['pool_gb']:.1f} GB pool out of "
            f"{choice['model_expert_gb']:.1f} GB of expert weights.")


# --------------------------------------------------------------------------- strategy
# The smallest MLX affine quantisation. Below this there is nothing to fall back to, so it is
# the hard floor on how small any model can be made.
FLOOR = (2, 128)


def _bpp(bits: int, group: int) -> float:
    return (bits + 32.0 / group) / 8.0


# MEASURED, on OLMoE-1B-7B-4bit against wikitext-2 and man pages:
#     3-bit g128   2.62 GB   perplexity +18.5% (wiki) / +36.0% (man)
#     2-bit g64    2.01 GB              +83.1%        / +122.6%
#     2-bit g128   1.81 GB              +99.3%        / +145.0%
# 2-bit is not a cheaper point on a curve, it is a cliff. On a 4x0.6B model compressed to 2-bit
# the engine ran at 34.9 tok/s, kept every expert resident, and emitted ".\n1\n1". Speed without
# a quality floor is not a feature, so the default floor is 3 bits and dropping below it has to
# be asked for. Larger models may tolerate 2-bit better; that is untested here and not assumed.
DEFAULT_MIN_BITS = 3


def choose_strategy(manifest: dict, budget_gb: float | None = None, top_k: int = 8,
                    reserve_gb: float = RESERVE_GB, non_expert_gb: float = 0.0,
                    min_bits: int = DEFAULT_MIN_BITS,
                    resident_reserve_gb: float | None = None,
                    headroom_gb: float | None = None) -> dict:
    """Decide HOW to run this model on this machine, not just how much of it to keep.

    Three outcomes, in the order they are preferred:

      native     it already fits. Run it untouched; anything else is pure overhead.
      compress   it does not fit, but it fits in fewer bits. Every expert stays in RAM, so
                 nothing is ever fetched and nothing stalls -- the engine costs no wall clock at
                 all, and the price is quantisation error instead.
      stream     it does not fit even at the floor precision. Keep what fits, fetch the rest.
                 Exact, but it pays a GPU stall at every layer: measured 98.5 ms of a 119.4 ms
                 token on Qwen3-30B.

    The order matters because it is the reverse of how much the user pays. `native` is free,
    `compress` costs accuracy, `stream` costs speed. Reaching for the engine when the model
    already fits, or streaming when compressing would have done, is the engine making things
    worse and calling it a feature.
    """
    sh = model_shape(manifest)
    avail = available_gb() if budget_gb is None else float(budget_gb)
    # TWO ROOMS, BECAUSE THE THREE MODES DO NOT COST THE SAME TO RUN.
    #     `native` and `compress` both keep every expert in RAM, so neither ever services a miss
    #     and neither allocates the admission buffer that is the largest transient in a streamed
    #     step. Measured, that difference is about 1.5 GB (see RESIDENT_WORKING_MEMORY_GB).
    #     Charging a resident model for machinery it does not run pushed models into streaming
    #     that had room to run whole -- the slower path, for memory nobody was going to use.
    #     Defaults to `reserve_gb` so a caller that says nothing gets exactly the old behaviour.
    res_reserve = reserve_gb if resident_reserve_gb is None else float(resident_reserve_gb)
    headroom = MIN_HEADROOM_GB if headroom_gb is None else float(headroom_gb)
    room = avail - reserve_gb - headroom - non_expert_gb
    resident_room = avail - res_reserve - headroom - non_expert_gb
    q = (manifest["layers"][str(sh["layer_keys"][0])].get("quant")
         or {"bits": 4, "group_size": 64})
    src = _bpp(int(q["bits"]), int(q["group_size"]))
    params = sh["expert_bytes"] / src
    total_gb = sh["expert_bytes"] / 1e9

    per_layer_all = sh["bytes_per_expert"] * sh["n_layers"] / 1e9
    if total_gb <= resident_room:
        return {"mode": "native", "expert_gb": total_gb, "available_gb": avail,
                "room_gb": resident_room, "source_bits": int(q["bits"]),
                "reason": "the model already fits; streaming or compressing it would only "
                          "make it slower or worse"}

    # Cheapest precision that still fits, preferring the fewest bits given up.
    for bits, group in sorted(
            [c for c in [(8, 64), (6, 64), (4, 64), (4, 128), (3, 64), (3, 128), (2, 64),
                         (2, 128)] if c[0] >= min_bits],
            key=lambda c: -_bpp(*c)):
        gb = params * _bpp(bits, group) / 1e9
        if gb <= resident_room:
            # Also compute the streaming fallback HERE, where `room` already has the non-expert
            # weights subtracted. Letting the caller recompute it from choose_capacity() -- which
            # does not know about them -- made the two disagree: choose_strategy said "does not
            # fit" while choose_capacity said every expert fit, and a declined compression then
            # printed "100% of experts in RAM, the rest streamed from disk".
            fb = max(0, min(sh["n_experts"], int(room / per_layer_all)))
            return {"mode": "compress", "bits": bits, "group_size": group,
                    "stream_capacity": fb, "stream_residency": fb / sh["n_experts"],
                    "expert_gb": gb, "original_gb": total_gb, "available_gb": avail,
                    "room_gb": room, "source_bits": int(q["bits"]),
                    "ratio": gb / total_gb,
                    "reason": f"it does not fit at {q['bits']} bits but does at {bits}; every "
                              f"expert stays resident, so there is no fetch and no stall"}

    # The floor the USER allowed, not the floor MLX supports. With --exact (min_bits above every
    # candidate) no compression is permitted at all, so the honest floor is the source precision
    # -- reporting a 2-bit floor there would explain the decision with a number that was never
    # on the table.
    allowed = [c for c in [(2, 128), (2, 64), (3, 128), (3, 64)] if c[0] >= min_bits]
    if allowed:
        allowed_floor = min(allowed, key=lambda c: _bpp(*c))
        floor_gb = params * _bpp(*allowed_floor) / 1e9
        floor_why = f"even at {allowed_floor[0]}-bit the model needs {floor_gb:.1f} GB"
    else:
        allowed_floor = (int(q["bits"]), int(q["group_size"]))
        floor_gb = total_gb
        floor_why = (f"compression was declined, and untouched the model needs "
                     f"{floor_gb:.1f} GB")
    cap = max(0, min(sh["n_experts"], int(room / per_layer_all)))
    if cap < top_k:
        raise MemoryError(
            f"this model needs {top_k * per_layer_all:.1f} GB just to hold top-{top_k} experts "
            f"per layer, and only {max(room,0):.1f} GB is free. Even at the floor precision it "
            f"would need {floor_gb:.1f} GB at {allowed_floor[0]}-bit. It cannot run on this "
            f"machine right now.")
    return {"mode": "stream", "capacity": cap, "n_experts": sh["n_experts"],
            # The shape the speed word needs. Without these the description could only speak in
            # residency, which stopped meaning anything once the zero-copy path landed.
            "n_layers": sh["n_layers"], "bytes_per_expert": sh["bytes_per_expert"],
            "top_k": int(top_k),
            "residency": cap / sh["n_experts"], "expert_gb": cap * per_layer_all,
            "original_gb": total_gb, "floor_gb": floor_gb, "available_gb": avail,
            "room_gb": room, "source_bits": int(q["bits"]),
            "floor_bits": allowed_floor[0],
            "reason": f"{floor_why}, more than the {max(room,0):.1f} GB free, so part of it "
                      f"must come from disk"}


def describe_strategy(s: dict, disk_gbs: float | None = None) -> str:
    """One line a person can act on. The speed word is the part that was missing: 'Exact, but
    slower' told nobody whether slower meant 9 tok/s or 1.6.

    `disk_gbs` is this Mac's measured read rate when the caller has it, so this line and
    `doctor`'s own verdict cannot disagree about the same plan on the same machine."""
    if s["mode"] == "native":
        return (f"Fits in memory whole ({s['expert_gb']:.1f} GB of experts, "
                f"{s['room_gb']:.1f} GB to spare). Full speed, nothing streamed.")
    if s["mode"] == "compress":
        return (f"Shrunk from {s['original_gb']:.1f} GB to {s['expert_gb']:.1f} GB "
                f"({s['bits']}-bit) so it all fits in memory. Full speed, some accuracy lost.")
    from .preflight import speed_tier                  # lazy: preflight imports this module
    shape = {"n_layers": int(s.get("n_layers") or 0), "n_experts": int(s["n_experts"]),
             "top_k": int(s.get("top_k") or 8),
             "bytes_per_expert": int(s.get("bytes_per_expert") or 0)}
    tier, _ = speed_tier(shape, {"capacity": int(s["capacity"])}, disk_gbs) \
        if shape["n_layers"] and shape["bytes_per_expert"] else ("", "")
    tier = f"{tier}: " if tier else ""
    return (f"Streamed, {tier}{s['capacity']} of {s['n_experts']} experts in memory "
            f"({s['residency']*100:.0f}%, {s['expert_gb']:.1f} GB), the rest read from disk. "
            f"The weights are untouched; decode is bit-identical to the original.")
