"""Serve several requests in one forward pass.

WHY THIS IS THE LARGEST REMAINING WIN, AND WHY IT IS NOT ABOUT ARITHMETIC
    A streamed layer pays a host round-trip every step: the router's output has to be read back
    to decide which experts to fetch, and reading it drains the GPU queue. Measured on
    Qwen3-30B-A3B-3bit at 44/128, that stall is 58.7% of wall-clock time against 34.2% for the
    disk read itself. Nothing about a single sequence can remove it -- the routing genuinely is
    not known until the router runs.

    But the stall is paid per STEP, not per token. Step B sequences together and one stall
    produces B tokens. The disk read amortises the same way, and better than linearly: B
    sequences reveal B tokens of routing at once, so their misses are one batched request
    instead of B small ones, and read rate is a strong function of request size -- 2.2 GB/s for
    one expert against 3.8 GB/s for eight.

    Measured on this model, decode only:

        B     ms/pass    ms/token    tok/s    speed-up
        1      58.24      58.24      17.17     1.00x
        2      77.02      38.51      25.97     1.51x
        4      97.37      24.34      41.08     2.39x
        8     147.49      18.44      54.24     3.16x
       16     252.99      15.81      63.24     3.68x

WHAT IT COSTS, WHICH IS THE PART THAT DECIDES WHETHER IT CAN BE USED HERE
    The expert pool does NOT grow: B sequences share it, and at one layer they need the union of
    their selections, not the sum. The KV cache does grow, linearly in B, and on a machine picked
    for not having enough memory that is the binding term. `plan_batch` prices it against the
    same budget everything else is planned against, and refuses rather than overrunning.

    There is a second limit that is easy to miss. A pool of C slots cannot serve a step that
    needs more than C distinct experts, and B sequences at top-k can ask for up to B*k. Past
    that, `_chunks` splits the step into several passes, which is correct but gives back exactly
    the amortisation the batch was for. `plan_batch` reports the size where that begins.
"""
from __future__ import annotations

import mlx.core as mx
from mlx_lm.models.cache import BatchKVCache


def plan_batch(session, want: int, prompt_tokens: int, reply_tokens: int) -> dict:
    """How many of `want` requests can share a pass, and what stopped it being more.

    Returns `size`, `reason`, and the arithmetic behind both, so a caller can explain the answer
    rather than just obey it.
    """
    want = max(1, int(want))
    kv = session.kv_bytes_per_token or 0
    per_seq_gb = (prompt_tokens + reply_tokens) * kv / 1e9
    spare = session.budget_gb - session.footprint_gb - session.working_memory_gb
    by_memory = want if per_seq_gb <= 0 else int(max(0, spare) // per_seq_gb)

    # A step that needs more distinct experts than a layer holds is split by _chunks, and a split
    # step pays the host round-trip once per piece -- which is the cost the batch existed to
    # amortise. Beyond this size the batch stops paying for itself.
    cap = getattr(session, "capacity", 0) or 0
    top_k = getattr(session, "top_k", 0) or 0
    by_pool = want if (cap <= 0 or top_k <= 0) else max(1, cap // top_k)

    size = max(1, min(want, by_memory, by_pool))
    if size == want:
        reason = "requested"
    elif by_memory <= by_pool:
        reason = "memory"
    else:
        reason = "pool"
    return {"size": size, "reason": reason, "requested": want,
            "by_memory": by_memory, "by_pool": by_pool,
            "gb_per_sequence": round(per_seq_gb, 3), "spare_gb": round(spare, 2)}


def _pad_left(prompts: list[list[int]], pad_id: int):
    """Left-pad to a common width, which is the layout BatchKVCache is built for.

    Left rather than right because every sequence must end at the same column: decode appends one
    token per step to all of them at once, and a right-padded batch would be writing the next
    token into the middle of the shorter rows.
    """
    width = max(len(p) for p in prompts)
    pads = [width - len(p) for p in prompts]
    rows = [[pad_id] * n + list(p) for n, p in zip(pads, prompts)]
    return mx.array(rows), pads, width


def generate_batch(model, prompts: list[list[int]], max_tokens,
                   eos_ids=(), pad_id: int = 0, prefill_step: int = 128,
                   sampler=None, on_token=None) -> list[list[int]]:
    """Generate for every prompt at once, one forward pass per step.

    `sampler(logits) -> mx.array` of shape (B,), or None for greedy. Greedy is what the
    equivalence test uses, because a shared sampler state would make batched and sequential runs
    differ for reasons that have nothing to do with batching.

    Padding is masked, not merely ignored: BatchKVCache records each row's left padding and
    builds the attention mask from it, so a short prompt never attends to the filler in front of
    it. That is what allows prompts of different lengths in one pass at all.
    """
    if not prompts:
        return []
    B = len(prompts)
    # Requests in one pass rarely want the same reply length, so the limit is per row.
    limits = ([int(max_tokens)] * B if isinstance(max_tokens, int)
              else [int(v) for v in max_tokens])
    if len(limits) != B:
        raise ValueError(f"got {len(limits)} limits for {B} prompts")
    eos = set(int(e) for e in eos_ids)
    tokens, pads, width = _pad_left(prompts, pad_id)

    n_layers = len(model.layers) if hasattr(model, "layers") else len(model.model.layers)
    caches = [BatchKVCache(list(pads)) for _ in range(n_layers)]

    # Prefill everything except the final column, in bounded chunks -- the widest forward pass is
    # the largest memory spike the server can produce, batched or not.
    lead = tokens[:, :-1]
    for i in range(0, lead.shape[1], prefill_step):
        model(lead[:, i:i + prefill_step], cache=caches)
        mx.eval([c.keys for c in caches])

    out = [[] for _ in range(B)]
    done = [False] * B
    cur = tokens[:, -1:]
    for _ in range(max(limits) if limits else 0):
        logits = model(cur, cache=caches)[:, -1, :]
        nxt = mx.argmax(logits, axis=-1) if sampler is None else sampler(logits)
        mx.eval(nxt)
        ids = [int(v) for v in nxt]
        live = 0
        for b, t in enumerate(ids):
            if done[b]:
                continue
            if t in eos:
                done[b] = True
                if on_token is not None:
                    on_token(b, None)                 # None means "this one stopped on its own"
                continue
            out[b].append(t)
            if on_token is not None:
                on_token(b, t)
            if len(out[b]) >= limits[b]:
                done[b] = True
            else:
                live += 1
        if not live:
            break
        # A finished row keeps stepping rather than being removed. Dropping it would reshape the
        # batch and every cache in it mid-flight; feeding it its own last token costs one column
        # of a pass that was being made anyway, and its output is discarded above.
        cur = mx.array([[t] for t in ids])
    return out
