"""Putting a model's cache back after a pass whose tokens were not all kept.

Speculative paths -- the MTP head, prompt-lookup drafting -- run the model over tokens that may
turn out to be wrong, and then have to leave the cache holding exactly the tokens that were kept.
mlx_lm's `trim_prompt_cache` does that for attention caches (a KV cache is a write position; trim
moves it back) and does NOTHING at all when any layer cannot be trimmed: it returns 0 and leaves
every layer, attention ones included, holding the rejected tokens.

THE BUG THAT FOUND THIS. Qwen3.6's linear-attention layers keep a recurrent state, not a KV cache.
With "guess ahead" on, every rejected guess -- text copied from the prompt -- stayed in the model's
state as though it had been written. The reply drifted into repeating the user's own sentence and
the model's own reasoning scaffold, verbatim, dozens of times; 105 tokens flagged by the meter.
The measurement that justified the feature had been taken on Qwen3-30B, whose cache trims.

WHAT WORKS INSTEAD. The recurrent layers rebind their state on every step (`cache[0] = ...`), so
the arrays from before a pass are never touched by it: a list of references taken beforehand IS a
snapshot, costs nothing, and puts the layer back exactly. Attention layers go back by offset. A
layer of any other kind cannot be put back, and a speculative path must refuse to run on it
rather than corrupt it.
"""
from __future__ import annotations


def supported(cache) -> str:
    """"" if every layer of this cache can be put back after a pass, else why not."""
    for c in cache:
        if hasattr(c, "cache") and isinstance(getattr(c, "cache"), list):
            continue                                   # ArraysCache: state rebound per step
        if hasattr(c, "offset") and hasattr(c, "trim") and c.is_trimmable():
            continue                                   # KVCache, QuantizedKVCache: a write position
        return (f"a {type(c).__name__} layer cannot be put back after a rejected guess, so a "
                f"speculative pass would leave tokens the model never emitted in its state")
    return ""


def snapshot(cache) -> list:
    """Enough to put every cache entry back exactly as it is now. Immutable arrays are held by
    reference, so this costs nothing; a KV cache is a write position."""
    out = []
    for c in cache:
        if hasattr(c, "cache") and isinstance(getattr(c, "cache"), list):
            out.append(("arrays", list(c.cache)))
        elif hasattr(c, "offset") and hasattr(c, "trim") and c.is_trimmable():
            out.append(("offset", int(c.offset)))
        else:
            raise TypeError(supported([c]))
    return out


def restore(cache, snap: list) -> None:
    """Every layer back to the snapshot. Never raises the write position, only lowers it."""
    for c, (kind, v) in zip(cache, snap):
        if kind == "arrays":
            c.cache = list(v)
        else:
            c.trim(max(0, int(c.offset) - int(v)))


def all_trimmable(cache) -> bool:
    """True when mlx_lm's own trim is enough -- no layer holds recurrent state."""
    return all(hasattr(c, "is_trimmable") and c.is_trimmable() for c in cache)
