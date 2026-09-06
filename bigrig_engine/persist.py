"""Conversations that survive a restart.

WHY. The conversation cache is what makes turn two fast: the attention state for everything
already said is kept, so a follow-up only reads its own new tokens. It lived in memory, so a
restart -- a pool rebuild, an update, a reboot, a `rig serve` closed by accident -- threw it away,
and the next turn of a long agent session paid the whole prefill again. On a streamed model that
is tens of seconds of reading experts for text the machine had already processed once.

WHAT IS KEPT. Exactly the entries the in-memory cache holds, no more: each one is the attention
state for one conversation prefix, written with mlx_lm's own cache serialiser, plus the token ids
it belongs to. Bit-identical on the way back (checked in the tests: every array equal, every
dtype and shape equal). Because the set on disk mirrors the set in memory it is bounded by the
same budget -- half a gigabyte by default -- and an entry the cache evicts is removed from disk
at the next flush.

WHEN. On the server's idle tick, the one moment no reply is in flight, and on close. Never in
the reply path. A 200 MB conversation writes in about 70 ms.

WHAT IS NEVER RESTORED. A saved state is only legal for the numbers that produced it. The
fingerprint covers the model's config and weight sizes, the KV-cache precision and where it
starts, the serving mode (compressed weights compute a different state), the reroute tolerance,
and the mlx_lm version whose format wrote the file. Anything else -- a different model, a
re-quantised copy, `--kv-bits 8` after a 4-bit run -- and the saved files are discarded rather
than restored. Restoring a state the current configuration would not have computed is a quality
loss dressed up as a speed-up, and this never does it.

PRIVACY. The files hold the tokens of the conversations, on this machine, under BigRig's own
data directory, readable by this user. `bigrig sessions` shows what is kept and clears it;
`--no-persist` keeps everything in memory only, as before.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import time

from . import home

SUFFIX = ".safetensors"
# An in-progress write. It keeps the .safetensors ending because mx.save_safetensors appends one
# to any name without it -- the first version wrote to `x.safetensors.tmp`, got
# `x.safetensors.tmp.safetensors` back, and then swept its own file up as a stranger.
TMP_SUFFIX = ".tmp" + SUFFIX
# Sanity ceiling on a single file's token list in metadata; a real conversation is far under.
_MAX_TOKENS = 1_000_000


# ------------------------------------------------------------------------------ where and what
def root() -> str:
    return os.path.join(home(), "data", "sessions")


def model_dir(model_name: str) -> str:
    return os.path.join(root(), model_name)


def _config_hash(model_path: str) -> str:
    """The model's config.json and the sizes of its weight shards, hashed. Catches a different
    quantisation, a different architecture, and a re-downloaded copy whose config happens to
    match but whose weights do not."""
    from .session import _config_dir
    d = _config_dir(model_path)
    h = hashlib.sha1()
    try:
        with open(os.path.join(d, "config.json"), "rb") as f:
            h.update(f.read())
    except OSError:
        h.update(b"no-config")
    try:
        for name in sorted(os.listdir(d)):
            if name.endswith(".safetensors"):
                h.update(f"{name}:{os.path.getsize(os.path.join(d, name))}".encode())
    except OSError:
        pass
    return h.hexdigest()


def fingerprint(session) -> str:
    """Everything that has to be identical for a saved state to be the state this session would
    have computed itself. Sixteen hex characters; also the directory the files live in."""
    import mlx_lm
    from .session import KV_GROUP_SIZE
    parts = {
        "model": session.name,
        "config": _config_hash(session.model_dir),
        "kv_bits": session.kv_bits,
        "kv_group_size": KV_GROUP_SIZE,
        "kv_quant_start": session.kv_quant_start,
        "mode": (getattr(session, "strategy", None) or {}).get("mode"),
        "source_precision": getattr(session, "source_precision", None),
        "reroute": getattr(session, "reroute_tol", None) or None,
        "mlx_lm": getattr(mlx_lm, "__version__", "?"),
    }
    return hashlib.sha1(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _key(tokens) -> str:
    return hashlib.sha1(json.dumps([int(t) for t in tokens]).encode()).hexdigest()[:24]


def _path(d: str, tokens) -> str:
    return os.path.join(d, _key(tokens) + SUFFIX)


def _rm(path: str) -> int:
    """Remove a file; return the bytes it held, 0 if it was not there."""
    try:
        n = os.path.getsize(path)
        os.remove(path)
        return n
    except OSError:
        return 0


# ------------------------------------------------------------------------------ flush / restore
def flush(session) -> dict:
    """Write the conversations the cache holds that are not yet on disk, and drop the files of
    those it has let go. Returns counts; never raises past its own boundary.

    The cache is walked through its public lookup, exact match only: an entry that comes back
    with a non-empty remainder was evicted (or only a prefix of something longer survives), and
    its file goes. An entry already on disk is not rewritten -- entries are immutable once
    stored, a continued conversation is a NEW, longer entry.
    """
    pc = getattr(session, "_prompt_cache", None)
    out = {"written": 0, "removed": 0, "bytes": 0, "on_disk": 0}
    if pc is None or not getattr(session, "persist", False) or not getattr(pc, "known", None):
        if pc is not None and getattr(pc, "dirty", False):
            pc.dirty = False
        return out
    if not pc.dirty:
        return out
    from mlx_lm.models.cache import save_prompt_cache
    fp = fingerprint(session)
    d = os.path.join(model_dir(session.name), fp)
    os.makedirs(d, exist_ok=True)
    for key, info in list(pc.known.items()):
        path = _path(d, key)
        try:
            cache, rest, protected = pc.fetch_nearest_cache(session._cache_key, list(key))
        except Exception:                          # noqa: BLE001
            cache, rest, protected = None, [1], False
        if cache is None or rest:
            pc.known.pop(key, None)
            if _rm(path):
                out["removed"] += 1
            continue
        if info.get("saved"):
            continue
        tmp = path[: -len(SUFFIX)] + TMP_SUFFIX
        try:
            save_prompt_cache(tmp, cache, {
                "fingerprint": fp,
                "tokens": json.dumps([int(t) for t in key]),
                "proven": "1" if (protected or info.get("proven")) else "0",
                "n_tokens": str(len(key)),
                "saved_at": str(int(time.time())),
            })
            os.replace(tmp, path)
            info["saved"] = True
            out["written"] += 1
            out["bytes"] += os.path.getsize(path)
        except Exception:                          # noqa: BLE001 -- a save must never hurt serving
            _rm(tmp)
    # Files nothing in memory accounts for: evicted while this process was not looking, or left
    # by a run that ended without a flush. The disk set mirrors the memory set, nothing more.
    keep = {_key(k) + SUFFIX for k in pc.known}
    try:
        for name in os.listdir(d):
            if name.endswith(SUFFIX) and name not in keep:
                if _rm(os.path.join(d, name)) and not name.endswith(TMP_SUFFIX):
                    out["removed"] += 1
    except OSError:
        pass
    pc.dirty = False
    out["on_disk"] = len(keep)
    return out


def restore(session) -> dict:
    """Load the saved conversations that match this session's fingerprint into its cache, most
    recently saved first, within the cache's own byte budget. Saved states from a DIFFERENT
    fingerprint under this model are deleted: they can never be legal again for the numbers
    this session computes, and keeping them would let the disk set grow past the budget.

    Returns {"restored": n, "bytes": b, "discarded": m}. Never raises past its own boundary;
    a file that fails to load is removed, because it will never load.
    """
    pc = getattr(session, "_prompt_cache", None)
    out = {"restored": 0, "bytes": 0, "discarded": 0, "skipped": 0}
    if pc is None or not getattr(session, "persist", False) or not hasattr(pc, "known"):
        return out
    mdir = model_dir(session.name)
    if not os.path.isdir(mdir):
        return out
    fp = fingerprint(session)
    # Sibling fingerprints are a configuration this session does not have. Gone.
    try:
        for name in os.listdir(mdir):
            p = os.path.join(mdir, name)
            if os.path.isdir(p) and name != fp:
                out["discarded"] += sum(1 for f in os.listdir(p) if f.endswith(SUFFIX))
                shutil.rmtree(p, ignore_errors=True)
    except OSError:
        pass
    d = os.path.join(mdir, fp)
    if not os.path.isdir(d):
        return out
    from mlx_lm.models.cache import load_prompt_cache
    files = []
    for name in os.listdir(d):
        p = os.path.join(d, name)
        if name.endswith(TMP_SUFFIX):
            _rm(p)                                 # a write that never completed
            continue
        if name.endswith(SUFFIX):
            try:
                files.append((os.path.getmtime(p), p))
            except OSError:
                pass
    files.sort(reverse=True)                       # newest first: the conversation most likely to continue
    budget = int(getattr(pc, "max_bytes", 0) or 0)
    total = 0
    chosen = []
    for _, p in files:
        try:
            cache, meta = load_prompt_cache(p, return_metadata=True)
            if meta.get("fingerprint") != fp:
                raise ValueError("fingerprint")
            tokens = json.loads(meta["tokens"])
            if not isinstance(tokens, list) or not tokens or len(tokens) > _MAX_TOKENS:
                raise ValueError("tokens")
            tokens = [int(t) for t in tokens]
            nbytes = sum(int(c.nbytes) for c in cache)
        except Exception:                          # noqa: BLE001 -- unreadable: never will be
            _rm(p)
            out["discarded"] += 1
            continue
        if budget and total + nbytes > budget:
            out["skipped"] += 1                    # would be evicted at once; the next flush drops it
            continue
        total += nbytes
        chosen.append((p, tokens, cache, nbytes, meta.get("proven") == "1"))
    # Inserted OLDEST first, the order they were stored in live. A longer conversation then pops
    # the prefix it grew from exactly as it did when it was first stored, so what the cache holds
    # after a restart is what it held before one.
    for p, tokens, cache, nbytes, proven in reversed(chosen):
        try:
            pc.insert_cache(session._cache_key, tokens, cache, proven=proven)
        except Exception:                          # noqa: BLE001
            out["discarded"] += 1
            _rm(p)
            continue
        pc.known[tuple(tokens)] = {"proven": proven, "saved": True}
        out["restored"] += 1
        out["bytes"] += nbytes
    pc.dirty = bool(out["skipped"])                # so the next flush removes what did not fit
    return out


# ------------------------------------------------------------------------------ the command
def summary() -> list:
    """One row per model with saved conversations: name, files, bytes, age of the newest."""
    rows = []
    r = root()
    if not os.path.isdir(r):
        return rows
    for model in sorted(os.listdir(r)):
        mdir = os.path.join(r, model)
        if not os.path.isdir(mdir):
            continue
        n = b = 0
        newest = 0.0
        for fpdir in os.listdir(mdir):
            p = os.path.join(mdir, fpdir)
            if not os.path.isdir(p):
                continue
            for f in os.listdir(p):
                if f.endswith(SUFFIX):
                    fp = os.path.join(p, f)
                    try:
                        n += 1
                        b += os.path.getsize(fp)
                        newest = max(newest, os.path.getmtime(fp))
                    except OSError:
                        pass
        if n:
            rows.append({"model": model, "conversations": n, "bytes": b, "newest": newest})
    return rows


def clear(model_name: str | None = None) -> dict:
    """Delete saved conversations: one model's, or all. Returns what was removed."""
    rows = summary()
    if model_name is not None:
        rows = [r for r in rows if r["model"] == model_name]
    n = sum(r["conversations"] for r in rows)
    b = sum(r["bytes"] for r in rows)
    for r in rows:
        shutil.rmtree(model_dir(r["model"]), ignore_errors=True)
    if model_name is None:
        shutil.rmtree(root(), ignore_errors=True)
    return {"conversations": n, "bytes": b, "models": len(rows)}
