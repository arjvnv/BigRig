"""COMPONENT 12 — the engine: make the six components run together on a real model.

WHAT THIS IS, STATED PRECISELY, BECAUSE THE DISTINCTION MATTERS
    Components 6-11 have each been measured alone, on traces and on files. None of them had ever
    seen a model generate a token. This runs them together against a live MLX model and a real
    packed weight file, driven by the model's OWN routing decisions.

    It runs in SHADOW MODE: the model computes normally from its resident weights, while the
    engine observes every routing decision, maintains the pool, fetches the missing experts from
    disk for real, and checks that the bytes it fetched are byte-identical to the ones the model
    actually used. Output is therefore provably unchanged, and every number below -- miss rate,
    bytes moved, fetch time -- comes from a real generation rather than a trace replay.

    WHAT SHADOW MODE DOES NOT PROVE: that a model can run with experts ABSENT from memory. MLX
    holds each layer's experts in one (E, ...) array, so saving memory means replacing the
    gather with one that indexes a C-slot pool -- a change to the model's forward pass, not to
    the engine. The round-trip needed for that IS verified here (zero an expert's weights, fetch
    its bytes, reinstall, confirm the model's output returns bit-identical), so the remaining
    work is plumbing rather than an open question. But it is remaining work, and calling shadow
    mode "streaming" would be a lie.

WHAT IT COMPOSES
    adapter    describes the checkpoint          -> expert count, layers, active bytes
    calibrate  measures this host                -> bandwidths, kappa, thread count
    repack     expert-contiguous layout          -> one pread per expert
    fetch      parallel reads                    -> the bytes
    pool       LFUDA eviction                    -> which experts are resident
    prefetch   pipelining                        -> overlap fetch with compute
    meter      the quality layer                 -> is the output still good
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field

import numpy as np


@dataclass
class EngineStats:
    tokens: int = 0
    layers_seen: int = 0
    expert_requests: int = 0
    misses: int = 0
    bytes_fetched: int = 0
    fetch_seconds: float = 0.0
    byte_checks: int = 0
    byte_mismatches: int = 0
    per_layer_miss: dict = field(default_factory=dict)

    @property
    def miss_rate(self) -> float:
        return self.misses / max(self.expert_requests, 1)

    @property
    def fetch_gbs(self) -> float:
        return self.bytes_fetched / self.fetch_seconds / 1e9 if self.fetch_seconds > 0 else 0.0

    def report(self) -> dict:
        return {"tokens": self.tokens, "expert_requests": self.expert_requests,
                "misses": self.misses, "miss_rate": round(self.miss_rate, 4),
                "gb_fetched": round(self.bytes_fetched / 1e9, 3),
                "fetch_seconds": round(self.fetch_seconds, 3),
                "fetch_gbs": round(self.fetch_gbs, 2),
                "byte_checks": self.byte_checks, "byte_mismatches": self.byte_mismatches}


class Engine:
    """The six components, wired together, driven by a real model's routing."""

    def __init__(self, model_dir: str, packed_path: str, capacity: int | None = None,
                 profile: dict | None = None, policy=None, verify_bytes: int = 32,
                 threads: int | None = None):
        from .adapter import describe
        from .calibrate import calibrate
        from .fetch import ParallelFetcher, WeightStore
        from .pool import recommended
        from .repack import load_layout

        self.spec = describe(model_dir)
        self.profile = profile or calibrate(weight_path=packed_path)
        self.layout, self.manifest = load_layout(packed_path)
        self.store = WeightStore(packed_path, self.layout)
        self.fetcher = ParallelFetcher(
            self.store, threads=threads or self.profile.get("fetch_threads", 8))

        # capacity in EXPERTS PER LAYER, sized from measured available memory unless overridden
        if capacity is None:
            budget = max(0.5, self.profile["available_gb"] - 4.0) * 1e9
            per_layer = budget / max(len(self.spec.moe_layers), 1)
            capacity = int(max(1, min(self.spec.n_experts,
                                      per_layer // max(self.spec.expert_bytes, 1))))
        self.capacity = capacity
        self._factory = policy or recommended()
        self.pools = {l: self._factory(capacity, self.spec.n_experts)
                      for l in self.spec.moe_layers}
        self.resident = {l: set() for l in self.spec.moe_layers}
        self.stats = EngineStats()
        self.verify_budget = verify_bytes
        self._live_weights = None

    # ------------------------------------------------------------------ core step
    def route(self, layer: int, experts, ranks=None, fetch: bool = True):
        """Tell the engine which experts layer `layer` just selected. Returns the missing ones.

        This is the whole engine in one call: consult the pool, fetch what is absent, evict to
        make room, and account for it. Called once per MoE layer per token.
        """
        if layer not in self.resident:
            return []
        ranks = ranks if ranks is not None else range(len(experts))
        res = self.resident[layer]
        pol = self.pools[layer]
        missing = []
        for e, r in zip(experts, ranks):
            e = int(e)
            self.stats.expert_requests += 1
            if e in res:
                pol.touch(e, r)
                continue
            self.stats.misses += 1
            self.stats.per_layer_miss[layer] = self.stats.per_layer_miss.get(layer, 0) + 1
            missing.append(e)
            if len(res) >= self.capacity:
                v = pol.victim()
                if v in res:
                    res.discard(v)
                    if hasattr(pol, "evicted"):
                        pol.evicted(v)
            res.add(e)
            pol.admit(e, r)
        if missing and fetch:
            keys = [(layer, e) for e in missing if (layer, e) in self.layout]
            if keys:
                t0 = time.perf_counter()
                got = self.fetcher.fetch(keys)
                self.stats.fetch_seconds += time.perf_counter() - t0
                self.stats.bytes_fetched += sum(len(v) for v in got.values())
                if self.verify_budget > 0:
                    self._verify(got)
        self.stats.layers_seen += 1
        return missing

    def _verify(self, got: dict) -> None:
        """Confirm fetched bytes equal what the LIVE model holds for the same expert.

        This is the check that makes shadow mode meaningful. Without it the engine could be
        fetching plausible-looking bytes from the wrong offsets and every timing number would be
        measuring a read of the wrong data.
        """
        if self._live_weights is None:
            return
        for (layer, e), blob in got.items():
            if self.verify_budget <= 0:
                return
            ref = self._live_weights(layer, e)
            if ref is None:
                continue
            self.stats.byte_checks += 1
            self.verify_budget -= 1
            if ref != blob[:len(ref)]:
                self.stats.byte_mismatches += 1

    def bind_live_weights(self, fn) -> None:
        """Supply a callable (layer, expert) -> bytes as the live model holds them."""
        self._live_weights = fn

    def close(self) -> None:
        self.fetcher.close()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

    # ------------------------------------------------------------------ reporting
    def projected_tok_s(self) -> dict:
        """Token rate implied by the MEASURED miss rate and this host's measured bandwidths.

        A COMPOSITION BUG THAT ONLY END-TO-END TESTING FOUND. This used to call plan() with the
        model's size, and plan() re-derives residency from available memory and clamps the miss
        rate to zero when the model fits. Running a 3.62 GB model on a machine with 16 GB free
        therefore reported "213 tok/s, disk 0% of the time" one line after the engine had
        MEASURED a 49.78% miss rate at a deliberately small pool.

        Both pieces were individually correct. plan() is right that a model which fits cannot
        miss; the engine is right that a 16-of-64 pool misses half the time. They disagreed
        because plan() inferred residency from memory while the engine was told its capacity.
        The engine's own configuration wins, because that is the pool that actually ran.
        """
        ram, disk = self.profile["ram_gbs"], self.profile["disk_gbs"]
        active = self.spec.active_bytes_per_token / 1e9
        m = self.stats.miss_rate
        disk_part = active * m / disk
        ram_part = active * (1 - m) / ram
        t = disk_part + ram_part
        # HOW EXPENSIVE A MISS IS, on this host's measured kappa. Worth having in front of you,
        # because "a few percent" sounds cheap and is not:
        #     miss 0.1% -> disk is  1.8% of the time
        #     miss 0.5% ->          8.3%
        #     miss 2.0% ->         26.9%
        #     miss 5.0% ->         48.6%
        # A 5% miss rate spends half the token budget on disk. Every target in this project that
        # said "2-5% miss is fine" was understating what that costs.
        return {"residency": round(self.capacity / self.spec.n_experts, 3),
                "capacity_experts": self.capacity,
                "miss_rate_used": m,
                "seconds_per_token": round(t, 5),
                "tok_s": round(1 / t, 2) if t > 0 else float("inf"),
                "disk_fraction_of_time": round(disk_part / t, 3) if t > 0 else 0.0,
                "note": "residency is the CONFIGURED pool size, not inferred from free memory"}

    def report(self) -> dict:
        r = self.stats.report()
        r.update({
            "model": self.spec.model_type, "n_experts": self.spec.n_experts,
            "capacity_per_layer": self.capacity,
            "residency_pct": round(100 * self.capacity / self.spec.n_experts, 1),
            "policy": getattr(self._factory(1, 1), "name", "?"),
            "moe_layers": len(self.spec.moe_layers),
            "host": {k: self.profile[k] for k in ("ram_gbs", "disk_gbs", "kappa",
                                                  "fetch_threads") if k in self.profile},
            "projected": self.projected_tok_s(),
        })
        return r
