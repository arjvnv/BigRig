"""Name the next layer's experts early, so the disk read can start before they are asked for.

WHY THIS EXISTS, AND WHY AN EARLIER MEASUREMENT SAID IT COULD NOT
    Streaming an expert off disk costs about 0.6 ms and the layer that needs it can do nothing
    until it lands. Starting that read one layer early would hide it behind compute -- but the
    routing is data-dependent and is not known until the router runs.

    Measured on a 2,271-token trace, predicting from what has already been ROUTED does not work
    at all:

        the same layer's previous token named   0.00% of the misses
        the same layer's last eight tokens      0.68%
        the previous layer, same token          5.91%

    That is not a weak signal, it is the definition of a miss: an expert the recent past used is
    still resident, so the ones that must be read are exactly the ones with no recent history.

    The hidden state is a different signal, and it works. A ridge regression -- a linear map,
    fitted on 170 decode steps -- recovers the NEXT layer's experts:

        recall@8   64.5%      recall@16   80.0%      recall@32   89.5%

    against a same-layer upper bound of 65.4 / 80.8 / 90.0%, which is what a predictor that had
    already seen the routing would get. Predicting one layer ahead costs almost nothing.

WHY A WRONG PREDICTION CANNOT CHANGE AN ANSWER
    The prediction is used for ONE thing: deciding what to start reading. `ensure()` is untouched
    -- it still takes the real router's output, still admits exactly the experts that were
    actually selected, and still fetches anything missing. A prediction that is wrong wastes some
    disk bandwidth. It cannot put a different expert into the computation, so it cannot move a
    logit. This is the same separation SpecPrefetch describes: prediction errors affect transfer
    efficiency rather than model outputs.
"""
from __future__ import annotations

import json
import os
import time

from . import home
import numpy as np

PREDICT_DIR = os.path.join(home(), "data", "results")

# Ridge, not least squares. The hidden state's dimensions are strongly correlated, so the normal
# equations are near-singular and an unregularised solve produces enormous weights that fit the
# training steps and predict nothing. Chosen by sweeping; the recall curve is flat from 30 to 300.
RIDGE = 1e2


def predictor_path(model_name: str) -> str:
    safe = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in model_name)
    return os.path.join(PREDICT_DIR, f"predict_{safe}.npz")


def load(model_name: str):
    """Per-layer weight matrices, or None. Never another model's -- the hidden space differs."""
    p = predictor_path(model_name)
    if not os.path.exists(p):
        return None
    try:
        z = np.load(p)
        meta = json.loads(str(z["meta"]))
        return {"W": [z[f"W{i}"] for i in range(int(meta["n_layers"]))], "meta": meta}
    except (OSError, ValueError, KeyError):
        return None


def save(model_name: str, Ws: list, meta: dict) -> str:
    os.makedirs(PREDICT_DIR, exist_ok=True)
    p = predictor_path(model_name)
    np.savez_compressed(p, meta=json.dumps(meta), **{f"W{i}": w for i, w in enumerate(Ws)})
    return p


def recall_at(scores: np.ndarray, truth: list, m: int) -> float:
    """Share of the experts actually chosen that appear in the predictor's top m."""
    if not len(truth):
        return 0.0
    m = min(m, scores.shape[1])
    top = np.argpartition(-scores, m - 1, axis=-1)[:, :m]
    return float(np.mean([len(set(t) & set(p)) / max(len(set(t)), 1)
                          for t, p in zip(truth, top)]))


def fit(X: np.ndarray, Y: np.ndarray, ridge: float = RIDGE) -> np.ndarray:
    """Closed-form ridge from hidden states to a one-hot-per-expert target."""
    d = X.shape[1]
    A = X.T @ X + ridge * np.eye(d, dtype=np.float32)
    return np.linalg.solve(A, X.T @ Y).astype(np.float32)


def capture(session, prompts, tokens: int = 90, verbose: bool = True) -> tuple:
    """(hidden state entering each layer's MoE, experts each layer chose), decode steps only.

    Prefill is excluded deliberately: it routes many tokens at once through _chunks and the
    hidden states do not line up one-to-one with the selections.
    """
    import mlx.core as mx
    mods = session.handle.mods
    n_layers = len(mods)
    xs = [[] for _ in range(n_layers)]
    es = [[] for _ in range(n_layers)]
    by_id = {id(m): i for i, m in enumerate(mods)}
    cls = type(mods[0])
    orig = cls.__call__

    def hook(self, x, indices):
        i = by_id.get(id(self))
        if i is not None:
            gi = np.array(indices)
            gi = gi.reshape(-1, gi.shape[-1])
            if gi.shape[0] == 1:                     # one row == one decode step
                xf = x.reshape(-1, x.shape[-1])
                xs[i].append(np.array(xf.astype(mx.float32))[0])
                es[i].append(gi[0])
        return orig(self, x, indices)

    cls.__call__ = hook
    try:
        for j, p in enumerate(prompts):
            for _t, _i in session.stream_text([{"role": "user", "content": p}],
                                              max_tokens=tokens, think=False):
                pass
            if verbose:
                print(f"    prompt {j + 1}/{len(prompts)}: "
                      f"{min(len(c) for c in xs)} decode steps captured", flush=True)
    finally:
        cls.__call__ = orig
    n = min(len(c) for c in xs) if xs else 0
    return [np.stack(c[:n]).astype(np.float32) for c in xs], [c[:n] for c in es], n


def train(session, prompts, tokens: int = 90, holdout: float = 0.3,
          verbose: bool = True) -> dict:
    """Fit one predictor per layer, mapping this layer's hidden state to the NEXT layer's experts.

    The last layer gets a zero matrix: there is no next layer to prefetch for, and a predictor
    that is never used should be obviously never used rather than quietly wrong.
    """
    xs, es, n = capture(session, prompts, tokens=tokens, verbose=verbose)
    if n < 40:
        raise ValueError(f"only {n} decode steps captured; need at least 40 to fit and hold out")
    n_layers = len(xs)
    E = int(session.plan["n_experts"])
    cut = int(n * (1 - holdout))
    Ws, rec8, rec16, rec32 = [], [], [], []
    for l in range(n_layers):
        if l + 1 >= n_layers:
            Ws.append(np.zeros((xs[l].shape[1], E), dtype=np.float32))
            continue
        Y = np.zeros((n, E), dtype=np.float32)
        for r, ids in enumerate(es[l + 1]):
            Y[r, np.asarray(ids, dtype=np.int64)] = 1.0
        W = fit(xs[l][:cut], Y[:cut])
        Ws.append(W)
        P = xs[l][cut:] @ W
        truth = [ids for ids in es[l + 1][cut:]]
        rec8.append(recall_at(P, truth, 8))
        rec16.append(recall_at(P, truth, 16))
        rec32.append(recall_at(P, truth, 32))
    meta = {"model": session.name, "n_layers": n_layers, "n_experts": E,
            "hidden": int(xs[0].shape[1]), "steps": n, "holdout_steps": n - cut,
            "ridge": RIDGE, "recall_at_8": float(np.mean(rec8)),
            "recall_at_16": float(np.mean(rec16)), "recall_at_32": float(np.mean(rec32)),
            "trained_at": int(time.time())}
    return {"W": Ws, "meta": meta}
