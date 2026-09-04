"""Never change a user's model without them knowing.

THE RULE
    Three ways to run a model that does not fit, and exactly one of them alters the weights:

        native     it fits. Nothing happens to it.
        stream     bit-identical, slower. Nothing happens to it either.
        compress   THE WEIGHTS CHANGE. Output is no longer what they downloaded.

    Only `compress` needs consent, and it needs it every time it is a NEW decision -- not every
    run, which is nagging, and not never, which is what the first implementation did.

WHAT WAS WRONG BEFORE
    The engine picked `compress` on its own and said "compressing to 3-bit" the first time. The
    compressed copy was then cached, so every later run served a quantised model in silence, and
    the per-run summary said "100% of experts kept in RAM" -- true, and misleading, because it
    never mentioned those experts were no longer the ones downloaded.

THE THREE PLACES A CHOICE CAN COME FROM, in priority order
    1. an explicit flag         --compress / --exact. Always wins; scripts must be able to pin it.
    2. a remembered choice      made once, stored beside the blob, re-checked for still being valid.
    3. an interactive prompt    when a terminal is attached, or when the caller supplies its
                                own reader (a GUI, a web prompt, a test harness).

    With none of those and no terminal, this REFUSES. A script that silently degrades a model
    is the failure this module exists to prevent, so the non-interactive path errors out with
    the two flags spelled out rather than guessing.
"""
from __future__ import annotations

import json
import os
import sys

CHOICE_VERSION = 1


class ConsentRequired(Exception):
    """Raised when a weight-altering decision is needed and nobody can be asked."""


def choice_path(blob_path: str) -> str:
    return os.path.expanduser(blob_path) + ".choice.json"


def load_choice(blob_path: str) -> dict | None:
    """The remembered decision, or None. A damaged file is treated as no decision.

    Never raises: a corrupt cache must degrade to asking again, never to a crash and never to
    proceeding as though consent had been given.
    """
    p = choice_path(blob_path)
    if not os.path.exists(p):
        return None
    try:
        with open(p) as f:
            c = json.load(f)
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return None
    if not isinstance(c, dict) or c.get("version") != CHOICE_VERSION:
        return None
    if c.get("mode") not in ("compress", "exact"):
        return None
    if c["mode"] == "compress" and not (
            isinstance(c.get("bits"), int) and isinstance(c.get("group_size"), int)):
        return None
    return c


def save_choice(blob_path: str, mode: str, bits: int = 0, group_size: int = 0,
                source_bits: int = 0) -> dict:
    if mode not in ("compress", "exact"):
        raise ValueError(f"mode must be 'compress' or 'exact', got {mode!r}")
    c = {"version": CHOICE_VERSION, "mode": mode}
    if mode == "compress":
        c.update({"bits": int(bits), "group_size": int(group_size),
                  "source_bits": int(source_bits)})
    p = choice_path(blob_path)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = p + ".tmp"
    with open(tmp, "w") as f:
        json.dump(c, f, indent=1)
    os.replace(tmp, p)          # atomic: a half-written choice file must never be readable
    return c


def forget_choice(blob_path: str) -> bool:
    p = choice_path(blob_path)
    if os.path.exists(p):
        os.remove(p)
        return True
    return False


def is_interactive() -> bool:
    """A terminal on BOTH ends. stdout alone is not enough -- output can be piped to a file
    while stdin is still a tty, and a prompt written to a log nobody reads is not consent."""
    try:
        return sys.stdin.isatty() and sys.stderr.isatty()
    except (ValueError, AttributeError):
        return False


def render_question(model_name: str, strategy: dict, source_bits: int) -> str:
    """The question, with only numbers we actually know.

    Sizes are exact -- they are arithmetic on the manifest. Speed and quality figures for THIS
    model are not known and are deliberately not invented; what is offered instead is the shape
    of the trade and a pointer to the one model where it was measured.
    """
    orig = strategy.get("original_gb", 0.0)
    small = strategy.get("expert_gb", 0.0)
    room = max(0.0, strategy.get("room_gb", 0.0))
    bits, grp = strategy.get("bits"), strategy.get("group_size")
    return (
        f"\n  {model_name} needs {orig:.1f} GB of expert weights. "
        f"You have {room:.1f} GB free.\n"
        f"  It cannot run as-is. Two ways forward:\n\n"
        f"    [1] Shrink it to fit     {small:.1f} GB at {bits}-bit "
        f"(it was {source_bits}-bit)\n"
        f"        Full speed, because every expert stays in RAM.\n"
        f"        THE WEIGHTS CHANGE -- output will differ from what you downloaded.\n"
        f"        The quality monitor stays on and flags it if generation degrades.\n\n"
        f"    [2] Keep it exact        streamed from disk, the weights untouched\n"
        f"        Slower: it reads experts it does not have room for as it goes.\n\n"
        f"  Measured on OLMoE-1B-7B, 3-bit cost ~18% higher perplexity and ran 4.4x faster\n"
        f"  than streaming. This model has not been measured; that is a reference, not a promise.\n"
    )


def ask(model_name: str, strategy: dict, source_bits: int, reader=None, writer=None) -> str:
    """Put the question to a human. Returns 'compress' or 'exact'.

    `reader`/`writer` are injectable so the whole flow is testable without a terminal.
    """
    reader = reader or input
    writer = writer or (lambda s: sys.stderr.write(s))
    writer(render_question(model_name, strategy, source_bits))
    for _ in range(3):
        try:
            a = (reader("  Which? [1/2] ") or "").strip().lower()
        except (EOFError, KeyboardInterrupt):
            writer("\n  No answer given; keeping your model exact.\n")
            return "exact"
        if a in ("1", "compress", "shrink", "s"):
            return "compress"
        if a in ("2", "exact", "e", ""):
            return "exact"
        writer("  Please answer 1 (shrink) or 2 (keep exact).\n")
    # Three unusable answers: choose the option that cannot harm them.
    writer("  No clear answer; keeping your model exact.\n")
    return "exact"


def resolve(strategy: dict, blob_path: str, source_bits: int, model_name: str,
            preference: str | None = None, interactive: bool = False,
            remember: bool = True, reader=None, writer=None) -> tuple:
    """Settle on a strategy the user has actually agreed to.

    Returns (strategy, decision) where `decision` records how it was reached, so the caller can
    say so on screen. Only ever downgrades `compress` to `stream` -- never the reverse, and never
    silently upgrades anything into altering weights.
    """
    mode = strategy["mode"]
    if mode != "compress":
        # Nothing here alters the model, so there is nothing to consent to.
        return strategy, {"source": "no-choice-needed", "mode": mode}

    def to_stream(why):
        s = dict(strategy)
        s["mode"] = "stream"
        s["declined_compression"] = True
        # Use the fallback capacity choose_strategy computed with the SAME memory accounting
        # it used to reject the model. Recomputing it elsewhere is how the two drifted apart.
        if "stream_capacity" in strategy:
            s["capacity"] = strategy["stream_capacity"]
            s["residency"] = strategy["stream_residency"]
        s["reason"] = (f"{why}; running exact instead, streaming what does not fit "
                       f"({strategy.get('original_gb', 0):.1f} GB model)")
        return s

    if preference == "exact":
        return to_stream("you asked for exact output"), {"source": "flag", "mode": "exact"}
    if preference == "compress":
        if remember:
            save_choice(blob_path, "compress", strategy["bits"], strategy["group_size"],
                        source_bits)
        return strategy, {"source": "flag", "mode": "compress"}

    saved = load_choice(blob_path)
    if saved:
        if saved["mode"] == "exact":
            return to_stream("you previously chose exact output"), {"source": "remembered",
                                                                    "mode": "exact"}
        # A remembered compress choice is only honoured if the precision it names still fits.
        # Free memory changes between runs; silently serving a different precision than the one
        # consented to is the same failure in a smaller costume.
        if (saved.get("bits"), saved.get("group_size")) == (strategy["bits"],
                                                            strategy["group_size"]):
            return strategy, {"source": "remembered", "mode": "compress"}
        # otherwise fall through and ask again about the new precision

    # A caller that supplies its own reader is asserting it can obtain an answer -- that is how
    # a GUI, a web prompt or a test harness participates without a terminal. Absent that, a real
    # terminal on both ends is required. `interactive=True` is a REQUEST to ask, never a claim
    # that asking will work.
    can_ask = reader is not None or (interactive and is_interactive())
    if not can_ask:
        raise ConsentRequired(
            f"{model_name} does not fit in {max(0.0, strategy.get('room_gb', 0)):.1f} GB and "
            f"running it means choosing:\n"
            f"    --compress   shrink it to {strategy['bits']}-bit "
            f"({strategy.get('expert_gb', 0):.1f} GB), full speed, THE WEIGHTS CHANGE\n"
            f"    --exact      keep it bit-identical, stream from disk, slower\n"
            f"  Refusing to choose for you: a script that quietly serves a degraded model is "
            f"worse than one that stops.")

    picked = ask(model_name, strategy, source_bits, reader=reader, writer=writer)
    if remember:
        if picked == "compress":
            save_choice(blob_path, "compress", strategy["bits"], strategy["group_size"],
                        source_bits)
        else:
            save_choice(blob_path, "exact")
    if picked == "exact":
        return to_stream("you chose exact output"), {"source": "asked", "mode": "exact"}
    return strategy, {"source": "asked", "mode": "compress"}
