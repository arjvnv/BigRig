"""BigRig -- run large Mixture-of-Experts models on a Mac that does not have room for them."""
import os

__all__ = ["home"]


def home() -> str:
    """Where models, packed blobs and measurements live.

    THREE CASES, IN THIS ORDER, AND THE THIRD IS THE ONE THAT MATTERS FOR ANYONE WHO INSTALLS.
        BIGRIG_HOME wins outright, because someone with a fast external disk should be able to
        put 60 GB of weights on it.

        Otherwise, if this package sits next to a pyproject.toml, it is a source checkout being
        run in place and the checkout is the home -- which keeps a developer's models and every
        measurement they have taken exactly where they left them.

        Otherwise it is an installed package, and the home is ~/.bigrig. It must NOT be derived
        from the package's own location: `pip install bigrig` puts the package in
        site-packages, so deriving from __file__ would download 60 GB of model weights INTO
        site-packages, where reinstalling the package or deleting the virtualenv destroys them.
        Verified before publishing rather than after -- a clean install resolved home() to
        `.../vtest/lib/python3.12/site-packages`.

    The paths used to be spelled `~/tollgate/data/...` by hand in eight modules, which meant the
    product could not be renamed, moved or installed anywhere without stranding every
    measurement it had ever taken. There is one of these and everything derives from it.
    """
    env = os.environ.get("BIGRIG_HOME")
    if env:
        return os.path.expanduser(env)
    pkg = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(pkg)
    if os.path.exists(os.path.join(root, "pyproject.toml")):
        return root                                   # a source checkout, run in place
    return os.path.expanduser("~/.bigrig")


# HOW MUCH WORK METAL BATCHES INTO ONE COMMAND BUFFER, AND WHY 8.
#     Every streamed layer reads its router's output back to the host to decide what to fetch,
#     and that read waits for whatever is queued. The larger the command buffer, the more it
#     waits for. MLX's default batches far more than a layer's worth, so each of the 48 reads a
#     token drains work belonging to layers that had not been asked about yet.
#
#     Measured on Qwen3-30B-A3B-3bit, 14 samples a value, each value run twice in interleaved
#     order so neither gets the warm machine:
#
#         ops/buffer     median tok/s     spread     vs default
#           default          12.96          2.44        1.00x
#                 1          13.73          0.65        1.06x
#                 2          16.19          0.68        1.25x
#                 4          18.43          1.26        1.42x
#                 8          20.18          0.82        1.56x   <-
#                10          19.31          1.20        1.49x
#                16          19.33          1.34        1.49x
#                24          17.47          2.24        1.35x
#
#     Unimodal, and both rounds agree at every value. Replies are byte-identical at all of them,
#     which they must be: this schedules work, it does not change it. Time to first token is
#     slightly better too, and the run-to-run spread falls from 2.44 to 0.82.
#
#     `setdefault`, so anyone who sets it themselves keeps their value. Set it to MLX's default
#     behaviour by exporting a large number if you want the old scheduling back.
os.environ.setdefault("MLX_MAX_OPS_PER_BUFFER", "8")
