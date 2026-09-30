"""Which way does the car turn in the dip? Contact sheet to compare with the
article's Fig. 9, before running the model.

    python examples/car_orientation.py --stl CAR.stl --out car_orientation.png

Rows: the four ways the body can turn 360 deg over the dip (pitch about the
transverse axis, nose down first / tail down first; roll about the long axis,
left side / right side first). Columns: the "short" motion at 0-35 s, seen
from the side (the car travels to the right, the bath is blue). The roof is
red and the nose green. The same is printed as text for every row and time.
Use the row that matches Fig. 9 as ``--rotation R --sense S`` (and
``--reverse`` if the car should start with the nose pointing backwards) in
car_movie.py.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from car_article import add_car_args, car_motion, load_car        # noqa: E402

VARIANTS = (("pitch", 1, "pitch, nose down first"),
            ("pitch", -1, "pitch, tail down first"),
            ("roll", 1, "roll, left side down first"),
            ("roll", -1, "roll, right side down first"))
TIMES = (0, 5, 10, 15, 20, 25, 30, 35)


def word(d):
    k = int(np.argmax(np.abs(d)))
    s = d[k] > 0
    return [("forward (travel)", "backward"), ("sideways", "sideways"),
            ("up", "down")][k][0 if s else 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stl", required=True)
    ap.add_argument("--out", default="car_orientation.png")
    ap.add_argument("--rock", type=float, default=5.0)
    add_car_args(ap)
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    mesh, info = load_car(a.stl, a.orient, a.reverse)
    V = np.asarray(mesh.vertices)
    rng = np.random.default_rng(0)
    V = V[rng.choice(len(V), min(40000, len(V)), replace=False)]
    roof = V[:, 2] > np.percentile(V[:, 2], 88)
    nose = V[:, 0] > np.percentile(V[:, 0], 88)
    col = np.where(roof, 0, np.where(nose, 1, 2))
    colors = np.array(["#d62728", "#2ca02c", "#9a9893"])
    fig, axs = plt.subplots(len(VARIANTS), len(TIMES), figsize=(2.3 * len(TIMES), 2.5 * len(VARIANTS)))
    for r, (rot, sg, name) in enumerate(VARIANTS):
        mo = car_motion(a.rock, 0.0, "short", rot, sg)
        print(f"--rotation {rot} --sense {sg:+d}{' --reverse' if a.reverse else ''}  ({name}):")
        for c, t in enumerate(TIMES):
            R, T = mo.pose(t)
            W = V @ R.T + T
            ax = axs[r, c]
            ax.axhspan(-5, 0.0, color="#8080ff", alpha=0.3, lw=0)
            for k in (2, 1, 0):
                m = col == k
                ax.plot(W[m, 0] - T[0], W[m, 2], ",", color=colors[k], alpha=0.6)
            ax.set_xlim(-3.2, 3.2)
            ax.set_ylim(-3.4, 4.2)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(f"t = {t} s", fontsize=10)
            if c == 0:
                ax.set_ylabel(f"{rot} {sg:+d}\n{name}", fontsize=8)
            txt = f"nose {word(R[:, 0])}, roof {word(R[:, 2])}"
            ax.text(0, -3.2, txt, ha="center", fontsize=6)
            print(f"   t = {t:2d} s: nose {word(R[:, 0]):17s} roof {word(R[:, 2]):17s} "
                  f"centre {'under' if T[2] < 0 else 'above'} the surface")
    fig.suptitle("Car turning in the dip (side view, travel to the right; roof red, nose green). "
                 "Compare with the article's Fig. 9.", fontsize=11)
    fig.tight_layout()
    fig.savefig(a.out, dpi=110)
    print("wrote", a.out, flush=True)


if __name__ == "__main__":
    main()
