"""Geometric check of the cut-cell volumes on the car door.

Volume of the corner pocket below the lowest rim point of the lowest drain
hole (the water the 45 deg case can hold at most), for binary cells (k=0)
and side-aware cut-cell volumes (k=2, 4, 8). No time stepping: this only
tests the volume-level relation, so it is fast and should converge with dx.

    python examples/door_pocket.py --stl door.stl --tilt 45 \\
           --dx 0.006 0.005 0.004 0.003 --k 0 2 4 8 --out runs_v3/pocket

The partial top layer is counted with a linear fraction of each cell's
volume, so the result is not quantised to whole cell layers.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
from scipy import ndimage

from door_article import DOOR_HOLES, door_setup, peak_rss_gb
from drainsim import compartments as cp
from drainsim.volfrac import effective_volumes


def pocket(stl, dx, tilt, ks=(0, 2, 4, 8)):
    t0 = time.time()
    mesh, g, motion = door_setup(stl, dx, tilt)
    cp.carve_holes(g, DOOR_HOLES)
    up = np.asarray(motion.frame(motion.t_end)[0], float)
    rims = []
    for H in DOOR_HOLES:
        a = np.asarray(H["axis"], float)
        a /= np.linalg.norm(a)
        upp = up - (up @ a) * a
        rims.append(np.asarray(H["center"]) @ up
                    - 0.5 * H["diameter"] * np.linalg.norm(upp))
    hp = min(rims)
    h = g.centers() @ up
    e = 0.5 * dx * np.abs(up).sum()
    fl = g.fluid.ravel()
    lab, n = ndimage.label((fl & (h - e < hp)).reshape(g.shape))
    lb = lab.ravel()
    size = np.bincount(lb, minlength=n + 1).astype(float)
    size[np.unique(lb[g.boundary_mask().ravel()])] = 0     # open to outside
    size[0] = 0
    cells = lb == int(np.argmax(size))
    frac = np.clip((hp - (h[cells] - e)) / (2 * e), 0, 1)
    res = dict(dx=dx, tilt=tilt, cells=int(g.ncells), pocket_cells=int(cells.sum()),
               rim_height=float(hp), setup_s=time.time() - t0)
    for k in ks:
        t = time.time()
        v, st = effective_volumes(g, k, return_stats=True)
        res[f"k{k}_l"] = float((v[cells] * frac).sum() * 1e3)
        res[f"k{k}_s"] = time.time() - t
        res[f"k{k}_dropped_ml"] = float(st.get("dropped", 0.0) * 1e6)
    res["peak_rss_gb"] = peak_rss_gb()
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stl", required=True)
    ap.add_argument("--tilt", type=float, default=45.0)
    ap.add_argument("--dx", type=float, nargs="+", default=[0.006, 0.005, 0.004])
    ap.add_argument("--k", type=int, nargs="+", default=[0, 2, 4, 8])
    ap.add_argument("--out", default="examples/out/door_pocket")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    out = []
    for dx in args.dx:
        r = pocket(args.stl, dx, args.tilt, args.k)
        out.append(r)
        print(f"dx={dx*1e3:.2f} mm: " + "  ".join(
            f"k={k}: {r[f'k{k}_l']:.4f} l ({r[f'k{k}_s']:.0f} s)" for k in args.k),
            f"  pocket {r['pocket_cells']} cells, peak RSS {r['peak_rss_gb'] or 0:.1f} GB",
            flush=True)
        with open(os.path.join(args.out, f"pocket_{args.tilt:g}.json"), "w") as f:
            json.dump(out, f, indent=1)
