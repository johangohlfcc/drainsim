"""Cascade rack: three stepped cups on one rigid backplate (presentation case).

It shows spill routing: water that leaves one cavity goes on to the next one.

- **A**, the top cup: no hole.
- **B**, the middle cup: a 20 mm hole in its bottom, above C.
- **C**, the bottom tray: a 40 mm hole to the bath.

Sequence:

1. Dip upright and lift: all cups are full.
2. Hold at 0°: B and C drain through their holes, and A stays full.
3. Tilt slowly to 40° (lowering +x): A pours into B, B drains and
   overflows into C, and C drains into the bath.
4. Hold.

The same motion is run with the previous method (equilibrium, escaped
liquid is lost) for comparison.

    python examples/cascade_rack.py --dx 0.01            # curves + PNG only
    python examples/cascade_rack.py --dx 0.006 --movie out.mp4

The geometry is generated here (zero-thickness sheets, holes cut in the
mesh). No customer data is involved.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from drainsim.grid import Grid
from drainsim.model import Simulation
from drainsim.motion import Keyframes
from drainsim.physics import ThroatModel

# cups: name -> (x0, x1, z_bottom, height, hole x or None, hole diameter);
# y is -D..D
D_Y = 0.08
CUPS = {"A": (-0.30, -0.10, 0.62, 0.10, None, None),
        "B": (-0.14, 0.14, 0.36, 0.10, 0.09, 0.020),
        "C": (0.02, 0.46, 0.08, 0.08, 0.40, 0.040)}
PLATE = (-0.34, 0.50, 0.0, 0.80)           # backplate x0, x1, z0, z1 at y = -D_Y


# ----------------------------------------------------------------- geometry
def _quad(p0, p1, p2, p3):
    return [p0, p1, p2, p3], [[0, 1, 2], [0, 2, 3]]


def _plate_with_hole(x0, x1, y0, y1, z, hx, hy, r, n=48):
    """Zero-thickness rectangle z = const with a circular hole (hx, hy, r)."""
    corners = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
    ang_c = np.arctan2(corners[:, 1] - hy, corners[:, 0] - hx)
    ang = np.sort(np.concatenate([np.linspace(-np.pi, np.pi, n, endpoint=False), ang_c]))
    d = np.column_stack([np.cos(ang), np.sin(ang)])
    # ray from the hole centre to the rectangle boundary
    with np.errstate(divide="ignore", invalid="ignore"):
        tx = np.where(d[:, 0] > 0, (x1 - hx) / d[:, 0], np.where(d[:, 0] < 0, (x0 - hx) / d[:, 0], np.inf))
        ty = np.where(d[:, 1] > 0, (y1 - hy) / d[:, 1], np.where(d[:, 1] < 0, (y0 - hy) / d[:, 1], np.inf))
    t = np.minimum(tx, ty)
    outer = np.column_stack([hx + t * d[:, 0], hy + t * d[:, 1]])
    inner = np.column_stack([hx + r * d[:, 0], hy + r * d[:, 1]])
    m = len(ang)
    V = np.vstack([np.column_stack([inner, np.full(m, z)]),
                   np.column_stack([outer, np.full(m, z)])])
    F = []
    for k in range(m):
        a, b = k, (k + 1) % m
        F += [[a, m + a, m + b], [a, m + b, b]]
    return V, np.array(F)


def rack_mesh():
    import trimesh
    parts = []
    x0, x1, z0, z1 = PLATE
    v, f = _quad([x0, -D_Y, z0], [x1, -D_Y, z0], [x1, -D_Y, z1], [x0, -D_Y, z1])
    parts.append(trimesh.Trimesh(v, f, process=False))
    for name, (a, b, zb, h, hx, hd) in CUPS.items():
        zt = zb + h
        if hx is None:
            v, f = _quad([a, -D_Y, zb], [b, -D_Y, zb], [b, D_Y, zb], [a, D_Y, zb])
        else:
            v, f = _plate_with_hole(a, b, -D_Y, D_Y, zb, hx, 0.0, 0.5 * hd)
        parts.append(trimesh.Trimesh(v, f, process=False))
        for q in ([[a, -D_Y, zb], [b, -D_Y, zb], [b, -D_Y, zt], [a, -D_Y, zt]],   # back
                  [[a, D_Y, zb], [b, D_Y, zb], [b, D_Y, zt], [a, D_Y, zt]],      # front
                  [[a, -D_Y, zb], [a, D_Y, zb], [a, D_Y, zt], [a, -D_Y, zt]],    # -x
                  [[b, -D_Y, zb], [b, D_Y, zb], [b, D_Y, zt], [b, -D_Y, zt]]):   # +x
            v, f = _quad(*q)
            parts.append(trimesh.Trimesh(v, f, process=False))
    return trimesh.util.concatenate(parts)


def rack_holes():
    return [dict(center=(hx, 0.0, zb), diameter=hd, axis=(0.0, 0.0, 1.0))
            for (a, b, zb, h, hx, hd) in CUPS.values() if hx is not None]


# ------------------------------------------------------------------- motion
def rack_motion(tilt=40.0, above=0.30, t_dip=4.0, t_hold0=38.0, t_tilt=4.0, t_end=70.0):
    """Dip upright (whole rack under), hold at 0°, tilt about the plate centre."""
    travel = above + PLATE[3] + 0.05
    z0 = above - PLATE[2]
    T = [0.0, t_dip, 2 * t_dip, t_hold0, t_hold0 + t_tilt, t_end]
    ang = [0, 0, 0, 0, tilt, tilt]
    z = [z0, z0 - travel, z0, z0, z0, z0]
    pivot = np.array([0.5 * (PLATE[0] + PLATE[1]), 0.0, 0.5 * (PLATE[2] + PLATE[3])])
    return Keyframes(T, np.array([[0, a, 0] for a in ang], float),
                     np.array([[0, 0, q] for q in z], float), ndim=3, pivot=pivot)


def cup_masks(sim):
    """Fluid nodes inside each cup (object frame), by node position."""
    X = sim.X
    grid = sim.grid
    out = {}
    for name, (a, b, zb, h, hx, hd) in CUPS.items():
        out[name] = ((X[:, 0] > a) & (X[:, 0] < b) & (np.abs(X[:, 1]) < D_Y)
                     & (X[:, 2] > zb) & (X[:, 2] < zb + h + 0.5 * grid.dx)
                     & sim.fl)
    return out


def setup(dx, tilt=40.0, cd=0.65, **mkw):
    mesh = rack_mesh()
    grid = Grid.from_mesh(mesh, dx=dx, pad=3 * dx)
    mo = rack_motion(tilt, **mkw)
    new = Simulation(grid.copy(), mo, dt_max=0.05, holes=rack_holes(),
                     throat_model=ThroatModel(Cd=cd))
    old = Simulation(grid.copy(), mo, dt_max=0.05, split=False, spill_routing=False,
                     holes=None, subcell_connect=False)
    return mesh, grid, mo, new, old


def cup_volumes(sim, masks):
    return {k: float((sim.L[m] * sim.v[m]).sum()) for k, m in masks.items()}


def run_curves(dx, out, tilt=40.0, dt_rec=0.25, **mkw):
    t0 = time.time()
    mesh, grid, mo, new, old = setup(dx, tilt, **mkw)
    masks = {"new": cup_masks(new), "old": cup_masks(old)}
    print(f"grid {grid.shape} ({grid.ncells/1e3:.0f} k cells), throats "
          f"{[(round(t.diameter*1e3,1), t.a, t.b) for t in new.comp.throats]}", flush=True)
    rec = {"t": [], "new": {k: [] for k in CUPS}, "old": {k: [] for k in CUPS}}
    for t in np.arange(0.0, mo.t_end + 1e-9, dt_rec):
        for s in (new, old):
            if t > s.t + 1e-9:
                s.run(t_end=float(t))
        rec["t"].append(float(t))
        for key, s in (("new", new), ("old", old)):
            for k, v in cup_volumes(s, masks[key]).items():
                rec[key][k].append(v * 1e3)
    os.makedirs(out, exist_ok=True)
    json.dump(rec, open(os.path.join(out, "cascade_curves.json"), "w"))
    print(f"done in {time.time()-t0:.0f} s", flush=True)
    return rec, mo


def plot_curves(rec, mo, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = {"A": "#2a78d6", "B": "#eb6834", "C": "#1baf7a"}
    fig, ax = plt.subplots(figsize=(9, 4.6))
    T = mo.times
    for a, b, lab in ((0, T[2], "dip"), (T[3], T[4], "tilt to 40°")):
        ax.axvspan(a, b, color="#f1f0ec", lw=0, zorder=0)
        ax.text(0.5 * (a + b), 1.0, lab, transform=ax.get_xaxis_transform(), ha="center",
                va="bottom", fontsize=9, color="#52514e")
    t = rec["t"]
    for k, c in cols.items():
        ax.plot(t, rec["new"][k], color=c, lw=2.2, label=f"cup {k}")
        ax.plot(t, rec["old"][k], color=c, lw=1.4, ls=(0, (4, 3)))
    ax.plot([], [], color="#52514e", lw=2.2, label="new model")
    ax.plot([], [], color="#52514e", lw=1.4, ls=(0, (4, 3)), label="previous method")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("water in the cup (l)")
    ax.set_xlim(0, t[-1])
    ax.set_ylim(bottom=0)
    ax.grid(color="#e6e5e0", lw=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, fontsize=9, ncol=2, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    print("wrote", path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dx", type=float, default=0.01)
    ap.add_argument("--tilt", type=float, default=40.0)
    ap.add_argument("--out", default="examples/out/cascade_rack")
    a = ap.parse_args()
    rec, mo = run_curves(a.dx, a.out, a.tilt)
    plot_curves(rec, mo, os.path.join(a.out, "cascade_curves.png"))
