"""Transient drainage of the car door through its three drain holes.

Experiment (FCC, water): the door hangs horizontally (0° tilt). The three
19 mm drain holes are plugged, the door is dipped and lifted, then the three
plugs are pulled at the same moment and the outflow is weighed
(``examples/data/door_drain_exp_0deg.csv``, four runs).

Model: the same dip with the holes plugged (``open_at``), a pause, then all
three holes open at ``t_open``. The water left in the door is the volume
inside at ``t_open`` minus the cumulative flow through the three holes, so it
is directly comparable with the scale.

    python examples/door_drain.py --stl door.stl --dx 0.006 --out runs/drain
    python examples/door_drain.py --stl door.stl --dx 0.005 --cd 0.6 0.7

Writes ``<out>/drain_results.json`` and ``<out>/door_drain.png``.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from door_article import DOOR_HOLES, door_setup, peak_rss_gb
from drainsim.model import Simulation
from drainsim.physics import ThroatModel

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.join(HERE, "data", "door_drain_exp_0deg.csv")


def load_experiment(path=EXP):
    runs = {}
    for line in open(path):
        if line.startswith("#") or line.startswith("run"):
            continue
        k, t, m, vd, vl = line.split(",")
        runs.setdefault(int(k), []).append((float(t), float(vd), float(vl)))
    return {k: np.array(v) for k, v in runs.items()}


def fractions(t, V, fr=(0.5, 0.9, 0.99)):
    """Times at which the fractions ``fr`` of the initial volume have left."""
    V0, Vend = V[0], V[-1]
    out = []
    for f in fr:
        target = V0 - f * (V0 - Vend)
        i = np.flatnonzero(V <= target)
        out.append(float(t[i[0]]) if i.size else np.nan)
    return out


def run_drain(stl, dx, tilt=0.0, cd=0.6, subcells=4, t_settle=5.0, t_after=40.0,
              dt_max=0.05, verbose=True, progress=False, hang=5.0, levels=None):
    t0 = time.time()
    t_open = 8.0 + t_settle                     # 4 s down, 4 s up, then pause
    mesh, grid, motion = door_setup(stl, dx, tilt, t_drain=t_settle + t_after, hang=hang,
                                    levels=levels)
    holes = [dict(h, open_at=t_open) for h in DOOR_HOLES]
    sim = Simulation(grid, motion, dt_max=dt_max, holes=holes, subcells=subcells,
                     throat_model=ThroatModel(Cd=cd))
    hole_ids = [i for i, t in enumerate(sim.comp.throats) if t.axis is not None]
    if verbose:
        print(f"dx={dx*1e3:.1f} mm tilt={tilt} hang={hang} Cd={cd} k={subcells}: grid {grid.shape} "
              f"({grid.ncells/1e6:.2f} M cells), {len(hole_ids)} explicit holes "
              f"(open at t = {t_open:g} s), {len(sim.comp.throats)} throats, "
              f"setup {time.time()-t0:.0f} s", flush=True)
    sim.run(t_end=t_open, progress=progress)
    V_open = sim.hist.liquid_retained[-1]
    n0 = len(sim.hist.t)
    sim.run(t_end=motion.t_end, progress=progress)
    H = sim.hist.arrays()
    t = H["t"][n0 - 1:] - t_open
    Q = np.abs(np.asarray(H["throat_flow"])[n0 - 1:][:, hole_ids]).sum(1)
    Q[0] = 0.0
    out = np.concatenate([[0.0], np.cumsum(Q[1:] * np.diff(t))])
    V = V_open - out
    t50, t90, t99 = fractions(t, V)
    res = dict(dx=dx, tilt=tilt, hang=hang, cd=cd, subcells=subcells, dt_max=dt_max,
               cells=int(grid.ncells), V_open_l=V_open * 1e3,
               V_end_l=float(V[-1] * 1e3), out_l=float(out[-1] * 1e3),
               retained_total_l=float(H["liquid_retained"][-1] * 1e3),
               t50=t50, t90=t90, t99=t99,
               rate_initial_lps=float(np.interp(2.0, t, Q) * 1e3),
               wall_s=time.time() - t0, peak_rss_gb=peak_rss_gb(),
               t=t.tolist(), V_left_l=(V * 1e3).tolist(), Q_lps=(Q * 1e3).tolist())
    if verbose:
        print(f"    V at opening {res['V_open_l']:.3f} l, left after {t[-1]:.0f} s "
              f"{res['V_end_l']*1e3:.0f} ml; 50/90/99 % out at {t50:.1f} / {t90:.1f} / "
              f"{t99:.1f} s; rate at 2 s {res['rate_initial_lps']:.3f} l/s; "
              f"{res['wall_s']:.0f} s, peak RSS {res['peak_rss_gb'] or 0:.1f} GB",
              flush=True)
    return sim, res


def figure(results, path, exp=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.signal import savgol_filter
    exp = load_experiment() if exp is None else exp
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    for k, e in exp.items():
        lab = "experiment (4 runs)" if k == 1 else None
        ax[0].plot(e[:, 0], e[:, 2], color="#8a8985", lw=1.2, label=lab)
        Q = -savgol_filter(e[:, 1], 11, 2, deriv=1, delta=0.1)
        m = (e[:, 0] > 0.8) & (e[:, 1] > 0.05)
        ax[1].plot(e[m, 1], Q[m], color="#8a8985", lw=1.2, label=lab)
    cols = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
    for r, c in zip(results, cols):
        lab = f"model {r['dx']*1e3:g} mm, Cd = {r['cd']:g}, " + (
            f"k = {r['subcells']}" if r["subcells"] >= 2 else "binary cells")
        t, V, Q = np.array(r["t"]), np.array(r["V_left_l"]), np.array(r["Q_lps"])
        ls = "-" if r["subcells"] >= 2 else (0, (4, 2))
        ax[0].plot(t, V, color=c, lw=2, ls=ls, label=lab)
        Vd = V - V[-1]
        m = (t > 0.05) & (Vd > 0.05)
        ax[1].plot(Vd[m], Q[m], color=c, lw=2, ls=ls, label=lab)
    for a in ax:
        a.grid(color="#e6e5e0", lw=0.8)
        a.set_axisbelow(True)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    ax[0].set_xlim(-1, 30)
    ax[0].set_ylim(0, 12)
    ax[0].set_xlabel("time since the holes were opened (s)")
    ax[0].set_ylabel("water left in the door (l)")
    ax[0].set_title("Horizontal door, three 19 mm holes opened at t = 0",
                    fontsize=11, loc="left")
    ax[0].legend(frameon=False, fontsize=8)
    ax[1].set_xlim(11.5, 0)
    ax[1].set_ylim(0, 1.2)
    ax[1].set_xlabel("drainable water still in the door (l)")
    ax[1].set_ylabel("outflow rate (l/s)")
    ax[1].set_title("Outflow rate vs remaining water", fontsize=11, loc="left")
    ax[1].legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    print("wrote", path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stl", required=True)
    ap.add_argument("--dx", type=float, nargs="+", default=[0.006])
    ap.add_argument("--cd", type=float, nargs="+", default=[0.6])
    ap.add_argument("--subcells", type=int, default=4)
    ap.add_argument("--tilt", type=float, default=0.0)
    ap.add_argument("--dt-max", type=float, default=0.05)
    ap.add_argument("--t-after", type=float, default=40.0,
                    help="seconds simulated after the holes open")
    ap.add_argument("--progress", action="store_true")
    ap.add_argument("--hang", type=float, default=5.0,
                    help="lean of the door on its hooks, degrees (see door_setup)")
    ap.add_argument("--levels", type=int, default=None,
                    help="octree with this many coarser levels (dx at the walls)")
    ap.add_argument("--out", default="examples/out/door_drain")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    results = []
    for dx in args.dx:
        for cd in args.cd:
            _, r = run_drain(args.stl, dx, args.tilt, cd, args.subcells,
                             t_after=args.t_after, dt_max=args.dt_max,
                             progress=args.progress, hang=args.hang, levels=args.levels)
            results.append(r)
            with open(os.path.join(args.out, "drain_results.json"), "w") as f:
                json.dump(results, f)
    figure(results, os.path.join(args.out, "door_drain.png"))
