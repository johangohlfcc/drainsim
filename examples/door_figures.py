"""Fig. 7 / Fig. 8 replicas for the car-door drainage case.

    python examples/door_figures.py --stl door.stl --runs examples/out/door \\
           --fig8-dx 0.004

Fig. 7: water left after drainage vs tilt, for every resolution found in
``<runs>/*/results.json``, with the experimental means and the article's
simulation values. Fig. 8: rendering of the trapped water after drainage
(world frame, seen from the cabin side), from a dedicated run at
``--fig8-dx``.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from door_article import EXPERIMENT, run_case

# Article: text values (7 refinements, 0.35 mm) and experiment; the
# 3-refinement (5.5 mm) values are read approximately off Fig. 7.
ARTICLE = {"7 ref. (0.35 mm), article": {22.5: 0.039, 45.0: 0.226},
           "3 ref. (5.5 mm), article (approx.)": {22.5: 0.020, 45.0: 0.145}}
EXP_SIGMA = {22.5: 0.005, 45.0: 0.015}          # approx. from the error bars


def load(runs):
    res = []
    for f in glob.glob(os.path.join(runs, "*", "results.json")):
        res += json.load(open(f))
    return [r for r in res if r.get("split", True) and r.get("spill_routing", True)]


def fig7(res, out):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for r in res:
        r.setdefault("subcells", 0)                    # v1/v2 results: binary
    dxs = sorted({r["dx"] for r in res}, reverse=True)
    keys = sorted({(r["dx"], r["subcells"]) for r in res}, key=lambda x: (-x[0], x[1]))
    cmap = plt.get_cmap("viridis")
    for dx, k in keys:
        pts = sorted([(r["tilt"], r["retained_l"]) for r in res
                      if r["dx"] == dx and r["subcells"] == k])
        if pts:
            t, v = zip(*pts)
            c = cmap(dxs.index(dx) / max(1, len(dxs) - 1))
            vf = k >= 2
            ax[0].plot(np.array(t) + (0.6 if vf else -0.6), v, "o", color=c, ms=7,
                       mfc=c if vf else "none",
                       label=f"this model, {dx*1e3:g} mm"
                             + (f", k={k}" if vf else ", binary"))
    for (lab, vals), mk in zip(ARTICLE.items(), ["D", "s"]):
        ax[0].plot(list(vals), list(vals.values()), mk, mfc="none", mec="#d62728",
                   ms=8, label=lab)
    ax[0].errorbar(list(EXPERIMENT), list(EXPERIMENT.values()),
                   yerr=[EXP_SIGMA[t] for t in EXPERIMENT], fmt="o", mfc="white",
                   mec="k", ecolor="k", capsize=6, ms=9, label="experiment (±1σ approx.)")
    ax[0].set_xticks([22.5, 45.0])
    ax[0].set_xlim(10, 57)
    ax[0].set_ylim(0, 0.3)
    ax[0].set_xlabel("degrees tilt (°)")
    ax[0].set_ylabel("water trapped after drainage (l)")
    ax[0].set_title("Fig. 7 replica", fontsize=10)
    ax[0].legend(fontsize=7, loc="upper left")
    # transients at the finest resolution
    fin = min(dxs)
    kfin = max(r["subcells"] for r in res if r["dx"] == fin)
    for r in res:
        if r["dx"] == fin and r["subcells"] == kfin:
            ax[1].plot(r["t"], r["liquid_l"],
                       label=f"{r['tilt']:g}°, {fin*1e3:g} mm, k={kfin}")
    for tilt, v in EXPERIMENT.items():
        ax[1].axhline(v, ls=":", color="k", lw=0.8)
        ax[1].text(ax[1].get_xlim()[1] if False else 30, v, f" exp. {tilt:g}°",
                   fontsize=7, va="bottom")
    ax[1].axvspan(0, 8, color="#eeeeee", zorder=0)
    ax[1].set_xlabel("time (s)  (grey: dip down and up)")
    ax[1].set_ylabel("water inside the door, not bath (l)")
    ax[1].set_title("transient: filling through the holes and drainage", fontsize=10)
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig7_replica.png"), dpi=140)
    print("wrote", os.path.join(out, "fig7_replica.png"))


def fig8(stl, dx, out, t_drain="auto"):
    from drainsim.worldviz import export_world
    from drainsim.render3d import render_animation
    paths = []
    for tilt in (22.5, 45.0):
        sim, H, res = run_case(stl, dx, tilt, t_drain, snapshot_every=None)
        vt = os.path.join(out, f"fig8_vtk_{tilt:g}")
        export_world(sim, H, vt, prefix="door", object_mesh=stl, volume=False,
                     times=[max(H.snapshots)])
        png = os.path.join(out, f"fig8_{tilt:g}.png")
        render_animation(vt, "door", png, size=(1100, 800), azimuth=180,
                         elevation=5, zoom=1.15, object_opacity=0.35,
                         show_bath=False, fit="object", show_air=False)
        paths.append((tilt, png, res["retained_l"]))
    import matplotlib.image as mpimg
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    for a, (tilt, png, v) in zip(ax, paths):
        a.imshow(mpimg.imread(png))
        a.set_axis_off()
        a.set_title(f"({'a' if tilt < 30 else 'b'}) {tilt:g}°: {v*1e3:.0f} ml trapped "
                    f"(exp. {EXPERIMENT[tilt]*1e3:.0f} ml), {dx*1e3:g} mm", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig8_replica.png"), dpi=130)
    print("wrote", os.path.join(out, "fig8_replica.png"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stl", required=True)
    ap.add_argument("--runs", default="examples/out/door")
    ap.add_argument("--fig8-dx", type=float, default=0.004)
    ap.add_argument("--no-fig8", action="store_true")
    args = ap.parse_args()
    res = load(args.runs)
    if res:
        fig7(res, args.runs)
    else:
        print("no results.json found under", args.runs, "- skipping Fig. 7")
    if not args.no_fig8:
        fig8(args.stl, args.fig8_dx, args.runs)
