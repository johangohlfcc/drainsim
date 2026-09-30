"""Rerun the article's door cases with drainsim and compare with its Table 1
and Figs. 5-8.

Cases, with the article's motion (0.15 m above the bath, 0.4 m down and up
in 8 s):

- **filling:** drain holes plugged. The result is the water trapped at the
  end of the motion.
- **drainage:** holes open. The result is the water left once drainage has
  settled (drainsim is transient); the value at the end of the motion is
  kept as well.

As in the article's post-processing, liquid bodies smaller than 0.01 l are
not counted.

    python examples/article_comparison.py --stl door.stl --dx 0.012 0.010 0.008 --out runs_article
    python examples/article_comparison.py --stl door.stl --figures --out runs_article

Each run appends one line to ``<out>/results.jsonl``, so several processes
(for example one per dx) can write to the same folder.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time

import numpy as np

from door_article import DOOR_HOLES, door_setup, peak_rss_gb, run_case
from drainsim.model import Simulation
from drainsim.physics import ThroatModel

MIN_BODY_L = 0.01                                     # article post-processing
# The untilted door's lowest point is 0.15 m above the bath, and the door is
# then tilted about its centre. With the lowest point of the *tilted* door at
# 0.15 m instead, the access opening never goes under at 22.5° and the
# filling case holds no water; the article and the experiment report 5.16 l.
PLACE_UNTILTED = True
# The door hangs on two hooks and leans towards -y (cabin side up); the
# article used about 5 degrees. It lifts the lower edge of the clip slot in
# the aperture lip relative to the pool (0° filling).
HANG = 5.0
LEVELS = None          # octree levels (--levels); None = uniform grid
NARROW = None          # narrow-passage refinement (--narrow K)

# ---- article data --------------------------------------------------------
# Table 1 (tilt 0°): refinements, min cell (mm), cells, time (h:mm on 16 cores)
ART_TABLE1 = [(2, 11.0, "0.18M", "0:00", 8.11, 8.11),
              (3, 5.5, "0.47M", "0:01", 9.86, 0.00),
              (4, 2.75, "1.60M", "0:03", 10.32, 0.00),
              (5, 1.375, "6.44M", "0:37", 10.56, 0.00),
              (6, 0.69, "28.2M", "5:19", 10.75, 0.00),
              (7, 0.35, "127.5M", "22:06", 10.81, 0.00)]
# Fig. 5 (filling) and Fig. 7 (drainage): refinement 3..7. 0° is from Table 1;
# the other values are read off the figures (about +-0.1 l for Fig. 5,
# +-0.003 l for Fig. 7). Refinement 7 at 22.5°/45° drainage is from the text.
ART_REF_DX = {3: 5.5, 4: 2.75, 5: 1.375, 6: 0.69, 7: 0.35}
ART_FILL = {0.0: {3: 9.86, 4: 10.32, 5: 10.56, 6: 10.75, 7: 10.81},
            22.5: {3: 4.08, 4: 4.60, 5: 4.78, 6: 4.95, 7: 5.15}}
ART_DRAIN = {22.5: {3: 0.017, 4: 0.022, 5: 0.027, 6: 0.033, 7: 0.039},
             45.0: {3: 0.145, 4: 0.201, 5: 0.208, 6: 0.222, 7: 0.226}}
# experiments: mean and +-1 sigma (sigma read off the error bars)
EXP_FILL = {0.0: (10.75, 0.14), 22.5: (5.16, 0.13)}
EXP_DRAIN = {0.0: (0.0, 0.0), 22.5: (0.056, 0.005), 45.0: (0.246, 0.016)}


def trapped(sim, min_l=MIN_BODY_L):
    """Liquid held by the door (not the bath) in bodies of at least min_l."""
    P = sim.pockets(min_volume=min_l * 1e-3)["liquid"]
    return float(sum(v for v, _ in P) * 1e3), [(float(v * 1e3), c.tolist()) for v, c in P]


PROGRESS_MIN = 0.0          # minutes between progress lines (0 = none), --progress


class Progress:
    """Prints a progress line every PROGRESS_MIN minutes of wall time:
    model time, steps, retained liquid, wall time, ETA and peak memory."""

    def __init__(self, tag, t_end, t0):
        self.tag, self.t_end, self.t0 = tag, float(t_end), t0
        self.t_run = time.time()
        self.last = self.t_run
        self.n = 0

    def __call__(self, sim):
        self.n += 1
        now = time.time()
        if PROGRESS_MIN <= 0 or now - self.last < 60.0 * PROGRESS_MIN:
            return
        self.last = now
        el = now - self.t_run
        frac = min(sim.t / self.t_end, 1.0) if self.t_end > 0 else 0.0
        eta = el / frac * (1 - frac) / 60 if frac > 0 else float("nan")
        print(f"[{self.tag}] t = {sim.t:6.2f}/{self.t_end:g} s, {self.n} steps, "
              f"liquid {sim.hist.liquid_retained[-1]*1e3:.3f} l, wall "
              f"{(now - self.t0)/60:.1f} min, ETA {eta:.0f} min (if the steps stay "
              f"as fast), peak {peak_rss_gb()} GB", flush=True)


def _say(tag, msg, t0):
    if PROGRESS_MIN > 0:
        print(f"[{tag}] {msg} ({(time.time()-t0)/60:.1f} min, peak {peak_rss_gb()} GB)",
              flush=True)


def run_fill(stl, dx, tilt, subcells=4):
    t0 = time.time()
    tag = f"fill {tilt:g} deg, {dx*1e3:g} mm"
    _say(tag, "setup: voxelising the door", t0)
    mesh, grid, mo = door_setup(stl, dx, tilt, place_untilted=PLACE_UNTILTED, hang=HANG,
                                levels=LEVELS if not (NARROW and LEVELS is None) else 0)
    _say(tag, f"setup: grid {grid.shape} = {grid.ncells/1e6:.2f} M cells; building the "
              "sub-cell graph, compartments and throats", t0)
    # drain holes plugged: every link through them is cut (no throats needed)
    sim = Simulation(grid, mo, dt_max=0.1, plugs=DOOR_HOLES, subcells=subcells,
                     throat_model=ThroatModel(Cd=0.65), narrow=NARROW)
    _say(tag, f"setup done: {sim.N/1e6:.2f} M nodes; running the motion", t0)
    sim.run(t_end=mo.t_end, progress=Progress(tag, mo.t_end, t0))
    v, bodies = trapped(sim)
    return sim, dict(case="fill", tilt=tilt, dx=dx, subcells=subcells, hang=HANG, levels=LEVELS, narrow=NARROW,
                     connect=bool(sim.fine.any()), nodes=int(sim.N),
                     cells=int(grid.ncells), trapped_l=v,
                     total_l=float(sim.hist.liquid_retained[-1] * 1e3),
                     bodies=bodies[:8], steps=len(sim.hist.t) - 1,
                     wall_s=time.time() - t0, peak_rss_gb=peak_rss_gb())


def run_drain(stl, dx, tilt, subcells=4):
    t0 = time.time()
    tag = f"drain {tilt:g} deg, {dx*1e3:g} mm"
    _say(tag, "setup and run (progress lines follow once it runs)", t0)
    prog = Progress(tag, 8.0 + 30.0, t0)       # motion + at least 30 s of drainage
    try:
        sim, H, r = run_case(stl, dx, tilt, "auto", verbose=False, subcells=subcells,
                             place_untilted=PLACE_UNTILTED, hang=HANG, progress=prog,
                             levels=LEVELS, narrow=NARROW)
    except ValueError:            # a hole is closed in a very coarse voxel grid
        sim, H, r = run_case(stl, dx, tilt, "auto", verbose=False, subcells=subcells,
                             place_untilted=PLACE_UNTILTED, hang=HANG, holes=None,
                             progress=prog, levels=LEVELS, narrow=NARROW)
    v, bodies = trapped(sim)
    return sim, dict(case="drain", tilt=tilt, dx=dx, subcells=subcells, hang=HANG, levels=LEVELS, narrow=NARROW,
                     connect=bool(sim.fine.any()), nodes=int(sim.N),
                     cells=r["cells"], trapped_l=v, total_l=r["retained_l"],
                     end_of_motion_l=r["retained_end_of_motion_l"], t_end=r["t_end"],
                     bodies=bodies[:8], steps=r["steps"], wall_s=time.time() - t0,
                     peak_rss_gb=peak_rss_gb())


CASES = [("fill", 0.0), ("fill", 22.5), ("drain", 0.0), ("drain", 22.5), ("drain", 45.0)]


# ------------------------------------------------------------------ figures
def load(out):
    res = []
    for f in glob.glob(os.path.join(out, "results*.jsonl")):
        for line in open(f):
            if line.strip():
                res.append(json.loads(line))
    return res


def _pick(res, case, tilt, dx):
    r = [x for x in res if x["case"] == case and x["tilt"] == tilt and x["dx"] == dx]
    return r[-1] if r else None


def table1(res, out):
    rows = []
    dxs = sorted({r["dx"] for r in res if r["tilt"] == 0.0}, reverse=True)
    for dx in dxs:
        f, d = _pick(res, "fill", 0.0, dx), _pick(res, "drain", 0.0, dx)
        if not f:
            continue
        rows.append(dict(dx=dx, cells=f["cells"], nodes=f.get("nodes"), tf=f["wall_s"],
                         vf=f["trapped_l"],
                         td=d["wall_s"] if d else None, vd=d["trapped_l"] if d else None,
                         ve=d.get("end_of_motion_l") if d else None))
    fmt = lambda x, f: "–" if x is None else f.format(x)
    md = ["| cell size | cells | time filling / drainage (1 core) | filling (l) | drainage (l) |",
          "|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['dx']*1e3:g} mm | {r['cells']/1e6:.2f}M | {r['tf']/60:.1f} / "
                  f"{fmt(r['td'] and r['td']/60, '{:.1f}')} min | {r['vf']:.2f} | {fmt(r['vd'], '{:.2f}')} |")
    md.append("")
    md.append("| article: refinements | min cell | cells | time (16 cores, h:mm) | filling (l) | drainage (l) |")
    md.append("|---|---|---|---|---|---|")
    for ref, mm, cells, tm, vf, vd in ART_TABLE1:
        md.append(f"| {ref} | {mm:g} mm | {cells} | {tm} | {vf:.2f} | {vd:.2f} |")
    md.append("")
    md.append("Experiment (0°): filling 10.75 ± 0.14 l; drainage: no trapped water.")
    open(os.path.join(out, "table1_comparison.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    head = ["", "(min) cell size", "cells (+ sub-cells)", "time (fill / drain)", "filling (l)",
            "drainage (l)"]
    cells, kind = [], []
    cells.append(["drainsim v4, 1 core", "", "", "", "", ""]); kind.append("h")
    for r in rows:
        t = f"{r['tf']/60:.1f} min" if r["td"] is None else \
            f"{r['tf']/60:.1f} / {r['td']/60:.1f} min"
        nc = f"{r['cells']/1e6:.2f}M" if not r.get("nodes") or r["nodes"] == r["cells"] else \
            f"{r['cells']/1e6:.2f}M + {(r['nodes'] - r['cells'])/1e6:.2f}M"
        cells.append(["uniform grid", f"{r['dx']*1e3:g} mm", nc, t,
                      f"{r['vf']:.2f}", fmt(r["vd"], "{:.2f}")]); kind.append("d")
    cells.append(["article, 16 cores", "", "", "", "", ""]); kind.append("h")
    for ref, mm, c, tm, vf, vd in ART_TABLE1:
        cells.append([f"refinement {ref}", f"{mm:g} mm", c, f"{tm} h:mm", f"{vf:.2f}",
                      f"{vd:.2f}"]); kind.append("a")
    cells.append(["experiment", "", "", "", "10.75 ± 0.14", "0.00"]); kind.append("e")
    fig, ax = plt.subplots(figsize=(10.5, 0.3 * (len(cells) + 1) + 0.4))
    ax.axis("off")
    tb = ax.table(cellText=cells, colLabels=head, loc="center", cellLoc="center",
                  colWidths=[0.18, 0.13, 0.19, 0.19, 0.16, 0.15], bbox=[0, 0, 1, 1])
    tb.auto_set_font_size(False)
    tb.set_fontsize(10)
    for (i, j), c in tb.get_celld().items():
        c.set_edgecolor("#d9d8d4")
        if i == 0:
            c.set_text_props(weight="bold")
            continue
        k = kind[i - 1]
        if k == "h":
            c.set_facecolor("#f1f0ec")
            c.set_text_props(weight="bold")
        if k == "e":
            c.set_text_props(weight="bold")
    ax.set_title("Table 1 recreated: door at 0°, filling (plugs in) and drainage (plugs out)",
                 fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "table1_comparison.png"), dpi=160, bbox_inches="tight",
                pad_inches=0.15)
    plt.close(fig)


def fig_points(res, out, case, tilts, art, exp, ylabel, fname, title, ylim, max_dx=None):
    """Article-style figure: one marker per grid at each tilt. Values above
    the axis are drawn as a triangle at the top with the value written."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    cmap = plt.get_cmap("Blues")
    dxs = sorted({r["dx"] for r in res if r["case"] == case and r["tilt"] in tilts
                  and (max_dx is None or r["dx"] <= max_dx + 1e-9)}, reverse=True)
    n = len(dxs)
    top = ylim[1]
    for i, dx in enumerate(dxs):
        pts = [(tl, _pick(res, case, tl, dx)) for tl in tilts]
        pts = [(tl, r["trapped_l"]) for tl, r in pts if r]
        if not pts:
            continue
        col = cmap(0.35 + 0.65 * i / max(n - 1, 1))
        t, v = map(np.array, zip(*pts))
        off = t - 3.0 + 0.0 * i
        ok = v <= top
        lab = f"drainsim {dx*1e3:g} mm"
        if case == "drain" and dx > 0.011:
            lab += " (holes open only\nat sub-cell level)"
        ax.plot(off[ok], v[ok], "o", ms=8, color=col, mec="white", mew=0.8, label=lab)
        if not ok.any():
            ax.plot([], [], "^", ms=8, color=col, mec="white", mew=0.8, label=lab)
        for x, y in zip(off[~ok], v[~ok]):
            ax.plot([x], [top * 0.985], "^", ms=8, color=col, mec="white", mew=0.8, clip_on=False)
            ax.annotate(f"{y:.2f}", (x, top * 0.985), xytext=(-4, -3), textcoords="offset points",
                        ha="right", va="top", fontsize=7, color="#52514e")
    art_cols = plt.get_cmap("Oranges")
    for j, ref in enumerate(sorted(ART_REF_DX)):
        pts = [(tl, art[tl][ref]) for tl in tilts if tl in art]
        if pts:
            t, v = zip(*pts)
            ax.plot(np.array(t) + 3.0, v, "s", ms=7, color=art_cols(0.35 + 0.65 * j / 4),
                    mec="white", mew=0.8, label=f"article ref. {ref} ({ART_REF_DX[ref]:g} mm)")
    for tl in tilts:
        m, s = exp[tl]
        ax.errorbar([tl], [m], yerr=[s], fmt="o", mfc="white", mec="k", ecolor="k",
                    capsize=6, ms=9, zorder=5, label="experiment (±1σ)" if tl == tilts[0] else None)
    ax.set_xticks(list(tilts))
    ax.set_xticklabels([f"{t:g}°" for t in tilts])
    ax.set_xlim(min(tilts) - 13, max(tilts) + 13)
    ax.set_ylim(*ylim)
    ax.set_xlabel("tilt")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color="#e6e5e0", lw=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    ax.set_title(title, fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(out, fname), dpi=150)
    plt.close(fig)
    print("wrote", os.path.join(out, fname))


# What sets the filling volume (found by tracing the minimax spill path):
# the lowest exits of the door cavity are the lower edge of the aperture in
# the inner panel (z ~ 1.589 m at x ~ -0.23 m), and two ~25 x 14 mm clip
# slots in the panel, at x ~ -0.10 m (bottom tab, with ~5 mm notches beside
# it, down to z ~ 1.589 m) and x ~ +0.225 m (the lowest exit at 22.5°).
# Cut-cell voxels (v3) close the slots and notches on some grids and not on
# others; the sub-cell graph (v4) resolves them to dx/k on every grid. The
# door hangs on two hooks and leans ~5° towards -y (HANG), which lifts these
# edges relative to the pool.


def fig_convergence(res, out, fname="convergence_comparison.png", prev=None,
                    prev_label="v3.3: cut cells closed, no hang"):
    """Trapped water against cell size for the four cases, drainsim and the
    article. ``prev``: results of an earlier version, drawn in grey."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    panels = [("fill", 0.0, ART_FILL, EXP_FILL, (9.5, 12.0)),
              ("fill", 22.5, ART_FILL, EXP_FILL, (3.5, 7.5)),
              ("drain", 22.5, ART_DRAIN, EXP_DRAIN, (0.0, 0.16)),
              ("drain", 45.0, ART_DRAIN, EXP_DRAIN, (0.0, 0.40))]
    fig, axs = plt.subplots(1, 4, figsize=(15, 4.0))

    def series(rr, case, tl, drop_unresolved):
        d = {}
        for r in sorted(rr, key=lambda r: r["dx"]):
            if r["case"] == case and r["tilt"] == tl:
                if drop_unresolved and case == "drain" and r["dx"] > 0.011:
                    continue       # v3 at 12 mm: the 19 mm drain holes are shut
                d[r["dx"] * 1e3] = r["trapped_l"]
        x = np.array(sorted(d))
        return x, np.array([d[k] for k in x])

    for ax, (case, tl, art, exp, yl) in zip(axs, panels):
        if prev:
            x, v = series(prev, case, tl, True)
            if x.size:
                ax.plot(x, np.clip(v, *yl), "o--", color="#a3a29d", mfc="white", ms=5,
                        lw=1.0, label=prev_label, zorder=2)
        x, v = series(res, case, tl, False)
        ax.plot(x, np.clip(v, *yl), "o-", color="#2a78d6", ms=7, lw=1.4,
                label="drainsim v4: sub-cell graph, 5° hang", zorder=4)
        ra = sorted(ART_REF_DX)
        ax.plot([ART_REF_DX[k] for k in ra], [art[tl][k] for k in ra], "s-",
                color="#eb6834", ms=6, lw=1.2, label="article (min cell)", zorder=3)
        m, s_ = exp[tl]
        ax.axhspan(m - s_, m + s_, color="#52514e", alpha=0.12, lw=0)
        ax.axhline(m, color="#52514e", lw=1.0, ls="--", label="experiment ±1σ")
        ax.set_xscale("log")
        ax.set_xticks([0.35, 0.7, 1.4, 2.75, 5.5, 11])
        ax.set_xticklabels(["0.35", "0.7", "1.4", "2.75", "5.5", "11"])
        ax.minorticks_off()
        ax.invert_xaxis()
        ax.set_ylim(*yl)
        ax.set_xlabel("cell size (mm)")
        ax.set_title(f"{'filling' if case == 'fill' else 'drainage'} {tl:g}°", fontsize=11,
                     loc="left")
        ax.grid(color="#e6e5e0", lw=0.8)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axs[0].set_ylabel("trapped water (l)")
    axs[1].legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(out, fname), dpi=150)
    plt.close(fig)
    print("wrote", os.path.join(out, fname))


def render_still(stl, case, tilt, dx, path, size=(1300, 900)):
    """Door and trapped water after filling (plugs in) or drainage, world
    frame, seen from the cabin side, like the article's Figs. 6 and 8."""
    sim = (run_fill if case == "fill" else run_drain)(stl, dx, tilt)[0]
    render_sim(sim, stl, path, size)
    return trapped(sim)[0]


STILLS = {("fill", 0.0), ("fill", 22.5), ("drain", 22.5), ("drain", 45.0)}


def still_path(out, case, tilt, dx):
    return os.path.join(out, f"{case}_{tilt:g}_{dx*1e3:g}mm.png")


def render_sim(sim, stl, path, size=(1800, 1250)):
    """Render a finished run (Figs. 6/8 still) without running it again."""
    import trimesh
    from door_drain_movie import Scene
    t = sim.t
    # as in the article's post-processing: bodies below 0.01 l are not shown
    bd = sim._bodies()
    small = np.flatnonzero((bd["vol"] < MIN_BODY_L * 1e-3) & ~bd["bath"])
    sim.L[np.isin(bd["body"], small)] = 0.0
    holes = [dict(center=th.centroid, diameter=th.diameter, open_at=-np.inf)
             for th in sim.comp.throats if th.axis is not None]
    sc = Scene(sim, trimesh.load(stl, force="mesh"), holes, size, np.array([t]),
               ssaa=2, door_opacity=0.35)
    # frame the whole door (not the default near-bath box)
    from drainsim.worldviz import to_world
    W = to_world(trimesh.load(stl, force="mesh").vertices[::10], sim.motion, t)
    box = np.array([W.min(0), W.max(0)]).T
    ren = sc.ren
    ren.ResetCamera(*box.ravel())
    cam = ren.GetActiveCamera()
    cam.SetViewUp(0, 0, 1)
    ctr = box.mean(1)
    cam.SetFocalPoint(*ctr)
    cam.SetPosition(ctr[0], ctr[1] - 1.0, ctr[2])
    cam.Azimuth(195)
    cam.Elevation(8)
    ren.ResetCamera(*box.ravel())
    cam.Zoom(1.3)
    sc.bath.VisibilityOff()
    for p in sc.plugs:
        p.VisibilityOff()
    img = sc.render(t, "", np.zeros(len(holes)), 1e-3)
    from PIL import Image
    Image.fromarray(img).save(path)


def _water_box(img, pad=40):
    """Pixel box around the rendered water (blue), or None."""
    im = img[..., :3].astype(int)
    m = (im[..., 2] > im[..., 0] + 50) & (im[..., 2] > im[..., 1] + 25)
    if m.sum() < 20:
        return None
    ys, xs = np.nonzero(m)
    y0, y1 = np.percentile(ys, [1, 99]).astype(int)
    x0, x1 = np.percentile(xs, [1, 99]).astype(int)
    h, w = m.shape
    return max(y0 - pad, 0), min(y1 + pad, h), max(x0 - pad, 0), min(x1 + pad, w)


def fig_stills(stl, out, case, tilts, dx, fname, title, inset=None, size=(1800, 1250),
               res=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    inset = (case == "drain") if inset is None else inset
    fig, ax = plt.subplots(1, len(tilts), figsize=(6.4 * len(tilts), 4.8))
    for k, (a, tl) in enumerate(zip(np.atleast_1d(ax), tilts)):
        p = still_path(out, case, tl, dx)
        r = _pick(res or [], case, tl, dx)
        if r is not None and os.path.exists(p):      # rendered right after its run
            v = r["trapped_l"]
            print("using", p)
        else:                                        # run it again and render
            v = render_still(stl, case, tl, dx, p, size=size)
        img = mpimg.imread(p)
        a.imshow(img)
        a.set_axis_off()
        a.set_title(f"({'ab'[k]}) {tl:g}°: {v:.3f} l trapped" if v < 1 else
                    f"({'ab'[k]}) {tl:g}°: {v:.2f} l trapped", fontsize=11)
        box = _water_box((img * 255).astype(np.uint8) if img.dtype != np.uint8 else img)
        if inset and box:
            y0, y1, x0, x1 = box
            ia = a.inset_axes([0.62, 0.02, 0.37, 0.37])
            ia.imshow(img[y0:y1, x0:x1])
            ia.set_xticks([]); ia.set_yticks([])
            for sp in ia.spines.values():
                sp.set_edgecolor("#2a78d6"); sp.set_linewidth(1.2)
            a.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                      ec="#2a78d6", lw=1.0))
    fig.suptitle(title, fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(out, fname), dpi=150)
    plt.close(fig)
    print("wrote", os.path.join(out, fname))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stl", required=True)
    ap.add_argument("--dx", type=float, nargs="*", default=[])
    ap.add_argument("--cases", nargs="*", default=None,
                    help="subset like fill:0 drain:45 (default: all five)")
    ap.add_argument("--figures", action="store_true")
    ap.add_argument("--stills-dx-fill", type=float, default=None,
                    help="cell size for the Fig. 6 renders (default: finest filling run)")
    ap.add_argument("--stills-dx-drain", type=float, default=None,
                    help="cell size for the Fig. 8 renders (default: finest drainage run)")
    ap.add_argument("--no-stills", action="store_true")
    ap.add_argument("--hang", type=float, default=HANG,
                    help="lean of the door on its hooks, degrees (default %(default)s)")
    ap.add_argument("--exclude-dx", type=float, nargs="*", default=[],
                    help="grids left out of the figures")
    ap.add_argument("--prev", default=None,
                    help="results folder of an earlier version, drawn in grey in the "
                         "convergence figure")
    ap.add_argument("--progress", type=float, default=0.0, metavar="MIN",
                    help="print a progress line every MIN minutes (0 = off)")
    ap.add_argument("--levels", type=int, default=None,
                    help="octree with this many coarser levels (dx = the finest cell); "
                         "no stills (they need the uniform grid)")
    ap.add_argument("--narrow", type=int, default=0, metavar="K",
                    help="K sub-cells per edge in cells at narrow passages (0 = off; "
                         "uses the octree, no stills)")
    ap.add_argument("--out", default="examples/out/article_comparison")
    a = ap.parse_args()
    HANG = a.hang
    LEVELS = a.levels
    NARROW = dict(k=a.narrow) if a.narrow else None
    if LEVELS is not None or NARROW:
        a.no_stills = True
    PROGRESS_MIN = a.progress
    os.makedirs(a.out, exist_ok=True)
    cases = CASES if not a.cases else [(c.split(":")[0], float(c.split(":")[1])) for c in a.cases]
    log = os.path.join(a.out, f"results_{os.getpid()}.jsonl")
    for dx in a.dx:
        for case, tilt in cases:
            sim, r = (run_fill if case == "fill" else run_drain)(a.stl, dx, tilt)
            print(f"{case:5s} {tilt:4g}° dx={dx*1e3:g} mm: {r['trapped_l']:.4f} l "
                  f"(all {r['total_l']:.4f} l), {r['cells']/1e6:.2f} M cells, "
                  f"{r['nodes']/1e6:.2f} M nodes, {r['wall_s']:.0f} s, "
                  f"peak {r['peak_rss_gb']} GB", flush=True)
            with open(log, "a") as f:
                f.write(json.dumps(r) + "\n")
            if (case, tilt) in STILLS and not a.no_stills:
                # the Figs. 6/8 still now, so --figures need not run it again
                try:
                    p = still_path(a.out, case, tilt, dx)
                    render_sim(sim, a.stl, p)
                    print("wrote", p, flush=True)
                except Exception as e:                   # keep the result anyway
                    print(f"still for {case} {tilt:g} {dx*1e3:g} mm failed: {e!r}", flush=True)
            del sim
    if a.figures:
        res = [r for r in load(a.out)
               if not any(abs(r["dx"] - x) < 1e-9 for x in a.exclude_dx)]
        table1(res, a.out)
        fig_points(res, a.out, "fill", (0.0, 22.5), ART_FILL, EXP_FILL,
                   "water trapped after filling (l)", "fig5_comparison.png",
                   "Fig. 5 recreated: filling (plugs in)", (0, 12))
        fig_points(res, a.out, "drain", (22.5, 45.0), ART_DRAIN, EXP_DRAIN,
                   "water trapped after drainage (l)", "fig7_comparison.png",
                   "Fig. 7 recreated: drainage (plugs out)", (0, 0.4))
        fig_convergence(res, a.out, prev=load(a.prev) if a.prev else None)
        if not a.no_stills:
            fdx = a.stills_dx_fill or min(r["dx"] for r in res if r["case"] == "fill")
            ddx = a.stills_dx_drain or min(r["dx"] for r in res if r["case"] == "drain")
            fig_stills(a.stl, a.out, "fill", (0.0, 22.5), fdx, "fig6_comparison.png",
                       f"Fig. 6 recreated: trapped water after filling (plugs in), drainsim {fdx*1e3:g} mm",
                       res=res)
            fig_stills(a.stl, a.out, "drain", (22.5, 45.0), ddx, "fig8_comparison.png",
                       f"Fig. 8 recreated: trapped water after drainage (plugs out), drainsim {ddx*1e3:g} mm",
                       res=res)
