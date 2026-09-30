"""Presentation movie: four of the early test cases, side by side (2 x 2).

The synthetic 2D cases used to develop and check the model, run with the
current version of drainsim and drawn in the world frame (the object moves,
the bath stays):

1. **Drainage through a hole**: a vented box empties through a floor
   opening; the level follows the orifice law (dashed: analytic,
   Torricelli with the capillary hold-up).
2. **Spill routing**: a full cup tilts; the overflow is routed into the cup
   below instead of disappearing.
3. **Trapped air**: dip-in and dip-out; an upside-down cup traps an air
   pocket under the bath, an open cup carries liquid out.
4. **Door-like section**: dip, hold, tilt out and drain; drain holes empty
   the cavity over time, a shelf and a small cup keep some liquid.

Colours: bath light blue, liquid held by the object dark blue, trapped air
orange, walls dark grey. Volumes in 2D are per metre of depth.

    python examples/early_cases_movie.py --out early_cases.mp4
    python examples/early_cases_movie.py --preview 3 8 15 25 --out early_cases.mp4
"""
from __future__ import annotations

import argparse
import os
import subprocess
import time

import numpy as np

from drainsim import cases
from drainsim import shapes as sh
from drainsim.model import Simulation
from drainsim.motion import Keyframes
from drainsim.physics import Fluid, ThroatModel, torricelli_level
from drainsim.worldviz import pose3

COL = dict(bath=np.array([0.62, 0.79, 0.95]), liq=np.array([0.10, 0.33, 0.75]),
           air=np.array([1.0, 0.62, 0.15]), solid=np.array([0.20, 0.20, 0.22]),
           bg=np.array([1.0, 1.0, 1.0]))
T_END = 30.0


# -------------------------------------------------------------------- cases
def case_torricelli():
    g, interior = cases.torricelli_box(dx=0.004, hole=0.010)
    sim = Simulation(g, cases.static(t_end=T_END), initial_L=interior.astype(float),
                     dt_max=0.02)
    th = min(sim.comp.throats, key=lambda t: t.centroid[1])
    W = 0.38                                    # inner width of the box
    floor = 0.31                                # inner floor height
    hc = ThroatModel().holdup_head(th.diameter, Fluid(), 2)
    h0 = float((sim.L * sim.v)[interior].sum() / W) + floor - th.centroid[1]

    def metric(s, ax):
        vol = float((s.L * s.v)[interior].sum())
        lvl = floor + vol / W
        ana = th.centroid[1] + float(torricelli_level(s.t, h0, 0.6, th.area, W, h_cap=hc))
        _line(ax, "ana", [0.5 - W / 2, 0.5 + W / 2], [ana, ana], s)
        return f"level {lvl - th.centroid[1]:.3f} m   analytic {ana - th.centroid[1]:.3f} m"
    return dict(sim=sim, title="1  Drainage through a hole",
                sub="orifice law with capillary hold-up; dashed: analytic", metric=metric)


def case_spill():
    g = cases.two_cups(dx=0.004)
    X = g.centers()
    fl = g.fluid.ravel()
    upper = fl & (X[:, 0] > 0.12) & (X[:, 0] < 0.38) & (X[:, 1] > 0.52) & (X[:, 1] < 0.68)
    lower = fl & (X[:, 0] > 0.37) & (X[:, 0] < 0.78) & (X[:, 1] > 0.12) & (X[:, 1] < 0.30)
    # hold 3 s, tilt the right side down by 40 deg over 12 s, hold
    mo = Keyframes([0, 3, 15, T_END], [0, 0, -40, -40], np.zeros((4, 2)), 2,
                   pivot=np.array([0.45, 0.4]), bath_level=-10.0)
    sim = Simulation(g, mo, initial_L=upper.astype(float), dt_max=0.05)

    def metric(s, ax):
        up = float((s.L * s.v)[upper].sum()) * 1e3
        lo = float((s.L * s.v)[lower].sum()) * 1e3
        return f"upper cup {up:5.1f} l/m    lower cup {lo:5.1f} l/m"
    return dict(sim=sim, title="2  Spill routing",
                sub="liquid leaving the upper cup is routed to the cup below", metric=metric)


def case_air():
    g = cases.inverted_cup(dx=0.004)
    # down 0.55 m in 6 s, hold 4 s, up in 6 s, hang
    mo = Keyframes([0, 6, 10, 16, T_END], [0, 0, 0, 0, 0],
                   np.array([[0, 0.12], [0, -0.43], [0, -0.43], [0, 0.12], [0, 0.12]]), 2,
                   pivot=np.zeros(2))
    sim = Simulation(g, mo, dt_max=0.05)

    def metric(s, ax):
        a = s.hist.air_trapped[-1] * 1e3
        r = s.hist.liquid_above_bath[-1] * 1e3
        return f"trapped air {a:5.1f} l/m    carried out {r:5.1f} l/m"
    return dict(sim=sim, title="3  Trapped air and carried liquid",
                sub="upside-down cup traps air; open cup carries liquid out", metric=metric)


def case_door():
    g = cases.door_section(dx=0.003, hole=0.008)
    mo = Keyframes([0, 6, 8, 14, T_END], [0, 0, 0, 25, 25],
                   np.array([[0, 0.15], [0, -0.75], [0, -0.75], [0.3, 0.15], [0.3, 0.15]]),
                   2, pivot=np.array([0, 0.35]))
    sim = Simulation(g, mo, dt_max=0.05)

    def metric(s, ax):
        r = s.hist.liquid_retained[-1] * 1e3
        return f"liquid held by the section {r:6.1f} l/m"
    return dict(sim=sim, title="4  Door-like section",
                sub="dip, tilt out and drain through 8 mm holes", metric=metric,
                thicken=1)


# ------------------------------------------------------------------ drawing
def _line(ax, key, xs, ys, s):
    """A line fixed to the object (moves with it), created once."""
    import matplotlib.transforms as mt
    store = ax.__dict__.setdefault("_obj_lines", {})
    if key not in store:
        store[key], = ax.plot(xs, ys, "--", color="#0b0b0b", lw=1.2, zorder=4)
    store[key].set_data(xs, ys)
    store[key].set_transform(_affine(s, s.t) + ax.transData)


def _affine(sim, t):
    import matplotlib.transforms as mt
    R, T, p = pose3(sim.motion, t)
    R2 = R[:2, :2]
    off = (p + T - R @ p)[:2]
    M = np.array([[R2[0, 0], R2[0, 1], off[0]], [R2[1, 0], R2[1, 1], off[1]], [0, 0, 1]])
    return mt.Affine2D(M)


def colours(sim, thicken=0):
    g = sim.grid
    n = g.ncells
    L = np.clip(sim.L[:n], 0, 1)
    B = sim.B[:n]
    A = sim.A[:n]
    fl = sim.fl[:n]
    up, zb = sim.motion.frame(sim.t)
    below = (sim.X[:n] @ up) < zb
    air = np.where(fl & ~A & below, 1 - L, 0.0)
    # held liquid: dark blue, opacity = fill fraction (empty cells are see-through)
    rgb = np.repeat(COL["liq"][None], n, 0)
    alpha = L.copy()
    rgb[B] = COL["bath"]
    m = air > 0                                # gas below the bath surface
    base = np.where(B[m, None], COL["bath"][None], COL["liq"][None])
    rgb[m] = COL["air"][None] * air[m, None] + base * (1 - air[m])[:, None]
    alpha[m] = 1.0
    alpha[alpha < 0.02] = 0.0
    solid = g.solid.ravel()
    rgb[solid] = COL["solid"]
    alpha[solid] = 1.0
    nx, ny = g.shape
    if thicken:                                # display only: fatten thin walls
        from scipy.ndimage import binary_dilation
        grow = binary_dilation(solid.reshape(nx, ny), iterations=thicken).ravel()
        grow &= alpha == 0
        rgb[grow] = COL["solid"]
        alpha[grow] = 1.0
    c = np.column_stack([rgb, alpha])
    return c.reshape(nx, ny, 4).transpose(1, 0, 2).reshape(-1, 4)


class Panel:
    def __init__(self, fig, rect, case, times):
        self.case = case
        s = case["sim"]
        g = s.grid
        ax = fig.add_axes(rect)
        # the camera follows the object (centre of its solid cells) at a fixed
        # zoom that fits its largest rotated extent over the whole motion
        X = g.centers()
        S = X[g.solid.ravel()]
        olo, ohi = S.min(0) - 2 * g.dx, S.max(0) + 2 * g.dx
        self.ocen = 0.5 * (olo + ohi)
        corners = np.array([[olo[0], olo[1]], [ohi[0], olo[1]], [ohi[0], ohi[1]],
                            [olo[0], ohi[1]]])
        from drainsim.worldviz import to_world
        self.to_world = to_world
        half = np.zeros(2)
        for t in np.linspace(0.0, T_END, 241):
            P = to_world(corners, s.motion, t)[:, :2]
            c = to_world(self.ocen[None], s.motion, t)[0, :2]
            half = np.maximum(half, np.abs(P - c).max(0))
        half *= 1.12
        r = (rect[2] * fig.get_figwidth()) / (rect[3] * fig.get_figheight())
        if half[0] / half[1] < r:
            half[0] = half[1] * r
        else:
            half[1] = half[0] / r
        self.half = half
        if s.motion.bath_level > -5.0:                   # bath in view
            ax.axhspan(-1e3, s.motion.bath_level, color=COL["bath"], lw=0, zorder=0)
            ax.axhline(s.motion.bath_level, color="#3b78c2", lw=1, zorder=0.5)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("#d9d8d4")
        x = g.origin[0] + np.arange(g.shape[0] + 1) * g.dx
        y = g.origin[1] + np.arange(g.shape[1] + 1) * g.dx
        X, Y = np.meshgrid(x, y)
        self.mesh = ax.pcolormesh(X, Y, np.zeros((g.shape[1], g.shape[0])), shading="flat",
                                  zorder=1, rasterized=True)
        self.mesh.set_array(None)
        self.ax = ax
        x0, y1 = rect[0], rect[1] + rect[3]
        fig.text(x0, y1 + 0.052, case["title"], fontsize=15, weight="bold", color="#0b0b0b")
        fig.text(x0, y1 + 0.030, case["sub"], fontsize=10.5, color="#52514e")
        self.txt = fig.text(x0, y1 + 0.008, "", fontsize=10.5, color="#0b0b0b",
                            family="monospace")

    def draw(self):
        s = self.case["sim"]
        self.mesh.set_facecolor(colours(s, self.case.get("thicken", 0)))
        self.mesh.set_transform(_affine(s, s.t) + self.ax.transData)
        c = self.to_world(self.ocen[None], s.motion, s.t)[0, :2]
        self.ax.set_xlim(c[0] - self.half[0], c[0] + self.half[0])
        self.ax.set_ylim(c[1] - self.half[1], c[1] + self.half[1])
        self.txt.set_text(self.case["metric"](s, self.ax))


def main(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    t0 = time.time()
    cs = [case_torricelli(), case_spill(), case_air(), case_door()]
    fps = a.fps
    times = np.arange(0.0, T_END + 1e-9, 1.0 / fps)
    if a.preview:
        times = np.array(sorted(a.preview))
    W, H = a.size
    dpi = 100
    fig = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi)
    fig.patch.set_facecolor("white")
    canvas = FigureCanvasAgg(fig)
    rects = [[0.02, 0.535, 0.46, 0.34], [0.52, 0.535, 0.46, 0.34],
             [0.02, 0.055, 0.46, 0.34], [0.52, 0.055, 0.46, 0.34]]
    panels = [Panel(fig, r, c, times) for r, c in zip(rects, cs)]
    fig.text(0.02, 0.965, "drainsim: early test cases", fontsize=18, weight="bold",
             color="#0b0b0b")
    clock = fig.text(0.98, 0.965, "", fontsize=16, ha="right", family="monospace",
                     color="#0b0b0b")
    fig.text(0.98, 0.012, "bath  ■ light blue    held liquid  ■ dark blue    trapped air  "
             "■ orange    walls  ■ grey", fontsize=10, ha="right", color="#52514e")
    ff = None
    if not a.preview:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        ff = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{W}x{H}", "-r", str(fps), "-i", "-", "-c:v", "libx264",
             "-pix_fmt", "yuv420p", "-crf", str(a.crf), "-preset", a.preset,
             "-movflags", "+faststart", a.out], stdin=subprocess.PIPE)
    for i, t in enumerate(times):
        for c in cs:
            if t > c["sim"].t + 1e-9:
                c["sim"].run(t_end=float(t))
        for p in panels:
            p.draw()
        clock.set_text(f"t = {t:5.2f} s")
        canvas.draw()
        img = np.ascontiguousarray(np.asarray(canvas.buffer_rgba())[:, :, :3])
        if a.preview:
            from PIL import Image
            path = a.out.replace(".mp4", "") + f"_t{t:05.2f}.png"
            Image.fromarray(img).save(path)
            print("wrote", path, flush=True)
        else:
            ff.stdin.write(img.tobytes())
        if i % 120 == 0:
            print(f"frame {i}/{len(times)}  t = {t:.2f} s  ({time.time()-t0:.0f} s)", flush=True)
    if ff is not None:
        ff.stdin.close()
        ff.wait()
        print("wrote", a.out, flush=True)
    print(f"total {time.time()-t0:.0f} s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--size", type=int, nargs=2, default=[1920, 1080])
    ap.add_argument("--crf", type=int, default=16)
    ap.add_argument("--preset", default="slow")
    ap.add_argument("--preview", type=float, nargs="*")
    ap.add_argument("--out", default="examples/out/early_cases.mp4")
    main(ap.parse_args())
