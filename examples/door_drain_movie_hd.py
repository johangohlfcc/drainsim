"""Full-HD movie of the door drainage case next to the measurements.

The door (hanging on its hooks, 5 degrees lean) is dipped with its three
drain holes plugged, lifted and held; then the plugs are pulled and the
water runs out. Left, the 3D view:

- the door, translucent; below the bath surface it is tinted blue and a
  dark waterline marks where the surface cuts it;
- the water held by the door (trapped liquid above the surface);
- the plugs, which drop into the bath when pulled;
- the jets: tapered streams (they narrow as they speed up), turning into
  drops when the flow becomes a trickle, with ripples where they hit the
  bath; the camera closes in slowly while the door drains.

Right: water left in the door against time since the plugs were pulled,
model against the four measurements (range shaded), the outflow rate below,
and the 50 / 90 / 99 % drained times, filled in as they are reached.

    python examples/door_drain_movie_hd.py --stl door.stl --out door_drain_hd.mp4
    python examples/door_drain_movie_hd.py --stl door.stl --preview 3 12 15 22 34 --out door_drain_hd.mp4
    python examples/door_drain_movie_hd.py --stl door.stl --workers 8 --out door_drain_hd.mp4

``--workers N`` renders the frames in N processes and joins the parts. Each
worker runs the model from t = 0 up to its own last frame (the frames before
its first one are only not drawn), so the physics is repeated: the workers
speed up the drawing, not the model, and hold N simulations in memory. For a
recording that runs the model once, see ``car_movie.py --record/--render``.
Needs vtk and ffmpeg.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import warnings

import numpy as np

from door_article import DOOR_HOLES, door_setup
from door_drain import fractions, load_experiment
from door_drain_movie import Scene
from drainsim.model import Simulation
from drainsim.physics import ThroatModel
from drainsim.worldviz import to_world

warnings.filterwarnings("ignore")
G = 9.81
C_MODEL = "#2a78d6"
C_EXP = "#8a8985"
WATER = (0.13, 0.45, 0.88)


def smoothstep(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


# ------------------------------------------------------------------ 3D view
class DrainScene(Scene):
    def __init__(self, sim, mesh, holes, size, t_all, t_open, **kw):
        super().__init__(sim, mesh, holes, size, t_all, **kw)
        vtk = self.vtk
        self.t_open = t_open
        self.zb = float(sim.motion.bath_level)
        ren = self.ren
        # softer, three-point lighting
        ren.RemoveAllLights()
        lk = vtk.vtkLightKit()
        lk.SetKeyLightIntensity(0.95)
        lk.SetKeyToFillRatio(2.6)
        lk.SetKeyToHeadRatio(3.0)
        lk.AddLightsToRenderer(ren)
        # water: deeper blue, glossy
        lp = self.liq.GetProperty()
        lp.SetColor(*WATER)
        lp.SetSpecular(0.55)
        lp.SetSpecularPower(45)
        lp.SetOpacity(0.93)
        # bath: pale box plus a clearer surface sheet
        self.bath.GetProperty().SetOpacity(0.14)
        self.bath.GetProperty().SetColor(0.62, 0.79, 0.96)
        b = self.bath.GetMapper().GetInput().GetBounds()
        pl = vtk.vtkPlaneSource()
        pl.SetOrigin(b[0], b[2], self.zb)
        pl.SetPoint1(b[1], b[2], self.zb)
        pl.SetPoint2(b[0], b[3], self.zb)
        pl.Update()
        self.surf = self._actor(pl.GetOutput(), (0.42, 0.66, 0.92), 0.30, specular=0.4)
        ren.AddActor(self.surf)
        # the door under the surface tinted, and the waterline
        self._underwater(None)
        # hide the old cylinder jets; new ones: tubes, drops, ripples
        for j in self.jets:
            j.VisibilityOff()
        self.jet_pd = [vtk.vtkPolyData() for _ in holes]
        self.jet_act = []
        for pd in self.jet_pd:
            a = self._actor(pd, WATER, 1.0, specular=0.6)
            a.GetProperty().SetSpecularPower(60)
            ren.AddActor(a)
            self.jet_act.append(a)
        self.drop_pts = vtk.vtkPoints()
        self.drop_pd = vtk.vtkPolyData()
        self.drop_pd.SetPoints(self.drop_pts)
        sph = vtk.vtkSphereSource()
        sph.SetThetaResolution(16)
        sph.SetPhiResolution(12)
        sph.SetRadius(1.0)
        gl = vtk.vtkGlyph3D()
        gl.SetInputData(self.drop_pd)
        gl.SetSourceConnection(sph.GetOutputPort())
        gl.SetScaleModeToDataScalingOff()
        gl.SetScaleFactor(0.0045)
        m = vtk.vtkPolyDataMapper()
        m.SetInputConnection(gl.GetOutputPort())
        m.ScalarVisibilityOff()
        self.drops_act = vtk.vtkActor()
        self.drops_act.SetMapper(m)
        dp = self.drops_act.GetProperty()
        dp.SetColor(*WATER)
        dp.SetSpecular(0.6)
        dp.SetSpecularPower(60)
        ren.AddActor(self.drops_act)
        self.ring_pd = vtk.vtkPolyData()
        rm = vtk.vtkPolyDataMapper()
        rm.SetInputData(self.ring_pd)
        rm.SetColorModeToDirectScalars()
        rm.SetScalarModeToUseCellData()
        self.ring_act = vtk.vtkActor()
        self.ring_act.SetMapper(rm)
        self.ring_act.GetProperty().SetLineWidth(2.0 * self.ssaa)
        self.ring_act.GetProperty().LightingOff()
        ren.AddActor(self.ring_act)
        self.drops = []                 # (birth time, hole index)
        self.acc = np.zeros(len(holes))
        self.splash = []                # (time, x, y) of drop impacts
        self.t_last = None
        cam = ren.GetActiveCamera()
        self._cam0 = (np.array(cam.GetFocalPoint()), np.array(cam.GetPosition()))

    # jets ----------------------------------------------------------------
    def _jet(self, k, c, q, d):
        """Tapered stream from c down to the bath: the contracted jet speeds
        up in free fall and narrows as r ~ v^-1/2."""
        vtk = self.vtk
        L = c[2] - self.zb
        A = 0.25 * np.pi * d * d
        v0 = max(q / (0.62 * A), 0.05)
        r0 = np.sqrt(q / (np.pi * v0))
        s = np.linspace(0.0, L, 32)
        v = np.sqrt(v0 * v0 + 2 * G * s)
        r = r0 * np.sqrt(v0 / v)
        pts = vtk.vtkPoints()
        for si in s:
            pts.InsertNextPoint(c[0], c[1], c[2] - si)
        line = vtk.vtkPolyLine()
        line.GetPointIds().SetNumberOfIds(len(s))
        for i in range(len(s)):
            line.GetPointIds().SetId(i, i)
        cells = vtk.vtkCellArray()
        cells.InsertNextCell(line)
        pd = vtk.vtkPolyData()
        pd.SetPoints(pts)
        pd.SetLines(cells)
        from vtk.util.numpy_support import numpy_to_vtk
        rr = numpy_to_vtk(r.astype(float), deep=True)
        rr.SetName("r")
        pd.GetPointData().SetScalars(rr)
        tf = vtk.vtkTubeFilter()
        tf.SetInputData(pd)
        tf.SetNumberOfSides(24)
        tf.SetVaryRadiusToVaryRadiusByAbsoluteScalar()
        tf.CappingOn()
        tf.Update()
        self.jet_pd[k].DeepCopy(tf.GetOutput())
        return r[-1]

    def _rings(self, items):
        """Ripples: (x, y, radius, alpha) circles on the bath surface."""
        vtk = self.vtk
        pts = vtk.vtkPoints()
        lines = vtk.vtkCellArray()
        col = vtk.vtkUnsignedCharArray()
        col.SetNumberOfComponents(4)
        n = 48
        th = np.linspace(0, 2 * np.pi, n, endpoint=False)
        for x, y, rad, al in items:
            base = pts.GetNumberOfPoints()
            for a in th:
                pts.InsertNextPoint(x + rad * np.cos(a), y + rad * np.sin(a), self.zb + 1e-3)
            pl = vtk.vtkPolyLine()
            pl.GetPointIds().SetNumberOfIds(n + 1)
            for i in range(n):
                pl.GetPointIds().SetId(i, base + i)
            pl.GetPointIds().SetId(n, base)
            lines.InsertNextCell(pl)
            col.InsertNextTuple4(245, 250, 255, int(255 * max(0.0, min(al, 1.0))))
        pd = vtk.vtkPolyData()
        pd.SetPoints(pts)
        pd.SetLines(lines)
        pd.GetCellData().SetScalars(col)
        self.ring_pd.DeepCopy(pd)

    def update(self, t, label, hole_flows, qmax):
        super().update(t, label, hole_flows, qmax)
        vm = self._vm
        self._update_water(t, vm)
        for j in self.jets:
            j.VisibilityOff()
        dt = 0.0 if self.t_last is None else max(t - self.t_last, 0.0)
        self.t_last = t
        rings = []
        q_drop = 2e-6                        # below 2 ml/s: drops, not a stream
        v_drop = 0.08e-6                     # drop volume (display)
        for k, h in enumerate(self.holes):
            opened = t >= h["open_at"] - 1e-9
            c = to_world(np.asarray(h["center"])[None], self.sim.motion, t)[0]
            q = float(hole_flows[k]) if opened else 0.0
            # plug: pops out and falls into the bath
            age = t - h["open_at"]
            if opened and age < 0.8:
                s = min(0.5 * G * age * age, c[2] - self.zb + 0.05)
                tr = self.vtk.vtkTransform()
                tr.PostMultiply()
                tr.SetMatrix(vm)
                tr.Translate(0.0, 0.0, -s)
                self.plugs[k].SetUserMatrix(tr.GetMatrix())
                self.plugs[k].SetVisibility(True)
            if q > q_drop and c[2] > self.zb + 0.01:
                self.jet_act[k].VisibilityOn()
                rj = self._jet(k, c, q, h["diameter"])
                amp = np.sqrt(q / qmax)
                for i in range(3):
                    ph = (t * 1.7 + i / 3.0) % 1.0
                    rings.append((c[0], c[1], rj + (0.012 + 0.05 * ph) * (0.6 + amp),
                                  0.75 * (1 - ph)))
                self.acc[k] = 0.0
            else:
                self.jet_act[k].VisibilityOff()
                if opened and q > 0 and dt > 0:
                    self.acc[k] += q * dt
                    while self.acc[k] >= v_drop:
                        self.acc[k] -= v_drop
                        self.drops.append((t - self.acc[k] / max(q, 1e-12), k))
        # drops: free fall from the hole; a small ripple where they land
        self.drop_pts.Reset()
        alive = []
        for tb, k in self.drops:
            c = to_world(np.asarray(self.holes[k]["center"])[None], self.sim.motion, t)[0]
            s = 0.5 * G * max(t - tb, 0.0) ** 2
            if c[2] - s <= self.zb:
                self.splash.append((tb + np.sqrt(2 * max(c[2] - self.zb, 0) / G), c[0], c[1]))
                continue
            self.drop_pts.InsertNextPoint(c[0], c[1], c[2] - 0.004 - s)
            alive.append((tb, k))
        self.drops = alive
        self.drop_pts.Modified()
        self.drop_pd.Modified()
        keep = []
        for ts, x, y in self.splash:
            a = t - ts
            if 0 <= a < 0.6:
                rings.append((x, y, 0.004 + 0.06 * a, 0.7 * (1 - a / 0.6)))
                keep.append((ts, x, y))
            elif a < 0:
                keep.append((ts, x, y))
        self.splash = keep
        self._rings(rings)
        # camera: steady through the dip, then a slow push towards the holes
        f0, p0 = self._cam0
        u = smoothstep((t - self.t_open + 1.0) / 10.0)
        ctr = np.mean([to_world(np.asarray(h["center"])[None], self.sim.motion, t)[0]
                       for h in self.holes], axis=0)
        f = f0 + 0.20 * u * (ctr - f0)
        az = np.radians(8.0 * u)
        d = p0 - f0
        Rz = np.array([[np.cos(az), -np.sin(az), 0], [np.sin(az), np.cos(az), 0], [0, 0, 1]])
        p = f + (Rz @ d) * (1.0 - 0.08 * u)
        cam = self.ren.GetActiveCamera()
        cam.SetFocalPoint(*f)
        cam.SetPosition(*p)
        cam.SetViewUp(0, 0, 1)
        self.ren.ResetCameraClippingRange()


# ------------------------------------------------------------------ panel
class DrainPanel:
    def __init__(self, size, t_open, t_end, exp, info_model):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        dpi = 100 * size[1] / 900
        self.fig = plt.figure(figsize=(size[0] / dpi, size[1] / dpi), dpi=dpi)
        self.fig.patch.set_facecolor("white")
        self.canvas = FigureCanvasAgg(self.fig)
        ax = self.fig.add_axes([0.15, 0.40, 0.80, 0.42])
        ax2 = self.fig.add_axes([0.15, 0.17, 0.80, 0.15], sharex=ax)
        self.ax, self.ax2 = ax, ax2
        x0, x1 = -t_open, t_end - t_open
        # measurements: range of the four runs, and their mean
        tg = np.linspace(0, x1, 400)
        Vs, Qs = [], []
        for k, e in exp.items():
            Vs.append(np.interp(tg, e[:, 0], e[:, 2], right=e[-1, 2]))
            ts, vl = e[:, 0], e[:, 2]
            tq = np.arange(0, ts[-1], 0.25)
            vq = np.interp(tq, ts, vl)
            q = -np.gradient(vq, tq)
            q = np.convolve(q, np.ones(5) / 5, mode="same")
            Qs.append(np.interp(tg, tq, np.maximum(q, 0), right=0.0))
        Vs, Qs = np.array(Vs), np.array(Qs)
        for a in (ax, ax2):
            for xa, xb, lab in ((x0, x0 + 4, "dip"), (x0 + 4, x0 + 8, "lift"),
                                (x0 + 8, 0.0, "hold")):
                a.axvspan(xa, xb, color="#f3f2ee", zorder=0, lw=0)
            a.axvline(0, color="#52514e", lw=0.8, ls=(0, (3, 3)))
            a.grid(color="#e8e7e2", lw=0.8)
            a.set_axisbelow(True)
            for s in ("top", "right"):
                a.spines[s].set_visible(False)
        for xa, xb, lab in ((x0, x0 + 4, "dip"), (x0 + 4, x0 + 8, "lift"),
                            (x0 + 8, 0.0, "hold"), (0.0, x1, "draining")):
            ax.text(0.5 * (xa + xb), 1.01, lab, transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=10, color="#52514e")
        ax.fill_between(tg, Vs.min(0), Vs.max(0), color=C_EXP, alpha=0.25, lw=0,
                        label="experiment, 4 runs (range)")
        ax.plot(tg, Vs.mean(0), color=C_EXP, lw=1.2, ls=(0, (4, 2)), label="experiment, mean")
        ax2.fill_between(tg, Qs.min(0), Qs.max(0), color=C_EXP, alpha=0.25, lw=0)
        ax2.plot(tg, Qs.mean(0), color=C_EXP, lw=1.2, ls=(0, (4, 2)))
        self.line, = ax.plot([], [], color=C_MODEL, lw=2.6, label="model", zorder=5)
        self.dot, = ax.plot([], [], "o", color=C_MODEL, ms=8, mec="white", mew=1.8, zorder=6)
        self.qline, = ax2.plot([], [], color=C_MODEL, lw=2.2, zorder=5)
        self.qdot, = ax2.plot([], [], "o", color=C_MODEL, ms=6, mec="white", mew=1.4, zorder=6)
        self.cur = [a.axvline(x0, color=C_MODEL, lw=0.8, alpha=0.4) for a in (ax, ax2)]
        ax.set_xlim(x0, x1)
        ax.set_ylim(0, 12.5)
        ax2.set_ylim(0, max(1.15, 1.1 * Qs.max()))
        ax.set_ylabel("water in the door (l)")
        ax2.set_ylabel("outflow (l/s)")
        ax2.set_xlabel("time since the plugs were pulled (s)")
        import matplotlib.pyplot as plt2
        plt2.setp(ax.get_xticklabels(), visible=False)
        ax.legend(frameon=False, loc="upper right", fontsize=10)
        ax.text(0.3, 0.35, "plugs pulled", fontsize=9, color="#52514e", va="bottom")
        self.fig.text(0.15, 0.955, "Door drainage: model vs experiment", fontsize=16,
                      color="#0b0b0b", weight="bold")
        self.fig.text(0.15, 0.925, "door on its hooks (5° lean), three 19 mm drain holes",
                      fontsize=11, color="#52514e")
        self.fig.text(0.15, 0.899, info_model, fontsize=11, color="#52514e")
        self.info = self.fig.text(0.15, 0.862, "", fontsize=12, color=C_MODEL,
                                  family="monospace")
        # 50 / 90 / 99 % drained: experiment (mean of the runs) and model
        fr = np.array([fractions(e[:, 0], e[:, 2]) for e in exp.values()])
        self.t_exp = np.nanmean(fr, axis=0)
        self.fig.text(0.15, 0.095, "drained               50 %     90 %     99 %",
                      fontsize=11, color="#0b0b0b", family="monospace")
        self.fig.text(0.15, 0.068, "experiment (mean)  " + "".join(
            f"{x:6.1f} s " for x in self.t_exp), fontsize=11, color="#52514e",
            family="monospace")
        self.mtxt = self.fig.text(0.15, 0.041, "", fontsize=11, color=C_MODEL,
                                  family="monospace")
        self.xs, self.ys, self.qs = [], [], []
        self.V0 = None
        self.t_mod = [None, None, None]
        self.marks = []

    def _milestones(self, x, y):
        if x < 0 or not self.xs:
            return
        if self.V0 is None:
            self.V0 = y
        for i, f in enumerate((0.5, 0.9, 0.99)):
            if self.t_mod[i] is None and y <= self.V0 * (1 - f):
                self.t_mod[i] = x
                m, = self.ax.plot([x], [y], "D", color=C_MODEL, ms=6, mec="white", zorder=7)
                self.ax.annotate(f"{int(f*100)} %", (x, y), xytext=(6, 6),
                                 textcoords="offset points", fontsize=9, color=C_MODEL)
        self.mtxt.set_text("model              " + "".join(
            (f"{v:6.1f} s " if v is not None else "     –   ") for v in self.t_mod))

    def render(self, x, y, q, info):
        self.xs.append(x)
        self.ys.append(y)
        self.qs.append(q if x >= 0 else np.nan)
        self._milestones(x, y)
        self.line.set_data(self.xs, self.ys)
        self.dot.set_data([x], [y])
        self.qline.set_data(self.xs, self.qs)
        self.qdot.set_data([x], [q if x >= 0 else np.nan])
        for c in self.cur:
            c.set_xdata([x, x])
        self.info.set_text(info)
        self.canvas.draw()
        a = np.asarray(self.canvas.buffer_rgba())
        return np.ascontiguousarray(a[:, :, :3])


# ------------------------------------------------------------------ driver
def render_frames(a, i0=None, i1=None, part=None):
    import trimesh
    size3d = tuple(int(v) // 2 * 2 for v in a.size3d)
    size2d = (int(a.width2d) // 2 * 2, size3d[1])
    t0 = time.time()
    t_open = 8.0 + a.t_settle
    t_end = t_open + a.t_after
    mesh, grid, motion = door_setup(a.stl, a.dx, 0.0, t_drain=a.t_settle + a.t_after,
                                    hang=a.hang)
    holes = [dict(h, open_at=t_open) for h in DOOR_HOLES]
    sim = Simulation(grid, motion, dt_max=0.05, holes=holes, subcells=a.subcells,
                     throat_model=ThroatModel(Cd=a.cd), threads=a.threads)
    hole_ids = [i for i, t in enumerate(sim.comp.throats) if t.axis is not None]
    hole_geo = [dict(center=sim.comp.throats[i].centroid,
                     diameter=sim.comp.throats[i].diameter, open_at=t_open)
                for i in hole_ids]
    print(f"setup {time.time()-t0:.0f} s, grid {grid.shape}, {sim.N/1e6:.2f} M nodes, "
          f"{len(hole_ids)} holes", flush=True)
    times = np.arange(0.0, t_end + 1e-9, 1.0 / a.fps)
    if a.preview:
        times = np.array(sorted(a.preview))
    if i0 is None:
        i0, i1 = 0, len(times)
    scene = DrainScene(sim, trimesh.load(a.stl, force="mesh"), hole_geo, size3d,
                       np.arange(0, t_end, 1 / a.fps), t_open, ssaa=a.ssaa,
                       door_opacity=a.door_opacity, edges=True, smooth=a.smooth,
                       contain=True)
    info_model = (f"drainsim: {a.dx*1e3:g} mm grid, {a.dx*1e3/a.subcells:g} mm sub-cells, "
                  f"Cd = {a.cd:g}")
    panel = DrainPanel(size2d, t_open, t_end, load_experiment(), info_model)
    W, Hh = size3d[0] + size2d[0], size3d[1]
    ff = None
    if not a.preview:
        ff = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{W}x{Hh}", "-r", str(a.fps), "-i", "-", "-c:v", "libx264",
             "-pix_fmt", "yuv420p", "-crf", str(a.crf), "-preset", a.preset,
             "-movflags", "+faststart", part or a.out], stdin=subprocess.PIPE)
    V_open = None
    out_cum = 0.0
    nrec = 1
    qmax = 1e-3 / len(hole_ids)
    tr = time.time()
    for i, t in enumerate(times[:i1]):
        if V_open is None and t > t_open + 1e-9 and sim.t < t_open - 1e-9:
            sim.run(t_end=t_open)
        if V_open is None and abs(sim.t - t_open) < 1e-9:
            V_open = sim.hist.liquid_retained[-1]
            nrec = len(sim.hist.t)
        if t > sim.t + 1e-9:
            sim.run(t_end=float(t))
        H = sim.hist
        tt = np.asarray(H.t)
        for k in range(nrec, len(tt)):
            if tt[k] > t_open + 1e-9:
                fl = np.abs(np.asarray(H.throat_flow[k])[hole_ids])
                out_cum += fl.sum() * (tt[k] - tt[k - 1])
        nrec = len(tt)
        flows = np.abs(np.asarray(H.throat_flow[-1])[hole_ids]) if len(tt) > 1 \
            else np.zeros(len(hole_ids))
        # before the opening: all liquid outside the bath (as V_open above and
        # door_drain_movie.py), not only what is held above the surface
        V = H.liquid_retained[-1] if V_open is None else V_open - out_cum
        phase = ("dip, plugs in" if t < 4 else "lift" if t < 8 else "hold" if t < t_open
                 else "plugs pulled: draining")
        label = f"t = {t:5.2f} s    {phase}"
        qtot = flows.sum() if t >= t_open else 0.0
        info = f"{V*1e3:5.2f} l in the door" + (f"   {qtot*1e3:4.2f} l/s out" if t >= t_open else "")
        if i < i0:
            # before this part: keep the curves (and the drop state) going
            panel.xs.append(t - t_open)
            panel.ys.append(V * 1e3)
            panel.qs.append(qtot * 1e3 if t >= t_open else np.nan)
            panel._milestones(t - t_open, V * 1e3)
            continue
        img3 = scene.render(t, label, flows, qmax)
        img2 = panel.render(t - t_open, V * 1e3, qtot * 1e3, info)
        frame = np.hstack([img3, img2[:Hh, :, :]])
        if a.preview:
            from PIL import Image
            p = a.out.replace(".mp4", "") + f"_t{t:05.2f}.png"
            Image.fromarray(frame).save(p)
            print("wrote", p, flush=True)
        else:
            ff.stdin.write(frame.tobytes())
        if (i - i0) % 48 == 0:
            print(f"[part {part and os.path.basename(part)}] frame {i}/{i1}  t = {t:.2f} s  "
                  f"V = {V*1e3:.2f} l  ({(time.time()-tr)/max(i-i0+1,1):.1f} s/frame)",
                  flush=True)
    if ff is not None:
        ff.stdin.close()
        ff.wait()
    print(f"done frames {i0}-{i1}: {(time.time()-t0)/60:.1f} min", flush=True)


def main(a):
    if a.preview or a.workers <= 1 and a.frames is None:
        render_frames(a)
        return
    if a.frames is not None:
        render_frames(a, a.frames[0], a.frames[1], a.part)
        return
    t_end = 8.0 + a.t_settle + a.t_after
    n = len(np.arange(0.0, t_end + 1e-9, 1.0 / a.fps))
    W = max(1, min(a.workers, n))
    cuts = np.linspace(0, n, W + 1).round().astype(int)
    out = os.path.abspath(a.out)
    pdir = out.replace(".mp4", "") + "_parts"
    os.makedirs(pdir, exist_ok=True)
    base = [sys.executable, "-X", "faulthandler", "-u", os.path.abspath(__file__)] + \
        [x for x in sys.argv[1:]]
    # drop --workers from the part command lines (--out stays: each part is
    # given its own --part, which render_frames prefers to --out)
    clean, skip = [], 0
    for x in base:
        if skip:
            skip -= 1
            continue
        if x in ("--workers",):
            skip = 1
            continue
        clean.append(x)
    procs, parts = [], []
    for w in range(W):
        part = os.path.join(pdir, f"part{w:02d}.mp4")
        parts.append(part)
        log = open(os.path.join(pdir, f"part{w:02d}.log"), "w")
        procs.append((subprocess.Popen(clean + ["--frames", str(cuts[w]), str(cuts[w + 1]),
                                                "--part", part],
                                       stdout=log, stderr=subprocess.STDOUT), log))
    print(f"rendering {n} frames in {W} processes (logs in {pdir})", flush=True)
    bad = False
    for p, log in procs:
        p.wait()
        log.close()
        bad |= p.returncode != 0
    if bad:
        print("a part failed; see the part logs", flush=True)
        sys.exit(1)
    lst = os.path.join(pdir, "parts.txt")
    with open(lst, "w") as f:
        for p in parts:
            f.write(f"file '{p}'\n")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
                    "-i", lst, "-c", "copy", "-movflags", "+faststart", out], check=True)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stl", required=True)
    ap.add_argument("--dx", type=float, default=0.006)
    ap.add_argument("--cd", type=float, default=0.63,
                    help="discharge coefficient (0.63 fits the experiment at 6 mm, 5° hang)")
    ap.add_argument("--subcells", type=int, default=4)
    ap.add_argument("--hang", type=float, default=5.0,
                    help="lean of the door on its hooks, degrees")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--t-settle", type=float, default=5.0)
    ap.add_argument("--t-after", type=float, default=25.0)
    ap.add_argument("--preview", type=float, nargs="*")
    ap.add_argument("--size3d", type=int, nargs=2, default=[1200, 1080], metavar=("W", "H"))
    ap.add_argument("--width2d", type=int, default=720)
    ap.add_argument("--ssaa", type=int, default=2)
    ap.add_argument("--crf", type=int, default=15)
    ap.add_argument("--preset", default="slow")
    ap.add_argument("--door-opacity", type=float, default=0.32)
    ap.add_argument("--smooth", type=int, default=30)
    ap.add_argument("--workers", type=int, default=1,
                    help="render processes; each re-runs the model up to its last frame "
                         "(it speeds up the drawing only, and uses N times the memory)")
    ap.add_argument("--frames", type=int, nargs=2, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--part", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--out", default="examples/out/door_drain/door_drain_hd.mp4")
    main(ap.parse_args())
