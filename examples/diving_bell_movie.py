"""Presentation case: compressible air pockets, a diving bell lowered to 10 m.

Two identical inverted cups (inner radius 10 cm, inner height 30 cm) are
lowered from the surface to 10 m and raised again (1 m/s, 2 s pauses):

- **left**: the previous model, where trapped air keeps its volume;
- **right**: compressible air (``Simulation(..., compressible_air=True)``).
  The pocket follows Boyle's law with the hydrostatic pressure at its water
  surface, so at 10 m it is roughly half the size.

Left of the movie is the 3D view: glass cups, the air pockets in white, the
bath in blue, a depth pole with 1 m marks, and a camera that follows the
cups. On the right, the pocket volume is plotted against depth, with the
exact Boyle curve.

    python examples/diving_bell_movie.py --out diving_bell.mp4
    python examples/diving_bell_movie.py --preview 0.5 6 12 --out bell.mp4

No customer geometry is involved: the cups are generated here.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import time
import warnings

import numpy as np

from drainsim.grid import Grid
from drainsim.model import Simulation
from drainsim.motion import Keyframes
from drainsim.physics import Fluid
from drainsim.worldviz import _dilate_into_solid, _iso_obj, planar_fraction, to_world

warnings.filterwarnings("ignore", category=DeprecationWarning)
C_NEW, C_OLD, C_EXACT = "#2a78d6", "#eb6834", "#52514e"
RI, HC, WALL = 0.10, 0.30, 0.004
OFFSET = 0.24                     # x offset of each cup in the picture (m)


def bell_mesh(Ri=RI, Hc=HC, t=WALL):
    import trimesh
    tube = trimesh.creation.annulus(r_min=Ri, r_max=Ri + t, height=Hc + t, sections=96)
    tube.apply_translation([0, 0, (Hc + t) / 2])
    cap = trimesh.creation.cylinder(radius=Ri + t, height=t, sections=96)
    cap.apply_translation([0, 0, Hc + t / 2])
    return trimesh.util.concatenate([tube, cap])


def boyle_fraction(D, Hc=HC, fp=Fluid()):
    """Exact air fraction ha/Hc of a cylindrical bell with its rim at depth D."""
    D = np.asarray(D, float)
    Dtop = D - Hc
    a, b, c = fp.rho * fp.g, fp.p_atm + fp.rho * fp.g * Dtop, -Hc * fp.p_atm
    ha = (-b + np.sqrt(b * b - 4 * a * c)) / (2 * a)
    return np.where(D <= 0, 1.0, ha / Hc)


def motion(depth=10.0, speed=1.0, pause=2.0, start=0.05):
    tdn = (depth + start) / speed
    T = [0.0, 0.5, 0.5 + tdn, 0.5 + tdn + pause, 0.5 + 2 * tdn + pause,
         1.0 + 2 * tdn + 2 * pause]
    z = [start, start, -depth, -depth, start, start]
    return Keyframes(T, np.zeros((len(T), 3)), np.array([[0, 0, q] for q in z]),
                     ndim=3, pivot=np.zeros(3))


class BellScene:
    def __init__(self, mesh, sims, size, ssaa=2, depth=10.0):
        import vtk
        from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray
        self.vtk, self.n2v, self.n2id = vtk, numpy_to_vtk, numpy_to_vtkIdTypeArray
        self.sims = sims
        self.size, self.ssaa = tuple(size), max(1, int(ssaa))
        W, H = self.size[0] * self.ssaa, self.size[1] * self.ssaa
        ren = vtk.vtkRenderer()
        ren.SetBackground(0.97, 0.98, 1.0)
        ren.SetBackground2(0.80, 0.86, 0.93)
        ren.GradientBackgroundOn()
        ren.SetUseDepthPeeling(1)
        ren.SetMaximumNumberOfPeels(8)
        win = vtk.vtkRenderWindow()
        win.SetOffScreenRendering(1)
        win.SetSize(W, H)
        win.SetAlphaBitPlanes(1)
        win.SetMultiSamples(0)
        win.AddRenderer(ren)
        self.ren, self.win = ren, win
        # cups (glass) with outline edges
        self.cups, self.pockets, self.labels = [], [], []
        cup_pd = self._poly(mesh.vertices, mesh.faces, split=True)
        fe = vtk.vtkFeatureEdges()
        fe.SetInputData(cup_pd)
        fe.BoundaryEdgesOn()
        fe.FeatureEdgesOn()
        fe.SetFeatureAngle(40)
        fe.ManifoldEdgesOff()
        fe.NonManifoldEdgesOff()
        fe.ColoringOff()
        fe.Update()
        names = ("previous model: air keeps its volume", "new: compressible air (Boyle)")
        for k, _ in enumerate(sims):
            a = self._actor(cup_pd, (0.80, 0.86, 0.92), 0.22, specular=0.6)
            e = self._actor(fe.GetOutput(), (0.30, 0.35, 0.42), 0.8, specular=0.0)
            e.GetProperty().SetLineWidth(1.2 * self.ssaa)
            p = vtk.vtkPolyData()
            pa = self._actor(p, (0.96, 0.97, 1.0), 0.9, specular=0.5)
            pa.GetProperty().SetSpecularPower(40)
            for x in (a, e, pa):
                ren.AddActor(x)
            lab = vtk.vtkBillboardTextActor3D()
            lab.SetInput(names[k])
            tp = lab.GetTextProperty()
            tp.SetFontSize(int(17 * self.size[1] / 720 * self.ssaa))
            tp.SetColor(0.1, 0.1, 0.1)
            tp.SetJustificationToCentered()
            ren.AddActor(lab)
            self.cups.append((a, e))
            self.pockets.append((p, pa))
            self.labels.append(lab)
        # bath: translucent box and a slightly stronger surface
        self._box((-1.3, 1.3, -1.2, 1.4, -depth - 2.0, 0.0), (0.50, 0.72, 0.93), 0.12)
        self._box((-1.3, 1.3, -1.2, 1.4, -0.002, 0.0), (0.60, 0.80, 0.97), 0.15)
        # depth pole with 1 m marks
        pole = vtk.vtkCylinderSource()
        pole.SetRadius(0.012)
        pole.SetHeight(depth + 1.5)
        pole.SetResolution(24)
        pole.Update()
        tr = vtk.vtkTransform()
        tr.Translate(0.0, 0.35, -(depth + 1.5) / 2 + 0.3)
        tr.RotateX(90)
        pa = self._actor(pole.GetOutput(), (0.35, 0.35, 0.38), 1.0)
        pa.SetUserTransform(tr)
        ren.AddActor(pa)
        for d in range(0, int(depth) + 2):
            ring = vtk.vtkCylinderSource()
            ring.SetRadius(0.03)
            ring.SetHeight(0.012)
            ring.SetResolution(24)
            ring.Update()
            t2 = vtk.vtkTransform()
            t2.Translate(0.0, 0.35, -float(d))
            t2.RotateX(90)
            ra = self._actor(ring.GetOutput(), (0.85, 0.25, 0.20), 1.0)
            ra.SetUserTransform(t2)
            ren.AddActor(ra)
            if d > 0:
                tx = vtk.vtkBillboardTextActor3D()
                tx.SetInput(f"{d} m")
                tx.SetPosition(0.06, 0.35, -float(d))
                tp = tx.GetTextProperty()
                tp.SetFontSize(int(16 * self.size[1] / 720 * self.ssaa))
                tp.SetColor(0.15, 0.15, 0.2)
                ren.AddActor(tx)
        self.txt = vtk.vtkTextActor()
        tp = self.txt.GetTextProperty()
        tp.SetFontSize(int(22 * self.size[1] / 720 * self.ssaa))
        tp.SetColor(0.1, 0.1, 0.1)
        self.txt.SetPosition(18 * self.ssaa, 18 * self.ssaa)
        ren.AddViewProp(self.txt)
        self.cam = ren.GetActiveCamera()
        self.cam.SetViewUp(0, 0, 1)
        self.cam.SetViewAngle(30)
        self.w2i = vtk.vtkWindowToImageFilter()
        self.w2i.SetInput(win)
        self.w2i.SetInputBufferTypeToRGB()
        self.w2i.ReadFrontBufferOff()

    # ------------------------------------------------------------ helpers
    def _poly(self, pts, faces, split=False, normals=True):
        vtk = self.vtk
        p = vtk.vtkPoints()
        p.SetData(self.n2v(np.ascontiguousarray(pts, float), deep=True))
        F = np.asarray(faces, np.int64)
        ca = vtk.vtkCellArray()
        off = np.arange(0, F.shape[1] * len(F) + 1, F.shape[1], dtype=np.int64)
        ca.SetData(self.n2id(off, deep=True), self.n2id(np.ascontiguousarray(F.ravel()), deep=True))
        pd = vtk.vtkPolyData()
        pd.SetPoints(p)
        pd.SetPolys(ca)
        if not normals:
            return pd
        nf = vtk.vtkPolyDataNormals()
        nf.SetInputData(pd)
        if split:
            nf.SplittingOn()
            nf.SetFeatureAngle(30)
        else:
            nf.SplittingOff()
        nf.ConsistencyOn()
        nf.Update()
        return nf.GetOutput()

    def _actor(self, pd, color, opacity, specular=0.2):
        vtk = self.vtk
        m = vtk.vtkPolyDataMapper()
        m.SetInputData(pd)
        m.ScalarVisibilityOff()
        a = vtk.vtkActor()
        a.SetMapper(m)
        pr = a.GetProperty()
        pr.SetColor(*color)
        pr.SetOpacity(opacity)
        pr.SetSpecular(specular)
        pr.SetSpecularPower(20)
        return a

    def _box(self, b, color, opacity):
        vtk = self.vtk
        cube = vtk.vtkCubeSource()
        cube.SetBounds(*b)
        cube.Update()
        a = self._actor(cube.GetOutput(), color, opacity, specular=0.0)
        self.ren.AddActor(a)

    def _smooth(self, pts, faces, iters=20):
        vtk = self.vtk
        pd = self._poly(pts, faces, normals=False)
        sm = vtk.vtkWindowedSincPolyDataFilter()
        sm.SetInputData(pd)
        sm.SetNumberOfIterations(iters)
        sm.SetPassBand(0.05)
        sm.NonManifoldSmoothingOn()
        sm.NormalizeCoordinatesOn()
        sm.Update()
        from vtk.util.numpy_support import vtk_to_numpy
        return vtk_to_numpy(sm.GetOutput().GetPoints().GetData()).astype(float)

    # ------------------------------------------------------------- frame
    def render(self, t, label):
        vtk = self.vtk
        zc = None
        for k, sim in enumerate(self.sims):
            dxk = (2 * k - 1) * OFFSET
            vm = vtk.vtkMatrix4x4()
            from drainsim.worldviz import pose3
            R, T, p = pose3(sim.motion, t)
            M = np.eye(4)
            M[:3, :3] = R
            M[:3, 3] = p + T - R @ p + np.array([dxk, 0, 0])
            for i in range(4):
                for j in range(4):
                    vm.SetElement(i, j, M[i, j])
            for a in self.cups[k]:
                a.SetUserMatrix(vm)
            # air pocket (trapped air, not connected to the atmosphere)
            up, _ = sim.motion.frame(t)
            h = sim.X @ up
            sim._vis_e = 0.5 * sim.grid.dx * np.abs(up).sum()
            air = np.where(sim.fl & ~sim.A, 1.0 - sim.L, 0.0)
            fld = planar_fraction(sim, air, -h)
            pts, faces, _ = _iso_obj(_dilate_into_solid(fld, sim.grid), sim.grid)
            pd, _ = self.pockets[k]
            if len(pts):
                pts = self._smooth(pts, faces)
                W = to_world(pts, sim.motion, t) + np.array([dxk, 0, 0])
                pd.DeepCopy(self._poly(W, faces))
            else:
                pd.DeepCopy(vtk.vtkPolyData())
            c = to_world(np.array([[0, 0, HC]]), sim.motion, t)[0]
            self.labels[k].SetPosition(c[0] + dxk, c[1], c[2] + 0.09)
            zc = c[2] - HC / 2
        # camera follows the cups
        f = np.array([0.0, 0.0, zc + 0.02])
        self.cam.SetFocalPoint(*f)
        self.cam.SetPosition(*(f + np.array([0.0, -1.55, 0.30])))
        self.cam.SetClippingRange(0.2, 20.0)
        self.txt.SetInput(label)
        self.win.Render()
        self.w2i.Modified()
        self.w2i.Update()
        from vtk.util.numpy_support import vtk_to_numpy
        img = self.w2i.GetOutput()
        w, hh, _ = img.GetDimensions()
        a = vtk_to_numpy(img.GetPointData().GetScalars()).reshape(hh, w, -1)
        a = np.ascontiguousarray(a[::-1, :, :3])
        if self.ssaa > 1:
            from PIL import Image
            a = np.asarray(Image.fromarray(a).resize(self.size, Image.LANCZOS))
        return a


class BellPanel:
    def __init__(self, size, depth):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        dpi = 100 * size[1] / 720
        self.fig = plt.figure(figsize=(size[0] / dpi, size[1] / dpi), dpi=dpi)
        self.canvas = FigureCanvasAgg(self.fig)
        ax = self.fig.add_axes([0.14, 0.12, 0.80, 0.72])
        self.ax = ax
        ax.axvspan(2.0, 3.0, color="#f1f0ec", lw=0, zorder=0)
        ax.text(2.5, 0.43, "typical\nED tank\ndepth", ha="center", va="bottom",
                fontsize=9, color="#52514e")
        D = np.linspace(0, depth, 400)
        ax.plot(D, boyle_fraction(D), color=C_EXACT, lw=1.2, ls=(0, (4, 3)),
                label="exact (Boyle)")
        self.l_old, = ax.plot([], [], color=C_OLD, lw=2.4, label="previous model")
        self.l_new, = ax.plot([], [], color=C_NEW, lw=2.4, label="new: compressible air")
        self.d_old, = ax.plot([], [], "o", color=C_OLD, ms=8, mec="white", mew=1.5)
        self.d_new, = ax.plot([], [], "o", color=C_NEW, ms=8, mec="white", mew=1.5)
        ax.set_xlim(0, depth)
        ax.set_ylim(0.4, 1.06)
        ax.grid(color="#e6e5e0", lw=0.8)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.set_xlabel("depth of the cup rim (m)")
        ax.set_ylabel("trapped air volume / volume at the surface")
        ax.legend(frameon=False, loc="upper right", fontsize=10)
        self.fig.text(0.14, 0.95, "Diving bell: trapped air vs depth", fontsize=13,
                      color="#0b0b0b")
        self.info = self.fig.text(0.14, 0.905, "", fontsize=10.5, color="#52514e")
        self.D, self.old, self.new = [], [], []

    def render(self, D, f_old, f_new, info):
        if f_old is not None:
            self.D.append(D)
            self.old.append(f_old)
            self.new.append(f_new)
            self.l_old.set_data(self.D, self.old)
            self.l_new.set_data(self.D, self.new)
            self.d_old.set_data([D], [f_old])
            self.d_new.set_data([D], [f_new])
        self.info.set_text(info)
        self.canvas.draw()
        return np.ascontiguousarray(np.asarray(self.canvas.buffer_rgba())[:, :, :3])


def main(out, dx=0.008, depth=10.0, fps=24, size3d=(1040, 720), size2d=(720, 720),
         ssaa=2, crf=16, preset="slow", preview=None):
    t0 = time.time()
    size3d = tuple(int(v) // 2 * 2 for v in size3d)
    size2d = (int(size2d[0]) // 2 * 2, size3d[1])
    mesh = bell_mesh()
    grid = Grid.from_mesh(mesh, dx=dx, pad=3 * dx)
    mo = motion(depth)
    sims = [Simulation(grid.copy(), mo, dt_max=0.05, split=False, compressible_air=c)
            for c in (False, True)]
    print(f"setup {time.time()-t0:.0f} s, grid {grid.shape}", flush=True)
    scene = BellScene(mesh, sims, size3d, ssaa=ssaa, depth=depth)
    panel = BellPanel(size2d, depth)
    times = np.arange(0.0, mo.t_end + 1e-9, 1.0 / fps)
    if preview:
        times = np.array(sorted(preview))
    W, Hh = size3d[0] + size2d[0], size3d[1]
    ff = None
    if not preview:
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        ff = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{W}x{Hh}", "-r", str(fps), "-i", "-", "-c:v", "libx264",
             "-pix_fmt", "yuv420p", "-crf", str(crf), "-preset", preset,
             "-movflags", "+faststart", out], stdin=subprocess.PIPE)
    n0 = [None, None]
    t_sim = t_draw = 0.0
    for i, t in enumerate(times):
        a = time.time()
        for s in sims:
            if t > s.t + 1e-9:
                s.run(t_end=float(t))
        t_sim += time.time() - a
        rim = to_world(np.zeros((1, 3)), mo, t)[0, 2]
        D = max(-rim, 0.0)
        fr = []
        for k, s in enumerate(sims):
            V = s.hist.air_trapped[-1]
            gas = s.hist.air_gas[-1]
            if D > HC + 0.02 and n0[k] is None:
                n0[k] = gas                      # cup fully under: charge known
            # plot only while the whole cup is under water (then all its air
            # is a trapped pocket); near the surface it vents and emerges
            full = n0[k] is not None and D > HC + 0.02
            fr.append(V / n0[k] if full else None)
        T = mo.times
        phase = ("at the surface" if D <= 0 else "lowering" if t < T[2] else
                 "holding at depth" if t < T[3] else "raising")
        a = time.time()
        img3 = scene.render(t, f"t = {t:5.2f} s   rim depth {D:5.2f} m   {phase}")
        p = sims[1].P[sims[1].fl & ~sims[1].A & (sims[1].L < 0.5)]
        pk = (p.max() / 1e5) if p.size else Fluid().p_atm / 1e5
        info = (f"rim depth {D:4.2f} m,  pocket pressure {pk:4.2f} bar,  new model: "
                + (f"{fr[1]*100:5.1f} % of the surface volume" if fr[1] is not None else "—"))
        img2 = (panel.render(D, fr[0], fr[1], info) if fr[0] is not None
                else panel.render(D, None, None, info))
        frame = np.hstack([img3, img2[:Hh]])
        t_draw += time.time() - a
        if preview:
            from PIL import Image
            pth = out.replace(".mp4", "") + f"_t{t:05.2f}.png"
            Image.fromarray(frame).save(pth)
            print("wrote", pth, flush=True)
        else:
            ff.stdin.write(frame.tobytes())
        if i % 48 == 0:
            print(f"frame {i}/{len(times)}  t = {t:.2f}  depth {D:.2f}  "
                  f"sim {t_sim:.0f} s, draw {t_draw:.0f} s", flush=True)
    if ff is not None:
        ff.stdin.close()
        ff.wait()
        print("wrote", out)
    print(f"total {time.time()-t0:.0f} s: simulation {t_sim:.0f} s, drawing {t_draw:.0f} s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dx", type=float, default=0.008)
    ap.add_argument("--depth", type=float, default=10.0)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--size3d", type=int, nargs=2, default=[1040, 720])
    ap.add_argument("--width2d", type=int, default=720)
    ap.add_argument("--ssaa", type=int, default=2)
    ap.add_argument("--crf", type=int, default=16)
    ap.add_argument("--preset", default="slow")
    ap.add_argument("--preview", type=float, nargs="*")
    ap.add_argument("--out", default="examples/out/diving_bell/diving_bell.mp4")
    a = ap.parse_args()
    main(a.out, a.dx, a.depth, a.fps, tuple(a.size3d), (a.width2d, a.size3d[1]),
         a.ssaa, a.crf, a.preset, a.preview)
