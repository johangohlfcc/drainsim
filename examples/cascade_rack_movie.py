"""Movie of the cascade rack: the previous method and the new model side by side.

Left rack: the previous method (equilibrium, escaped liquid is lost). Right
rack: the new model (spill routing plus orifice drainage). Both have the same
motion: dip upright, hold at 0° while B and C drain, tilt to 40°, hold.

Drawn:
- water in blue;
- jets through the holes, with width ~ sqrt(flow);
- spill streams where a cup overflows its rim (new model only; in the
  previous method that water simply disappears).

Right panel: water in each cup against time. Solid lines are the new model,
dashed lines the previous method.

    python examples/cascade_rack_movie.py --dx 0.01 --preview 6 20 39.5 41 45
    python examples/cascade_rack_movie.py --dx 0.005 --size3d 1600 1080 --width2d 1000 --out cascade.mp4
"""
from __future__ import annotations

import argparse
import os
import subprocess
import time
import warnings

import numpy as np

from cascade_rack import CUPS, D_Y, PLATE, cup_masks, cup_volumes, setup
from drainsim.worldviz import (_dilate_into_solid, _iso_obj, planar_fraction, trapped_display,
                               pose3, to_world)

warnings.filterwarnings("ignore", category=DeprecationWarning)
CUP_COL = {"A": "#2a78d6", "B": "#eb6834", "C": "#1baf7a"}
WATER = (0.10, 0.33, 0.78)


class RackScene:
    def __init__(self, mesh, sims, offsets, names, size, ssaa=2, follow=0.8,
                 t_ref=20.0, depth=1.5):
        import vtk
        from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray
        self.vtk, self.n2v, self.n2id = vtk, numpy_to_vtk, numpy_to_vtkIdTypeArray
        self.sims, self.offsets = sims, [np.array([o, 0, 0], float) for o in offsets]
        self.size, self.ssaa = tuple(size), max(1, int(ssaa))
        self.follow, self.t_ref = follow, t_ref
        W, H = self.size[0] * self.ssaa, self.size[1] * self.ssaa
        ren = vtk.vtkRenderer()
        ren.SetBackground(1, 1, 1)
        ren.SetBackground2(0.86, 0.89, 0.93)
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
        rack_pd = self._poly(mesh.vertices, mesh.faces, split=True)
        fe = vtk.vtkFeatureEdges()
        fe.SetInputData(rack_pd)
        fe.BoundaryEdgesOn()
        fe.FeatureEdgesOn()
        fe.SetFeatureAngle(40)
        fe.ManifoldEdgesOff()
        fe.NonManifoldEdgesOff()
        fe.ColoringOff()
        fe.Update()
        self.racks, self.water, self.labels, self.jets, self.streams = [], [], [], [], []
        for k, sim in enumerate(sims):
            a = self._actor(rack_pd, (0.74, 0.76, 0.80), 0.30, specular=0.3)
            e = self._actor(fe.GetOutput(), (0.25, 0.27, 0.32), 0.9, specular=0.0)
            e.GetProperty().SetLineWidth(1.3 * self.ssaa)
            e.GetProperty().LightingOff()
            wp = vtk.vtkPolyData()
            wa = self._actor(wp, WATER, 0.95, specular=0.4)
            for x in (a, e, wa):
                ren.AddActor(x)
            self.racks.append((a, e))
            self.water.append(wp)
            lab = vtk.vtkBillboardTextActor3D()
            lab.SetInput(names[k])
            tp = lab.GetTextProperty()
            tp.SetFontSize(int(18 * self.size[1] / 720 * self.ssaa))
            tp.SetColor(0.1, 0.1, 0.1)
            tp.SetJustificationToCentered()
            ren.AddActor(lab)
            self.labels.append(lab)
            self.jets.append([self._cyl() for _ in sim.comp.throats if _.axis is not None])
            self.streams.append({n: self._cyl() for n in CUPS})
        # bath
        lo = np.array([min(offsets) - 0.8, -0.8, -depth - 1.5])
        hi = np.array([max(offsets) + 0.8, 0.8, 0.0])
        cube = vtk.vtkCubeSource()
        cube.SetBounds(lo[0], hi[0], lo[1], hi[1], lo[2], hi[2])
        cube.Update()
        ren.AddActor(self._actor(cube.GetOutput(), (0.55, 0.74, 0.93), 0.20, specular=0.0))
        # camera on both racks at rest, plus the bath surface
        box = np.array([[min(offsets) + PLATE[0] - 0.05, max(offsets) + PLATE[1] + 0.05],
                        [-D_Y, D_Y], [-0.15, 0.0]])
        rest = to_world(np.array([[0, 0, PLATE[3]]]), sims[0].motion, t_ref)[0, 2]
        box[2, 1] = rest + 0.30
        ren.ResetCamera(*box.ravel())
        cam = ren.GetActiveCamera()
        cam.SetViewUp(0, 0, 1)
        ctr = box.mean(1)
        cam.SetFocalPoint(*ctr)
        cam.SetPosition(ctr[0], ctr[1] - 1.0, ctr[2])
        cam.Azimuth(-18)
        cam.Elevation(14)
        ren.ResetCamera(*box.ravel())
        cam.Zoom(1.25)
        self._cam0 = (np.array(cam.GetFocalPoint()), np.array(cam.GetPosition()))
        self.txt = vtk.vtkTextActor()
        tp = self.txt.GetTextProperty()
        tp.SetFontSize(int(22 * self.size[1] / 720 * self.ssaa))
        tp.SetColor(0.1, 0.1, 0.1)
        self.txt.SetPosition(18 * self.ssaa, 18 * self.ssaa)
        ren.AddViewProp(self.txt)
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

    def _cyl(self):
        vtk = self.vtk
        c = vtk.vtkCylinderSource()
        c.SetResolution(16)
        c.SetHeight(1.0)
        c.SetRadius(1.0)
        c.Update()
        a = self._actor(c.GetOutput(), WATER, 0.85, specular=0.4)
        a.VisibilityOff()
        self.ren.AddActor(a)
        return a

    def _smooth(self, pts, faces, iters=20):
        vtk = self.vtk
        sm = vtk.vtkWindowedSincPolyDataFilter()
        sm.SetInputData(self._poly(pts, faces, normals=False))
        sm.SetNumberOfIterations(iters)
        sm.SetPassBand(0.05)
        sm.NonManifoldSmoothingOn()
        sm.NormalizeCoordinatesOn()
        sm.Update()
        from vtk.util.numpy_support import vtk_to_numpy
        return vtk_to_numpy(sm.GetOutput().GetPoints().GetData()).astype(float)

    @staticmethod
    def fall(sim, x_obj, t):
        """Free-fall distance from x_obj (object frame) to a solid cell or the
        bath, marching down through the voxel grid."""
        g = sim.grid
        up, zb = sim.motion.frame(t)
        up = np.asarray(up, float)
        to_bath = float(x_obj @ up - zb)
        if to_bath <= 0:
            return 0.0
        s = np.arange(g.dx, to_bath, 0.5 * g.dx)
        if s.size == 0:
            return to_bath
        P = x_obj[None] - s[:, None] * up[None]
        ci = np.floor((P - g.origin) / g.dx).astype(np.int64)
        ins = np.all((ci >= 0) & (ci < np.array(g.shape)), axis=1)
        hit = np.zeros(len(s), bool)
        i = np.flatnonzero(ins)
        hit[i] = g.solid[ci[i, 0], ci[i, 1], ci[i, 2]]
        k = np.flatnonzero(hit)
        return float(s[k[0]]) if k.size else to_bath

    def _column(self, actor, top, length, rate, rmax=0.012, qref=1e-3):
        if rate <= 1e-6 or length <= 0.01:
            actor.VisibilityOff()
            return
        r = max(0.0025, rmax * np.sqrt(min(rate / qref, 2.0)))
        tr = self.vtk.vtkTransform()
        tr.Translate(top[0], top[1], top[2] - length / 2)
        tr.RotateX(90)
        tr.Scale(r, length, r)
        actor.SetUserTransform(tr)
        actor.VisibilityOn()

    # ------------------------------------------------------------- frame
    def render(self, t, label, flows, spills):
        """flows[k]: hole flow rates (m3/s) per sim; spills[k]: dict cup ->
        overflow rate (m3/s) over its rim, or None (no streams)."""
        vtk = self.vtk
        for k, sim in enumerate(self.sims):
            off = self.offsets[k]
            R, T, p = pose3(sim.motion, t)
            M = np.eye(4)
            M[:3, :3] = R
            M[:3, 3] = p + T - R @ p + off
            vm = vtk.vtkMatrix4x4()
            for i in range(4):
                for j in range(4):
                    vm.SetElement(i, j, M[i, j])
            for a in self.racks[k]:
                a.SetUserMatrix(vm)
            up, zb = sim.motion.frame(t)
            h = sim.X @ up
            sim._vis_e = 0.5 * sim.grid.dx * np.abs(up).sum()
            fld = trapped_display(sim, h, zb)[0]    # trapped liquid, above the surface
            pts, faces, _ = _iso_obj(fld, sim.grid)       # stays inside the cups
            if len(pts):
                pts = self._smooth(pts, faces)
                self.water[k].DeepCopy(self._poly(to_world(pts, sim.motion, t) + off, faces))
            else:
                self.water[k].DeepCopy(vtk.vtkPolyData())
            # label at a fixed height above the rack, following only the dip
            c = to_world(np.array([[0.5 * (PLATE[0] + PLATE[1]), 0, 0.5 * (PLATE[2] + PLATE[3])]]),
                         sim.motion, t)[0] + off
            self.labels[k].SetPosition(c[0], c[1], c[2] + 0.62)
            # jets through the holes
            holes = [th for th in sim.comp.throats if th.axis is not None]
            for j, th in enumerate(holes):
                x = np.asarray(th.centroid, float) - 0.5 * sim.grid.dx * np.asarray(up)
                w = to_world(x[None], sim.motion, t)[0] + off
                q = flows[k][j] if flows[k] is not None else 0.0
                self._column(self.jets[k][j], w, self.fall(sim, x, t), q)
            # spill streams from the lowest rim edge of each cup
            for name, (x0, x1, zb_c, hc, hx, hd) in CUPS.items():
                st = self.streams[k][name]
                q = spills[k][name] if spills[k] is not None else 0.0
                if q <= 1e-6:
                    st.VisibilityOff()
                    continue
                zt = zb_c + hc
                edges = [((x1 + 0.012, 0, zt)), ((x0 - 0.012, 0, zt)),
                         ((0.5 * (x0 + x1), D_Y + 0.012, zt)),
                         ((0.5 * (x0 + x1), -D_Y - 0.012, zt))]
                E = np.array(edges)
                hE = E @ np.asarray(up)
                e = E[int(np.argmin(hE))]
                w = to_world(e[None], sim.motion, t)[0] + off
                self._column(st, w, self.fall(sim, e, t), q)
        # camera follows the dip partly
        if self.follow:
            dz = pose3(self.sims[0].motion, t)[1][2] - pose3(self.sims[0].motion, self.t_ref)[1][2]
            f0, p0 = self._cam0
            o = np.array([0, 0, self.follow * dz])
            cam = self.ren.GetActiveCamera()
            cam.SetFocalPoint(*(f0 + o))
            cam.SetPosition(*(p0 + o))
            self.ren.ResetCameraClippingRange()
        self.txt.SetInput(label)
        self.win.Render()
        self.w2i.Modified()
        self.w2i.Update()
        from vtk.util.numpy_support import vtk_to_numpy
        img = self.w2i.GetOutput()
        w_, hh, _ = img.GetDimensions()
        a = vtk_to_numpy(img.GetPointData().GetScalars()).reshape(hh, w_, -1)
        a = np.ascontiguousarray(a[::-1, :, :3])
        if self.ssaa > 1:
            from PIL import Image
            a = np.asarray(Image.fromarray(a).resize(self.size, Image.LANCZOS))
        return a


class RackPanel:
    def __init__(self, size, mo):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        dpi = 100 * size[1] / 720
        self.fig = plt.figure(figsize=(size[0] / dpi, size[1] / dpi), dpi=dpi)
        self.canvas = FigureCanvasAgg(self.fig)
        ax = self.fig.add_axes([0.13, 0.12, 0.82, 0.72])
        self.ax = ax
        T = mo.times
        for a, b, lab in ((0, T[2], "dip"), (T[3], T[4], "tilt")):
            ax.axvspan(a, b, color="#f1f0ec", lw=0, zorder=0)
            ax.text(0.5 * (a + b), 1.0, lab, transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=9, color="#52514e")
        self.lines = {}
        for n, c in CUP_COL.items():
            self.lines[("new", n)], = ax.plot([], [], color=c, lw=2.4, label=f"cup {n}")
            self.lines[("old", n)], = ax.plot([], [], color=c, lw=1.5, ls=(0, (4, 3)))
        ax.plot([], [], color="#52514e", lw=2.4, label="new model")
        ax.plot([], [], color="#52514e", lw=1.5, ls=(0, (4, 3)), label="previous method")
        self.cur = ax.axvline(0, color="#52514e", lw=0.8, alpha=0.6)
        ax.set_xlim(0, mo.t_end)
        ax.set_ylim(0, 6.5)
        ax.grid(color="#e6e5e0", lw=0.8)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.set_xlabel("time (s)")
        ax.set_ylabel("water in the cup (l)")
        ax.legend(frameon=False, fontsize=9.5, ncol=2, loc="upper right")
        self.fig.text(0.13, 0.95, "Cascade rack: where does the escaping water go?",
                      fontsize=13, color="#0b0b0b")
        self.info = self.fig.text(0.13, 0.905, "", fontsize=10.5, color="#52514e")
        self.data = {k: [] for k in self.lines}
        self.t = []

    def render(self, t, vols, info):
        self.t.append(t)
        for (m, n), ln in self.lines.items():
            self.data[(m, n)].append(vols[m][n] * 1e3)
            ln.set_data(self.t, self.data[(m, n)])
        self.cur.set_xdata([t, t])
        self.info.set_text(info)
        self.canvas.draw()
        return np.ascontiguousarray(np.asarray(self.canvas.buffer_rgba())[:, :, :3])


def main(out, dx=0.01, tilt=40.0, fps=24, size3d=(1040, 720), size2d=(720, 720),
         ssaa=2, crf=16, preset="slow", preview=None):
    t0 = time.time()
    size3d = tuple(int(v) // 2 * 2 for v in size3d)
    size2d = (int(size2d[0]) // 2 * 2, size3d[1])
    mesh, grid, mo, new, old = setup(dx, tilt)
    sims = [old, new]
    masks = {"old": cup_masks(old), "new": cup_masks(new)}
    holes = [i for i, th in enumerate(new.comp.throats) if th.axis is not None]
    print(f"setup {time.time()-t0:.0f} s, grid {grid.shape}", flush=True)
    scene = RackScene(mesh, sims, offsets=(-0.75, 0.75),
                      names=("previous method: escaped water is lost",
                             "new: spill routing to the next cup"),
                      size=size3d, ssaa=ssaa, t_ref=mo.times[3])
    panel = RackPanel(size2d, mo)
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
    prev = None
    ema = {n: 0.0 for n in CUPS}
    t_sim = t_draw = 0.0
    T = mo.times
    for i, t in enumerate(times):
        a = time.time()
        for s in sims:
            if t > s.t + 1e-9:
                s.run(t_end=float(t))
        t_sim += time.time() - a
        vols = {"old": cup_volumes(old, masks["old"]), "new": cup_volumes(new, masks["new"])}
        q = (np.abs(np.asarray(new.hist.throat_flow[-1])[holes])
             if len(new.hist.t) > 1 else np.zeros(len(holes)))
        # overflow over each rim of the new model, from the volume balance
        # between frames: in - out_hole - dV/dt (A has no inflow; A -> B -> C)
        spill = {n: 0.0 for n in CUPS}
        submerged = t < T[2] + 0.5
        if prev is not None and not submerged:
            dt = t - prev[0]
            dV = {n: (vols["new"][n] - prev[1][n]) / dt for n in CUPS}
            qB, qC = q[0], q[1]
            sA = max(-dV["A"], 0.0)
            sB = max(sA - qB - dV["B"], 0.0)
            sC = max(sB + qB - qC - dV["C"], 0.0)
            for n, v in zip(CUPS, (sA, sB, sC)):
                ema[n] = 0.6 * ema[n] + 0.4 * v
                spill[n] = ema[n] if ema[n] > 2e-5 else 0.0
        prev = (t, dict(vols["new"]))
        phase = ("dip" if t < T[2] else "holding at 0°: B and C drain" if t < T[3]
                 else "tilting to 40°" if t < T[4] else "holding at 40°")
        a = time.time()
        img3 = scene.render(t, f"t = {t:5.2f} s   {phase}",
                            [None, q if not submerged else None], [None, spill])
        info = "new model:  " + ",  ".join(f"{n} {vols['new'][n]*1e3:4.2f} l" for n in CUPS)
        img2 = panel.render(t, vols, info)
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
            print(f"frame {i}/{len(times)}  t = {t:.2f}  sim {t_sim:.0f} s, draw {t_draw:.0f} s",
                  flush=True)
    if ff is not None:
        ff.stdin.close()
        ff.wait()
        print("wrote", out)
    print(f"total {time.time()-t0:.0f} s: simulation {t_sim:.0f} s, drawing {t_draw:.0f} s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dx", type=float, default=0.01)
    ap.add_argument("--tilt", type=float, default=40.0)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--size3d", type=int, nargs=2, default=[1040, 720])
    ap.add_argument("--width2d", type=int, default=720)
    ap.add_argument("--ssaa", type=int, default=2)
    ap.add_argument("--crf", type=int, default=16)
    ap.add_argument("--preset", default="slow")
    ap.add_argument("--preview", type=float, nargs="*")
    ap.add_argument("--out", default="examples/out/cascade_rack/cascade_rack.mp4")
    a = ap.parse_args()
    main(a.out, a.dx, a.tilt, a.fps, tuple(a.size3d), (a.width2d, a.size3d[1]),
         a.ssaa, a.crf, a.preset, a.preview)
