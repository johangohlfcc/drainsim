"""Real-time movie of the plugged-hole drainage case, next to the measurements.

Left: world-frame 3D view. The drawn water is kept inside the real
geometry: vertices of its (smoothed) surface that poke through a wall are
moved just inside it, using the STL surface and its normals (display only). The door (semi-transparent) is dipped with its
three drain holes plugged (red), lifted, held, and the plugs are pulled
(jets drawn below the holes, width ~ sqrt of the flow). Water held by the
door is blue; the bath is the pale box.

Right: water in the door vs time since the plugs were pulled. The four
measured runs are grey; the model line grows as the movie plays.

    python examples/door_drain_movie.py --stl door.stl --dx 0.006 --cd 0.65 \\
           --out examples/out/door_drain/door_drain_movie.mp4

``--preview 2 6 10 15 20`` renders stills at those times (s) instead.
Needs vtk and ffmpeg. Real time: one frame per 1/fps s of physical time.
"""
from __future__ import annotations

import argparse
import os
import warnings
import subprocess
import time

import numpy as np

from door_article import DOOR_HOLES, door_setup
from door_drain import load_experiment
from drainsim.model import Simulation
from drainsim.physics import ThroatModel
from drainsim.worldviz import (_dilate_into_solid, _iso_obj, bath_geometry,
                               planar_fraction, trapped_display, pose3, to_world)

warnings.filterwarnings("ignore", category=DeprecationWarning)
C_MODEL = "#2a78d6"
C_EXP = "#8a8985"


# ------------------------------------------------------------------ 3D view
class Scene:
    def __init__(self, sim, mesh, holes, size, t_all, ssaa=2, door_opacity=0.35,
                 edges=True, smooth=30, contain=True, water_behind=False):
        import vtk
        self.vtk = vtk
        self.sim = sim
        self.holes = holes
        self.size = tuple(size)
        self.ssaa = max(1, int(ssaa))
        self.smooth = int(smooth)
        self.contain = None
        if contain:
            self._setup_contain()
        size = (self.size[0] * self.ssaa, self.size[1] * self.ssaa)
        ren = vtk.vtkRenderer()
        ren.SetBackground(1, 1, 1)
        ren.SetBackground2(0.86, 0.89, 0.93)
        ren.GradientBackgroundOn()
        ren.SetUseDepthPeeling(1)
        ren.SetMaximumNumberOfPeels(8)
        win = vtk.vtkRenderWindow()
        win.SetOffScreenRendering(1)
        win.SetSize(*size)
        win.SetAlphaBitPlanes(1)
        win.SetMultiSamples(0)
        win.AddRenderer(ren)
        self.ren, self.win = ren, win
        # water in its own bottom layer; the door and everything else are
        # drawn over it. The water then always reads as seen through the
        # translucent sheet metal (no z-fighting where it touches the walls).
        self.wren = ren
        if water_behind:
            top = vtk.vtkRenderer()
            top.SetLayer(1)
            top.SetUseDepthPeeling(1)
            top.SetMaximumNumberOfPeels(8)
            top.SetActiveCamera(ren.GetActiveCamera())
            win.SetNumberOfLayers(2)
            win.AddRenderer(top)
            ren = top
            self.ren = top

        # door: one static polydata, moved with a user matrix
        door_pd = self._poly(mesh.vertices, mesh.faces, split=True)
        self.door = self._actor(door_pd, (0.72, 0.73, 0.76), door_opacity,
                                specular=0.3)
        ren.AddActor(self.door)
        # sharp edges of the sheet metal as thin dark lines (crisper outline)
        self.edges = None
        if edges:
            fe = vtk.vtkFeatureEdges()
            fe.SetInputData(door_pd)
            fe.BoundaryEdgesOn()
            fe.FeatureEdgesOn()
            fe.SetFeatureAngle(40)
            fe.ManifoldEdgesOff()
            fe.NonManifoldEdgesOff()
            fe.ColoringOff()
            fe.Update()
            self.edges = self._actor(fe.GetOutput(), (0.25, 0.26, 0.30), 0.55,
                                     specular=0.0)
            self.edges.GetProperty().SetLineWidth(1.0 * self.ssaa)
            self.edges.GetProperty().LightingOff()
            ren.AddActor(self.edges)
        # bath box (static, world frame)
        bpts, bq = bath_geometry(sim, t_all, margin=0.05)
        self.bath = self._actor(self._poly(bpts, bq, quads=True),
                                (0.55, 0.74, 0.93), 0.20, specular=0.0)
        ren.AddActor(self.bath)
        # liquid held by the door
        self.liq_poly = vtk.vtkPolyData()
        self.liq = self._actor(self.liq_poly, (0.10, 0.33, 0.78), 1.0)
        self.wren.AddActor(self.liq)
        # plugs and jets
        self.plugs, self.jets = [], []
        for h in holes:
            s = vtk.vtkSphereSource()
            s.SetRadius(0.6 * h["diameter"])
            s.SetThetaResolution(16)
            s.SetPhiResolution(12)
            s.SetCenter(*h["center"])
            s.Update()
            a = self._actor(s.GetOutput(), (0.85, 0.15, 0.15), 1.0)
            ren.AddActor(a)
            self.plugs.append(a)
            c = vtk.vtkCylinderSource()
            c.SetResolution(16)
            c.SetHeight(1.0)
            c.SetRadius(1.0)
            c.Update()
            j = self._actor(c.GetOutput(), (0.10, 0.33, 0.78), 0.85)
            j.VisibilityOff()
            ren.AddActor(j)
            self.jets.append(j)

        # camera on the lower (wettable) part of the door and the bath top
        V = mesh.vertices
        W = np.vstack([to_world(V[:: 50], sim.motion, t) for t in t_all[:: 24]])
        lo, hi = W.min(0), W.max(0)
        zb = sim.motion.bath_level
        box = np.array([[lo[0], hi[0]], [lo[1], hi[1]],
                        [max(lo[2], zb - 0.35), min(hi[2], zb + 0.85)]])
        ren.ResetCamera(*box.ravel())
        cam = ren.GetActiveCamera()
        cam.SetViewUp(0, 0, 1)
        ctr = box.mean(1)
        cam.SetFocalPoint(*ctr)
        cam.SetPosition(ctr[0], ctr[1] - 1.0, ctr[2])
        cam.Azimuth(200)
        cam.Elevation(12)
        ren.ResetCamera(*box.ravel())
        cam.Zoom(1.35)

        self.txt = vtk.vtkTextActor()
        tp = self.txt.GetTextProperty()
        tp.SetFontSize(int(round(22 * self.size[1] / 720 * self.ssaa)))
        tp.SetColor(0.1, 0.1, 0.1)
        self.txt.SetPosition(18 * self.ssaa, 18 * self.ssaa)
        ren.AddViewProp(self.txt)
        self.w2i = vtk.vtkWindowToImageFilter()
        self.w2i.SetInput(win)
        self.w2i.SetInputBufferTypeToRGB()
        self.w2i.ReadFrontBufferOff()

    def _underwater(self, top):
        """Make the bath surface easy to read:
        - the part of the car below the surface is drawn blue-tinted (the
          same sheet, clipped at the surface plane, which is fixed in the
          world);
        - the waterline, where the surface cuts the car, is a dark blue line
          on top of everything."""
        vtk = self.vtk
        zb = float(self.sim.motion.bath_level)
        up_plane = vtk.vtkPlane()
        up_plane.SetOrigin(0, 0, zb)
        up_plane.SetNormal(0, 0, 1)
        dn_plane = vtk.vtkPlane()
        dn_plane.SetOrigin(0, 0, zb)
        dn_plane.SetNormal(0, 0, -1)
        self._below = []
        for src, col in ((self.door, (0.30, 0.52, 0.80)), (self.edges, (0.10, 0.25, 0.50))):
            if src is None:
                continue
            src.GetMapper().AddClippingPlane(up_plane)
            m = vtk.vtkPolyDataMapper()
            m.SetInputData(src.GetMapper().GetInput())
            m.ScalarVisibilityOff()
            m.AddClippingPlane(dn_plane)
            a = vtk.vtkActor()
            a.SetMapper(m)
            a.GetProperty().DeepCopy(src.GetProperty())
            a.GetProperty().SetColor(*col)
            a.GetProperty().SetOpacity(min(1.0, src.GetProperty().GetOpacity() * 1.15))
            self.ren.AddActor(a)
            self._below.append(a)
        self._car_pd = self.door.GetMapper().GetInput()
        self._cut_plane = vtk.vtkPlane()
        self._cutter = vtk.vtkCutter()
        self._cutter.SetInputData(self._car_pd)
        self._cutter.SetCutFunction(self._cut_plane)
        self.wline_poly = vtk.vtkPolyData()
        self.wline = self._actor(self.wline_poly, (0.02, 0.18, 0.55), 1.0, specular=0.0)
        self.wline.GetProperty().SetLineWidth(3.0 * self.ssaa)
        self.wline.GetProperty().LightingOff()
        (top or self.ren).AddActor(self.wline)

    def _update_water(self, t, vm):
        """Move the underwater copies and cut the waterline for time t."""
        for a in self._below:
            a.SetUserMatrix(vm)
        R, T, p = pose3(self.sim.motion, t)
        c = p + T - R @ p
        n = R.T @ np.array([0.0, 0.0, 1.0])          # surface normal, object frame
        zb = float(self.sim.motion.bath_level)
        self._cut_plane.SetNormal(*n)
        self._cut_plane.SetOrigin(*(n * (zb - c[2])))
        self._cutter.Modified()
        self._cutter.Update()
        self.wline_poly.DeepCopy(self._cutter.GetOutput())
        self.wline.SetUserMatrix(vm)

    def _setup_contain(self, spacing=1e-3):
        """Data to keep the drawn water inside the real geometry: a dense
        random sampling of the surface triangles (about 2 points per
        ``spacing``^2) with the normal of the triangle each point lies on."""
        from scipy.spatial import cKDTree
        tri = self.sim.grid.triangles
        if tri is None or self.sim.grid.ndim != 3:
            return
        e1, e2 = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
        cr = np.cross(e1, e2)
        area = 0.5 * np.linalg.norm(cr, axis=1)
        nrm = cr / np.maximum(2 * area, 1e-300)[:, None]
        cnt = np.maximum(1, np.ceil(2 * area / spacing ** 2)).astype(np.int64)
        ti = np.repeat(np.arange(len(tri)), cnt)
        rng = np.random.default_rng(0)
        r1 = np.sqrt(rng.random(ti.size))
        r2 = rng.random(ti.size)
        P = (tri[ti, 0] * (1 - r1)[:, None] + tri[ti, 1] * (r1 * (1 - r2))[:, None]
             + tri[ti, 2] * (r1 * r2)[:, None])
        P = np.vstack([P, tri.reshape(-1, 3)])
        ti = np.concatenate([ti, np.repeat(np.arange(len(tri)), 3)])
        self.contain = dict(tree=cKDTree(P), pts=P, nrm=nrm[ti])

    def _contain(self, pts, fld):
        """Keep the drawn water surface inside the real geometry.

        For every vertex close to a wall, the nearest point q on the STL
        surface and its normal n are looked up. The water side of that wall
        is the side of the nearest wet cell centre. A vertex on the other side
        (the voxel surface pokes through the sheet) or closer than ``eps``
        to it is moved to q + eps on the water side. Display only.
        """
        from scipy.spatial import cKDTree
        cd = self.contain
        sim, g = self.sim, self.sim.grid
        eps = max(7e-4, g.dx / 10)
        d, idx = cd["tree"].query(pts, distance_upper_bound=1.5 * g.dx)
        near = np.flatnonzero(np.isfinite(d))
        if near.size == 0:
            return pts
        wet = np.flatnonzero(sim.fl & (fld > 0.5))
        if wet.size == 0:
            wet = np.flatnonzero(sim.fl & (fld > 0))
        if wet.size == 0:
            return pts
        _, iw = cKDTree(sim.X[wet]).query(pts[near])
        cw = sim.X[wet[iw]]
        q = cd["pts"][idx[near]]
        n = cd["nrm"][idx[near]]
        s_ref = np.sign(np.einsum("ij,ij->i", n, cw - q))
        s_ref[s_ref == 0] = 1.0
        s_p = np.einsum("ij,ij->i", n, pts[near] - q) * s_ref   # >0: water side
        move = (s_p < eps)
        out = pts.copy()
        m = near[move]
        out[m] = q[move] + (eps * s_ref[move])[:, None] * n[move]
        return out

    def _smooth_pts(self, pts, faces):
        vtk = self.vtk
        pd = self._poly(pts, faces, normals=False)
        sm = vtk.vtkWindowedSincPolyDataFilter()
        sm.SetInputData(pd)
        sm.SetNumberOfIterations(self.smooth)
        sm.SetPassBand(0.05)
        sm.BoundarySmoothingOff()
        sm.FeatureEdgeSmoothingOff()
        sm.NonManifoldSmoothingOn()
        sm.NormalizeCoordinatesOn()
        sm.Update()
        from vtk.util.numpy_support import vtk_to_numpy
        return vtk_to_numpy(sm.GetOutput().GetPoints().GetData()).astype(float)

    def _poly(self, pts, faces, quads=False, split=False, smooth=0, normals=True):
        vtk = self.vtk
        from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray
        p = vtk.vtkPoints()
        p.SetData(numpy_to_vtk(np.ascontiguousarray(pts, float), deep=True))
        faces = np.asarray(faces, np.int64)
        n = faces.shape[1]
        cells = np.hstack([np.full((len(faces), 1), n, np.int64), faces]).ravel()
        ca = vtk.vtkCellArray()
        ca.SetCells(len(faces), numpy_to_vtkIdTypeArray(cells, deep=True))
        pd = vtk.vtkPolyData()
        pd.SetPoints(p)
        pd.SetPolys(ca)
        if not normals:
            return pd
        src = pd
        if smooth > 0:                 # display only: soften the voxel staircase
            sm = vtk.vtkWindowedSincPolyDataFilter()
            sm.SetInputData(pd)
            sm.SetNumberOfIterations(smooth)
            sm.SetPassBand(0.05)
            sm.BoundarySmoothingOff()
            sm.FeatureEdgeSmoothingOff()
            sm.NonManifoldSmoothingOn()
            sm.NormalizeCoordinatesOn()
            sm.Update()
            src = sm.GetOutput()
        nf = vtk.vtkPolyDataNormals()
        nf.SetInputData(src)
        if split:                      # keep sharp sheet-metal edges sharp
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

    def _matrix(self, t):
        R, T, p = pose3(self.sim.motion, t)
        M = np.eye(4)
        M[:3, :3] = R
        M[:3, 3] = p + T - R @ p
        vm = self.vtk.vtkMatrix4x4()
        for i in range(4):
            for j in range(4):
                vm.SetElement(i, j, M[i, j])
        return vm, R

    def render(self, t, label, hole_flows, qmax):
        self.update(t, label, hole_flows, qmax)
        return self.grab()

    def update(self, t, label, hole_flows, qmax):
        """Move/refresh everything in the scene for time t (no rendering)."""
        sim = self.sim
        vm, R = self._matrix(t)
        self._vm = vm
        self.door.SetUserMatrix(vm)
        if self.edges is not None:
            self.edges.SetUserMatrix(vm)
        # liquid surface: flat-plane reconstruction of the held water
        up, zb = sim.motion.frame(t)
        h = sim.X @ up
        sim._vis_e = 0.5 * sim.grid.dx * np.abs(up).sum()
        fld = trapped_display(sim, h, zb)[0]        # trapped liquid, above the surface
        pts, faces, _ = _iso_obj(_dilate_into_solid(fld, sim.grid), sim.grid)
        if len(pts):
            if self.smooth > 0:
                pts = self._smooth_pts(pts, faces)
            if self.contain is not None:
                pts = self._contain(pts, fld)
            pd = self._poly(to_world(pts, sim.motion, t), faces)
            self.liq_poly.DeepCopy(pd)
        else:
            self.liq_poly.DeepCopy(self.vtk.vtkPolyData())
        # plugs / jets
        for k, hole in enumerate(self.holes):
            opened = t >= hole["open_at"] - 1e-9
            self.plugs[k].SetUserMatrix(vm)
            self.plugs[k].SetVisibility(not opened)
            q = hole_flows[k] if opened else 0.0
            j = self.jets[k]
            c = to_world(np.asarray(hole["center"])[None], sim.motion, t)[0]
            L = min(0.30, c[2] - sim.motion.bath_level)       # down to the bath
            if q > 1e-7 and L > 0.01:                          # hole above the bath
                r = 0.5 * hole["diameter"] * np.sqrt(min(q / qmax, 4.0))
                tr = self.vtk.vtkTransform()
                tr.Translate(c[0], c[1], c[2] - L / 2)
                tr.RotateX(90)                   # cylinder axis y -> z
                tr.Scale(r, L, r)
                j.SetUserTransform(tr)
                j.VisibilityOn()
            else:
                j.VisibilityOff()
        self.txt.SetInput(label)

    def grab(self):
        """Render and return the frame as an (H, W, 3) uint8 array."""
        self.win.Render()
        self.w2i.Modified()
        self.w2i.Update()
        img = self.w2i.GetOutput()
        from vtk.util.numpy_support import vtk_to_numpy
        w, hgt, _ = img.GetDimensions()
        a = vtk_to_numpy(img.GetPointData().GetScalars()).reshape(hgt, w, -1)
        a = np.ascontiguousarray(a[::-1, :, :3])
        if self.ssaa > 1:              # supersampling anti-aliasing
            from PIL import Image
            a = np.asarray(Image.fromarray(a).resize(self.size, Image.LANCZOS))
        return a


# ------------------------------------------------------------------ 2D panel
class Panel:
    def __init__(self, size, t_open, t_end, exp):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        dpi = 100 * size[1] / 720          # fonts scale with the frame height
        self.fig = plt.figure(figsize=(size[0] / dpi, size[1] / dpi), dpi=dpi)
        self.canvas = FigureCanvasAgg(self.fig)
        ax = self.fig.add_axes([0.14, 0.12, 0.82, 0.74])
        self.ax = ax
        x0 = -t_open
        for x_a, x_b, lab in ((x0, x0 + 4, "dip"), (x0 + 4, x0 + 8, "lift"),
                              (x0 + 8, 0.0, "hold"), (0.0, t_end - t_open, "drain")):
            ax.axvspan(x_a, x_b, color="#f1f0ec" if lab != "drain" else "#ffffff",
                       zorder=0, lw=0)
            ax.text(0.5 * (x_a + x_b), 12.35, lab, ha="center", va="bottom",
                    fontsize=10, color="#52514e")
        ax.axvline(0, color="#52514e", lw=0.8, ls=(0, (3, 3)))
        ax.text(0.3, 11.6, "plugs pulled", fontsize=9, color="#52514e")
        for k, e in exp.items():
            ax.plot(e[:, 0], e[:, 2], color=C_EXP, lw=1.3,
                    label="experiment (4 runs)" if k == 1 else None)
        self.line, = ax.plot([], [], color=C_MODEL, lw=2.2, label="model")
        self.dot, = ax.plot([], [], "o", color=C_MODEL, ms=7, mec="white", mew=1.5)
        self.cursor = ax.axvline(x0, color=C_MODEL, lw=0.8, alpha=0.5)
        ax.set_xlim(x0, t_end - t_open)
        ax.set_ylim(0, 12.3)
        ax.grid(color="#e6e5e0", lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.set_xlabel("time since the plugs were pulled (s)")
        ax.set_ylabel("water in the door, above the bath (l)")
        ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(1.0, 0.93),
                  fontsize=10)
        self.fig.text(0.14, 0.955, "Horizontal door, three 19 mm holes",
                      fontsize=13, color="#0b0b0b")
        self.info = self.fig.text(0.14, 0.905, "", fontsize=10.5, color="#52514e")
        self.xs, self.ys = [], []

    def render(self, x, y, info):
        self.xs.append(x)
        self.ys.append(y)
        self.line.set_data(self.xs, self.ys)
        self.dot.set_data([x], [y])
        self.cursor.set_xdata([x, x])
        self.info.set_text(info)
        self.canvas.draw()
        a = np.asarray(self.canvas.buffer_rgba())
        return np.ascontiguousarray(a[:, :, :3])


# ------------------------------------------------------------------ driver
def main(stl, dx, cd, subcells, out, fps=24, t_settle=5.0, t_after=25.0,
         size3d=(1040, 720), size2d=(720, 720), preview=None, ssaa=2, crf=16,
         preset="slow", door_opacity=0.35, edges=True, smooth=30, contain=True,
         water_behind=False):
    size3d = tuple(int(v) // 2 * 2 for v in size3d)        # even for yuv420p
    size2d = (int(size2d[0]) // 2 * 2, size3d[1])
    import trimesh
    t0 = time.time()
    t_open = 8.0 + t_settle
    t_end = t_open + t_after
    mesh, grid, motion = door_setup(stl, dx, 0.0, t_drain=t_settle + t_after)
    holes = [dict(h, open_at=t_open) for h in DOOR_HOLES]
    sim = Simulation(grid, motion, dt_max=0.05, holes=holes, subcells=subcells,
                     throat_model=ThroatModel(Cd=cd))
    hole_ids = [i for i, t in enumerate(sim.comp.throats) if t.axis is not None]
    hole_geo = [dict(center=sim.comp.throats[i].centroid,
                     diameter=sim.comp.throats[i].diameter, open_at=t_open)
                for i in hole_ids]
    print(f"setup {time.time()-t0:.0f} s, grid {grid.shape}, {len(hole_ids)} holes",
          flush=True)
    times = np.arange(0.0, t_end + 1e-9, 1.0 / fps)
    if preview:
        times = np.array(sorted(preview))
    full_mesh = trimesh.load(stl, force="mesh")
    scene = Scene(sim, full_mesh, hole_geo, size3d, np.arange(0, t_end, 1 / fps),
                  ssaa=ssaa, door_opacity=door_opacity, edges=edges, smooth=smooth,
                  contain=contain, water_behind=water_behind)
    panel = Panel(size2d, t_open, t_end, load_experiment())
    W = size3d[0] + size2d[0]
    Hh = size3d[1]
    ff = None
    if not preview:
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        ff = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo",
             "-pix_fmt", "rgb24", "-s", f"{W}x{Hh}", "-r", str(fps), "-i", "-",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(crf),
             "-preset", preset, "-movflags", "+faststart", out], stdin=subprocess.PIPE)
    t_sim = t_draw = 0.0
    V_open = None
    out_cum = 0.0
    nrec = 1
    qmax = 1e-3 / len(hole_ids)                  # jet width scale (1 l/s total)
    for i, t in enumerate(times):
        a = time.time()
        # always stop exactly at the opening, so the volume at opening is
        # taken there even when frames are far apart (e.g. --preview)
        if V_open is None and t > t_open + 1e-9 and sim.t < t_open - 1e-9:
            sim.run(t_end=t_open)
        if V_open is None and abs(sim.t - t_open) < 1e-9:
            V_open = sim.hist.liquid_retained[-1]
            nrec = len(sim.hist.t)
        if t > sim.t + 1e-9:
            sim.run(t_end=float(t))
        t_sim += time.time() - a
        H = sim.hist
        # cumulative outflow through the holes since the last frame
        tt = np.asarray(H.t)
        for k in range(nrec, len(tt)):
            if tt[k] > t_open + 1e-9:
                fl = np.abs(np.asarray(H.throat_flow[k])[hole_ids])
                out_cum += fl.sum() * (tt[k] - tt[k - 1])
        nrec = len(tt)
        flows = np.abs(np.asarray(H.throat_flow[-1])[hole_ids]) if len(tt) > 1 \
            else np.zeros(len(hole_ids))
        if V_open is None:
            V = H.liquid_retained[-1]        # before the opening
        else:
            V = V_open - out_cum
        phase = ("dip (plugs in)" if t < 4 else "lift (plugs in)" if t < 8 else
                 "hold (plugs in)" if t < t_open else "plugs pulled: draining")
        a = time.time()
        img3 = scene.render(t, f"t = {t:5.2f} s    {phase}", flows, qmax)
        info = (f"model {dx*1e3:g} mm, Cd = {cd:g}:  {V*1e3:5.2f} l in door"
                + (f",  outflow {flows.sum()*1e3:4.2f} l/s" if t >= t_open else ""))
        img2 = panel.render(t - t_open, V * 1e3, info)
        frame = np.hstack([img3, img2[:Hh, :, :]])
        t_draw += time.time() - a
        if preview:
            from PIL import Image
            p = out.replace(".mp4", "") + f"_t{t:05.2f}.png"
            Image.fromarray(frame).save(p)
            print("wrote", p, flush=True)
        else:
            ff.stdin.write(frame.tobytes())
        if i % 48 == 0:
            print(f"frame {i}/{len(times)}  t = {t:.2f} s  V = {V*1e3:.2f} l  "
                  f"sim {t_sim:.0f} s, draw {t_draw:.0f} s", flush=True)
    if ff is not None:
        ff.stdin.close()
        ff.wait()
        print("wrote", out)
    print(f"total {time.time()-t0:.0f} s: simulation {t_sim:.0f} s, drawing {t_draw:.0f} s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stl", required=True)
    ap.add_argument("--dx", type=float, default=0.006)
    ap.add_argument("--cd", type=float, default=0.65)
    ap.add_argument("--subcells", type=int, default=4)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--t-after", type=float, default=25.0)
    ap.add_argument("--preview", type=float, nargs="*")
    ap.add_argument("--size3d", type=int, nargs=2, default=[1040, 720],
                    metavar=("W", "H"), help="3D view in pixels (H = movie height)")
    ap.add_argument("--width2d", type=int, default=720, help="plot width in pixels")
    ap.add_argument("--ssaa", type=int, default=2,
                    help="render the 3D view this many times larger, then downscale")
    ap.add_argument("--crf", type=int, default=16, help="x264 quality (lower = better)")
    ap.add_argument("--preset", default="slow")
    ap.add_argument("--door-opacity", type=float, default=0.35)
    ap.add_argument("--no-edges", action="store_true", help="no feature-edge lines")
    ap.add_argument("--smooth", type=int, default=30,
                    help="display smoothing iterations of the water surface (0 = off)")
    ap.add_argument("--water-behind", action="store_true",
                    help="always draw the door over the water (paler water)")
    ap.add_argument("--no-contain", action="store_true",
                    help="do not clip the drawn water to the real geometry")
    ap.add_argument("--out", default="examples/out/door_drain/door_drain_movie.mp4")
    a = ap.parse_args()
    main(a.stl, a.dx, a.cd, a.subcells, a.out, a.fps, t_after=a.t_after,
         preview=a.preview, size3d=a.size3d, size2d=(a.width2d, a.size3d[1]),
         ssaa=a.ssaa, crf=a.crf, preset=a.preset, door_opacity=a.door_opacity,
         edges=not a.no_edges, smooth=a.smooth, contain=not a.no_contain,
         water_behind=a.water_behind)
