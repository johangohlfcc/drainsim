"""Movie of the car-body dip: trapped air, trapped liquid and surface flow.

The car of the article's Sect. 3.3 goes through the shortened dip of
``car_article.py`` (dip-in, 5 s under the surface, dip-out, then hanging).
Left, a world-frame 3D view that follows the car:

- the car, translucent grey;
- **trapped air** under the bath surface, yellow;
- **liquid** held by the car above the bath, blue (drawn on top so it stays
  visible through the sheet metal);
- the **wall film** deposited on withdrawal, coloured by thickness (µm, log
  scale) and fading to transparent as it drains;
- **drops** released from hanging film (enlarged).

Right, the curves: trapped air, liquid in the car, film on the walls and the
cumulative dripped volume, with a cursor.

The model runs with fixed steps of ``--sim-dt`` (0.1 s) through the motion
and the first ``--slow-hang`` seconds of the hanging, then ``--dt-hang``
(0.5 s) while the car hangs still. Frames in between move the car smoothly
and show the state of the last step. The movie is real time up to the end
of the dip-out, then ``--speedup`` (2×) for the first ``--slow-hang`` (10 s)
of the hanging and ``--speedup-late`` (10×) for the rest of the ``--hang``
(120 s) drip-off: about 35 + 5 + 11 = 51 s of movie.

    python examples/car_movie.py --stl xc90.stl --dx 0.02 --out car_dip.mp4
    python examples/car_movie.py --stl xc90.stl --dx 0.02 --preview 8 17 27 34 45 --out car_dip.mp4

The car is drawn from a decimated copy of the mesh (``--display-faces``).
Needs vtk and ffmpeg.

**Parallel rendering** (for fine grids): record the model state once, then
render the frames in several processes.

    python examples/car_movie.py --stl xc90.stl --dx 0.007 --record runs_car/rec7
    python examples/car_movie.py --render runs_car/rec7 --workers 10 --out car_7mm.mp4

``--record`` runs only the model and writes, per model step, the liquid and
trapped-air fields (sparse, 8 bit), the film thickness and the new drops
(``steps/s00000.npz`` ...), plus the static data the drawing needs. The
render may start while the recording is still running: the workers wait for
the steps they need. Each worker writes a part of the movie; the parts are
joined without re-encoding.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import warnings
from types import SimpleNamespace

import numpy as np

from car_article import (MOTIONS, add_car_args, car_grid, car_motion, explicit_holes,
                         gap_channels, ips_motion, load_car, narrow_opt,
                         peak_rss_bytes)
from door_dip_film_movie import FilmScene
from drainsim.grid import Grid
from drainsim.model import Simulation
from drainsim.physics import ThroatModel
from drainsim.worldviz import (_dilate_into_solid, _iso_obj, planar_fraction, pose3, to_world,
                                trapped_display)

warnings.filterwarnings("ignore")
C_AIR, C_LIQ, C_FILM, C_DRIP = "#e3a008", "#2a78d6", "#eb6834", "#1baf7a"


def display_mesh(mesh, target):
    """Decimated copy of the mesh for drawing (quadric decimation, VTK)."""
    import trimesh
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray, vtk_to_numpy
    F = np.asarray(mesh.faces, np.int64)
    if len(F) <= target:
        return mesh
    t0 = time.time()
    pts = vtk.vtkPoints()
    pts.SetData(numpy_to_vtk(np.ascontiguousarray(mesh.vertices, float), deep=True))
    ca = vtk.vtkCellArray()
    ca.SetData(numpy_to_vtkIdTypeArray(np.arange(0, 3 * len(F) + 1, 3, dtype=np.int64), deep=True),
               numpy_to_vtkIdTypeArray(np.ascontiguousarray(F.ravel()), deep=True))
    pd = vtk.vtkPolyData()
    pd.SetPoints(pts)
    pd.SetPolys(ca)
    cl = vtk.vtkCleanPolyData()
    cl.SetInputData(pd)
    cl.PointMergingOn()
    cl.SetTolerance(0.0)
    cl.Update()
    n0 = cl.GetOutput().GetNumberOfPolys()
    dec = vtk.vtkQuadricDecimation()
    dec.SetInputConnection(cl.GetOutputPort())
    dec.SetTargetReduction(max(0.0, 1.0 - target / max(n0, 1)))
    dec.Update()
    out = dec.GetOutput()
    V = vtk_to_numpy(out.GetPoints().GetData()).astype(float)
    C = vtk_to_numpy(out.GetPolys().GetConnectivityArray()).reshape(-1, 3)
    print(f"display mesh: {len(F)} -> {len(C)} triangles ({time.time()-t0:.0f} s)", flush=True)
    return trimesh.Trimesh(V, C, process=False)


def grid_field(sim, f):
    """Node field -> grid field; sub-cell nodes are folded into their host
    cell (max), so thin pools and pockets in cut cells are drawn too."""
    n = sim.ncells
    g = np.array(f[:n], float)
    if sim.N > n:
        np.maximum.at(g, sim.host[n:], f[n:])
    return g


class CarScene(FilmScene):
    def __init__(self, sim, mesh, size, t_all, cam_dist=10.0, azimuth=-55.0,
                 elevation=14.0, look="hd", orbit=0.0, **kw):
        kw.setdefault("contain", False)
        kw.setdefault("edges", True)
        kw.setdefault("smooth", 10)
        super().__init__(sim, mesh, [], size, t_all, look="plain", **kw)
        # trapped air (under the bath), drawn on top like the liquid
        self.air_poly = self.vtk.vtkPolyData()
        self.air = self._actor(self.air_poly, (0.89, 0.63, 0.03), 0.95)
        rens = self.win.GetRenderers()
        rens.InitTraversal()
        top = None
        for _ in range(rens.GetNumberOfItems()):
            r = rens.GetNextItem()
            if r.GetLayer() == self.win.GetNumberOfLayers() - 1:
                top = r
        (top or self.ren).AddActor(self.air)
        self._underwater(top)
        self.cam_dist = cam_dist
        self.azimuth = np.radians(azimuth)
        self.elevation = np.radians(elevation)
        self.orbit = np.radians(orbit)
        self._t_end = float(np.max(t_all)) if len(t_all) else 1.0
        if look == "hd":
            self._look_hd()

    def _camera(self, t):
        R, T, p = pose3(self.sim.motion, t)
        # follow the car along the bath and halfway in height, so the bath
        # surface stays in view while the car goes in and comes out
        c = np.array([T[0], 0.0, 0.5 * T[2]])
        az = self.azimuth + self.orbit * min(max(t / self._t_end, 0.0), 1.0)
        d = np.array([np.sin(az) * np.cos(self.elevation),
                      -np.cos(az) * np.cos(self.elevation),
                      np.sin(self.elevation)])
        cam = self.ren.GetActiveCamera()
        cam.SetViewUp(0, 0, 1)
        cam.SetFocalPoint(*c)
        cam.SetPosition(*(c + self.cam_dist * d))
        cam.SetViewAngle(30)
        self.ren.ResetCameraClippingRange()

    def _surface(self, fld):
        pts, faces, _ = _iso_obj(_dilate_into_solid(grid_field(self.sim, fld), self.sim.grid),
                                 self.sim.grid)
        return pts, faces

    def update(self, t, label, t_state):
        sim = self.sim
        vm, R = self._matrix(t)
        self._vm = vm
        self.door.SetUserMatrix(vm)
        if self.edges is not None:
            self.edges.SetUserMatrix(vm)
        self._update_water(t, vm)
        up, zb = sim.motion.frame(t_state)
        h = sim.X @ up
        sim._vis_e = 0.5 * sim.grid.dx * np.abs(up).sum()
        liq, air = trapped_display(sim, h, zb)
        for fld, poly in ((grid_field(sim, liq), self.liq_poly),
                          (grid_field(sim, air), self.air_poly)):
            pts, faces = _iso_obj(_dilate_into_solid(fld, sim.grid), sim.grid)
            if len(pts):
                if self.smooth > 0:
                    pts = self._smooth_pts(pts, faces)
                poly.DeepCopy(self._poly(to_world(pts, sim.motion, t), faces))
            else:
                poly.DeepCopy(self.vtk.vtkPolyData())
        self.txt.SetInput(label)
        self._camera(t)

    def _follow(self, t):                           # camera is set in update()
        return


class Timeline:
    """Model steps and movie frames of the car dip.

    Real time to the end of the dip-out, then the hanging: ``slow`` seconds
    at ``speedup`` times real time, the rest at ``speedup_late``. The model
    steps ``dt`` through the motion and the first ``slow`` seconds of the
    hanging, then ``dt_hang`` (the car is still; the film sub-steps on its
    own)."""

    def __init__(self, t_dipout, t_end, fps, speedup, slow, speedup_late, dt, dt_hang):
        self.t_dipout, self.t_end, self.fps = float(t_dipout), float(t_end), int(fps)
        self.speedup, self.speedup_late = float(speedup), float(speedup_late)
        self.t_slow = min(self.t_dipout + float(slow), self.t_end)
        self.dt, self.dt_hang = float(dt), float(dt_hang)
        self.phases = [(0.0, self.t_dipout, 1.0), (self.t_dipout, self.t_slow, self.speedup),
                       (self.t_slow, self.t_end, self.speedup_late)]
        self.phases = [p for p in self.phases if p[1] > p[0] + 1e-9]

    @classmethod
    def from_meta(cls, m):
        return cls(m["t_dipout"], m["t_end"], m["fps"], m["speedup"], m["slow_hang"],
                   m["speedup_late"], m["sim_dt"], m["dt_hang"])

    def meta(self):
        return dict(t_dipout=self.t_dipout, t_end=self.t_end, fps=self.fps,
                    speedup=self.speedup, slow_hang=self.t_slow - self.t_dipout,
                    speedup_late=self.speedup_late, sim_dt=self.dt, dt_hang=self.dt_hang)

    def frames(self):
        out = []
        for i, (a, b, sp) in enumerate(self.phases):
            last = i == len(self.phases) - 1
            out.append(np.arange(a, b + (1e-9 if last else -1e-9), sp / self.fps))
        return np.concatenate(out)

    def steps(self):
        a = np.arange(0.0, self.t_slow - 1e-9, self.dt)
        b = np.arange(self.t_slow, self.t_end - 1e-9, self.dt_hang)
        return np.unique(np.round(np.r_[a, b, self.t_end], 9))

    def speed(self, t):
        for a, b, sp in self.phases:
            if t < b - 1e-9:
                return sp
        return self.phases[-1][2]

    def tau(self, t):
        """Movie time (s) at model time t."""
        t = np.asarray(t, float)
        out = np.zeros_like(t)
        m0 = 0.0
        for a, b, sp in self.phases:
            out = np.where(t >= a, m0 + (np.minimum(t, b) - a) / sp, out)
            m0 += (b - a) / sp
        return out

    def label(self, t):
        phase = ("dip-in" if t < 15 else "under the surface" if t < self.t_dipout - 15
                 else "dip-out" if t < self.t_dipout else "hanging: draining and dripping")
        sp = self.speed(t)
        return f"t = {t:6.2f} s    {phase}" + (f"    ({sp:g}× speed)" if sp > 1 else "")


class CarPanel:
    """Curves against movie time (the cursor moves steadily), labelled in
    model time; the sped-up hanging is shaded and marked."""

    def __init__(self, size, tl):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        self.tl = tl
        dpi = 100 * size[1] / 720
        self.fig = plt.figure(figsize=(size[0] / dpi, size[1] / dpi), dpi=dpi)
        self.canvas = FigureCanvasAgg(self.fig)
        a1 = self.fig.add_axes([0.15, 0.53, 0.80, 0.33])
        a2 = self.fig.add_axes([0.15, 0.10, 0.80, 0.33], sharex=a1)
        self.axes = (a1, a2)
        T = tl.tau
        td = tl.t_dipout
        for ax in self.axes:
            ax.axvspan(0, T(15.0), color="#f1f0ec", lw=0, zorder=0)
            ax.axvspan(T(td - 15), T(td), color="#f1f0ec", lw=0, zorder=0)
            for a, b, sp in tl.phases:
                if sp > 1:
                    ax.axvspan(T(a), T(b), color="#f7f6f2", lw=0, zorder=0)
                    ax.axvline(T(a), color="#d9d8d4", lw=0.8, zorder=0.5)
            ax.grid(color="#e6e5e0", lw=0.8)
            ax.set_axisbelow(True)
            for sp_ in ("top", "right"):
                ax.spines[sp_].set_visible(False)
        for x, lab in ((7.5, "dip-in"), (td - 7.5, "dip-out")):
            a1.text(T(x), 1.0, lab, transform=a1.get_xaxis_transform(), ha="center",
                    va="bottom", fontsize=9, color="#52514e")
        fast = [(a, b, sp) for a, b, sp in tl.phases if sp > 1]
        if fast:
            txt = "hanging, " + " then ".join(f"{sp:g}×" for _, _, sp in fast)
            a1.text(0.5 * (T(fast[0][0]) + T(fast[-1][1])), 1.0, txt,
                    transform=a1.get_xaxis_transform(), ha="center", va="bottom",
                    fontsize=9, color="#52514e")
        ticks = [x for x in (0, 10, 20, 30, td, td + 10, td + 30, td + 60, td + 90, td + 120,
                             td + 180, tl.t_end) if x <= tl.t_end + 1e-9]
        ticks = sorted(set(round(x, 6) for x in ticks))
        tt = [ticks[0]]
        for x in ticks[1:]:                      # keep labels apart (movie seconds)
            if T(x) - T(tt[-1]) > 2.5 or x == ticks[-1]:
                if x == ticks[-1] and T(x) - T(tt[-1]) <= 2.5 and len(tt) > 1:
                    tt[-1] = x
                else:
                    tt.append(x)
        a2.set_xticks([float(T(x)) for x in tt])
        a2.set_xticklabels([f"{x:g}" for x in tt])
        a1.set_xlim(0, float(T(tl.t_end)))
        a1.set_ylabel("volume (l)")
        a2.set_ylabel("volume (ml)")
        a2.set_xlabel("time (s)")
        plt.setp(a1.get_xticklabels(), visible=False)
        self.l_air, = a1.plot([], [], color=C_AIR, lw=2.2, label="trapped air (below the surface)")
        self.l_liq, = a1.plot([], [], color=C_LIQ, lw=2.2, label="trapped liquid (above the surface)")
        a1.legend(frameon=False, fontsize=9, loc="upper right")
        self.l_film, = a2.plot([], [], color=C_FILM, lw=2.2, label="film on the walls")
        self.l_drip, = a2.plot([], [], color=C_DRIP, lw=2.2, label="dripped (cumulative)")
        a2.legend(frameon=False, fontsize=9, loc="upper left")
        self.cur = [ax.axvline(0, color="#52514e", lw=0.8, alpha=0.6) for ax in self.axes]
        self.fig.text(0.15, 0.955, "Car body dip: trapped air, liquid and wall film",
                      fontsize=13, color="#0b0b0b")
        self.info = self.fig.text(0.15, 0.91, "", fontsize=10.5, color="#52514e")
        self.d = dict(t=[], air=[], liq=[], film=[], drip=[])
        self.ymax = [1.0, 50.0]

    def add(self, t, air_l, liq_l, film_ml, drip_ml):
        for k, v in (("t", float(self.tl.tau(t))), ("air", air_l), ("liq", liq_l),
                     ("film", film_ml), ("drip", drip_ml)):
            self.d[k].append(v)
        self.ymax[0] = max(self.ymax[0], 1.15 * air_l, 1.15 * liq_l)
        self.ymax[1] = max(self.ymax[1], 1.15 * film_ml, 1.15 * drip_ml)

    def render(self, t, air_l, liq_l, film_ml, drip_ml, info):
        self.add(t, air_l, liq_l, film_ml, drip_ml)
        d = self.d
        self.l_air.set_data(d["t"], d["air"])
        self.l_liq.set_data(d["t"], d["liq"])
        self.l_film.set_data(d["t"], d["film"])
        self.l_drip.set_data(d["t"], d["drip"])
        self.axes[0].set_ylim(0, self.ymax[0])
        self.axes[1].set_ylim(0, self.ymax[1])
        x = float(self.tl.tau(t))
        for c in self.cur:
            c.set_xdata([x, x])
        self.info.set_text(info)
        self.canvas.draw()
        return np.ascontiguousarray(np.asarray(self.canvas.buffer_rgba())[:, :, :3])


def run_motion(a, info):
    """(motion, t_dipout, motion info) of a run: an IPS motion file
    (--motion-file, with --bath-stl) or a built-in dip (--motion)."""
    if getattr(a, "motion_file", None):
        mo, mi = ips_motion(a.motion_file, a.bath_stl, info["centre_file"],
                            info["scale_to_m"], a.hang)
        return mo, mi["t_dipout"], mi
    return car_motion(a.rock, a.hang, a.motion, a.rotation, a.sense), \
        MOTIONS[a.motion]["dipout"], None


def timeline(a, t_dipout, t_end):
    return Timeline(t_dipout, t_end, a.fps, a.speedup, a.slow_hang, a.speedup_late,
                    a.sim_dt, a.dt_hang)


# ---------------------------------------------------------------- recording
def _state_fields(sim, t_state, L, B, A):
    """Grid fields (liquid held by the car, trapped air) exactly as
    CarScene.update draws them, as sparse 8-bit arrays. L, B, A are copies
    of the model state at ``t_state`` (the model may run on meanwhile)."""
    octree = getattr(sim, "octree", False)
    view = SimpleNamespace(fl=sim.fl, lab=sim.lab, nbr=sim.nbr, v=sim.v, fine=sim.fine,
                           subcells=sim.subcells, ncells=sim.ncells,
                           N=sim.N, grid=sim.grid, nsize=sim.nsize)
    if not octree:
        view.host = sim.host
    up, zb = sim.motion.frame(t_state)
    h = sim.X @ up
    view._vis_e = 0.5 * sim.grid.dx * np.abs(up).sum()
    liq, air = trapped_display(sim, h, zb, L, B, A, view=view)
    if octree:
        # octree: node fields onto the uniform display grid, at the
        # nonzero display cells only (in ascending order)
        from drainsim.octview import display_grid
        dg = display_grid(sim)
        fields = (dg.field_nonzero(sim, liq), dg.field_nonzero(sim, air))
        ncells = dg.ncells
    else:
        fields = []
        for f in (grid_field(view, liq), grid_field(view, air)):
            i = np.flatnonzero(f)
            fields.append((i, f[i]))
        ncells = sim.grid.ncells
    out = {}
    for k, (i, f) in zip(("liq", "air"), fields):
        q = np.clip(np.rint(f * 255.0), 0, 255).astype(np.uint8)
        nz = q != 0
        out[k + "_i"] = i[nz].astype(np.int64 if ncells >= 2 ** 31 else np.int32)
        out[k + "_v"] = q[nz]
    return out


def _save_atomic(path, **arrays):
    tmp = path + ".tmp.npz"
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def record(a):
    """Run the model only; write the state of every model step."""
    t0 = time.time()
    rec = a.record
    os.makedirs(os.path.join(rec, "steps"), exist_ok=True)
    mesh, info = load_car(a.stl, a.orient, a.reverse)
    # the display mesh first: the decimation needs a lot of memory for a
    # short while, before the model takes its share
    dmesh = display_mesh(mesh, a.display_faces)
    mo, t_dipout, minfo = run_motion(a, info)
    lo = np.asarray(mesh.vertices).min(0) - a.pad
    hi = np.asarray(mesh.vertices).max(0) + a.pad
    grid = car_grid(mesh, a.dx, lo, hi, a.levels, octree=bool(a.narrow))
    del mesh
    sim = Simulation(grid, mo, dt_max=a.sim_dt, cells_per_step=1e9, subcells=a.subcells,
                     film=True, throat_model=ThroatModel(Cd=a.cd), threads=a.threads,
                     narrow=narrow_opt(a), holes=explicit_holes(a, info),
                     channels=gap_channels(a, info))
    print(f"setup {time.time()-t0:.0f} s: grid {grid.shape}, {sim.N/1e6:.2f} M nodes, "
          f"{sim.film.c.n} film elements, peak {_peak_gb()} GB", flush=True)
    if sim.channels is not None:
        # the channels to review, in the STL's frame (metres)
        from drainsim.channels import export_channels
        from drainsim.openings import from_model
        export_channels(os.path.join(rec, "channels"), sim.channels, sim.comp.n - sim.channels.n,
                        lambda p: from_model(p, info))
    vg = grid
    if a.levels or a.narrow:
        # octree: the movie is drawn on a uniform display grid
        from drainsim.octview import display_grid
        vg = display_grid(sim, a.display_max_cells)
        print(f"display grid {vg.shape} at {vg.dx*1e3:g} mm", flush=True)
    np.savez(os.path.join(rec, "static.npz"),
             solid=np.packbits(vg.solid.ravel()), shape=np.array(vg.shape),
             origin=vg.origin, dx=vg.dx, cx=sim.film.c.x, cnormal=sim.film.c.normal,
             dv=np.asarray(dmesh.vertices), df=np.asarray(dmesh.faces))
    del dmesh
    tl = timeline(a, t_dipout, mo.t_end)
    steps = tl.steps()
    nsteps = len(steps) - 1
    meta = dict(dx=a.dx, subcells=a.subcells, motion=a.motion, rock=a.rock, hang=a.hang,
                motion_file=minfo and minfo["xmo"], bath_stl=minfo and minfo["bath_stl"],
                holes=a.holes and dict(file=os.path.abspath(a.holes), min_mm=a.holes_min,
                                       max_mm=a.holes_max, n=len(sim.holes)),
                channels=a.channels and sim.channels is not None and dict(
                    min_mm=a.channel_min, max_mm=a.channel_max, spacing_mm=a.gap_spacing,
                    **sim.channels.stats),
                ips=minfo,
                rotation=a.rotation, sense=a.sense, orient=a.orient, reverse=a.reverse,
                cd=a.cd, levels=a.levels, narrow=a.narrow, nsteps=nsteps,
                step_times=steps.tolist(),
                nodes=int(sim.N),
                cells=int(grid.ncells), car=info, **tl.meta())
    with open(os.path.join(rec, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    _record_steps(a, sim, rec, tl, steps, t0=t0)


def resume(a):
    """Go on with a recording from a state saved by --save-at (--resume
    STATE): into the recording it was saved from, or into --record DIR (a
    new recording that gets the first one's static.npz and meta.json; its
    steps start after the saved one)."""
    import shutil
    from drainsim.model import Simulation
    t0 = time.time()
    sim, extra = Simulation.load_state(a.resume)
    src = extra["rec"]
    rec = a.record or src
    os.makedirs(os.path.join(rec, "steps"), exist_ok=True)
    if os.path.abspath(rec) != os.path.abspath(src):
        for f in ("static.npz", "meta.json"):
            shutil.copy2(os.path.join(src, f), os.path.join(rec, f))
    with open(os.path.join(rec, "meta.json")) as f:
        meta = json.load(f)
    tl = Timeline.from_meta(meta)
    steps = np.asarray(meta["step_times"], float)
    print(f"resumed {a.resume}: t = {sim.t:.2f} s, step {extra['k']} of {len(steps) - 1} "
          f"({time.time() - t0:.0f} s)", flush=True)
    _record_steps(a, sim, rec, tl, steps, k0=extra["k"] + 1, nd=extra["nd"], t0=t0)


def _save_points(a):
    """--save-at: the times (s) to save the state at, and whether at the end."""
    pts, end = [], False
    for x in getattr(a, "save_at", None) or []:
        if str(x).lower() == "end":
            end = True
        else:
            pts.append(float(x))
    return sorted(pts), end


def _record_steps(a, sim, rec, tl, steps, k0=0, nd=0, t0=None):
    """The model steps of a recording from step k0 on (the state written per
    step; the state saved at --save-at)."""
    t0 = time.time() if t0 is None else t0
    nsteps = len(steps) - 1
    save_t, save_end = _save_points(a)
    save_t = [x for x in save_t if x > steps[k0 - 1] + 1e-9] if k0 > 0 else save_t
    # the fields of a step are extracted and written in a writer thread
    # while the model runs the next steps (at most 2 steps queued)
    import queue
    import threading
    q = queue.Queue(maxsize=2)
    err = []

    def writer():
        from drainsim.par import serial_thread
        serial_thread()                     # the parallel kernels stay with the model
        while True:
            job = q.get()
            if job is None:
                return
            try:
                k, ts, L, B, A, rest = job
                _save_atomic(os.path.join(rec, "steps", f"s{k:05d}.npz"), **rest,
                             **_state_fields(sim, ts, L, B, A))
            except Exception as ex:                  # report, stop the run
                err.append(ex)
                return

    wt = threading.Thread(target=writer, daemon=True)
    wt.start()
    t_sim = t_wait = 0.0
    for k in range(k0, nsteps + 1):
        ts = steps[k]
        s0 = time.time()
        if k > 0:
            # fine steps through the motion and the start of the hanging,
            # then larger ones while the car hangs still
            sim.dt_max = tl.dt if ts <= tl.t_slow + 1e-9 else tl.dt_hang
            sim.run(t_end=float(ts))
        t_sim += time.time() - s0
        if err:
            raise err[0]
        f = sim.film
        drips = f.s.drips[nd:]
        nd = len(f.s.drips)
        D = np.array([[d[0], *np.asarray(d[1], float), d[2], d[3]] for d in drips]) \
            if drips else np.zeros((0, 6))
        H = sim.hist
        sc = np.array([sim.t, H.air_trapped[-1] * 1e3, H.liquid_trapped[-1] * 1e3,
                       f.volume * 1e6, f.drip_volume * 1e6,
                       sum(d[3] for d in f.s.drips)], float)
        rest = dict(scalars=sc, drips=D, film_h=(f.s.h * 1e6).astype(np.float16),
                    film_sub=np.packbits(f.s.sub))
        s0 = time.time()
        q.put((k, sim.t, sim.L.copy(), sim.B.copy(), sim.A.copy(), rest))
        t_wait += time.time() - s0
        while save_t and sim.t >= save_t[0] - 1e-9 or (save_end and k == nsteps):
            p = os.path.join(rec, f"state_t{sim.t:07.2f}.pkl")
            s1 = time.time()
            sim.save_state(p, extra=dict(k=k, nd=nd, rec=os.path.abspath(rec)))
            print(f"saved the state at t = {sim.t:.2f} s (step {k}) to {p} "
                  f"({time.time() - s1:.0f} s)", flush=True)
            if save_t and sim.t >= save_t[0] - 1e-9:
                save_t.pop(0)
            else:
                save_end = False
        if k % 50 == 0 or k == nsteps:
            el = time.time() - t0
            print(f"step {k}/{nsteps}  t = {sim.t:.1f} s  air {sc[1]:.2f} l  liquid {sc[2]:.2f} l"
                  f"  film {sc[3]:.0f} ml  model {t_sim/60:.1f} min, waiting for the writer "
                  f"{t_wait/60:.1f} min,  ETA {el / max(k - k0, 1) * (nsteps - k) / 60:.0f} min, "
                  f"peak {_peak_gb()} GB", flush=True)
    q.put(None)
    wt.join()
    if err:
        raise err[0]
    t_rec = t_wait
    with open(os.path.join(rec, "done"), "w") as f:
        f.write(f"{time.time()-t0:.0f}\n")
    print(f"recorded {nsteps + 1 - k0} steps in {(time.time()-t0)/60:.1f} min "
          f"(model {t_sim/60:.1f}, writing {t_rec/60:.1f})", flush=True)


def _peak_gb():
    b = peak_rss_bytes()
    return None if b is None else round(b / 2 ** 30, 2)


# ---------------------------------------------------------------- rendering
def iso_blocks(q, solid, origin, dx, level=0.5, B=64):
    """Iso-surface of an 8-bit grid field (uint8 array of the grid shape),
    extended into neighbouring solid cells as ``_dilate_into_solid`` does,
    computed block by block over the blocks that hold any of the field. The
    result is that of marching cubes over the whole zero-padded grid (as in
    ``_iso_obj``) at a fraction of the cost; vertices on block seams are
    merged."""
    from scipy import ndimage
    from skimage import measure
    shp = np.array(q.shape)
    nz = np.argwhere(q > 0)
    if len(nz) == 0:
        return np.zeros((0, 3)), np.zeros((0, 3), np.int64)
    # padded sample s = cell s-1; cube c spans samples c..c+1. After the
    # dilation a cell i reaches cubes i-1 .. i+2, i.e. at most one block over
    blk = np.unique((nz + 1) // B, axis=0)
    offs = np.array(np.meshgrid([-1, 0, 1], [-1, 0, 1], [-1, 0, 1])).reshape(3, -1).T
    cand = np.unique((blk[:, None, :] + offs[None]).reshape(-1, 3), axis=0)
    ncube = shp + 1
    cand = cand[np.all((cand >= 0) & (cand * B < ncube), axis=1)]
    lev = level * 255.0
    P, F, nv = [], [], 0
    for b in cand:
        c0 = b * B
        c1 = np.minimum(c0 + B, ncube)             # cubes c0 .. c1-1
        clo = np.maximum(c0 - 2, 0)                # cells used by the dilation
        chi = np.minimum(c1 + 1, shp)
        sl = tuple(slice(l, h) for l, h in zip(clo, chi))
        sub = q[sl]
        if not sub.any():
            continue
        d = np.where(solid[sl], ndimage.maximum_filter(sub, size=3), sub)
        g = np.zeros(tuple(c1 - c0 + 1), np.float32)   # samples c0 .. c1
        vs = np.maximum(c0 - 1, 0)                     # cells behind the samples
        ve = np.minimum(c1 - 1, shp - 1)
        if np.any(ve < vs):
            continue
        gi = tuple(slice(a - (c - 1), e - (c - 1) + 1) for a, e, c in zip(vs, ve, c0))
        di = tuple(slice(a - l, e - l + 1) for a, e, l in zip(vs, ve, clo))
        g[gi] = d[di]
        if g.max() < lev or g.min() >= lev:
            continue
        v, f, _, _ = measure.marching_cubes(g, lev)
        P.append(origin + (v + c0 - 0.5) * dx)
        F.append(f + nv)
        nv += len(v)
    if not P:
        return np.zeros((0, 3)), np.zeros((0, 3), np.int64)
    P = np.concatenate(P)
    F = np.concatenate(F)
    key = np.round(P / (dx * 1e-6)).astype(np.int64)      # merge seam vertices
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    F = inv.ravel()[F]
    F = F[(F[:, 0] != F[:, 1]) & (F[:, 1] != F[:, 2]) & (F[:, 0] != F[:, 2])]
    return P[first], F


_WAIT_STALL = 3600.0        # s without a new step before a render gives up (0: never)


def _wait_for_step(path, rec):
    """Wait for a step file of a recording that may still be running (a
    render can start with it). Fails at once if the recording has finished
    (its ``done`` marker exists) without that step, and when nothing has been
    written to the steps folder for ``_WAIT_STALL`` s, so a recording that died
    does not leave the render workers waiting for ever."""
    steps_dir = os.path.dirname(path)
    t_start = time.time()
    while not os.path.exists(path):
        if os.path.exists(os.path.join(rec, "done")):
            raise FileNotFoundError(f"{path}: the recording is finished and has no such step")
        if _WAIT_STALL > 0:
            try:
                last = max(t_start, os.path.getmtime(steps_dir))
            except OSError:
                last = t_start
            if time.time() - last > _WAIT_STALL:
                raise TimeoutError(
                    f"no new step in {steps_dir} for {_WAIT_STALL:.0f} s while waiting for "
                    f"{os.path.basename(path)}: the recording (--record) seems to have "
                    f"stopped (--wait-timeout changes the limit, 0 waits for ever)")
        time.sleep(5.0)


class _Step:
    """One recorded model step, loaded lazily."""

    def __init__(self, rec, k, shape, wait=True):
        path = os.path.join(rec, "steps", f"s{k:05d}.npz")
        if not os.path.exists(path):
            if not wait:
                raise FileNotFoundError(path)
            _wait_for_step(path, rec)
        self.z = np.load(path)
        self.shape = shape
        self.k = k

    def field(self, name, ncells):
        q = np.zeros(ncells, np.uint8)
        q[self.z[name + "_i"]] = self.z[name + "_v"]
        return q.reshape(self.shape)


class ReplayScene(CarScene):
    """CarScene fed from recorded steps instead of a live Simulation."""

    def __init__(self, proxy, *args, **kw):
        super().__init__(proxy, *args, **kw)
        self._k = None
        self._surf = {}

    def set_step(self, st, fields=True):
        """Take the state of a recorded step; fields=False only adds its
        drops (steps between two frames, or before the first frame)."""
        f = self.sim.film
        if fields:
            if self._k == st.k:
                return
            self._k = st.k
            g = self.sim.grid
            # surfaces in the object frame, smoothed once per model step; the
            # frames move them with the actor matrix (like the car)
            for name, poly in (("liq", self.liq_poly), ("air", self.air_poly)):
                pts, faces = iso_blocks(st.field(name, g.ncells), g.solid, g.origin, g.dx)
                if len(pts):
                    if self.smooth > 0:
                        pts = self._smooth_pts(pts, faces)
                    poly.DeepCopy(self._poly(pts, faces))
                else:
                    poly.DeepCopy(self.vtk.vtkPolyData())
        else:
            for row in st.z["drips"]:
                f.s.drips.append((float(row[0]), row[1:4].copy(), float(row[4]),
                                  int(row[5])))
            return
        f.s.h = st.z["film_h"].astype(np.float64) * 1e-6
        f.s.sub = np.unpackbits(st.z["film_sub"])[:len(f.s.h)].astype(bool)
        for row in st.z["drips"]:
            f.s.drips.append((float(row[0]), row[1:4].copy(), float(row[4]), int(row[5])))

    def update(self, t, label, t_state):
        vm, R = self._matrix(t)
        self._vm = vm
        self.door.SetUserMatrix(vm)
        if self.edges is not None:
            self.edges.SetUserMatrix(vm)
        self._update_water(t, vm)
        self.liq.SetUserMatrix(vm)
        self.air.SetUserMatrix(vm)
        self.txt.SetInput(label)
        self._camera(t)


def _replay_setup(rec, a):
    """Proxy Simulation, meshes and motion for rendering from a recording."""
    import trimesh
    with open(os.path.join(rec, "meta.json")) as f:
        meta = json.load(f)
    while not os.path.exists(os.path.join(rec, "static.npz")):
        time.sleep(5.0)
    z = np.load(os.path.join(rec, "static.npz"))
    shape = tuple(int(v) for v in z["shape"])
    ncells = int(np.prod(shape))
    solid = np.unpackbits(z["solid"])[:ncells].astype(bool).reshape(shape)
    grid = SimpleNamespace(solid=solid, fluid=~solid, shape=shape, ncells=ncells, ndim=3,
                           origin=np.asarray(z["origin"], float), dx=float(z["dx"]),
                           triangles=None)
    if meta.get("motion_file"):
        mo, _ = ips_motion(meta["motion_file"], meta["bath_stl"], meta["car"]["centre_file"],
                           meta["car"]["scale_to_m"], meta["hang"])
    else:
        mo = car_motion(meta["rock"], meta["hang"], meta["motion"],
                        meta.get("rotation", "pitch"), meta.get("sense", 1))
    film = SimpleNamespace(c=SimpleNamespace(x=z["cx"], normal=z["cnormal"], n=len(z["cx"])),
                           s=SimpleNamespace(h=np.zeros(len(z["cx"])),
                                             sub=np.zeros(len(z["cx"]), bool), drips=[]))
    proxy = SimpleNamespace(grid=grid, motion=mo, film=film, subcells=meta["subcells"])
    dmesh = trimesh.Trimesh(z["dv"], z["df"], process=False)
    return meta, proxy, dmesh, shape


def render_chunk(a):
    """Render frames i0..i1 of a recording into one movie part (or PNGs)."""
    rec = a.render
    meta, proxy, dmesh, shape = _replay_setup(rec, a)
    tl = timeline(a, meta["t_dipout"], meta["t_end"])
    t_dipout, t_end = tl.t_dipout, tl.t_end
    steps = np.asarray(meta["step_times"])
    times = tl.frames()
    if a.preview:
        times = np.array(sorted(a.preview))
        i0, i1 = 0, len(times)
    else:
        i0, i1 = a.frames
    size3d = tuple(int(v) // 2 * 2 for v in a.size3d)
    size2d = (int(a.width2d) // 2 * 2, size3d[1])
    scene = ReplayScene(proxy, dmesh, size3d, np.arange(0, t_end + 1e-9, 1.0), ssaa=a.ssaa,
                        door_opacity=a.car_opacity, drop_scale=a.drop_scale,
                        cam_dist=a.cam_dist, azimuth=a.azimuth, elevation=a.elevation,
                        look=a.look, orbit=a.orbit,
                        shell_cache=os.path.join(rec, "film_shells.npz"))
    panel = CarPanel(size2d, tl)
    W, Hh = size3d[0] + size2d[0], size3d[1]
    ks = [int(np.searchsorted(steps, t + 1e-9) - 1) for t in times]
    # the curves up to the first frame of this part, and the drops before it
    first = ks[i0]
    sc_all = {}
    for k in sorted(set(ks[:i0 + 1])):
        st = _Step(rec, k, shape)
        sc_all[k] = st.z["scalars"]
    for i in range(i0):
        panel.add(times[i], *sc_all[ks[i]][1:5])
    k3 = int(np.searchsorted(steps, steps[first] - 3.0))
    for k in range(k3, first):
        scene.set_step(_Step(rec, k, shape), fields=False)     # drops still falling
    ff = None
    if not a.preview:
        ff = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{W}x{Hh}", "-r", str(a.fps), "-i", "-", "-c:v", "libx264",
             "-pix_fmt", "yuv420p", "-crf", str(a.crf), "-preset", a.preset, a.part],
            stdin=subprocess.PIPE)
    t0 = time.time()
    cur = None
    for i in range(i0, i1):
        t = times[i]
        k = ks[i]
        if cur is None or cur.k != k:
            # steps skipped between frames still bring their drops
            if cur is not None:
                for kk in range(cur.k + 1, k):
                    scene.set_step(_Step(rec, kk, shape), fields=False)
            cur = _Step(rec, k, shape)
            scene.set_step(cur)
        sc = cur.z["scalars"]
        air_l, liq_l, film_ml, drip_ml, ndrops = sc[1], sc[2], sc[3], sc[4], int(sc[5])
        label = tl.label(t)
        frame_dt = tl.speed(t) / a.fps
        scene.update(t, label, steps[k])
        scene.update_film(t, frame_dt)
        img3 = scene.grab()
        res = f"{meta['dx']*1e3:g} mm" + (f" octree ({meta['levels']} levels)"
                                          if meta.get("levels") else "") + \
            (f", {meta['dx']*1e3/meta['narrow']:g} mm in narrow gaps" if meta.get("narrow") else "")
        info_txt = (f"model {res}:  air {air_l:5.2f} l,  liquid {liq_l:5.2f} l,  "
                    f"film {film_ml:5.0f} ml,  {ndrops} drops")
        img2 = panel.render(t, air_l, liq_l, film_ml, drip_ml, info_txt)
        frame = np.hstack([img3, img2[:Hh]])
        if a.preview:
            from PIL import Image
            p = a.out.replace(".mp4", "") + f"_t{t:06.2f}.png"
            Image.fromarray(frame).save(p)
            print("wrote", p, flush=True)
        else:
            ff.stdin.write(frame.tobytes())
        if (i - i0) % 24 == 0:
            print(f"[part {a.part_id}] frame {i}/{i1}  t = {t:.2f} s  "
                  f"({(time.time()-t0)/max(i-i0, 1):.1f} s/frame)", flush=True)
    if ff is not None:
        ff.stdin.close()
        ff.wait()
    print(f"[part {a.part_id}] done: frames {i0}-{i1}, {(time.time()-t0)/60:.1f} min", flush=True)


def render(a):
    """Split the frames over --workers processes and join the parts."""
    rec = a.render
    while not os.path.exists(os.path.join(rec, "meta.json")):
        time.sleep(5.0)
    with open(os.path.join(rec, "meta.json")) as f:
        meta = json.load(f)
    if a.preview:
        a.part_id = 0
        render_chunk(a)
        return
    t0 = time.time()
    times = timeline(a, meta["t_dipout"], meta["t_end"]).frames()
    n = len(times)
    W = max(1, min(a.workers, n))
    cuts = np.linspace(0, n, W + 1).round().astype(int)
    out = os.path.abspath(a.out)
    parts_dir = out.replace(".mp4", "") + "_parts"
    os.makedirs(parts_dir, exist_ok=True)
    procs, parts = [], []
    base = [sys.executable, "-X", "faulthandler", "-u", os.path.abspath(__file__),
            "--render", rec, "--fps", str(a.fps), "--speedup", str(a.speedup),
            "--speedup-late", str(a.speedup_late), "--slow-hang", str(a.slow_hang),
            "--size3d", str(a.size3d[0]), str(a.size3d[1]), "--width2d", str(a.width2d),
            "--ssaa", str(a.ssaa), "--crf", str(a.crf), "--preset", a.preset,
            "--car-opacity", str(a.car_opacity), "--drop-scale", str(a.drop_scale),
            "--cam-dist", str(a.cam_dist), "--azimuth", str(a.azimuth),
            "--elevation", str(a.elevation), "--look", a.look, "--orbit", str(a.orbit),
            "--sim-dt", str(a.sim_dt), "--dt-hang", str(a.dt_hang),
            "--wait-timeout", str(a.wait_timeout)]
    for w in range(W):
        part = os.path.join(parts_dir, f"part{w:02d}.mp4")
        parts.append(part)
        log = open(os.path.join(parts_dir, f"part{w:02d}.log"), "w")
        procs.append((subprocess.Popen(base + ["--frames", str(cuts[w]), str(cuts[w + 1]),
                                               "--part", part, "--part-id", str(w)],
                                       stdout=log, stderr=subprocess.STDOUT), log))
    print(f"rendering {n} frames in {W} processes (logs in {parts_dir})", flush=True)
    bad = False
    for p, log in procs:
        p.wait()
        log.close()
        bad |= p.returncode != 0
    if bad:
        print("a render part failed; see the part logs", flush=True)
        sys.exit(1)
    lst = os.path.join(parts_dir, "parts.txt")
    with open(lst, "w") as f:
        for p in parts:
            f.write(f"file '{p}'\n")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
                    "-i", lst, "-c", "copy", "-movflags", "+faststart", out], check=True)
    print(f"wrote {out}: {n} frames, {(time.time()-t0)/60:.1f} min with {W} processes",
          flush=True)


def main(a):
    t0 = time.time()
    size3d = tuple(int(v) // 2 * 2 for v in a.size3d)
    size2d = (int(a.width2d) // 2 * 2, size3d[1])
    mesh, info = load_car(a.stl, a.orient, a.reverse)
    mo, t_dipout, _ = run_motion(a, info)
    t_end = mo.t_end
    if a.levels or a.narrow:
        raise SystemExit("octree runs (--levels, --narrow) are recorded and rendered: use --record "
                         "DIR, then --render DIR")
    lo = np.asarray(mesh.vertices).min(0) - a.pad
    hi = np.asarray(mesh.vertices).max(0) + a.pad
    grid = Grid.from_mesh(mesh, a.dx, bounds=(lo, hi))
    sim = Simulation(grid, mo, dt_max=a.sim_dt, cells_per_step=1e9, subcells=a.subcells,
                     film=True, throat_model=ThroatModel(Cd=a.cd), holes=explicit_holes(a, info),
                     channels=gap_channels(a, info))
    print(f"setup {time.time()-t0:.0f} s: grid {grid.shape}, {sim.N/1e6:.2f} M nodes, "
          f"{sim.film.c.n} film elements", flush=True)
    dmesh = display_mesh(mesh, a.display_faces)
    del mesh
    # frame times: real time to the end of the dip-out, then sped up
    fps = a.fps
    tl = timeline(a, t_dipout, t_end)
    steps = tl.steps()
    times = tl.frames()
    if a.preview:
        times = np.array(sorted(a.preview))
    scene = CarScene(sim, dmesh, size3d, np.arange(0, t_end + 1e-9, 1.0), ssaa=a.ssaa,
                     door_opacity=a.car_opacity, drop_scale=a.drop_scale,
                     cam_dist=a.cam_dist, azimuth=a.azimuth, elevation=a.elevation,
                        look=a.look, orbit=a.orbit)
    panel = CarPanel(size2d, tl)
    W, Hh = size3d[0] + size2d[0], size3d[1]
    ff = None
    if not a.preview:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        ff = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{W}x{Hh}", "-r", str(fps), "-i", "-", "-c:v", "libx264",
             "-pix_fmt", "yuv420p", "-crf", str(a.crf), "-preset", a.preset,
             "-movflags", "+faststart", a.out], stdin=subprocess.PIPE)
    t_sim = t_draw = 0.0
    for i, t in enumerate(times):
        # advance the model in fixed steps; the car pose uses the exact t
        k = int(np.searchsorted(steps, t + 1e-9) - 1)
        ts = float(steps[k])
        sim.dt_max = tl.dt if ts <= tl.t_slow + 1e-9 else tl.dt_hang
        s0 = time.time()
        if ts > sim.t + 1e-9:
            sim.run(t_end=float(ts))
        t_sim += time.time() - s0
        H = sim.hist
        f = sim.film
        air_l = H.air_trapped[-1] * 1e3
        liq_l = H.liquid_trapped[-1] * 1e3
        film_ml = f.volume * 1e6
        drip_ml = f.drip_volume * 1e6
        ndrops = sum(d[3] for d in f.s.drips)
        label = tl.label(t)
        s0 = time.time()
        frame_dt = tl.speed(t) / fps
        scene.update(t, label, sim.t)
        scene.update_film(t, frame_dt)
        img3 = scene.grab()
        info_txt = (f"model {a.dx*1e3:g} mm:  air {air_l:5.2f} l,  liquid {liq_l:5.2f} l,  "
                    f"film {film_ml:5.0f} ml,  {ndrops} drops")
        img2 = panel.render(t, air_l, liq_l, film_ml, drip_ml, info_txt)
        frame = np.hstack([img3, img2[:Hh]])
        t_draw += time.time() - s0
        if a.preview:
            from PIL import Image
            p = a.out.replace(".mp4", "") + f"_t{t:06.2f}.png"
            Image.fromarray(frame).save(p)
            print("wrote", p, flush=True)
        else:
            ff.stdin.write(frame.tobytes())
        if i % 48 == 0:
            print(f"frame {i}/{len(times)}  t = {t:.2f} s  air {air_l:.2f} l  liquid "
                  f"{liq_l:.2f} l  film {film_ml:.0f} ml  drops {ndrops}  sim {t_sim/60:.1f} min, "
                  f"draw {t_draw/60:.1f} min", flush=True)
    if ff is not None:
        ff.stdin.close()
        ff.wait()
        print("wrote", a.out, flush=True)
    print(f"total {(time.time()-t0)/60:.1f} min: simulation {t_sim/60:.1f} min, "
          f"drawing {t_draw/60:.1f} min", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stl", default=None)
    ap.add_argument("--dx", type=float, default=0.02)
    ap.add_argument("--subcells", type=int, default=2)
    ap.add_argument("--motion", choices=("short", "full"), default="short")
    ap.add_argument("--motion-file", default=None, metavar="XMO",
                    help="an IPS motion (.xmo) instead of --motion: the STL is the car at "
                         "the motion's first pose, in the plant frame (needs --bath-stl; "
                         "the car keeps the file's axes); --hang adds hanging at the end")
    ap.add_argument("--bath-stl", default=None, metavar="STL",
                    help="the bath of --motion-file: its top is the bath surface")
    add_car_args(ap)
    ap.add_argument("--rock", type=float, default=5.0)
    ap.add_argument("--hang", type=float, default=120.0,
                    help="seconds of hanging after the dip-out (drip-off)")
    ap.add_argument("--sim-dt", type=float, default=0.1)
    ap.add_argument("--cd", type=float, default=0.65)
    ap.add_argument("--pad", type=float, default=0.06)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--speedup", type=float, default=2.0,
                    help="movie speed for the first --slow-hang seconds of the hanging")
    ap.add_argument("--slow-hang", type=float, default=10.0)
    ap.add_argument("--speedup-late", type=float, default=10.0,
                    help="movie speed for the rest of the hanging")
    ap.add_argument("--dt-hang", type=float, default=0.5,
                    help="model step once the car hangs still (after --slow-hang)")
    ap.add_argument("--preview", type=float, nargs="*")
    ap.add_argument("--size3d", type=int, nargs=2, default=[1600, 1080], metavar=("W", "H"))
    ap.add_argument("--width2d", type=int, default=1000)
    ap.add_argument("--ssaa", type=int, default=2)
    ap.add_argument("--crf", type=int, default=16)
    ap.add_argument("--preset", default="slow")
    ap.add_argument("--display-faces", type=int, default=800_000)
    ap.add_argument("--display-max-cells", type=float, default=100e6,
                    help="octree runs: the finest display grid (h * 2**j) with at most "
                         "this many cells")
    ap.add_argument("--car-opacity", type=float, default=0.25)
    ap.add_argument("--drop-scale", type=float, default=3.0)
    ap.add_argument("--cam-dist", type=float, default=10.0)
    ap.add_argument("--azimuth", type=float, default=-55.0)
    ap.add_argument("--elevation", type=float, default=14.0)
    ap.add_argument("--look", choices=("hd", "plain"), default="hd",
                    help="hd: three-point light, glossy liquid, bath surface sheet")
    ap.add_argument("--orbit", type=float, default=0.0,
                    help="camera turns this many degrees around the car over the movie")
    ap.add_argument("--out", default="runs_car/car_dip.mp4")
    ap.add_argument("--threads", type=int, default=2,
                    help="threads for the model's parallel passes (1 = off)")
    ap.add_argument("--record", default=None, metavar="DIR",
                    help="run the model only and write its state per step to DIR")
    ap.add_argument("--save-at", nargs="+", default=None, metavar="T",
                    help="--record: save the whole state at these times (s; 'end': after "
                         "the last step) to DIR/state_t<T>.pkl, to go on later with --resume")
    ap.add_argument("--resume", default=None, metavar="STATE",
                    help="go on with a recording from a saved state (into --record DIR if "
                         "given, else into the recording it was saved from)")
    ap.add_argument("--render", default=None, metavar="DIR",
                    help="render a recording (see --workers)")
    ap.add_argument("--workers", type=int, default=8,
                    help="render processes for --render")
    ap.add_argument("--frames", type=int, nargs=2, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--part", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--part-id", type=int, default=0, help=argparse.SUPPRESS)
    ap.add_argument("--wait-timeout", type=float, default=3600.0,
                    help="--render: give up after this many seconds without a new step "
                         "from the recording (0: wait for ever). The wait for the "
                         "recording's setup (meta.json, static.npz) is not limited: "
                         "the setup of a fine grid takes hours")
    a = ap.parse_args()
    if a.motion_file:
        if not a.bath_stl:
            ap.error("--motion-file needs --bath-stl")
        a.orient, a.reverse = "file", False        # the motion is in the file's frame
    _WAIT_STALL = a.wait_timeout
    if a.resume:
        resume(a)
    elif a.record:
        record(a)
    elif a.render and a.frames is not None:
        render_chunk(a)
    elif a.render:
        render(a)
    else:
        main(a)
