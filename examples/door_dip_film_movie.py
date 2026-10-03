"""Showcase movie: a full dip of the door with wall film, surface flow and drops.

The door (tilted, default 45°, drain holes open) is dipped and lifted. The
movie shows:

- the water held in the door (blue, kept inside the geometry, drawn on top
  of the translucent sheet and film so it stays visible);
- the jets through the drain holes;
- the wall film deposited on withdrawal, coloured by thickness (µm, log
  scale 8-40 µm by default) and fading to transparent as it thins, so
  the drainage shows as the door returning to bare grey;
- drops released from hanging film, falling to the bath or onto the door
  (drawn 3× enlarged).

The first ``--t-realtime`` seconds play in real time, then the dripping
tail is sped up ``--speedup`` times until ``--t-end``. Right panel: water in
the door (pools), film on the walls and cumulative dripped volume, with a
cursor.

    python examples/door_dip_film_movie.py --stl door.stl --dx 0.006 \\
           --size3d 1600 1080 --width2d 1000 --out dip_film_45.mp4

``--preview 9 12 20 60`` writes stills instead. Needs vtk and ffmpeg.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import time
import warnings

import numpy as np

from door_article import DOOR_HOLES, door_setup
from door_drain_movie import Scene
from drainsim.model import Simulation
from drainsim.physics import ThroatModel
from drainsim.worldviz import to_world

warnings.filterwarnings("ignore", category=DeprecationWarning)
C_POOL, C_FILM, C_DRIP = "#2a78d6", "#eb6834", "#1baf7a"
# Film colour scale (log): below H_MIN the film is fully transparent (the
# door looks bare), towards H_MAX it becomes a deep, fairly opaque orange.
# The opacity rises with the thickness, so the draining shows as the colour
# fading back to the door. The defaults span the thicknesses of a water film
# 10 s-2 min after withdrawal (median about 32 -> 10 µm): early film is
# clearly orange, and by the end most of the door is back to bare grey except
# where film collects at the lowest edges.
H_MIN_UM, H_MAX_UM = 8.0, 40.0
FILM_ALPHA, FILM_GAMMA = 0.85, 1.5
FILM_C0 = np.array([0.97, 0.70, 0.48])      # thin end: light orange
FILM_C1 = np.array([0.85, 0.28, 0.06])      # thick end: vivid orange-red
DOOR_RGB = np.array([0.72, 0.73, 0.76])
WATER_HD = (0.07, 0.29, 0.70)
AIR_HD = (0.95, 0.66, 0.04)


class FilmScene(Scene):
    """Scene plus the film (coloured shells on the STL) and falling drops."""

    def __init__(self, sim, mesh, holes, size, t_all, drop_scale=3.0, **kw):
        pools_on_top = kw.pop("pools_on_top", True)
        self.cull = kw.pop("cull", True)
        cam_box = kw.pop("cam_box", None)
        self.follow = float(kw.pop("follow", 0.0))
        self.t_ref = kw.pop("t_ref", 0.0)
        look = kw.pop("look", "hd")
        self.orbit_deg = float(kw.pop("orbit", 0.0))
        self.shell_cache = kw.pop("shell_cache", None)
        self._t_orbit = float(kw.pop("t_orbit", np.max(t_all) if len(t_all) else 1.0)) or 1.0
        super().__init__(sim, mesh, holes, size, t_all, **kw)
        import vtk
        top = None
        if pools_on_top:
            # draw the pools over the door and film (x-ray style) so they
            # stay visible behind several translucent film/sheet layers
            top = vtk.vtkRenderer()
            top.SetLayer(self.win.GetNumberOfLayers())
            top.SetActiveCamera(self.ren.GetActiveCamera())
            self.win.SetNumberOfLayers(self.win.GetNumberOfLayers() + 1)
            self.win.AddRenderer(top)
            self.wren.RemoveActor(self.liq)
            top.AddActor(self.liq)
            self.liq.GetProperty().SetOpacity(0.92)
        import matplotlib
        self.cmap = matplotlib.colormaps["Oranges"]
        self.drop_scale = drop_scale
        self._build_film_shells(mesh)
        # drops: glyphs at points
        self.drop_pts = vtk.vtkPoints()
        self.drop_pd = vtk.vtkPolyData()
        self.drop_pd.SetPoints(self.drop_pts)
        self.drop_r = vtk.vtkFloatArray()
        self.drop_r.SetName("r")
        self.drop_pd.GetPointData().SetScalars(self.drop_r)
        sph = vtk.vtkSphereSource()
        sph.SetRadius(1.0)
        sph.SetThetaResolution(12)
        sph.SetPhiResolution(8)
        gl = vtk.vtkGlyph3D()
        gl.SetInputData(self.drop_pd)
        gl.SetSourceConnection(sph.GetOutputPort())
        gl.SetScaleModeToScaleByScalar()
        gl.SetScaleFactor(1.0)
        gl.SetColorModeToColorByScale()
        m = vtk.vtkPolyDataMapper()
        m.SetInputConnection(gl.GetOutputPort())
        m.ScalarVisibilityOff()
        a = vtk.vtkActor()
        a.SetMapper(m)
        a.GetProperty().SetColor(0.10, 0.33, 0.78)
        a.GetProperty().SetSpecular(0.5)
        self.ren.AddActor(a)
        self.drops_actor = a
        self._drops = []                       # (t0, P0 world, fall dist, radius)
        self._ndrips = 0
        # colour bar for the film thickness
        lut = vtk.vtkLookupTable()
        lut.SetNumberOfTableValues(256)
        # the bar shows what the film looks like over the grey door
        for q in range(256):
            x = q / 255
            rgb = FILM_C0 + x * (FILM_C1 - FILM_C0)
            al = FILM_ALPHA * x ** FILM_GAMMA
            r, g, b = al * rgb + (1 - al) * DOOR_RGB
            lut.SetTableValue(q, r, g, b, 1.0)
        lut.SetScaleToLog10()
        lut.SetTableRange(H_MIN_UM, H_MAX_UM)
        lut.Build()
        bar = vtk.vtkScalarBarActor()
        bar.SetLookupTable(lut)
        bar.SetTitle("wall film (µm)")
        bar.SetNumberOfLabels(2)
        bar.SetLabelFormat("%.0f")
        bar.SetOrientationToHorizontal()
        bar.SetPosition(0.62, 0.03)
        bar.SetWidth(0.34)
        bar.SetHeight(0.08)
        bar.UnconstrainedFontSizeOn()
        for tp in (bar.GetTitleTextProperty(), bar.GetLabelTextProperty()):
            tp.SetColor(0.1, 0.1, 0.1)
            tp.SetFontSize(int(16 * self.size[1] / 720 * self.ssaa))
            tp.ItalicOff()
            tp.BoldOff()
            tp.ShadowOff()
        self.ren.AddViewProp(bar)
        self.film_bar = bar
        if cam_box is not None:
            self._frame_camera(np.asarray(cam_box, float))
        cam = self.ren.GetActiveCamera()
        self._cam0 = (np.array(cam.GetFocalPoint()), np.array(cam.GetPosition()))
        self._water_hd = False
        if look == "hd":
            self._look_hd()
            self.film_bar.SetPosition(0.64, 0.905)      # clear of the time label
            self.film_bar.SetWidth(0.32)
            self.film_bar.SetHeight(0.07)
            # the part under the bath surface tinted, and the waterline
            self._underwater(top if pools_on_top else None)
            self._water_hd = True

    def _look_hd(self):
        """Presentation look (as the HD door drainage movie): three-point
        lighting, glossy deep-blue liquid (and golden air, if drawn), a clear
        bath surface sheet over a pale bath, glossy jets and drops."""
        vtk = self.vtk
        rens = self.win.GetRenderers()
        rens.InitTraversal()
        for _ in range(rens.GetNumberOfItems()):
            ren = rens.GetNextItem()
            ren.RemoveAllLights()
            lk = vtk.vtkLightKit()
            lk.SetKeyLightIntensity(0.95)
            lk.SetKeyToFillRatio(2.6)
            lk.SetKeyToHeadRatio(3.0)
            lk.AddLightsToRenderer(ren)
        lp = self.liq.GetProperty()
        lp.SetColor(*WATER_HD)
        lp.SetSpecular(0.55)
        lp.SetSpecularPower(45)
        lp.SetOpacity(0.95)
        if getattr(self, "air", None) is not None:
            ap = self.air.GetProperty()
            ap.SetColor(*AIR_HD)
            ap.SetSpecular(0.5)
            ap.SetSpecularPower(40)
        for j in getattr(self, "jets", []):
            jp = j.GetProperty()
            jp.SetColor(*WATER_HD)
            jp.SetSpecular(0.6)
            jp.SetSpecularPower(60)
            jp.SetOpacity(0.9)
        if getattr(self, "drops_actor", None) is not None:
            self.drops_actor.GetProperty().SetColor(*WATER_HD)
            self.drops_actor.GetProperty().SetSpecularPower(60)
        bp = self.bath.GetProperty()
        bp.SetOpacity(0.12)
        bp.SetColor(0.62, 0.79, 0.96)
        bpd = self.bath.GetMapper().GetInput()
        b = bpd.GetBounds()
        zb = float(self.sim.motion.bath_level)
        # the sheet is the surface: the bath box loses its top face, which
        # lies in the same plane (two coplanar translucent faces flicker as
        # the camera moves: z-fighting)
        keep = vtk.vtkCellArray()
        for c in range(bpd.GetNumberOfCells()):
            ids = bpd.GetCell(c).GetPointIds()
            z = [bpd.GetPoint(ids.GetId(k))[2] for k in range(ids.GetNumberOfIds())]
            if not all(abs(zz - zb) < 1e-9 for zz in z):
                keep.InsertNextCell(ids)
        bpd.SetPolys(keep)
        bpd.Modified()
        pl = vtk.vtkPlaneSource()
        pl.SetOrigin(b[0], b[2], zb)
        pl.SetPoint1(b[1], b[2], zb)
        pl.SetPoint2(b[0], b[3], zb)
        pl.Update()
        self.surf = self._actor(pl.GetOutput(), (0.40, 0.64, 0.92), 0.28, specular=0.45)
        self.surf.GetProperty().SetSpecularPower(30)
        self.ren.AddActor(self.surf)

    def update(self, t, label, hole_flows, qmax):
        super().update(t, label, hole_flows, qmax)
        if self._water_hd:
            self._update_water(t, self._vm)

    def _frame_camera(self, box, azimuth=200.0, elevation=12.0, zoom=1.15):
        ren = self.ren
        ren.ResetCamera(*box.ravel())
        cam = ren.GetActiveCamera()
        cam.SetViewUp(0, 0, 1)
        ctr = box.mean(1)
        cam.SetFocalPoint(*ctr)
        cam.SetPosition(ctr[0], ctr[1] - 1.0, ctr[2])
        cam.Azimuth(azimuth)
        cam.Elevation(elevation)
        ren.ResetCamera(*box.ravel())
        cam.Zoom(zoom)

    def _follow(self, t):
        """Move the camera down with the door by ``follow`` times its
        vertical displacement from the hanging position (0 = fixed camera),
        and turn it ``orbit`` degrees around the vertical over the movie."""
        if self.follow == 0.0 and self.orbit_deg == 0.0:
            return
        from drainsim.worldviz import pose3
        off = np.zeros(3)
        if self.follow != 0.0:
            dz = pose3(self.sim.motion, t)[1][2] - pose3(self.sim.motion, self.t_ref)[1][2]
            off[2] = self.follow * dz
        f0, p0 = self._cam0
        a = np.radians(self.orbit_deg) * min(max(t / self._t_orbit, 0.0), 1.0)
        d = p0 - f0
        c, s_ = np.cos(a), np.sin(a)
        d = np.array([c * d[0] - s_ * d[1], s_ * d[0] + c * d[1], d[2]])
        cam = self.ren.GetActiveCamera()
        cam.SetFocalPoint(*(f0 + off))
        cam.SetPosition(*(f0 + d + off))
        self.ren.ResetCameraClippingRange()

    # ---------------------------------------------------------------- film
    SHELL_MAX_EDGE = 6e-3        # the shells are refined to max(cell, this)
    SHELL_DELTA = 6e-4           # offset of each shell from the sheet
    SHELL_KQ = 16                # carrier elements tried per shell vertex

    def _film_shell_arrays(self, mesh):
        """Two thin shells on the STL (one per side of each sheet), refined
        to about one cell, each vertex mapped to the nearest film carrier
        element on that side (carrier normal pointing the same way).
        Returns ([(vertices, normals, mapping) per side], faces)."""
        import trimesh
        from scipy.spatial import cKDTree
        sim, g = self.sim, self.sim.grid
        c = sim.film.c
        lo = g.origin - g.dx
        hi = g.origin + np.array(g.shape) * g.dx + g.dx
        tc = mesh.triangles_center
        keep = np.all((tc > lo) & (tc < hi), axis=1)
        sub = mesh.submesh([np.flatnonzero(keep)], append=True)
        v, f = trimesh.remesh.subdivide_to_size(sub.vertices, sub.faces,
                                                max_edge=max(g.dx, self.SHELL_MAX_EDGE))
        sub = trimesh.Trimesh(v, f, process=True)
        vn = sub.vertex_normals
        tree = cKDTree(c.x)
        d, idx = tree.query(sub.vertices, k=self.SHELL_KQ, distance_upper_bound=1.5 * g.dx)
        sides = []
        for sgn in (1.0, -1.0):
            ok = np.isfinite(d)
            idc = np.where(ok, idx, 0)
            dots = np.einsum("vkj,vj->vk", c.normal[idc], sgn * vn)
            good = ok & (dots > 0.3)
            first = np.where(good.any(1), good.argmax(1), -1)
            mapping = np.where(first >= 0, idc[np.arange(len(idc)), np.maximum(first, 0)], -1)
            sides.append((np.ascontiguousarray(sub.vertices + sgn * self.SHELL_DELTA * vn),
                          np.ascontiguousarray(sgn * vn), mapping))
        return sides, np.asarray(sub.faces, np.int64)

    def _shell_meta(self, mesh):
        """What the shells depend on (a cache made for anything else is not used)."""
        from drainsim.par import fingerprint
        g, c = self.sim.grid, self.sim.film.c
        return repr(dict(version=1, dx=float(g.dx), origin=[float(x) for x in g.origin],
                         shape=[int(x) for x in g.shape], max_edge=self.SHELL_MAX_EDGE,
                         delta=self.SHELL_DELTA, kq=self.SHELL_KQ,
                         carrier=fingerprint(c.x, c.normal),
                         mesh=fingerprint(np.asarray(mesh.vertices), np.asarray(mesh.faces))))

    def _cached_film_shells(self, mesh, path, wait=3600.0):
        """``_film_shell_arrays`` built once per recording and read from
        ``path`` by every later render (worker). The first one to take the
        lock builds and writes it (atomically); the others wait for the file."""
        meta = self._shell_meta(mesh)

        def load():
            try:
                with np.load(path) as z:
                    if str(z["meta"]) != meta:
                        return None
                    sides = [(z[f"verts{s}"], z[f"normals{s}"], z[f"map{s}"]) for s in (0, 1)]
                    return sides, z["faces"]
            except (OSError, KeyError, ValueError):
                return None

        lock = path + ".lock"
        t0 = time.time()
        while True:
            got = load()
            if got is not None:
                return got
            try:
                os.close(os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
            except FileExistsError:
                try:
                    stale = time.time() - os.path.getmtime(lock) > wait
                except OSError:
                    continue                                 # just released: read it
                if stale or time.time() - t0 > wait:
                    return self._film_shell_arrays(mesh)     # builder gone: build, don't write
                time.sleep(5.0)
                continue
            try:
                sides, faces = self._film_shell_arrays(mesh)
                tmp = f"{path}.{os.getpid()}.tmp.npz"
                np.savez(tmp, meta=np.array(meta), faces=faces,
                         **{f"{k}{s}": a for s, side in enumerate(sides)
                            for k, a in zip(("verts", "normals", "map"), side)})
                os.replace(tmp, path)
                return sides, faces
            finally:
                os.remove(lock)

    def _build_film_shells(self, mesh):
        """The film shells as VTK actors (see ``_film_shell_arrays``); with
        ``shell_cache`` (a file path) they are built once and read from it."""
        vtk = self.vtk
        from vtk.util.numpy_support import numpy_to_vtk
        if self.shell_cache:
            sides, faces = self._cached_film_shells(mesh, self.shell_cache)
        else:
            sides, faces = self._film_shell_arrays(mesh)
        self.shells = []
        for verts, normals, mapping in sides:
            pts = vtk.vtkPoints()
            pts.SetData(numpy_to_vtk(verts, deep=True))
            nrm = numpy_to_vtk(normals, deep=True)
            nrm.SetName("Normals")
            pd = vtk.vtkPolyData()
            pd.SetPoints(pts)
            pd.GetPointData().SetNormals(nrm)
            m = vtk.vtkPolyDataMapper()
            m.SetInputData(pd)
            m.SetColorModeToDirectScalars()
            m.ScalarVisibilityOn()
            a = vtk.vtkActor()
            a.SetMapper(m)
            a.GetProperty().SetSpecular(0.2)
            a.GetProperty().SetOpacity(0.999)          # force translucent pass
            self.ren.AddActor(a)
            self.shells.append(dict(pd=pd, map=mapping, actor=a, faces=faces))
        self._film_key = None

    def _film_colours(self, h_um):
        x = np.clip((np.log10(np.maximum(h_um, 1e-3)) - np.log10(H_MIN_UM))
                    / (np.log10(H_MAX_UM) - np.log10(H_MIN_UM)), 0, 1)
        rgb = FILM_C0[None] + x[:, None] * (FILM_C1 - FILM_C0)[None]
        alpha = FILM_ALPHA * x ** FILM_GAMMA
        return np.column_stack([rgb, alpha])

    # --------------------------------------------------------------- drops
    def _fall_distance(self, x_obj, t0):
        """Free-fall distance from a release point down to the door surface
        below it, or to the bath (object frame at the release time).

        The drop is marched straight down through the voxel grid in half-cell
        steps until it enters a solid cell. This is pure numpy: an earlier
        VTK ray-cast locator crashed on Windows because its data set had
        been freed.
        """
        sim = self.sim
        g = sim.grid
        up, zb = sim.motion.frame(t0)
        up = np.asarray(up, float)
        to_bath = float(x_obj @ up - zb)
        if to_bath <= 0:
            return 0.0
        s = np.arange(g.dx, to_bath, 0.5 * g.dx)        # skip the release cell
        if s.size == 0:
            return to_bath
        P = x_obj[None, :] - s[:, None] * up[None, :]
        ci = np.floor((P - g.origin) / g.dx).astype(np.int64)
        inside = np.all((ci >= 0) & (ci < np.array(g.shape)), axis=1)
        hit = np.zeros(len(s), bool)
        ok = np.flatnonzero(inside)
        hit[ok] = g.solid[ci[ok, 0], ci[ok, 1], ci[ok, 2]]
        k = np.flatnonzero(hit)
        return float(s[k[0]]) if k.size else to_bath

    def update_film(self, t, frame_dt):
        self._follow(t)
        sim = self.sim
        film = sim.film
        from drainsim.par import fingerprint
        from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray
        # the colours and the drawn triangles change only with the film (once
        # per model step); between those, the shells only move with the object
        key = fingerprint(film.s.h, film.s.sub)
        if key != self._film_key:
            self._film_key = key
            self._colour_shells(film, numpy_to_vtk, numpy_to_vtkIdTypeArray)
        for sh in self.shells:
            sh["actor"].SetUserMatrix(self._vm)
        self._update_drops(t, frame_dt)

    def _colour_shells(self, film, numpy_to_vtk, numpy_to_vtkIdTypeArray):
        # film under liquid (in a pool or in the bath) is not drawn: the
        # water is shown there instead. The colour of a vertex is that of its
        # carrier element (the same numbers as colouring every vertex).
        h_um = np.where(film.s.sub, 0.0, film.s.h * 1e6)
        ce = (self._film_colours(h_um) * 255).astype(np.uint8)
        c0 = (self._film_colours(np.zeros(1)) * 255).astype(np.uint8)[0]
        for sh in self.shells:
            col = np.where((sh["map"] >= 0)[:, None], ce[np.maximum(sh["map"], 0)], c0)
            arr = numpy_to_vtk(col, deep=True, array_type=self.vtk.VTK_UNSIGNED_CHAR)
            arr.SetName("film")
            sh["pd"].GetPointData().SetScalars(arr)
            # only triangles that show some film (keeps the translucent pass
            # cheap). Never hand the mapper an empty triangle list: some
            # Windows OpenGL drivers crash on it; hide the shell instead.
            if self.cull:
                F = sh["faces"][(col[:, 3] > 0)[sh["faces"]].any(1)]
            else:
                F = sh["faces"]
            if len(F) == 0:
                sh["actor"].VisibilityOff()
                continue
            ca = self.vtk.vtkCellArray()
            off = np.arange(0, 3 * len(F) + 1, 3, dtype=np.int64)
            ca.SetData(numpy_to_vtkIdTypeArray(off, deep=True),
                       numpy_to_vtkIdTypeArray(np.ascontiguousarray(F.ravel()),
                                               deep=True))
            sh["pd"].SetPolys(ca)
            sh["pd"].Modified()
            sh["actor"].VisibilityOn()

    def _update_drops(self, t, frame_dt):
        sim, film = self.sim, self.sim.film
        # new drips since the last frame
        g = 9.81
        for (t0, x, vol, n) in film.s.drips[self._ndrips:]:
            P0 = to_world(np.asarray(x)[None], sim.motion, t0)[0]
            D = self._fall_distance(np.asarray(x, float), t0)
            r = (3 * vol / max(n, 1) / (4 * np.pi)) ** (1 / 3) * self.drop_scale
            self._drops.append((t0, P0, D, r))
        self._ndrips = len(film.s.drips)
        self.drop_pts.Reset()
        self.drop_r.Reset()
        alive = []
        for (t0, P0, D, r) in self._drops:
            tf = np.sqrt(2 * D / g)
            age = t - t0
            if age < 0:
                alive.append((t0, P0, D, r))
                continue
            # show every drop in at least one frame, even when sped up
            if age > max(tf, frame_dt):
                continue
            s = min(0.5 * g * min(age, tf) ** 2, D)
            if frame_dt > tf:
                s = 0.5 * D
            self.drop_pts.InsertNextPoint(P0[0], P0[1], P0[2] - s)
            self.drop_r.InsertNextValue(r)
            alive.append((t0, P0, D, r))
        self._drops = alive
        self.drop_pts.Modified()
        self.drop_pd.Modified()


# ------------------------------------------------------------------ 2D panel
class DipPanel:
    def __init__(self, size, t_end, t_rt, speedup, t_motion=8.0, tilt=45.0):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        dpi = 100 * size[1] / 720
        self.fig = plt.figure(figsize=(size[0] / dpi, size[1] / dpi), dpi=dpi)
        self.canvas = FigureCanvasAgg(self.fig)
        a1 = self.fig.add_axes([0.15, 0.53, 0.80, 0.33])
        a2 = self.fig.add_axes([0.15, 0.10, 0.80, 0.33], sharex=a1)
        self.axes = (a1, a2)
        for ax in self.axes:
            ax.axvspan(0, t_motion, color="#f1f0ec", lw=0, zorder=0)
            ax.axvspan(t_rt, t_end, color="#f7f6f2", lw=0, zorder=0)
            ax.grid(color="#e6e5e0", lw=0.8)
            ax.set_axisbelow(True)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
        a1.text(0.5 * t_motion, 1.0, "dip", transform=a1.get_xaxis_transform(), ha="center",
                va="bottom", fontsize=9, color="#52514e")
        a1.text(0.5 * (t_rt + t_end), 1.0, f"played {speedup:g}× faster",
                transform=a1.get_xaxis_transform(), ha="center", va="bottom",
                fontsize=9, color="#52514e")
        a1.set_xlim(0, t_end)
        a1.set_ylabel("water in the door (l)")
        a2.set_ylabel("volume (ml)")
        a2.set_xlabel("time (s)")
        plt.setp(a1.get_xticklabels(), visible=False)
        self.l_pool, = a1.plot([], [], color=C_POOL, lw=2.2)
        self.l_film, = a2.plot([], [], color=C_FILM, lw=2.2, label="film on the walls")
        self.l_drip, = a2.plot([], [], color=C_DRIP, lw=2.2, label="dripped (cumulative)")
        a2.legend(frameon=False, fontsize=9, loc="upper right")
        self.cur = [ax.axvline(0, color="#52514e", lw=0.8, alpha=0.6) for ax in self.axes]
        self.fig.text(0.15, 0.955, f"Door dip at {tilt:g}°: pools, wall film and drops",
                      fontsize=13, color="#0b0b0b")
        self.info = self.fig.text(0.15, 0.91, "", fontsize=10.5, color="#52514e")
        self.t, self.pool, self.film, self.drip = [], [], [], []
        self.ymax = [0.3, 10.0]

    def render(self, t, pool_l, film_ml, drip_ml, info):
        self.t.append(t)
        self.pool.append(pool_l)
        self.film.append(film_ml)
        self.drip.append(drip_ml)
        self.l_pool.set_data(self.t, self.pool)
        self.l_film.set_data(self.t, self.film)
        self.l_drip.set_data(self.t, self.drip)
        self.ymax[0] = max(self.ymax[0], 1.15 * pool_l)
        self.ymax[1] = max(self.ymax[1], 1.15 * film_ml, 1.15 * drip_ml)
        self.axes[0].set_ylim(0, self.ymax[0])
        self.axes[1].set_ylim(0, self.ymax[1])
        for c in self.cur:
            c.set_xdata([t, t])
        self.info.set_text(info)
        self.canvas.draw()
        return np.ascontiguousarray(np.asarray(self.canvas.buffer_rgba())[:, :, :3])


# ------------------------------------------------------------------ driver
def frame_times(fps, t_rt, speedup, t_end):
    a = np.arange(0.0, t_rt, 1.0 / fps)
    b = np.arange(t_rt, t_end + 1e-9, speedup / fps)
    return np.concatenate([a, b])


def main(stl, dx, tilt, cd, subcells, out, fps=24, t_rt=28.0, speedup=5.0,
         t_down=4.0, t_up=4.0, full_dip=False, follow=0.0,
         t_end=128.0, size3d=(1040, 720), size2d=(720, 720), preview=None,
         ssaa=2, crf=16, preset="slow", door_opacity=0.35, drop_scale=3.0,
         pools_on_top=True, cull=True, look="hd", orbit=0.0):
    import trimesh
    t0 = time.time()
    size3d = tuple(int(v) // 2 * 2 for v in size3d)
    size2d = (int(size2d[0]) // 2 * 2, size3d[1])
    t_motion = t_down + t_up
    kw_setup = dict(t_down=t_down, t_up=t_up)
    cam_box = None
    if full_dip:
        # dip the whole door: travel = start height + vertical extent of the
        # tilted door + 5 cm, so its top goes 5 cm under
        from drainsim.motion import rot3
        V = trimesh.load(stl, force="mesh").vertices
        W = (V - V.mean(0)) @ rot3(np.array([0.0, tilt, 0.0])).T
        ext = float(W[:, 2].max() - W[:, 2].min())
        kw_setup["depth"] = 0.15 + ext + 0.05
    mesh, grid, motion = door_setup(stl, dx, tilt, t_drain=t_end - t_motion, **kw_setup)
    if full_dip:
        from drainsim.worldviz import to_world
        Wh = to_world(mesh.vertices[::20], motion, t_end)       # hanging position
        lo, hi = Wh.min(0), Wh.max(0)
        zb = motion.bath_level
        cam_box = np.array([[lo[0], hi[0]], [lo[1], hi[1]],
                            [min(lo[2], zb) - 0.15, hi[2] + 0.05]])
        print(f"full dip: travel {kw_setup['depth']:.2f} m in {t_down:g} s down / "
              f"{t_up:g} s up", flush=True)
    sim = Simulation(grid, motion, dt_max=0.05, holes=DOOR_HOLES, film=True,
                     subcells=subcells, throat_model=ThroatModel(Cd=cd))
    hole_ids = [i for i, t in enumerate(sim.comp.throats) if t.axis is not None]
    hole_geo = [dict(center=sim.comp.throats[i].centroid,
                     diameter=sim.comp.throats[i].diameter, open_at=-np.inf)
                for i in hole_ids]
    print(f"setup {time.time()-t0:.0f} s, grid {grid.shape}, {len(hole_ids)} holes, "
          f"{sim.film.c.n} film elements", flush=True)
    times = frame_times(fps, t_rt, speedup, t_end)
    if preview:
        times = np.array(sorted(preview))
    full_mesh = trimesh.load(stl, force="mesh")
    scene = FilmScene(sim, full_mesh, hole_geo, size3d, np.arange(0, t_end, 0.5),
                      cam_box=cam_box, follow=follow, t_ref=t_end,
                      ssaa=ssaa, door_opacity=door_opacity, drop_scale=drop_scale,
                      pools_on_top=pools_on_top, cull=cull, look=look, orbit=orbit,
                      t_orbit=t_end)
    panel = DipPanel(size2d, t_end, t_rt, speedup, t_motion=t_motion, tilt=tilt)
    W, Hh = size3d[0] + size2d[0], size3d[1]
    ff = None
    if not preview:
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        ff = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo",
             "-pix_fmt", "rgb24", "-s", f"{W}x{Hh}", "-r", str(fps), "-i", "-",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(crf),
             "-preset", preset, "-movflags", "+faststart", out], stdin=subprocess.PIPE)
    qmax = 1e-3 / max(len(hole_ids), 1)
    t_sim = t_draw = 0.0
    for i, t in enumerate(times):
        a = time.time()
        if t > sim.t + 1e-9:
            sim.run(t_end=float(t))
        t_sim += time.time() - a
        H = sim.hist
        flows = (np.abs(np.asarray(H.throat_flow[-1])[hole_ids]) if len(H.t) > 1
                 else np.zeros(len(hole_ids)))
        f = sim.film
        pool_l = H.liquid_retained[-1] * 1e3
        film_ml = f.volume * 1e6
        drip_ml = f.drip_volume * 1e6
        ndrops = sum(d[3] for d in f.s.drips)
        fast = t >= t_rt - 1e-9
        phase = ("going in" if t < t_down else "coming out" if t < t_motion
                 else "draining and dripping")
        label = f"t = {t:6.2f} s    {phase}" + (f"    ({speedup:g}× speed)" if fast else "")
        a = time.time()
        frame_dt = (speedup if fast else 1.0) / fps
        scene.update(t, label, flows, qmax)
        scene.update_film(t, frame_dt)
        img3 = scene.grab()
        info = (f"model {dx*1e3:g} mm:  pools {pool_l:5.3f} l,  film {film_ml:4.1f} ml,  "
                f"{ndrops} drops ({drip_ml:4.1f} ml)")
        img2 = panel.render(t, pool_l, film_ml, drip_ml, info)
        frame = np.hstack([img3, img2[:Hh]])
        t_draw += time.time() - a
        if preview:
            from PIL import Image
            p = out.replace(".mp4", "") + f"_t{t:06.2f}.png"
            Image.fromarray(frame).save(p)
            print("wrote", p, flush=True)
        else:
            ff.stdin.write(frame.tobytes())
        if i % 48 == 0:
            print(f"frame {i}/{len(times)}  t = {t:.2f} s  pools {pool_l:.3f} l  film "
                  f"{film_ml:.2f} ml  drops {ndrops}  sim {t_sim:.0f} s, draw {t_draw:.0f} s",
                  flush=True)
    if ff is not None:
        ff.stdin.close()
        ff.wait()
        print("wrote", out)
    print(f"total {time.time()-t0:.0f} s: simulation {t_sim:.0f} s, drawing {t_draw:.0f} s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stl", required=True)
    ap.add_argument("--dx", type=float, default=0.006)
    ap.add_argument("--tilt", type=float, default=45.0)
    ap.add_argument("--cd", type=float, default=0.65)
    ap.add_argument("--subcells", type=int, default=4)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--full-dip", action="store_true",
                    help="submerge the whole door (8 s down, 8 s up, camera follows; "
                         "defaults --t-realtime 36 --t-end 136)")
    ap.add_argument("--t-down", type=float, default=None)
    ap.add_argument("--t-up", type=float, default=None)
    ap.add_argument("--follow", type=float, default=None,
                    help="camera follows this fraction of the door's vertical motion")
    ap.add_argument("--t-realtime", type=float, default=None,
                    help="play 0..t in real time, then speed up")
    ap.add_argument("--speedup", type=float, default=5.0)
    ap.add_argument("--t-end", type=float, default=None)
    ap.add_argument("--preview", type=float, nargs="*")
    ap.add_argument("--size3d", type=int, nargs=2, default=[1040, 720], metavar=("W", "H"))
    ap.add_argument("--width2d", type=int, default=720)
    ap.add_argument("--ssaa", type=int, default=2)
    ap.add_argument("--crf", type=int, default=16)
    ap.add_argument("--preset", default="slow")
    ap.add_argument("--door-opacity", type=float, default=0.35)
    ap.add_argument("--drop-scale", type=float, default=3.0,
                    help="drawn drop size relative to the true drop size")
    ap.add_argument("--film-range", type=float, nargs=2, default=[H_MIN_UM, H_MAX_UM],
                    metavar=("MIN_UM", "MAX_UM"),
                    help="film colour scale: transparent at MIN, full colour at MAX")
    ap.add_argument("--film-alpha", type=float, default=FILM_ALPHA,
                    help="film opacity at MAX thickness")
    ap.add_argument("--no-cull", action="store_true",
                    help="always draw all film-shell triangles (slower; fallback if "
                         "the graphics driver misbehaves)")
    ap.add_argument("--pools-depth-sorted", action="store_true",
                    help="draw pools depth-sorted (behind film/door) instead of on top")
    ap.add_argument("--look", choices=("hd", "plain"), default="hd",
                    help="hd: three-point light, glossy water, bath surface sheet, "
                         "blue tint and waterline under the surface")
    ap.add_argument("--orbit", type=float, default=0.0,
                    help="camera turns this many degrees around the door over the movie")
    ap.add_argument("--out", default="examples/out/door_dip_film/dip_film_45.mp4")
    a = ap.parse_args()
    H_MIN_UM, H_MAX_UM = a.film_range
    FILM_ALPHA = a.film_alpha
    fd = a.full_dip
    t_down = a.t_down if a.t_down is not None else (8.0 if fd else 4.0)
    t_up = a.t_up if a.t_up is not None else (8.0 if fd else 4.0)
    t_rt = a.t_realtime if a.t_realtime is not None else t_down + t_up + 20.0
    t_end = a.t_end if a.t_end is not None else t_rt + 100.0
    follow = a.follow if a.follow is not None else (0.8 if fd else 0.0)
    main(a.stl, a.dx, a.tilt, a.cd, a.subcells, a.out, a.fps, t_rt, a.speedup,
         t_down=t_down, t_up=t_up, full_dip=fd, follow=follow, t_end=t_end,
         size3d=a.size3d, size2d=(a.width2d, a.size3d[1]), preview=a.preview,
         ssaa=a.ssaa, crf=a.crf, preset=a.preset, door_opacity=a.door_opacity,
         drop_scale=a.drop_scale, pools_on_top=not a.pools_depth_sorted,
         cull=not a.no_cull, look=a.look, orbit=a.orbit)
