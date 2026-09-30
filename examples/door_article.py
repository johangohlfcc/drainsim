"""Car-door drainage experiment of the article (Fig. 7 and Fig. 8), water.

Setup as in the article (Sect. 3.2): the door hangs 0.15 m above the water
(lowest point), is moved 0.4 m straight down and back up in 8 s, tilted in
its own plane by 22.5 or 45 deg (the front, +x, end lowered, so water
collects in the front bottom corner as in Fig. 8). The drain holes are open
(drainage case). After the motion the door hangs for ``t_drain`` seconds and
the water still inside is reported.

Only the part of the door that can get wet is voxelised: the mesh is cropped
to the region that is at most ``margin`` above the bath at the deepest point
of the dip (plus a pad). This keeps the coarse runs small.

Usage (from the drainsim folder):
    python examples/door_article.py --stl door.stl --dx 0.006 0.005 0.004 \\
           --tilt 22.5 45 --t-drain 30 --out examples/out/door
    python examples/door_article.py --stl door.stl --dx 0.003 --tilt 45 --film --vtk
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from drainsim.grid import Grid
from drainsim.model import Simulation
from drainsim.motion import Keyframes, rot3

EXPERIMENT = {22.5: 0.056, 45.0: 0.246}               # litres, article Sect. 3.1
# The three drain holes in the bottom flange of the inner panel. Centre and
# axis are fitted to the rim vertices of the mesh (circle of radius 9.5 mm).
DOOR_HOLES = [dict(center=(-0.4132, -0.0076, 1.4529), diameter=0.019,
                   axis=(-0.015, -0.753, 0.658)),
              dict(center=(-0.0382, -0.0097, 1.4584), diameter=0.019,
                   axis=(-0.012, -0.761, 0.649)),
              dict(center=(0.3370, -0.0092, 1.4643), diameter=0.019,
                   axis=(-0.007, -0.766, 0.643))]
ARTICLE_SIM = {22.5: {3: 0.020, 7: 0.039}, 45.0: {3: 0.145, 7: 0.226}}  # Fig. 7 (approx.)


def door_setup(stl, dx, tilt, t_drain=0.0, above=0.15, depth=0.4,
               t_down=4.0, t_up=4.0, margin=0.10, pad=0.03, place_untilted=False,
               hang=0.0, levels=None):
    """place_untilted: put the *untilted* door's lowest point ``above`` the
    bath and then tilt it about its centre, so the tilted door hangs lower.
    The default puts the lowest point of the *tilted* door ``above``.
    hang: lean of the door on its hooks (degrees about its own x axis, applied
    before the tilt; positive raises the +y (cabin) side, i.e. the top leans
    towards -y). The article used about 5 degrees.
    levels: an octree with this many coarser levels (cells of dx at the
    surface, up to dx * 2**levels in the bulk) instead of the uniform grid."""
    import trimesh
    mesh = trimesh.load(stl, force="mesh")
    V = mesh.vertices
    angles = np.array([hang, tilt, 0.0])                # about y: lowers +x
    R = rot3(angles)
    pivot = V.mean(0)
    W = (V - pivot) @ R.T + pivot
    z0 = above - (V[:, 2].min() if place_untilted else W[:, 2].min())
    # crop: everything that is below bath + margin at the deepest position
    wet = W[:, 2] + z0 - depth < margin
    lo = V[wet].min(0) - pad
    hi = V[wet].max(0) + pad
    lo[1], hi[1] = V[:, 1].min() - pad, V[:, 1].max() + pad
    if levels is not None:
        from drainsim.octree import Octree
        grid = Octree.from_mesh(mesh, dx, levels=int(levels), bounds=(lo, hi))
    else:
        grid = Grid.from_mesh(mesh, dx, bounds=(lo, hi))
    T = [0.0, t_down, t_down + t_up] + ([t_down + t_up + t_drain] if t_drain > 0 else [])
    shifts = np.array([[0, 0, z0], [0, 0, z0 - depth], [0, 0, z0], [0, 0, z0]])[:len(T)]
    motion = Keyframes(T, np.tile(angles, (len(T), 1)), shifts, ndim=3, pivot=pivot)
    return mesh, grid, motion


def peak_rss_gb():
    """Peak memory of this process (GB), or None if it cannot be read."""
    try:
        import psutil
        mi = psutil.Process().memory_info()
        if hasattr(mi, "peak_wset"):                   # Windows
            return mi.peak_wset / 1e9
    except ImportError:
        pass
    try:
        import resource, sys
        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return r / 1e9 if sys.platform == "darwin" else r / 1e6
    except ImportError:
        return None


def run_case(stl, dx, tilt, t_drain="auto", film=False, split=True,
             spill_routing=True, snapshot_every=None, verbose=True,
             holes=DOOR_HOLES, settle_tol=0.005, t_drain_min=30.0,
             t_drain_max=300.0, progress=False, subcells=4, place_untilted=False,
             hang=0.0, levels=None, narrow=None):
    """One door case.

    t_drain : seconds of hanging after the motion, or "auto": at least
              ``t_drain_min``, then in 10 s chunks until the retained water
              changes by less than ``settle_tol`` (relative, or 0.1 ml) over
              the last 10 s, at most ``t_drain_max``.
    subcells: k for the side-aware cut-cell volumes (k^3 sub-cells per cut
              cell); 0 or 1 = binary cells (every fluid cell dx^3).
    holes   : explicitly listed drain holes (default: the door's three
              19 mm holes); None = rely on automatic detection only.
    """
    t0 = time.time()
    auto = t_drain == "auto"
    if narrow and levels is None:
        levels = 0                      # the narrow refinement needs the octree
    mesh, grid, motion = door_setup(stl, dx, tilt, 0.0 if auto else float(t_drain),
                                    place_untilted=place_untilted, hang=hang,
                                    levels=levels)
    sim = Simulation(grid, motion, dt_max=0.1, film=film, split=split,
                     spill_routing=spill_routing, holes=holes if split else None,
                     subcells=subcells, narrow=narrow)
    t_setup = time.time() - t0
    if verbose:
        inner = [t for t in sim.comp.throats if t.a == t.b]
        print(f"dx={dx*1e3:.1f} mm tilt={tilt}: grid {grid.shape} "
              f"({grid.ncells/1e6:.2f} M cells), {sim.comp.n} compartments, "
              f"{len(sim.comp.throats)} throats ({len(inner)} holes inside a "
              f"compartment, {len(sim.holes)} given explicitly), setup {t_setup:.0f} s",
              flush=True)
        for t in sim.comp.throats:
            print(f"    throat {t.a}-{t.b}: d = {t.diameter*1e3:.1f} mm, "
                  f"A = {t.area*1e6:.0f} mm2, at {np.round(t.centroid, 3)}")
        vs = sim.volfrac_stats
        if vs.get("cut_cells"):
            fl = grid.fluid.ravel()
            gain = sim.v[sim.fl].sum() / (fl.sum() * grid.cell_volume) - 1
            extra = (f"{vs['fine_active']/1e6:.2f} M sub-cell nodes"
                     if "fine_active" in vs else f"dropped {vs.get('dropped', 0)*1e6:.1f} ml")
            print(f"    cut cells k={subcells}: {vs['cut_cells']/1e6:.2f} M, fluid volume "
                  f"+{gain:.2%} vs binary, {extra}, {vs['time_s']:.0f} s", flush=True)
        else:
            print("    volume fractions: off (binary cells)", flush=True)
    t_motion = motion.times[2]
    t_first = (t_motion + t_drain_min) if auto else motion.t_end
    kw = dict(snapshot_every=snapshot_every) if snapshot_every else \
        dict(snapshot_times=[t_motion])
    H = sim.run(t_end=t_first, progress=progress, **kw)
    if auto:
        while sim.t < t_motion + t_drain_max - 1e-9:
            v0 = sim.hist.liquid_retained[-1]
            sim.run(t_end=sim.t + 10.0, progress=progress,
                    **(dict(snapshot_every=snapshot_every) if snapshot_every else {}))
            v1 = sim.hist.liquid_retained[-1]
            if abs(v1 - v0) <= max(settle_tol * v1, 1e-7):
                break
    sim._snap(sim.t)
    a = H.arrays()
    # largest retained pools, their level relative to the lowest hole sill
    bd = sim._bodies()
    sills = [sim._hole_wet(t, 0.0)[2] for t in sim.comp.throats if t.axis is not None]
    ref = min(sills) if sills else np.nan
    pools = []
    for b in np.argsort(-np.where(bd["bath"], -1.0, bd["vol"]))[:5]:
        if bd["bath"][b] or bd["vol"][b] <= 0:
            continue
        cells = np.flatnonzero(bd["body"] == b)
        pools.append(dict(volume_l=float(bd["vol"][b] * 1e3),
                          level_minus_sill_mm=float((bd["level"][b] - ref) * 1e3),
                          centroid=np.round(sim.X[cells].mean(0), 3).tolist()))
    t_end = float(a["t"][-1])
    res = dict(dx=dx, tilt=tilt, split=split, spill_routing=spill_routing,
               film=bool(film), holes=len(sim.holes), cells=int(grid.ncells),
               t_end=t_end,
               retained_l=float(a["liquid_retained"][-1] * 1e3),
               retained_end_of_motion_l=float(
                   np.interp(t_motion, a["t"], a["liquid_retained"]) * 1e3),
               retained_10s_before_end_l=float(
                   np.interp(t_end - 10, a["t"], a["liquid_retained"]) * 1e3),
               film_l=float(a["film_volume"][-1] * 1e3),
               drops=int(a["n_drips"][-1]),
               steps=int(len(a["t"]) - 1), wall_s=time.time() - t0,
               setup_s=t_setup, subcells=int(subcells or 0),
               volfrac={k: v for k, v in sim.volfrac_stats.items()},
               peak_rss_gb=peak_rss_gb(), pools=pools,
               t=a["t"].tolist(), liquid_l=(a["liquid_retained"] * 1e3).tolist())
    if verbose:
        print(f"    -> retained {res['retained_l']:.4f} l at t = {t_end:.0f} s "
              f"(end of motion {res['retained_end_of_motion_l']:.3f} l, "
              f"10 s earlier {res['retained_10s_before_end_l']:.4f} l), "
              f"film {res['film_l']:.4f} l, {res['steps']} steps, "
              f"{res['wall_s']:.0f} s, peak RSS {res['peak_rss_gb'] or 0:.1f} GB",
              flush=True)
        for p in pools:
            print(f"       pool {p['volume_l']:.4f} l, level - lowest hole sill "
                  f"{p['level_minus_sill_mm']:+.1f} mm, at {p['centroid']}", flush=True)
    return sim, H, res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stl", required=True)
    ap.add_argument("--dx", type=float, nargs="+", default=[0.006, 0.005, 0.004])
    ap.add_argument("--tilt", type=float, nargs="+", default=[22.5, 45.0])
    ap.add_argument("--t-drain", default="auto",
                    help='seconds after the motion, or "auto" (until settled)')
    ap.add_argument("--no-holes", action="store_true",
                    help="do not give the drain holes explicitly")
    ap.add_argument("--progress", action="store_true")
    ap.add_argument("--subcells", type=int, default=4,
                    help="k for side-aware cut-cell volumes (0 = binary cells)")
    ap.add_argument("--film", action="store_true")
    ap.add_argument("--levels", type=int, default=None,
                    help="octree with this many coarser levels (dx at the walls)")
    ap.add_argument("--narrow", type=int, default=0, metavar="K",
                    help="K sub-cells per edge in cells at narrow passages (0 = off)")
    ap.add_argument("--equilibrium", action="store_true",
                    help="also run the equilibrium and legacy variants")
    ap.add_argument("--vtk", action="store_true", help="world-frame VTK + MP4")
    ap.add_argument("--out", default="examples/out/door")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    results = []
    for tilt in args.tilt:
        for dx in args.dx:
            variants = [dict()]
            if args.equilibrium:
                variants += [dict(split=False), dict(split=False, spill_routing=False)]
            for v in variants:
                td = args.t_drain if args.t_drain == "auto" else float(args.t_drain)
                sim, H, res = run_case(args.stl, dx, tilt, td, film=args.film,
                                       snapshot_every=0.2 if args.vtk else None,
                                       holes=None if args.no_holes else DOOR_HOLES,
                                       progress=args.progress,
                                       subcells=args.subcells, levels=args.levels,
                                       narrow=dict(k=args.narrow) if args.narrow else None, **v)
                results.append(res)
                with open(os.path.join(args.out, "results.json"), "w") as f:
                    json.dump(results, f)
                if args.vtk and not v:
                    from drainsim.worldviz import export_world
                    tag = f"door_{tilt:g}deg_{dx*1e3:g}mm"
                    export_world(sim, H, os.path.join(args.out, tag), prefix="door",
                                 object_mesh=args.stl, volume=False)
                    try:
                        from drainsim.render3d import render_animation
                        render_animation(os.path.join(args.out, tag), "door",
                                         os.path.join(args.out, tag + ".mp4"),
                                         fps=10, azimuth=180, elevation=10)
                    except ImportError:
                        pass
