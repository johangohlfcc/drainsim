"""Car body of the article (Sect. 3.3, Figs. 9-13, Table 2), with drainsim.

The article dips the Volvo car body with a 55 s rotating dip ("modified
version of an actual dipping motion"): the body pitches nose-down through a
half turn while it enters the bath, travels upside down through the bath
with a slight rocking motion, and pitches on through the second half turn
while it leaves. The article reports

* after dip-in (rotation under the liquid surface): 0.61 l of trapped air in
  12 pockets larger than 0.01 l (Figs. 10 and 11);
* after dip-out: 0.41 l of trapped liquid in 8 cavities (Figs. 12 and 13).

Motion here: read off Fig. 9 (12 snapshots, 5 s apart). World +x is the
travel direction, +z is up and the bath surface is z = 0. The car frame is
the mesh frame, centred on its bounding box: +X forward, +Z up. The pitch
is a rotation about the car's +Y axis: positive angles take the nose down.

    t (s)      0     5    10    15    20    25    30    35    40    45    50    55
    x (m)      0  2.66  4.16  6.36 11.49 16.62 21.75 26.87 32.00 35.00 36.60 38.20
    z (m)   2.20  1.60  0.10 -1.14 -1.16 -1.16 -1.16 -1.16 -1.20 -0.24  1.60  2.30
    pitch      0    60   120   180 180+r 180-r 180+r 180-r   180   230   300   360

(r = rocking amplitude, default 5 deg; ``--motion full``). The default
``--motion short`` keeps the dip-in and the dip-out and replaces the 25 s
transport with 5 s upside down under the surface (35 s in total):

    t (s)      0     5    10    15    20    25    30    35
    z (m)   2.20  1.60  0.10 -1.14 -1.20 -0.24  1.60  2.30
    pitch      0    60   120   180   180   230   300   360

After dip-out the car can hang still for ``--hang`` seconds, so that the
transient model can finish draining. The article's method is an equilibrium
method: ``--mode equilibrium`` runs drainsim without throats (every opening
instantaneous) for comparison.

Time steps: fixed ``--dt-max`` (0.1 s: 1.2 deg of rotation per step) unless
``--cells-per-step`` > 0 limits them by the speed of the surface relative
to the car (the article took 23 300 steps, about 0.03 deg each).

Usage (from the drainsim folder):

    python examples/car_article.py --stl ../xc90.stl --preview-motion --out runs_car
    python examples/car_article.py --stl ../xc90.stl --dx 0.04 --subcells 2 --out runs_car
    python examples/car_article.py --stl ../xc90.stl --render runs_car/car_20mm_k4_transient

Outputs in ``<out>/car_<dx>mm_k<k>_<mode>/``: summary.json, pockets.md,
history.png, fields_t*.npz (for the renders), and after ``--render`` the
figures fig10_air_top.png, fig11_air_side.png, fig12_liquid_top.png and
fig13_liquid_side.png.
"""
from __future__ import annotations

import argparse
import json
import os
import time
import warnings

import numpy as np

from drainsim.grid import Grid
from drainsim.model import Simulation
from drainsim.octree import Octree
from drainsim.motion import Keyframes, rot3
from drainsim.physics import ThroatModel

warnings.filterwarnings("ignore")

MIN_BODY_L = 0.01                                   # article post-processing
ARTICLE = dict(air_l=0.61, air_n=12, liquid_l=0.41, liquid_n=8)
# Fig. 9 keyframes: time, travel x (m), height z of the car centre (m), pitch
KEY_T = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55]
KEY_X = [0.0, 2.66, 4.16, 6.36, 11.49, 16.62, 21.75, 26.87, 32.00, 35.00, 36.60, 38.20]
KEY_Z = [2.20, 1.60, 0.10, -1.14, -1.16, -1.16, -1.16, -1.16, -1.20, -0.24, 1.60, 2.30]
KEY_P = [0, 60, 120, 180, 180, 180, 180, 180, 180, 230, 300, 360]
ROCK_SIGN = [0, 0, 0, 0, 1, -1, 1, -1, 0, 0, 0, 0]
# "short": the same dip-in (0-15 s) and dip-out (the article's 40-55 s),
# joined by 5 s upside down under the surface instead of the 25 s transport
# with rocking. Only the rotation and the height matter to the model (the
# bath is infinite), so the travel between them is left out.
SHORT = dict(T=[0, 5, 10, 15, 20, 25, 30, 35],
             X=[0.0, 2.66, 4.16, 6.36, 11.49, 14.49, 16.09, 17.69],
             Z=[2.20, 1.60, 0.10, -1.14, -1.20, -0.24, 1.60, 2.30],
             P=[0, 60, 120, 180, 180, 230, 300, 360],
             R=[0, 0, 0, 0, 0, 0, 0, 0],
             air=[15.0, 20.0], dipout=35.0)
FULL = dict(T=KEY_T, X=KEY_X, Z=KEY_Z, P=KEY_P, R=ROCK_SIGN,
            air=[15.0, 20.0, 30.0, 40.0], dipout=55.0)
MOTIONS = dict(short=SHORT, full=FULL)


# ------------------------------------------------------------------ geometry
def orient_matrix(V):
    """Rotation that puts the car body upright and nose forward: length
    along X (nose +X), width along Y, height along Z (roof +Z). The roof is
    the narrower end of the height axis (the greenhouse tapers), and it sits
    towards the rear (behind the long bonnet), which gives the nose."""
    ext = V.max(0) - V.min(0)
    L, W, H = np.argsort(ext)[::-1]
    R = np.zeros((3, 3))
    R[0, L] = R[1, W] = R[2, H] = 1.0
    if np.linalg.det(R) < 0:
        R[1] *= -1.0
    P = V @ R.T
    x, y, z = P[:, 0], P[:, 1], P[:, 2]

    def width(sel):
        return np.percentile(y[sel], 98) - np.percentile(y[sel], 2)
    top, bot = z > np.percentile(z, 90), z < np.percentile(z, 10)
    if width(top) > width(bot):                    # roof at -Z: roll 180 deg
        R = np.diag([1.0, -1.0, -1.0]) @ R
        P = V @ R.T
        x, z = P[:, 0], P[:, 2]
    roof = z > np.percentile(z, 90)
    mid = 0.5 * (np.percentile(x[roof], 2) + np.percentile(x[roof], 98))
    if mid > 0:                                    # roof towards +X: nose at -X
        R = np.diag([-1.0, -1.0, 1.0]) @ R
    return R


def load_car(path, orient="auto", reverse=False):
    """Mesh in metres, centred on its bounding box. Returns (mesh, info).

    orient : "auto" turns the body upright with the nose along +X (see
             ``orient_matrix``); "file" keeps the file's axes.
    reverse: turn it 180 deg about the vertical afterwards (nose along -X,
             i.e. travelling backwards)."""
    import trimesh
    t0 = time.time()
    m = trimesh.load(path, force="mesh", process=False)
    V = np.asarray(m.vertices, float)
    lo0, hi0 = V.min(0), V.max(0)
    scale = 1e-3 if (hi0 - lo0).max() > 100.0 else 1.0     # mm -> m
    centre = 0.5 * (lo0 + hi0)
    V = (V - centre) * scale
    R = orient_matrix(V) if orient == "auto" else np.eye(3)
    if reverse:
        R = np.diag([-1.0, -1.0, 1.0]) @ R
    m.vertices = V @ R.T
    info = dict(file=os.path.abspath(path), triangles=int(len(m.faces)),
                vertices=int(len(V)), bounds_lo_file=lo0.tolist(),
                bounds_hi_file=hi0.tolist(), centre_file=centre.tolist(),
                scale_to_m=scale, size_m=((hi0 - lo0) * scale).tolist(),
                orient=orient, reverse=bool(reverse), rotation_file_to_car=R.tolist(),
                load_s=time.time() - t0)
    info["size_m"] = (np.asarray(m.vertices).max(0) - np.asarray(m.vertices).min(0)).tolist()
    print(f"car: file axes -> car axes (rows: car X forward, Y left, Z up) {R.round(3).tolist()}"
          f"{' (reversed: nose along -X)' if reverse else ''}", flush=True)
    return m, info


def to_file_coords(X, info):
    """Object-frame points (m) back to the original mesh coordinates."""
    R = np.asarray(info.get("rotation_file_to_car", np.eye(3)))
    return (np.asarray(X) @ R) / info["scale_to_m"] + np.asarray(info["centre_file"])


def car_motion(rock=5.0, hang=0.0, kind="short", rotation="pitch", sense=1):
    """Keyframed dip. The body turns 360 deg over the dip: about its
    transverse axis (``rotation="pitch"``; sense +1: nose down first) or its
    long axis (``"roll"``; sense +1: the left side (+Y) goes down first)."""
    k = MOTIONS[kind]
    T, X, Z = list(k["T"]), list(k["X"]), list(k["Z"])
    P = [p + rock * r for p, r in zip(k["P"], k["R"])]
    if hang > 0:
        T.append(T[-1] + hang)
        X.append(X[-1])
        Z.append(Z[-1])
        P.append(P[-1])
    sg = 1.0 if float(sense) >= 0 else -1.0
    if rotation == "roll":
        ang = np.array([[-sg * p, 0.0, 0.0] for p in P])
    else:
        ang = np.array([[0.0, sg * p, 0.0] for p in P])
    shifts = np.array([[x, 0.0, z] for x, z in zip(X, Z)])
    return Keyframes(np.array(T, float), ang, shifts, ndim=3, pivot=np.zeros(3))



# ------------------------------------------------------------ IPS motions
def read_xmo(path):
    """Keyframes of an IPS motion file (.xmo): a list of (t, q, T, off) with
    the quaternion q = (w, x, y, z), the translation T and the offset off
    (None for a frame given without one), lengths in metres (the file has
    millimetres). A frame inside an <Offset> element is the pose of a frame
    displaced from the object: the object's origin is T + R(q) off."""
    import xml.etree.ElementTree as ET
    root = ET.parse(path).getroot()
    mot = root.find("Motion")
    if mot is None:
        raise ValueError(f"{path}: no <Motion>")

    def frame(f, off):
        q = f.find("Quaternion")
        tr = f.find("Translation")
        return (float(f.find("Time").get("t")),
                np.array([float(q.get(k)) for k in ("e1", "e2", "e3", "e4")]),
                np.array([float(tr.get(k)) for k in "xyz"]) * 1e-3, off)

    out = []
    for el in mot:
        if el.tag == "Frame":
            out.append(frame(el, None))
        elif el.tag == "Offset":
            off = np.array([float(el.get(k)) for k in "xyz"]) * 1e-3
            out.extend(frame(f, off) for f in el.findall("Frame"))
    return sorted(out, key=lambda f: f[0])


def bath_level_of(path, scale=1.0):
    """Height of the bath surface: the top of the bath mesh (plant frame)."""
    import trimesh
    V = np.asarray(trimesh.load(path, force="mesh", process=False).vertices, float) * scale
    return float(V[:, 2].max())


def ips_motion(xmo, bath_stl, centre, scale=1.0, hang=0.0):
    """The motion of an IPS dip (``read_xmo``) for a car loaded in the file's
    axes (``load_car(orient="file")``: plant coordinates, centred on
    ``centre``, the file's bounding-box centre, times ``scale`` to metres),
    the STL being the car at the motion's first pose.

    IPS moves a frame on the body (the pivot): its position and rotation are
    interpolated linearly in time between the keyframes; the first keyframe,
    given without an offset, is the object's own pose and is turned into the
    pivot's. Only rotations about the plant y axis (the car's transverse
    axis) are supported: the angles are unwrapped along the keyframes (each
    step the short way round), so a turn over is a full 360 deg. ``hang``
    seconds of hanging still are added at the end. World frame: plant axes,
    origin at the pivot's first position (x, y) and the bath surface (z).
    Returns (Keyframes, info)."""
    fr = read_xmo(xmo)
    offs = [f[3] for f in fr if f[3] is not None]
    off = offs[0] if offs else np.zeros(3)
    if any(np.abs(o - off).max() > 1e-6 for o in offs):
        raise ValueError(f"{xmo}: the offsets differ between keyframes")
    th = []
    for t, q, T, o in fr:
        q = q / np.linalg.norm(q)
        if abs(q[1]) > 1e-9 or abs(q[3]) > 1e-9:
            raise ValueError(f"{xmo}: t = {t:g} s turns about another axis than y")
        a = np.degrees(2.0 * np.arctan2(q[2], q[0]))
        if th:
            a = th[-1] + (a - th[-1] + 180.0) % 360.0 - 180.0
        th.append(a)
    th = np.array(th)
    # the pivot (the offset frame's origin) at each keyframe, plant frame
    F = np.array([T if o is not None else T - rot3([0.0, a, 0.0]) @ off
                  for (t, q, T, o), a in zip(fr, th)])
    c = np.asarray(centre, float) * scale
    p = F[0] - c                                   # the pivot in object coordinates
    zbath = bath_level_of(bath_stl, scale)
    W0 = np.array([F[0][0], F[0][1], zbath])
    times = [f[0] for f in fr]
    ang = [[0.0, a - th[0], 0.0] for a in th]
    shifts = [Fk - W0 - p for Fk in F]
    if hang > 0:
        times.append(times[-1] + hang)
        ang.append(ang[-1])
        shifts.append(shifts[-1])
    mo = Keyframes(np.array(times, float), np.array(ang), np.array(shifts), ndim=3, pivot=p,
                   bath_level=0.0)
    info = dict(xmo=os.path.abspath(xmo), bath_stl=os.path.abspath(bath_stl),
                times=[f[0] for f in fr], angles_deg=th.tolist(), pivot_plant=F.tolist(),
                offset_m=off.tolist(), bath_level_plant=zbath, world_origin_plant=W0.tolist(),
                t_dipout=float(fr[-1][0]), hang=float(hang))
    return mo, info

def narrow_opt(args):
    """``Simulation(narrow=...)`` from ``--narrow K`` (0: off)."""
    k = int(getattr(args, "narrow", 0) or 0)
    return dict(k=k) if k else None


def car_grid(mesh, dx, lo, hi, levels=0, verbose=True, octree=False):
    """Uniform grid (levels = 0) or octree with ``levels`` coarser levels
    above the wall cells of size dx (base cells dx * 2**levels). octree:
    an octree also for levels = 0 (needed by the narrow refinement)."""
    if not levels and not octree:
        return Grid.from_mesh(mesh, dx, bounds=(lo, hi))
    ot = Octree.from_mesh(mesh, dx, levels=int(levels), bounds=(lo, hi))
    if verbose:
        print(f"octree h = {dx*1e3:g} mm, {levels} levels: leaves per level "
              f"{ot.nleaves()}, closed {ot.cut.size}, {ot.stats['build_s']:.0f} s", flush=True)
    return ot


def add_car_args(ap):
    """Orientation, rotation and grid options shared by the car scripts."""
    ap.add_argument("--levels", type=int, default=0,
                    help="octree: coarser levels above the wall cells (dx = the finest "
                         "cell, next to the walls); 0 = uniform grid (default)")
    ap.add_argument("--narrow", type=int, default=0, metavar="K",
                    help="refine the closed cells at narrow passages (gaps, slots, holes "
                         "narrower than 2 sub-cells) to K sub-cells per edge; 0 = off "
                         "(default). Uses the octree.")
    ap.add_argument("--orient", choices=("auto", "file"), default="auto",
                    help="auto: body upright, nose forward (default); file: the file's axes")
    ap.add_argument("--reverse", action="store_true",
                    help="start with the nose pointing backwards (against the travel)")
    ap.add_argument("--rotation", choices=("pitch", "roll"), default="pitch",
                    help="360 deg about the transverse (pitch) or the long (roll) axis")
    ap.add_argument("--sense", type=int, choices=(1, -1), default=1,
                    help="pitch +1: nose goes down first, -1: tail first; "
                         "roll +1: left side first, -1: right side first")
    ap.add_argument("--holes", default=None, metavar="CSV",
                    help="explicit holes: the table of find_openings.py (holes found on "
                         "this STL), opened with their true size (see --holes-min/-max)")
    ap.add_argument("--holes-min", type=float, default=3.0, metavar="MM",
                    help="--holes: the smallest diameter used (mm, default 3)")
    ap.add_argument("--holes-max", type=float, default=80.0, metavar="MM",
                    help="--holes: the largest diameter used (mm, default 80)")
    ap.add_argument("--channels", action="store_true",
                    help="gap channels (7.1): the narrow gaps between plates that the grid "
                         "closes, found on this STL by ray casting, as compartments of their "
                         "own joined by slot throats (see --channel-min/-max)")
    ap.add_argument("--channel-min", type=float, default=3.0, metavar="MM",
                    help="--channels: the narrowest gap used (mm, default 3)")
    ap.add_argument("--channel-max", type=float, default=20.0, metavar="MM",
                    help="--channels: the widest gap used (mm, default 20; wider gaps are "
                         "open in the grid anyway, the closed part of any gap is used)")
    ap.add_argument("--gap-spacing", type=float, default=4.0, metavar="MM",
                    help="--channels: the spacing of the ray samples on the faces (mm)")
    ap.add_argument("--seals", default=None, metavar="CSV",
                    help="--channels: sealed places (columns x, y, z in the STL's frame in "
                         "metres, radius_mm): the gaps within them are closed")


def gap_channels(args, info):
    """``Simulation(channels=...)`` from ``--channels`` (None without): the
    gaps of the STL sampled by ray casting, in the frame of the loaded car."""
    if not getattr(args, "channels", False):
        return None
    import time
    from drainsim import openings as op
    t0 = time.time()
    V, F = op.read_stl(args.stl)
    S = op.gap_samples(V, F, +1, spacing=args.gap_spacing * 1e-3, verbose=True)
    del V, F
    if getattr(args, "seals", None):
        S, n = op.seal_samples(S, *op.read_seals_csv(args.seals))
        print(f"seals from {args.seals}: {n} gap samples closed", flush=True)
    S = op.samples_to_model(S, info)
    print(f"gap samples for channels ({time.time() - t0:.0f} s)", flush=True)
    return dict(samples=S, lo=args.channel_min * 1e-3, hi=args.channel_max * 1e-3)


def explicit_holes(args, info):
    """``Simulation(holes=...)`` from ``--holes`` (None without): the holes of
    the table (kind "hole") within the diameter range, in the frame of the
    loaded car."""
    if not getattr(args, "holes", None):
        return None
    from drainsim.openings import holes_for_model, read_holes_csv
    hs = holes_for_model(read_holes_csv(args.holes), info, dmin=args.holes_min * 1e-3,
                         dmax=args.holes_max * 1e-3)
    print(f"explicit holes: {len(hs)} from {args.holes} ({args.holes_min:g}-"
          f"{args.holes_max:g} mm)", flush=True)
    return hs


def preview_motion(mesh, mo, path, n=40000, times=None):
    """Side view of the motion like Fig. 9 (world x-z, bath below z = 0)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(0)
    V = np.asarray(mesh.vertices)
    V = V[rng.choice(len(V), min(n, len(V)), replace=False)]
    times = list(mo.times) if times is None else times
    span = max(22.0, float(mo.shifts[:, 0].max()) + 5) + 4
    fig, ax = plt.subplots(figsize=(max(8.0, 16 * span / 47), 4.2))
    ax.axhspan(-4.4, 0.0, color="#8080ff", alpha=0.35, lw=0)
    for t in times:
        R, T = mo.pose(t)
        W = V @ R.T + T
        ax.plot(W[:, 0], W[:, 2], ",", color="#52514e", alpha=0.25)
        ax.text(T[0], T[2] + 1.4, f"{t:g}s", ha="center", fontsize=8)
    ax.set_aspect("equal")
    ax.set_xlim(-4, max(22.0, float(mo.shifts[:, 0].max()) + 5))
    ax.set_ylim(-4.4, 4.5)
    ax.set_xlabel("world x (m), travel direction")
    ax.set_ylabel("z (m)")
    ax.set_title("Car body motion (compare with the article's Fig. 9)", loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print("wrote", path, flush=True)


# -------------------------------------------------------------- simulation
def peak_rss_bytes():
    """Peak memory of this process in bytes, or None if it cannot be read.
    Windows: psutil's peak working set. Linux and macOS: getrusage's
    ru_maxrss (kB on Linux, bytes on macOS), a true peak; psutil's rss there
    is only the current size."""
    try:
        import psutil
        mi = psutil.Process().memory_info()
        if hasattr(mi, "peak_wset"):
            return int(mi.peak_wset)
    except Exception:
        pass
    try:
        import resource
        import sys
        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(r) if sys.platform == "darwin" else int(r) * 1024
    except Exception:
        return None


def peak_rss_gb():
    b = peak_rss_bytes()
    return None if b is None else round(b / 1e9, 2)


def pockets_table(sim, info, kind, min_l=MIN_BODY_L):
    """Pockets of `kind` ('air' or 'liquid') above min_l, in the car frame."""
    P = sim.pockets(min_volume=min_l * 1e-3)[kind]
    rows = []
    half = 0.5 * np.asarray(info["size_m"])
    for v, c in P:
        cf = to_file_coords(c, info)
        # position in words: along the car (front = +X), side (+Y left), height
        fx = (c[0] + half[0]) / (2 * half[0])
        along = "front" if fx > 0.66 else ("rear" if fx < 0.33 else "middle")
        side = "centre" if abs(c[1]) < 0.15 else ("left (+Y)" if c[1] > 0 else "right (-Y)")
        fz = (c[2] + half[2]) / (2 * half[2])
        height = "low" if fz < 0.33 else ("high" if fz > 0.66 else "mid")
        rows.append(dict(litres=float(v * 1e3), centre_file=[float(q) for q in cf],
                         centre_car_m=[float(q) for q in c],
                         where=f"{along}, {side}, {height}"))
    return rows


def body_fields(sim, kind, min_l=MIN_BODY_L):
    """Per-grid-cell body id (-1 none) and fraction of the pockets of `kind`,
    for the renders. Fine sub-cell nodes are folded into their host cell."""
    from drainsim.model import _label_bodies
    liq_t, air_t = sim.trapped_fields()
    w = liq_t if kind == "liquid" else air_t
    mask = w > 1e-9
    body, nb = _label_bodies(mask, sim.lab, sim.nbr)
    idx = np.flatnonzero(body >= 0)
    vol = np.bincount(body[idx], weights=w[idx] * sim.v[idx], minlength=nb)
    order = np.argsort(-vol)
    keep = order[vol[order] > min_l * 1e-3]
    rank = -np.ones(nb, np.int64)
    rank[keep] = np.arange(keep.size)
    sel = idx[rank[body[idx]] >= 0]
    if getattr(sim, "octree", False):
        # octree: on the uniform display grid (drainsim.octview)
        from drainsim.octview import display_grid
        dg = display_grid(sim)
        ws = np.zeros(sim.N)
        ws[sel] = w[sel]
        frac = dg.field(sim, ws)
        arg = dg.argmax_node(sim, ws)
        bid = -np.ones(dg.ncells, np.int16)
        ok = arg >= 0
        bid[ok] = rank[body[arg[ok]]].astype(np.int16)
        return bid, frac, vol[keep] * 1e3
    n = sim.ncells
    bid = -np.ones(n, np.int16)
    frac = np.zeros(n, np.float32)
    host = sim.host[sel]
    np.maximum.at(frac, host, w[sel].astype(np.float32))
    bid[host] = rank[body[sel]].astype(np.int16)
    return bid, frac, vol[keep] * 1e3


def field_grid(sim):
    """(shape, origin, dx) of the grid the saved fields live on."""
    if getattr(sim, "octree", False):
        from drainsim.octview import display_grid
        dg = display_grid(sim)
        return dg.shape, dg.origin, dg.dx
    return sim.grid.shape, sim.grid.origin, sim.grid.dx


def run(args, dx):
    mesh, info = load_car(args.stl, args.orient, args.reverse)
    tag = f"car_{dx*1e3:g}mm" + (f"_oct{args.levels}" if args.levels else "") + \
        (f"_n{args.narrow}" if args.narrow else "") + \
        f"_k{args.subcells}_{args.mode}" + \
        ("" if args.motion == "short" else "_full")
    out = os.path.join(args.out, tag)
    os.makedirs(out, exist_ok=True)
    mk = MOTIONS[args.motion]
    mo = car_motion(args.rock, args.hang if args.mode == "transient" else 0.0, args.motion,
                    args.rotation, args.sense)
    lo = np.asarray(mesh.vertices).min(0) - args.pad
    hi = np.asarray(mesh.vertices).max(0) + args.pad
    t0 = time.time()
    grid = car_grid(mesh, dx, lo, hi, args.levels, octree=bool(args.narrow))
    del mesh                                   # not needed any more (memory)
    import gc
    gc.collect()
    print(f"[{tag}] grid {grid.shape} = {grid.ncells/1e6:.2f} M cells "
          f"({time.time()-t0:.0f} s)", flush=True)
    t0 = time.time()
    cps = args.cells_per_step if args.cells_per_step > 0 else 1e9     # 0: fixed dt
    sim = Simulation(grid, mo, dt_max=args.dt_max, cells_per_step=cps,
                     subcells=args.subcells, split=(args.mode == "transient"),
                     throat_model=ThroatModel(Cd=args.cd),
                     compressible_air=args.compressible, threads=args.threads,
                     narrow=narrow_opt(args), holes=explicit_holes(args, info),
                     channels=gap_channels(args, info))
    setup_s = time.time() - t0
    print(f"[{tag}] setup {setup_s:.0f} s: {sim.N/1e6:.2f} M nodes "
          f"({(sim.N - sim.ncells)/1e6:.2f} M sub-cell), {sim.comp.n} compartments, "
          f"{len(sim.comp.throats)} throats, stats {sim.volfrac_stats}, "
          f"peak {peak_rss_gb()} GB", flush=True)
    # report times: after dip-in (rotation done at 15 s; 20 s settled),
    # middle of the bath, before dip-out, end of dip-out, end of hanging
    t_air = list(mk["air"])
    t_liq = [mk["dipout"]] + ([mo.t_end] if mo.t_end > mk["dipout"] + 1e-9 else [])
    stops = sorted(set(t_air + t_liq + list(np.arange(1.0, mo.t_end + 1e-9, 1.0))))
    res = dict(tag=tag, dx=dx, subcells=args.subcells, mode=args.mode, motion=args.motion,
               rock=args.rock, t_dipout=mk["dipout"],
               hang=args.hang if args.mode == "transient" else 0.0,
               cells_per_step=args.cells_per_step, dt_max=args.dt_max, cd=args.cd,
               compressible_air=bool(args.compressible), car=info,
               cells=int(grid.ncells), nodes=int(sim.N), setup_s=setup_s,
               air={}, liquid={})
    t_run = time.time()
    for ts in stops:
        sim.run(t_end=ts)
        if ts in t_air:
            rows = pockets_table(sim, info, "air")
            res["air"][f"{ts:g}"] = rows
            bid, frac, vols = body_fields(sim, "air")
            np.savez_compressed(os.path.join(out, f"fields_air_t{ts:g}.npz"), bid=bid,
                                frac=frac, vols=vols, **dict(zip(("shape", "origin", "dx"),
                                                                 field_grid(sim))))
            print(f"[{tag}] t={ts:g} s: trapped air {sum(r['litres'] for r in rows):.3f} l "
                  f"in {len(rows)} pockets > {MIN_BODY_L} l", flush=True)
        if ts in t_liq:
            rows = pockets_table(sim, info, "liquid")
            res["liquid"][f"{ts:g}"] = rows
            bid, frac, vols = body_fields(sim, "liquid")
            np.savez_compressed(os.path.join(out, f"fields_liquid_t{ts:g}.npz"), bid=bid,
                                frac=frac, vols=vols, **dict(zip(("shape", "origin", "dx"),
                                                                 field_grid(sim))))
            print(f"[{tag}] t={ts:g} s: trapped liquid {sum(r['litres'] for r in rows):.3f} l "
                  f"in {len(rows)} cavities > {MIN_BODY_L} l", flush=True)
        el = time.time() - t_run
        if ts > 0 and (abs(ts / 5.0 - round(ts / 5.0)) < 1e-9 or ts == mo.t_end):
            eta = el / ts * (mo.t_end - ts)
            print(f"[{tag}] t={ts:5.1f}/{mo.t_end:g} s  steps {len(sim.hist.t)-1}  "
                  f"wall {el/60:6.1f} min  ETA {eta/60:6.1f} min  peak {peak_rss_gb()} GB",
                  flush=True)
    res["steps"] = len(sim.hist.t) - 1
    res["wall_s"] = time.time() - t_run
    res["peak_rss_gb"] = peak_rss_gb()
    H = sim.hist.arrays()
    res["history"] = dict(t=H["t"].tolist(), air_trapped_l=(H["air_trapped"] * 1e3).tolist(),
                          liquid_trapped_l=(H["liquid_trapped"] * 1e3).tolist(),
                          liquid_retained_l=(H["liquid_retained"] * 1e3).tolist())
    json.dump(res, open(os.path.join(out, "summary.json"), "w"), indent=1)
    write_tables(res, out)
    plot_history(res, os.path.join(out, "history.png"))
    return res


def write_tables(res, out):
    L = [f"# {res['tag']}", "",
         f"{res['cells']/1e6:.2f} M cells, {res['nodes']/1e6:.2f} M nodes, "
         f"{res['steps']} steps, {res['wall_s']/3600:.2f} h, peak {res['peak_rss_gb']} GB", ""]
    for kind, key in (("air", "air"), ("liquid", "liquid")):
        for t, rows in res[key].items():
            L.append(f"## {kind} at t = {t} s: {sum(r['litres'] for r in rows):.3f} l in "
                     f"{len(rows)} bodies > {MIN_BODY_L} l")
            L.append("")
            L.append("| # | litres | where (car frame) | centre, file coordinates |")
            L.append("|---|---|---|---|")
            for i, r in enumerate(rows):
                c = ", ".join(f"{q:.0f}" for q in r["centre_file"])
                L.append(f"| {i+1} | {r['litres']:.3f} | {r['where']} | {c} |")
            L.append("")
    L.append(f"Article: {ARTICLE['air_l']} l air in {ARTICLE['air_n']} pockets after dip-in; "
             f"{ARTICLE['liquid_l']} l liquid in {ARTICLE['liquid_n']} cavities after dip-out.")
    open(os.path.join(out, "pockets.md"), "w").write("\n".join(L) + "\n")


def plot_history(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    h = res["history"]
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
    ax[0].plot(h["t"], h["air_trapped_l"], color="#2a78d6")
    ax[0].set_ylabel("trapped air below the bath (l)")
    ax[1].plot(h["t"], h.get("liquid_trapped_l", h["liquid_retained_l"]), color="#eb6834")
    ax[1].set_ylabel("trapped liquid above the bath (l)")
    ax[1].set_yscale("symlog", linthresh=1.0)
    for a in ax:
        a.set_xlabel("time (s)")
        for t in (15, res.get("t_dipout", 55) - 15, res.get("t_dipout", 55)):
            a.axvline(t, color="#a3a29d", lw=0.8, ls="--")
        a.grid(color="#e6e5e0", lw=0.8)
    fig.suptitle(res["tag"], x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ------------------------------------------------------------------ renders
COLORS = ["#d62728", "#2ca02c", "#1f77b4", "#17becf", "#e7c416", "#8c1c13", "#1b5e20",
          "#e377c2", "#ff7f0e", "#9467bd", "#8c564b", "#7f7f7f", "#bcbd22", "#393b79",
          "#637939", "#843c39"]


def render_pockets(mesh, npz, path, view, title, size=(1800, 1000), max_faces=1_500_000):
    """Car transparent, each pocket a colour; view 'top' or 'side'
    (car frame, like the article's Figs. 10-13)."""
    import vtk
    from skimage import measure
    from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray
    d = np.load(npz)
    bid = d["bid"].reshape(tuple(d["shape"]))
    frac = d["frac"].reshape(tuple(d["shape"]))
    origin, dx, vols = d["origin"], float(d["dx"]), d["vols"]

    def poly(V, F):
        pts = vtk.vtkPoints()
        pts.SetData(numpy_to_vtk(np.ascontiguousarray(V, float), deep=True))
        ca = vtk.vtkCellArray()
        ca.SetData(numpy_to_vtkIdTypeArray(np.arange(0, 3 * len(F) + 1, 3, dtype=np.int64), deep=True),
                   numpy_to_vtkIdTypeArray(np.ascontiguousarray(F, np.int64).ravel(), deep=True))
        pd = vtk.vtkPolyData()
        pd.SetPoints(pts)
        pd.SetPolys(ca)
        return pd

    ren = vtk.vtkRenderer()
    ren.SetBackground(1, 1, 1)
    ren.SetUseDepthPeeling(1)
    ren.SetMaximumNumberOfPeels(40)
    F = np.asarray(mesh.faces)
    if len(F) > max_faces:                            # keep the render light
        F = F[np.random.default_rng(0).choice(len(F), max_faces, replace=False)]
    car = poly(np.asarray(mesh.vertices), F)
    m = vtk.vtkPolyDataMapper()
    m.SetInputData(car)
    a = vtk.vtkActor()
    a.SetMapper(m)
    a.GetProperty().SetColor(0.72, 0.72, 0.74)
    a.GetProperty().SetOpacity(0.07)
    a.GetProperty().LightingOff()
    ren.AddActor(a)
    legend = []
    for i in range(len(vols)):
        col = COLORS[i % len(COLORS)]
        rgb = [int(col[k:k + 2], 16) / 255 for k in (1, 3, 5)]
        mask = bid == i
        idx = np.argwhere(mask)
        lo = np.maximum(idx.min(0) - 2, 0)
        hi = np.minimum(idx.max(0) + 3, np.array(bid.shape))
        sub = np.where(mask, np.maximum(frac, 0.6), 0.0)[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        sub = np.pad(sub, 1)
        try:
            v, f, _, _ = measure.marching_cubes(sub, 0.5)
            V = origin + (v - 1 + lo + 0.5) * dx
            pd = poly(V, f)
        except (ValueError, RuntimeError):
            c = origin + (idx.mean(0) + 0.5) * dx
            s = vtk.vtkSphereSource()
            s.SetCenter(*c)
            s.SetRadius((3 * vols[i] * 1e-3 / (4 * np.pi)) ** (1 / 3))
            s.Update()
            pd = s.GetOutput()
        mm = vtk.vtkPolyDataMapper()
        mm.SetInputData(pd)
        aa = vtk.vtkActor()
        aa.SetMapper(mm)
        aa.GetProperty().SetColor(*rgb)
        ren.AddActor(aa)
        legend.append((col, vols[i]))
    rw = vtk.vtkRenderWindow()
    rw.SetOffScreenRendering(1)
    rw.SetSize(*size)
    rw.SetAlphaBitPlanes(1)
    rw.SetMultiSamples(0)
    rw.AddRenderer(ren)
    cam = ren.GetActiveCamera()
    cam.SetFocalPoint(0, 0, 0)
    if view == "top":
        cam.SetPosition(0, 0, 20)       # from above: front (+X) to the right,
        cam.SetViewUp(0, 1, 0)          # left side (+Y) up
    else:
        cam.SetPosition(0, -20, 0)      # seen from the right side (-Y)
        cam.SetViewUp(0, 0, 1)
    cam.ParallelProjectionOn()
    ren.ResetCamera()
    cam.Zoom(1.15)
    rw.Render()
    w = vtk.vtkWindowToImageFilter()
    w.SetInput(rw)
    w.Update()
    wr = vtk.vtkPNGWriter()
    tmp = path + ".raw.png"
    wr.SetFileName(tmp)
    wr.SetInputConnection(w.GetOutputPort())
    wr.Write()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    img = mpimg.imread(tmp)
    fig, ax = plt.subplots(figsize=(size[0] / 150, size[1] / 150 + 0.6))
    ax.imshow(img)
    ax.set_axis_off()
    ax.set_title(title, loc="left", fontsize=11)
    for col, v in legend[:16]:
        ax.plot([], [], "s", color=col, ms=9, label=f"{v:.3f} l")
    if legend:
        ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    os.remove(tmp)
    print("wrote", path, flush=True)


def render_run(stl, run_dir):
    mesh, info = load_car(stl)
    res = json.load(open(os.path.join(run_dir, "summary.json")))
    t_air = "20" if "20" in res["air"] else sorted(res["air"], key=float)[0]
    t_liq = f"{res.get('t_dipout', 55):g}"
    jobs = [("air", t_air, "top", "fig10_air_top.png", "Fig. 10: trapped air after dip-in, top view"),
            ("air", t_air, "side", "fig11_air_side.png", "Fig. 11: trapped air after dip-in, side view"),
            ("liquid", t_liq, "top", "fig12_liquid_top.png", "Fig. 12: trapped liquid after dip-out, top view"),
            ("liquid", t_liq, "side", "fig13_liquid_side.png", "Fig. 13: trapped liquid after dip-out, side view")]
    for kind, t, view, fn, title in jobs:
        rows = res[kind].get(t, [])
        tot = sum(r["litres"] for r in rows)
        render_pockets(mesh, os.path.join(run_dir, f"fields_{kind}_t{t}.npz"),
                       os.path.join(run_dir, fn), view,
                       f"{title} (t = {t} s): {tot:.2f} l in {len(rows)} bodies > 0.01 l, "
                       f"drainsim {res['dx']*1e3:g} mm, {res['mode']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stl", required=True)
    ap.add_argument("--dx", type=float, nargs="*", default=[])
    ap.add_argument("--subcells", type=int, default=2)
    ap.add_argument("--threads", type=int, default=2,
                    help="threads for the model's parallel passes (1 = off)")
    ap.add_argument("--mode", choices=("transient", "equilibrium"), default="transient")
    ap.add_argument("--rock", type=float, default=5.0, help="rocking amplitude, deg")
    ap.add_argument("--motion", choices=("short", "full"), default="short",
                    help="short: dip-in, 5 s under the surface, dip-out (35 s); "
                         "full: the article's 55 s with transport and rocking")
    ap.add_argument("--hang", type=float, default=20.0,
                    help="seconds of hanging after dip-out (transient mode)")
    ap.add_argument("--cells-per-step", type=float, default=0.0,
                    help="limit the step so the surface moves at most this many cells "
                         "per step; 0 (default): fixed steps of --dt-max")
    ap.add_argument("--dt-max", type=float, default=0.1)
    ap.add_argument("--cd", type=float, default=0.65)
    ap.add_argument("--pad", type=float, default=0.06)
    ap.add_argument("--compressible", action="store_true")
    ap.add_argument("--preview-motion", action="store_true")
    add_car_args(ap)
    ap.add_argument("--render", default=None, help="run folder to render (Figs. 10-13)")
    ap.add_argument("--out", default="runs_car")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.preview_motion:
        mesh, info = load_car(a.stl, a.orient, a.reverse)
        print(json.dumps(info, indent=1), flush=True)
        preview_motion(mesh, car_motion(a.rock, 0.0, a.motion, a.rotation, a.sense),
                       os.path.join(a.out, "motion_preview.png"))
    for dx in a.dx:
        run(a, dx)
    if a.render:
        render_run(a.stl, a.render)
