"""The biggest pockets of a saved state (car_movie.py --save-at), one by one,
to judge whether each is physical: trapped air (under the bath surface) or
liquid held above it.

    python tools/debug/pockets.py STATE [--kind air|liquid] [--top 10]
           [--run SECONDS] [--images DIR] [--throat-model rt_orientation=1 ...]

For each pocket: its volume, where it is (model frame), how deep under /
how high over the bath surface, its compartments and their openings, the
shortest way through throats from its compartment to the exterior (0) and
the narrowest throat on it. --geometry MM: whether the true walls hold
the pocket (a flood on its side of its level, MM voxels). --run: the state run on for that long (the
pockets matched by place) to see which shrink. --images: a picture of each
pocket in its geometry (the walls around it cut open, the pocket's nodes,
the up direction). --throat-model: ThroatModel fields to change before
running on (e.g. after a fix in the stepping, to see a pocket go).
"""
import argparse
import collections
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))
from drainsim.model import Simulation, _label_bodies           # noqa: E402


def pockets(sim, kind, top):
    """(volume, nodes) of the ``top`` biggest pockets of ``kind``."""
    liq_t, air_t = sim.trapped_fields()
    w = air_t if kind == "air" else liq_t
    body, nb = _label_bodies(w > 1e-9, sim.lab, sim.nbr)
    idx = np.flatnonzero(body >= 0)
    vol = np.bincount(body[idx], weights=(w * sim.v)[idx], minlength=nb)
    order = np.argsort(-vol)[:top]
    o = np.argsort(body[idx], kind="stable")
    st = np.searchsorted(body[idx][o], np.arange(nb + 1))
    return [(float(vol[b]), idx[o[st[b]:st[b + 1]]], w) for b in order if vol[b] > 0]


def comp_graph(sim):
    adj = collections.defaultdict(list)
    for i, t in enumerate(sim.comp.throats):
        if t.a != t.b:
            adj[t.a].append((t.b, i))
            adj[t.b].append((t.a, i))
    return adj


def way_out(sim, adj, k0):
    """The fewest-throats way from compartment k0 to the exterior: list of
    throat ids (empty: k0 is the exterior; None: no way)."""
    if k0 == 0:
        return []
    prev = {k0: None}
    q = collections.deque([k0])
    while q:
        c = q.popleft()
        if c == 0:
            break
        for b, i in adj[c]:
            if b not in prev:
                prev[b] = (c, i)
                q.append(b)
    if 0 not in prev:
        return None
    path, c = [], 0
    while prev[c] is not None:
        c, i = prev[c]
        path.append(i)
    return path[::-1]


def describe(sim, adj, vol, nodes, w, kind):
    X, up, zb = sim.X, sim.up, sim.zb
    wv = (w * sim.v)[nodes]
    c = (X[nodes] * wv[:, None]).sum(0) / wv.sum()
    h = X[nodes] @ up
    comps, cnt = np.unique(sim.lab[nodes], return_counts=True)
    lines = [f"{vol*1e3:7.3f} l at {np.round(c, 3)}; heights {h.min()-zb:+.3f} .. {h.max()-zb:+.3f} "
             f"m from the bath surface; extent {np.round(np.ptp(X[nodes], 0), 3)} m"]
    th = sim.comp.throats
    for k in comps[np.argsort(-cnt)][:3]:
        k = int(k)
        own = [i for i, t in enumerate(th) if k in (t.a, t.b)]
        if k == 0:
            lines.append("   compartment 0 (the exterior): held by the geometry around it "
                         "(fill-spill), not by an opening")
            continue
        d = sorted(th[i].diameter * 1e3 for i in own)
        lines.append(f"   compartment {k}: {sim.comp.volume[k]*1e3:.3f} l, {len(own)} openings"
                     f"{' (d mm: ' + ', '.join(f'{x:.1f}' for x in d[:12]) + ')' if d else ''}")
        p = way_out(sim, adj, k)
        if p is None:
            lines.append("      no way out through throats: sealed in the model")
        else:
            j = min(p, key=lambda i: th[i].area)
            lines.append(f"      way out: {len(p)} throats; the narrowest {th[j].area*1e4:.3f} cm2 "
                         f"(d {th[j].diameter*1e3:.1f} mm, {'hole' if th[j].axis is not None else 'grid'}) "
                         f"between {th[j].a} and {th[j].b} at {np.round(th[j].centroid, 3)}")
    return c, "\n".join(lines)


def geometry_check(sim, nodes, w, kind, res=0.0015, margin=0.3, width=False, tol=None,
                   target="exterior"):
    """Is the pocket held by the true geometry? The walls (sim.grid.triangles)
    around it voxelised at ``res``; from the pocket a flood through free
    space on the pocket's side of its free surface (air: at or above its
    lowest level, as air only rises; liquid: at or below its highest),
    step by step, until it reaches the open space around the car (a voxel
    in a node of the exterior clear of walls holding bath liquid, for air;
    in the atmosphere, for liquid) or the box ``margin`` beyond the pocket. Returns
    (result, where, how far): result "held" (no way out), "exterior" (a
    way out to the exterior) or "box" (a way leaves the box: undecided)."""
    from scipy import ndimage
    from scipy.spatial import cKDTree
    T = sim.grid.triangles
    up = np.asarray(sim.up, float)
    X = sim.X[nodes]
    lo, hi = X.min(0) - margin, X.max(0) + margin
    C = T.mean(1)
    tri = T[np.all((C > lo - 0.02) & (C < hi + 0.02), axis=1)]
    shape = np.ceil((hi - lo) / res).astype(int)
    if np.prod(shape) > 1.2e9:
        return None, None, None
    solid = np.zeros(shape, bool)
    if len(tri):
        e1, e2 = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
        L = np.maximum(np.linalg.norm(e1, axis=1), np.linalg.norm(e2, axis=1))
        n = np.clip(np.ceil(L / (0.4 * res)), 1, 400).astype(int)
        for k in np.unique(n):
            sel = np.flatnonzero(n == k)
            for c0 in range(0, sel.size, max(1, 2_000_000 // ((k + 1) ** 2))):
                ss = sel[c0:c0 + max(1, 2_000_000 // ((k + 1) ** 2))]
                u, v = np.meshgrid(np.arange(k + 1) / k, np.arange(k + 1) / k)
                m = (u + v) <= 1.0
                P = (tri[ss, None, 0] + u[m][None, :, None] * e1[ss, None]
                     + v[m][None, :, None] * e2[ss, None])
                q = np.floor((P.reshape(-1, 3) - lo) / res).astype(int)
                ok = np.all((q >= 0) & (q < shape), axis=1)
                solid[tuple(q[ok].T)] = True
    g = [lo[d] + (np.arange(shape[d]) + 0.5) * res for d in range(3)]
    H = (g[0][:, None, None] * up[0] + g[1][None, :, None] * up[1]).astype(np.float32) +         (g[2][None, None, :] * up[2]).astype(np.float32)
    h = X @ up
    # strictly on the pocket's side of its level (a voxel in): a pocket that
    # drained down to a sill must not step over it
    tol = res if tol is None else max(tol, res)
    free = ~solid & ((H >= h.min() + tol) if kind == "air" else (H <= h.max() - tol))
    del H
    dist = ndimage.distance_transform_edt(~solid).astype(np.float32) if width else None
    hs = 0.5 * sim.nsize

    def cube_free(k):
        """No wall voxel in node k's cube (not a cut cell straddling a wall)."""
        a0 = np.maximum(np.floor((sim.X[k] - hs[k] - lo) / res).astype(int), 0)
        a1 = np.minimum(np.ceil((sim.X[k] + hs[k] - lo) / res).astype(int), shape)
        if np.any(a1 <= a0):
            return False
        return not solid[a0[0]:a1[0], a0[1]:a1[1], a0[2]:a1[2]].any()
    # seeds: the pocket's nodes clear of the walls (a cut cell's centre can
    # lie beyond the wall it straddles)
    good = [k for k in nodes if cube_free(k)]
    seeds = np.floor((sim.X[good] - lo) / res).astype(int) if good else np.zeros((0, 3), int)
    seeds = seeds[np.all((seeds >= 0) & (seeds < shape), axis=1)] if len(seeds) else seeds
    seeds = seeds[free[tuple(seeds.T)]] if len(seeds) else seeds
    if len(seeds) == 0:
        return None, None, None
    own0 = (sim.lab[nodes] == 0).any() and target != "other"
    own_comps = np.unique(sim.lab[nodes])
    tree = getattr(sim, "_pk_tree", None)
    if tree is None:
        tree = sim._pk_tree = cKDTree(sim.X)
    def flood(mask):
        front = np.zeros(shape, bool)
        front[tuple(seeds.T)] = True
        reached = front.copy()
        st = ndimage.generate_binary_structure(3, 1)
        f = seeds
        for step in range(1, 20000):
            # dilate only around the current front
            b0 = np.maximum(f.min(0) - 1, 0)
            b1 = np.minimum(f.max(0) + 2, shape)
            sl = tuple(slice(b0[d], b1[d]) for d in range(3))
            sub = ndimage.binary_dilation(front[sl], st) & mask[sl] & ~reached[sl]
            front[sl] = False
            front[sl] = sub
            if not sub.any():
                return "held", None, None
            reached[sl] |= sub
            f = np.argwhere(sub) + b0
            if (f.min(0) == 0).any() or (f.max(0) == shape - 1).any():
                e = f[np.any((f == 0) | (f == shape - 1), axis=1)][0]
                return "box", lo + (e + 0.5) * res, step * res
            if not own0:
                P = lo + (f + 0.5) * res
                _, j = tree.query(P, k=4)
                inside = np.all(np.abs(sim.X[j] - P[:, None, :]) <= hs[j][:, :, None] * (1 + 1e-9), axis=2)
                nd = np.where(inside.any(1), j[np.arange(len(P)), inside.argmax(1)], -1)
                ndc = np.maximum(nd, 0)
                if target == "other":
                    # any fluid of another compartment (where the pocket could go)
                    hit = (nd >= 0) & sim.fl[ndc] & ~np.isin(sim.lab[ndc], own_comps)
                else:
                    # the open exterior: for air, bath liquid (the air would rise
                    # on through it); for liquid, the atmosphere (it drains on)
                    hit = (nd >= 0) & sim.fl[ndc] & (sim.lab[ndc] == 0) & \
                        (sim.B[ndc] if kind == "air" else sim.A[ndc])
                for q in np.flatnonzero(hit)[:50]:
                    if cube_free(nd[q]):                   # an exterior node clear of walls
                        return "exterior", P[q], step * res
        return "box", None, None

    r = flood(free)
    if not width or r[0] != "exterior":
        return r
    # the widest way out: the most clearance from the walls (voxels) the
    # flood can keep and still get out
    lo_c, hi_c = 0.0, 6.0
    for _ in range(6):
        mid = 0.5 * (lo_c + hi_c)
        if flood(free & (dist >= mid))[0] == "exterior":
            lo_c = mid
        else:
            hi_c = mid
    return r[0], r[1], r[2], 2 * lo_c * res


def image(sim, nodes, c, path, title):
    import pyvista as pv
    pv.OFF_SCREEN = True
    T = sim.grid.triangles
    R = max(0.12, 0.6 * float(np.ptp(sim.X[nodes], 0).max()))
    C = T.mean(1)
    sel = np.all(np.abs(C - c) < R, axis=1)
    up = np.asarray(sim.up, float)
    side = np.cross(up, [1.0, 0, 0]) if abs(up[0]) < 0.9 else np.cross(up, [0, 1.0, 0])
    side /= np.linalg.norm(side)
    pl = pv.Plotter(shape=(1, 2), off_screen=True, window_size=(2000, 900))
    for col, cut in enumerate((False, True)):
        pl.subplot(0, col)
        s = sel & (((C - c) @ side) < 0) if cut else sel
        tri = T[s]
        if len(tri):
            faces = np.hstack([np.full((len(tri), 1), 3), np.arange(3 * len(tri)).reshape(-1, 3)])
            pl.add_mesh(pv.PolyData(tri.reshape(-1, 3), faces.ravel()), color="lightgrey",
                        opacity=0.35 if not cut else 1.0)
        pl.add_points(sim.X[nodes], color="orange", point_size=6, render_points_as_spheres=True)
        pl.add_arrows(np.array([c + up * 0.6 * R]), np.array([up]), mag=0.3 * R, color="blue")
        pl.add_text(title + ("  (cut open)" if cut else ""), font_size=10)
        pl.camera.focal_point = c
        pl.camera.position = c + (side * 2.2 + up * 0.8) * R
        pl.camera.up = up
    pl.screenshot(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("state")
    ap.add_argument("--kind", choices=("air", "liquid"), default="air")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--run", type=float, default=0.0)
    ap.add_argument("--images", default=None)
    ap.add_argument("--throat-model", nargs="*", default=[])
    ap.add_argument("--geometry", type=float, default=0.0, metavar="MM",
                    help="check each pocket against the true geometry voxelised at MM")
    ap.add_argument("--width", action="store_true",
                    help="--geometry: also the width of the way out at its narrowest")
    ap.add_argument("--tol", type=float, default=0.0, metavar="MM",
                    help="--geometry: keep this far inside the pocket's level (default one "
                         "voxel; e.g. a sill's capillary hold-up)")
    ap.add_argument("--target", choices=("exterior", "other"), default="exterior",
                    help="--geometry: a way to the open exterior (default), or to any "
                         "other compartment's fluid (where the pocket could move)")
    ap.add_argument("--only", type=int, nargs="*", default=None,
                    help="check only these pockets (numbers in the list)")
    a = ap.parse_args()
    t0 = time.time()
    sim, extra = Simulation.load_state(a.state)
    print(f"{a.state}: t = {sim.t:.2f} s, {sim.N} nodes, {sim.comp.n} compartments "
          f"({time.time() - t0:.0f} s); up {np.round(sim.up, 3)}, bath level {sim.zb:.3f}", flush=True)
    for kv in a.throat_model:
        k, v = kv.split("=")
        old = getattr(sim.tm, k)
        setattr(sim.tm, k, type(old)(float(v)) if not isinstance(old, bool) else v not in ("0", "False"))
        print(f"   throat model {k} = {getattr(sim.tm, k)}")
    adj = comp_graph(sim)
    P = pockets(sim, a.kind, a.top)
    print(f"the {len(P)} biggest pockets of {a.kind}:")
    cents = []
    for n, (vol, nodes, w) in enumerate(P, 1):
        c, txt = describe(sim, adj, vol, nodes, w, a.kind)
        cents.append(c)
        print(f"#{n:2d} " + txt, flush=True)
        if a.geometry and (a.only is None or n in a.only):
            out = geometry_check(sim, nodes, w, a.kind, a.geometry * 1e-3, width=a.width,
                                 tol=a.tol * 1e-3 if a.tol else None, target=a.target)
            r, where, far = out[:3]
            if r is None:
                print("   true geometry: not checked")
            elif r == "held":
                print("   true geometry: HELD (no way out on its side of its level)", flush=True)
            elif r == "exterior":
                wd = f", {out[3]*1e3:.1f} mm wide at its narrowest" if len(out) > 3 else ""
                print(f"   true geometry: NOT HELD - a way on its side of its level reaches the "
                      f"exterior at {np.round(where, 3)} ({far*1e3:.0f} mm{wd})", flush=True)
            else:
                print(f"   true geometry: a way leaves the box at {None if where is None else np.round(where, 3)}"
                      f" (undecided)", flush=True)
        if a.images:
            os.makedirs(a.images, exist_ok=True)
            image(sim, nodes, c, os.path.join(a.images, f"{a.kind}_{n:02d}.png"),
                  f"{a.kind} #{n}: {vol*1e3:.2f} l, t = {sim.t:.1f} s")
    if a.run > 0:
        t1 = sim.t + a.run
        sim.run(t_end=t1)
        liq_t, air_t = sim.trapped_fields()
        w2 = (air_t if a.kind == "air" else liq_t) * sim.v
        print(f"\nafter {a.run:g} s more (t = {sim.t:.2f} s), in the same compartments "
              f"(the same nodes in the exterior):")
        for n, (vol, nodes, _) in enumerate(P, 1):
            comps = np.unique(sim.lab[nodes])
            inner = comps[comps != 0]
            v2 = w2[np.isin(sim.lab, inner)].sum() if inner.size else 0.0
            if (comps == 0).any():
                v2 += w2[nodes[sim.lab[nodes] == 0]].sum()
            print(f"#{n:2d} {vol*1e3:7.3f} l -> {v2*1e3:7.3f} l")


if __name__ == "__main__":
    main()
