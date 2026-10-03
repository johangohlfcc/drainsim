"""The biggest pockets of a saved state (car_movie.py --save-at), one by one,
to judge whether each is physical: trapped air (under the bath surface) or
liquid held above it.

    python tools/debug/pockets.py STATE [--kind air|liquid] [--top 10]
           [--run SECONDS] [--images DIR] [--throat-model rt_orientation=1 ...]

For each pocket: its volume, where it is (model frame), how deep under /
how high over the bath surface, its compartments and their openings, the
shortest way through throats from its compartment to the exterior (0) and
the narrowest throat on it. --run: the state run on for that long (the
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
