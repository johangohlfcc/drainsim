"""Grid throats sized from the true geometry (7.2).

The segmentation joins compartments by throats made of the grid faces
where they meet. In a narrow passage that the grid resolves only with a
few sub-cells, those faces are a fraction of the real opening: a 6.5 mm
passage at the end of a bumper beam became 4 sub-cell faces, 0.07 cm2,
d 1.1 mm (its capillary hold-up 27 mm, its Rayleigh-Taylor cut-off
closed), and the beam held its air through a whole dip.

``neck_size_throats`` measures each grid throat's neck on the triangles:
the widest way between its two compartments near it (the largest ball
that passes from one side to the other), and widens the throat to it if
that is at least ``min_width`` (3 mm: narrower openings are not resolved).
That is a lower bound of the true opening, so nothing opens wider than
the walls allow. (Measures from the throat's plane or from a ball at its
faces overstate a neck in open space and drained pockets the true walls
hold; they were tried and dropped.)
"""
from __future__ import annotations

import numpy as np


def neck_widths(sim, ids, res=0.001, margin=0.012, verbose=False):
    """The neck of each throat ``ids`` on the true geometry: the walls
    (sim.grid.triangles) voxelised at ``res`` in a box ``margin`` around
    the throat's faces; the voxels of the model's fluid nodes of its two
    compartments mark the two sides; the widest way between them: the
    largest clearance from the walls at which free space still joins a
    voxel of side a to one of side b. Returns the neck widths (m; twice
    that clearance; NaN where the sides do not join within the box)."""
    from scipy import ndimage
    from scipy.spatial import cKDTree
    th = sim.comp.throats
    T = np.asarray(sim.grid.triangles, float)
    Tc = T.mean(1)
    Ttree = cKDTree(Tc)
    Tr = np.linalg.norm(T - Tc[:, None, :], axis=2).max(1)
    rmax_t = float(Tr.max())
    ntree = cKDTree(sim.X)
    hs = 0.5 * sim.nsize
    hs_max = float(hs.max())
    def one(q):
        i = ids[q]
        t = th[i]
        F = 0.5 * (sim.X[t.cells_a] + sim.X[t.cells_b])
        lo = F.min(0) - margin
        hi = F.max(0) + margin
        shape = np.ceil((hi - lo) / res).astype(int)
        if np.prod(shape) > 3e6:
            return np.nan
        # the walls in the box: sample their triangles
        c = 0.5 * (lo + hi)
        rad = 0.5 * np.linalg.norm(hi - lo) + rmax_t
        tri = T[[j for j in Ttree.query_ball_point(c, rad)]]
        solid = np.zeros(shape, bool)
        if len(tri):
            e1, e2 = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
            L = np.maximum(np.linalg.norm(e1, axis=1), np.linalg.norm(e2, axis=1))
            n = np.clip(np.ceil(L / (0.5 * res)), 1, 2000).astype(int)
            for k in np.unique(n):
                sel = np.flatnonzero(n == k)
                u, v = np.meshgrid(np.arange(k + 1) / k, np.arange(k + 1) / k)
                m = (u + v) <= 1.0
                P = (tri[sel, None, 0] + u[m][None, :, None] * e1[sel, None]
                     + v[m][None, :, None] * e2[sel, None]).reshape(-1, 3)
                g = np.floor((P - lo) / res).astype(int)
                ok = np.all((g >= 0) & (g < shape), axis=1)
                solid[tuple(g[ok].T)] = True
        # the two sides: the voxels of the fluid nodes of each compartment
        side = np.zeros(shape, np.int8)
        nn = np.asarray(ntree.query_ball_point(c, 0.5 * np.linalg.norm(hi - lo) + hs_max), np.int64)
        nn = nn[sim.fl[nn] & ((sim.lab[nn] == t.a) | (sim.lab[nn] == t.b))]
        if nn.size:
            # every node's cube, all nodes of one cube size at once
            a0 = np.floor((sim.X[nn] - hs[nn, None] - lo) / res).astype(int)
            a1 = np.ceil((sim.X[nn] + hs[nn, None] - lo) / res).astype(int)
            ext = (a1 - a0).max(1)
            val = np.where(sim.lab[nn] == t.a, 1, 2).astype(np.int8)
            for e in np.unique(ext):
                m = ext == e
                o = np.stack(np.meshgrid(*[np.arange(e)] * 3, indexing="ij"), -1).reshape(-1, 3)
                g = (a0[m][:, None, :] + o[None]).reshape(-1, 3)
                v = np.repeat(val[m], len(o))
                ok = np.all((g >= 0) & (g < shape), axis=1)
                side[tuple(g[ok].T)] = v[ok]
        free = ~solid
        A = free & (side == 1)
        B = free & (side == 2)
        if not A.any() or not B.any():
            return np.nan
        dist = ndimage.distance_transform_edt(free)

        def joined(cl):
            lab, _ = ndimage.label(free & (dist >= cl))
            la = np.unique(lab[A & (dist >= cl)])
            lb = np.unique(lab[B & (dist >= cl)])
            la, lb = la[la > 0], lb[lb > 0]
            return np.intersect1d(la, lb).size > 0
        if not joined(0.0):
            return np.nan
        lo_c, hi_c = 0.0, float(dist.max())
        for _ in range(7):
            mid = 0.5 * (lo_c + hi_c)
            if joined(mid):
                lo_c = mid
            else:
                hi_c = mid
        return 2 * lo_c * res

    import os
    from concurrent.futures import ThreadPoolExecutor
    workers = max(1, min(16, (os.cpu_count() or 2) // 2))
    out = np.full(len(ids), np.nan)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for q, w in enumerate(ex.map(one, range(len(ids)), chunksize=16)):
            out[q] = w
            if verbose and q % 2000 == 0:
                print(f"   neck widths: {q} of {len(ids)}", flush=True)
    return out


def neck_size_throats(sim, res=None, margin=0.012, min_width=0.003, max_d=0.02, verbose=True):
    """Grid throats widened to their neck on the true geometry
    (``neck_widths``) where that is wider than the grid gives, but only for
    necks of at least ``min_width`` (openings narrower than that are not
    resolved: 3 mm by default) and throats narrower than ``max_d`` (wider
    ones the grid resolves). ``res``: the voxel size (default 1 mm, or an
    eighth of the finest cell on grids coarser than 8 mm: the necks the
    grid itself cannot resolve, at a cost that does not grow with it). A
    throat whose box would exceed 3 M voxels (a large opening, which may
    span a wide part and a narrow one) keeps its grid size: one width for
    the whole of it would set the capillary hold-up of the narrow part by
    the wide one. d = the neck width, area = at least its
    circle. Lower bounds of the true opening: a ball of that size passes
    from one side to the other. Updates the throat arrays. Returns the
    number widened."""
    th = sim.comp.throats
    ids = [i for i, t in enumerate(th) if t.axis is None and not getattr(t, "slot", False)
           and t.a != t.b and t.diameter < max_d]
    if not ids:
        return 0
    if res is None:
        h = float(getattr(sim.grid, "h", getattr(sim.grid, "dx", 0.008)))
        res = max(0.001, h / 8.0)
    w = neck_widths(sim, ids, res, margin, verbose=verbose)
    n = 0
    for q, i in enumerate(ids):
        t = th[i]
        if not np.isfinite(w[q]) or w[q] < min_width or w[q] <= t.diameter:
            continue
        t.diameter = float(w[q])
        t.area = float(max(t.area, 0.25 * np.pi * w[q] ** 2))
        n += 1
    sim._throat_csr()
    if verbose:
        print(f"throat necks from the geometry: {n} of {len(ids)} grid throats widened", flush=True)
    return n
