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
from numba import njit

from .voxel import _range, tri_box_overlap


@njit(cache=True, nogil=True)
def _mark_walls(tri, origin, dx, dims, mask):
    """mask[i, j, k] = 1 for every voxel whose box touches a triangle: the
    exact (separating-axis) test of ``voxel.surface_cells``, so a wall is
    closed for face-connected free space. Serial and without the GIL: the
    necks are measured in threads."""
    h = 0.5 * dx * (1.0 + 1e-6)
    lo = np.empty(3, np.int64)
    hi = np.empty(3, np.int64)
    c = np.empty(3)
    for t in range(tri.shape[0]):
        if _range(tri[t], origin, dx, dims, lo, hi):
            for i in range(lo[0], hi[0] + 1):
                c[0] = origin[0] + (i + 0.5) * dx
                for j in range(lo[1], hi[1] + 1):
                    c[1] = origin[1] + (j + 0.5) * dx
                    for k in range(lo[2], hi[2] + 1):
                        c[2] = origin[2] + (k + 0.5) * dx
                        if tri_box_overlap(c, h, tri[t, 0], tri[t, 1], tri[t, 2]):
                            mask[i, j, k] = 1


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
        if np.prod(hi - lo) > 3e-3:            # a large opening (3 M voxels of 1 mm)
            return np.nan
        # the walls in the box: every voxel a triangle touches
        c = 0.5 * (lo + hi)
        rad = 0.5 * np.linalg.norm(hi - lo) + rmax_t
        tri = T[np.asarray(Ttree.query_ball_point(c, rad), np.int64)]
        hi_g = lo + shape * res                    # the voxels' extent (rounded up)
        if len(tri):
            tlo, thi = tri.min(1), tri.max(1)
            tri = np.ascontiguousarray(tri[np.all((thi >= lo) & (tlo <= hi_g), axis=1)])
        wall = np.zeros(shape, np.uint8)
        if len(tri):
            _mark_walls(tri, lo, float(res), shape.astype(np.int64), wall)
        solid = wall.view(bool)
        # the two sides: the voxels of the fluid nodes of each compartment
        # (1, 2); the voxels inside the fluid nodes of other compartments
        # are closed (3): a way through a third compartment is not this
        # throat's neck
        side = np.zeros(shape, np.int8)
        nn = np.asarray(ntree.query_ball_point(c, 0.5 * np.linalg.norm(hi - lo) + hs_max), np.int64)
        nn = nn[sim.fl[nn]]
        mine = (sim.lab[nn] == t.a) | (sim.lab[nn] == t.b)
        for sel, inner in ((nn[~mine], True), (nn[mine], False)):
            if not sel.size:
                continue
            # every node's cube (the voxels it touches; another
            # compartment's: those wholly inside it), one cube size at once
            if inner:
                a0 = np.ceil((sim.X[sel] - hs[sel, None] - lo) / res - 1e-9).astype(int)
                a1 = np.floor((sim.X[sel] + hs[sel, None] - lo) / res + 1e-9).astype(int)
                val = np.full(sel.size, 3, np.int8)
            else:
                a0 = np.floor((sim.X[sel] - hs[sel, None] - lo) / res).astype(int)
                a1 = np.ceil((sim.X[sel] + hs[sel, None] - lo) / res).astype(int)
                val = np.where(sim.lab[sel] == t.a, 1, 2).astype(np.int8)
            ext = np.maximum((a1 - a0).max(1), 0)
            for e in np.unique(ext):
                if e == 0:
                    continue
                m = ext == e
                o = np.stack(np.meshgrid(*[np.arange(e)] * 3, indexing="ij"), -1).reshape(-1, 3)
                g = (a0[m][:, None, :] + o[None]).reshape(-1, 3)
                v = np.repeat(val[m], len(o))
                ok = np.all((g >= 0) & (g < shape) & (g < np.repeat(a1[m], len(o), 0)), axis=1)
                side[tuple(g[ok].T)] = v[ok]
        free = ~solid & (side != 3)
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
    throat whose box exceeds 3e-3 m^3 (3 M voxels of 1 mm: a large opening, which may
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
