"""Compartments and throats on a node graph (octree leaves and sub-cells).

The same method as ``compartments.segment`` on the uniform grid, with every
step expressed on the graph instead of the voxel array:

1. ``D`` = distance from each node centre to the surface (nearest surface
   sample, sampled at the smallest node size).
2. Markers = local maxima (plateaus) of ``D`` over the graph neighbours.
3. Watershed of ``-D`` from the markers: a priority flood on the graph,
   nodes far from the walls first.
4. Neighbouring regions are merged unless their boundary is a constriction:
   ``r_throat >= beta * min(r_peak)`` or ``r_throat >= d_free / 2``, where
   ``r_throat`` is the largest ``min(D_u, D_v)`` over the links between them;
   regions below ``min_volume`` merge into their widest neighbour.
5. Regions touching the domain boundary become compartment 0.
6. Narrow, locally unique openings between pre-merge regions that end up in
   one compartment are kept as internal throats (a == b).

Throats are connected groups of links between two compartments; each keeps
its links (node pairs), face areas, normals, net (projected) area, the
inscribed width of its faces and its centroid.
"""
from __future__ import annotations

import numpy as np
from numba import njit

from .compartments import Compartments, Throat, opening_frame, raster_disc_width


# --------------------------------------------------------------- basics
def links_of(indptr, indices):
    """Each undirected link once (u < v)."""
    N = indptr.size - 1
    src = np.repeat(np.arange(N, dtype=np.int64), np.diff(indptr))
    ok = (indices >= 0) & (src < indices)
    return src[ok], indices[ok]


def _touches_boundary(lab, n, boundary):
    """Per compartment: does it hold a node next to the domain boundary?"""
    tb = np.zeros(n, bool)
    tb[np.unique(lab[boundary & (lab >= 0)])] = True
    return tb


def face_geometry(X, size, u, v):
    """Face centre, unit normal (u -> v, grid axis) and area of links."""
    d = X[v] - X[u]
    ax = np.argmax(np.abs(d), axis=1)
    sg = np.sign(d[np.arange(len(d)), ax])
    su, sv = size[u], size[v]
    small_u = su <= sv
    fc = np.where(small_u[:, None], X[u], X[v]).copy()
    half = 0.5 * np.minimum(su, sv)
    fc[np.arange(len(d)), ax] += np.where(small_u, sg * half, -sg * half)
    nrm = np.zeros((len(d), 3))
    nrm[np.arange(len(d)), ax] = sg
    return fc, nrm, np.minimum(su, sv) ** 2, ax


def wall_distance(X, size, active, triangles, smin, cap):
    """Distance from each node centre to the surface triangles (sampled at
    ``smin``), capped at ``cap``."""
    from scipy.spatial import cKDTree
    from .grid import sample_triangles
    P = np.concatenate(list(sample_triangles(triangles, smin, spacing=1.0)))
    tree = cKDTree(P)
    D = np.full(len(X), cap)
    idx = np.flatnonzero(active)
    for s0 in range(0, idx.size, 2_000_000):
        sel = idx[s0:s0 + 2_000_000]
        d, _ = tree.query(X[sel], distance_upper_bound=cap, workers=-1)
        D[sel] = np.minimum(d, cap)
    return np.maximum(D, 0.5 * size)


def raster_width(pos, nrm, fsize, nvec, s):
    """Inscribed width of an opening from its faces (``pos`` face centres,
    ``nrm`` normals, ``fsize`` face edge lengths): the faces across the
    opening are split into sub-faces of size ``s``, projected on the plane
    normal to ``nvec`` and drawn on a raster; the width is that of the
    largest inscribed disc (as ``compartments.face_raster_diameter``)."""
    nvec, basis = opening_frame(nvec, 3)
    cap = np.abs(nrm @ nvec) > 0.5
    if cap.any():
        pos, nrm, fsize = pos[cap], nrm[cap], fsize[cap]
    pts = []
    for f in np.unique(fsize):
        m = fsize == f
        n = max(1, int(round(f / s)))
        if n == 1:
            pts.append(pos[m])
            continue
        g = ((np.arange(n) + 0.5) / n - 0.5) * f
        for ax in range(3):
            mm = m & (np.abs(nrm[:, ax]) > 0.5)
            if not mm.any():
                continue
            od = [d for d in range(3) if d != ax]
            offs = np.array(np.meshgrid(g, g, indexing="ij")).reshape(2, -1).T
            O = np.zeros((offs.shape[0], 3))
            O[:, od] = offs
            pts.append((pos[mm][:, None, :] + O[None]).reshape(-1, 3))
    return raster_disc_width(np.concatenate(pts), basis, s)


# ----------------------------------------------------------- segmentation
def _merge(lab, n, D, vol, u, v, beta, r_free, min_volume):
    def find(p, x):
        while p[x] != x:
            p[x] = p[p[x]]
            x = p[x]
        return x
    for _ in range(50):
        valid = lab >= 0
        peak = np.zeros(n)
        np.maximum.at(peak, lab[valid], D[valid])
        size = np.bincount(lab[valid], weights=vol[valid], minlength=n)
        la, lb = lab[u], lab[v]
        m = (la >= 0) & (lb >= 0) & (la != lb)
        if not m.any():
            break
        la, lb = la[m], lb[m]
        rf = np.minimum(D[u[m]], D[v[m]])
        lo = np.minimum(la, lb).astype(np.int64)
        hi = np.maximum(la, lb).astype(np.int64)
        ukey, inv = np.unique(lo * n + hi, return_inverse=True)
        rt = np.zeros(ukey.size)
        np.maximum.at(rt, inv, rf)
        pa, pb = ukey // n, ukey % n
        score = rt / np.maximum(np.minimum(peak[pa], peak[pb]), 1e-30)
        small = (size[pa] < min_volume) | (size[pb] < min_volume)
        order = np.argsort(-(score + 10.0 * small), kind="stable")
        par = np.arange(n)
        gpeak, gsize = peak.copy(), size.copy()
        merged = 0
        for e in order:
            x, y = find(par, pa[e]), find(par, pb[e])
            if x == y:
                continue
            is_small = gsize[x] < min_volume or gsize[y] < min_volume
            if rt[e] >= beta * min(gpeak[x], gpeak[y]) or rt[e] >= r_free or is_small:
                par[y] = x
                gpeak[x] = max(gpeak[x], gpeak[y])
                gsize[x] += gsize[y]
                merged += 1
        if merged == 0:
            break
        roots = np.array([find(par, i) for i in range(n)])
        _, newid = np.unique(roots, return_inverse=True)
        lab = np.where(lab >= 0, newid[np.maximum(lab, 0)], -1)
        n = int(newid.max()) + 1
    return lab, n


def _exterior_zero(lab, bnd):
    ext = np.unique(lab[bnd & (lab >= 0)])
    n = int(lab.max()) + 1 if lab.size and lab.max() >= 0 else 0
    new = -np.ones(max(n, 1), np.int64)
    new[ext] = 0
    present = np.zeros(max(n, 1), bool)
    present[lab[lab >= 0]] = True
    k = 1
    for r in range(n):
        if new[r] < 0 and present[r]:
            new[r] = k
            k += 1
    return np.where(lab >= 0, new[np.maximum(lab, 0)], -1)


def _groups(fc, pair_keys, radius):
    """Split links with the same compartment pair into spatially connected
    groups (face centres within ``radius``)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree
    order = np.argsort(pair_keys, kind="stable")
    ks = pair_keys[order]
    brk = np.r_[0, np.flatnonzero(np.diff(ks)) + 1, len(ks)]
    for s0, s1 in zip(brk[:-1], brk[1:]):
        sel = order[s0:s1]
        if sel.size == 1:
            yield sel
            continue
        pairs = cKDTree(fc[sel]).query_pairs(radius, output_type="ndarray")
        G = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])),
                       shape=(sel.size, sel.size)) if len(pairs) else \
            coo_matrix((sel.size, sel.size))
        ng, gl = connected_components(G, directed=False)
        for g in range(ng):
            yield sel[gl == g]


def _throat(a, b, ua, vb, X, size, s):
    """Throat from links (ua on side a, vb on side b)."""
    fc, nrm, w, fax = face_geometry(X, size, ua, vb)
    nv = (nrm * w[:, None]).sum(0)
    vs = np.linalg.norm(nv)
    area = vs if vs > 1e-3 * w.sum() else w.sum() / np.sqrt(3)
    if vs <= 1e-3 * w.sum():
        # the net area cancels: the axis with the largest total face area
        # (as compartments.node_throats does on the uniform grid)
        ax = np.argmax(np.bincount(fax, weights=w, minlength=3))
        nv = np.eye(3)[ax]
    dia = raster_width(fc, nrm, np.sqrt(w), nv, s)
    cen = (fc * w[:, None]).sum(0) / w.sum()
    return Throat(a=int(a), b=int(b), cells_a=ua.astype(np.int64), cells_b=vb.astype(np.int64),
                  normals=nrm, area=float(area), diameter=float(dia), centroid=cen,
                  weights=w)


class _Local:
    """Neighbourhood queries for ``_locally_unique``: a k-d tree of the
    active nodes and the CSR graph (made once per segmentation)."""

    def __init__(self, X, active, ptr, idx):
        from scipy.spatial import cKDTree
        self.ids = np.flatnonzero(active)
        self.tree = cKDTree(X[self.ids])
        self.ptr, self.idx = ptr, idx
        self.loc = -np.ones(len(X), np.int64)


def _locally_unique(ca, cb, lab0, pa, X, loc_q, rg, h):
    """Near the opening, are the two sides connected only through it?"""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    c = 0.5 * (X[ca].mean(0) + X[cb].mean(0))
    ext = np.max(np.linalg.norm(np.r_[X[ca], X[cb]] - c, axis=1))
    R = ext + max(2 * rg, 3 * h) + 2 * h
    nodes = loc_q.ids[np.asarray(loc_q.tree.query_ball_point(c, R), np.int64)]
    ex = np.zeros(len(nodes), bool)
    ex |= np.isin(nodes, ca)
    nodes = np.sort(nodes[~ex])
    if nodes.size == 0:
        return True
    loc = loc_q.loc
    loc[nodes] = np.arange(nodes.size)
    try:
        ptr, idx = loc_q.ptr, loc_q.idx
        cnt = ptr[nodes + 1] - ptr[nodes]
        src = np.repeat(nodes, cnt)
        pos = np.repeat(ptr[nodes] - np.r_[0, np.cumsum(cnt)[:-1]], cnt) + np.arange(cnt.sum())
        dst = idx[pos]
        ok = dst >= 0
        src, dst = src[ok], dst[ok]
        ok = loc[dst] >= 0
        G = coo_matrix((np.ones(ok.sum()), (loc[src[ok]], loc[dst[ok]])),
                       shape=(nodes.size,) * 2)
        _, cl = connected_components(G, directed=False)
        cbl = loc[cb]
        lb = set(cl[cbl[cbl >= 0]].tolist())
        sidea = nodes[lab0[nodes] == pa]
        la = set(cl[loc[sidea]].tolist())
    finally:
        loc[nodes] = -1
    return not (la & lb)


def voxel_distance(X, size, active, closed_centres, h, cap):
    """Distance from each node centre to the nearest closed level-0 cell,
    as ``compartments.distance_field`` measures it on the uniform grid:
    centre-to-centre distance minus half a cell, at least half the node
    size."""
    from scipy.spatial import cKDTree
    tree = cKDTree(closed_centres)
    D = np.full(len(X), cap)
    idx = np.flatnonzero(active)
    for s0 in range(0, idx.size, 2_000_000):
        sel = idx[s0:s0 + 2_000_000]
        d, _ = tree.query(X[sel], distance_upper_bound=cap + 0.5 * h, workers=-1)
        D[sel] = np.minimum(d - 0.5 * h, cap)
    # rounded, so that equal distances (plateaus, ties) are equal in the
    # last bit, as the uniform grid's distance transform gives them
    return np.round(np.maximum(D, 0.5 * size), 10)


@njit(cache=True)
def _flood_fifo(lab, pri, ptr, idx, active, seeds):
    """Watershed by flooding (as skimage.segmentation.watershed): the seeds
    (labelled nodes, highest ``pri`` first) are popped from a heap ordered by
    (-pri, age); a popped node labels its unlabelled active neighbours and
    pushes them with a new age. Equal priorities are therefore flooded
    first-in first-out, and fronts grow evenly over plateaus."""
    N = lab.shape[0]
    cap = N + seeds.shape[0] + 1
    hk = np.empty(cap, np.float64)
    ha = np.empty(cap, np.int64)
    hv = np.empty(cap, np.int64)
    size = 0
    age = 0
    for s in range(seeds.shape[0]):
        n = seeds[s]
        i = size
        size += 1
        hk[i] = -pri[n]
        ha[i] = age
        hv[i] = n
        age += 1
        while i > 0:
            q = (i - 1) // 2
            if hk[q] < hk[i] or (hk[q] == hk[i] and ha[q] < ha[i]):
                break
            hk[q], hk[i] = hk[i], hk[q]
            ha[q], ha[i] = ha[i], ha[q]
            hv[q], hv[i] = hv[i], hv[q]
            i = q
    while size > 0:
        c = hv[0]
        size -= 1
        hk[0] = hk[size]
        ha[0] = ha[size]
        hv[0] = hv[size]
        i = 0
        while True:
            l = 2 * i + 1
            if l >= size:
                break
            r = l + 1
            m = l
            if r < size and (hk[r] < hk[l] or (hk[r] == hk[l] and ha[r] < ha[l])):
                m = r
            if hk[i] < hk[m] or (hk[i] == hk[m] and ha[i] < ha[m]):
                break
            hk[m], hk[i] = hk[i], hk[m]
            ha[m], ha[i] = ha[i], ha[m]
            hv[m], hv[i] = hv[i], hv[m]
            i = m
        for jj in range(ptr[c], ptr[c + 1]):
            n = idx[jj]
            if n < 0 or lab[n] >= 0 or not active[n]:
                continue
            lab[n] = lab[c]
            i = size
            size += 1
            hk[i] = -pri[n]
            ha[i] = age
            hv[i] = n
            age += 1
            while i > 0:
                q = (i - 1) // 2
                if hk[q] < hk[i] or (hk[q] == hk[i] and ha[q] < ha[i]):
                    break
                hk[q], hk[i] = hk[i], hk[q]
                ha[q], ha[i] = ha[i], ha[q]
                hv[q], hv[i] = hv[i], hv[q]
                i = q
    return lab


def _links26(g, dims0):
    """Same-level edge and corner neighbours of the leaves (the links of
    the 26-neighbourhood that faces do not give), as (a, b) node pairs."""
    from .octree import _coords, _key
    A, B = [], []
    offs = [(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1)
            if abs(a) + abs(b) + abs(c) >= 2]
    for l in range(len(g.offsets) - 1):
        s0, s1 = int(g.offsets[l]), int(g.offsets[l + 1])
        if s1 <= s0:
            continue
        K = g.key[s0:s1]
        d = np.asarray(dims0, np.int64) >> l
        C = _coords(K, d)
        for o in offs:
            if o <= (0, 0, 0):                       # each pair once
                continue
            Q = C + np.array(o, np.int64)
            ok = np.all((Q >= 0) & (Q < d), axis=1)
            q = _key(Q[ok], d)
            pos = np.minimum(np.searchsorted(K, q), K.size - 1)
            f = K[pos] == q
            A.append(s0 + np.flatnonzero(ok)[f])
            B.append(s0 + pos[f])
    if not A:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    return np.concatenate(A), np.concatenate(B)


def _links26_to(g, dims0, nodes):
    """Edge and corner neighbours (same level) of the given leaf nodes:
    (node, neighbour) pairs, all 20 directions."""
    from .octree import _coords, _key
    A, B = [], []
    offs = [o for o in ((a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1))
            if abs(o[0]) + abs(o[1]) + abs(o[2]) >= 2]
    for l in range(len(g.offsets) - 1):
        s0, s1 = int(g.offsets[l]), int(g.offsets[l + 1])
        sel = nodes[(nodes >= s0) & (nodes < s1)]
        if sel.size == 0:
            continue
        K = g.key[s0:s1]
        d = np.asarray(dims0, np.int64) >> l
        C = _coords(g.key[sel], d)
        for o in offs:
            Q = C + np.array(o, np.int64)
            ok = np.all((Q >= 0) & (Q < d), axis=1)
            q = _key(Q[ok], d)
            pos = np.minimum(np.searchsorted(K, q), K.size - 1)
            f = K[pos] == q
            A.append(sel[np.flatnonzero(ok)[f]])
            B.append(s0 + pos[f])
    if not A:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    return np.concatenate(A), np.concatenate(B)


def _pairs26(g, dims0, nodes):
    """Same-level edge and corner neighbour pairs among the given leaf nodes
    (sorted node ids), each pair once."""
    from .octree import _coords, _key
    A, B = [], []
    offs = [o for o in ((a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1))
            if abs(o[0]) + abs(o[1]) + abs(o[2]) >= 2 and o > (0, 0, 0)]
    for l in range(len(g.offsets) - 1):
        s0, s1 = int(g.offsets[l]), int(g.offsets[l + 1])
        sel = nodes[(nodes >= s0) & (nodes < s1)]
        if sel.size == 0:
            continue
        K = g.key[sel]                      # sorted, as the node ids are
        d = np.asarray(dims0, np.int64) >> l
        C = _coords(K, d)
        for o in offs:
            Q = C + np.array(o, np.int64)
            ok = np.all((Q >= 0) & (Q < d), axis=1)
            q = _key(Q[ok], d)
            pos = np.minimum(np.searchsorted(K, q), K.size - 1)
            f = K[pos] == q
            A.append(sel[np.flatnonzero(ok)[f]])
            B.append(sel[pos[f]])
    if not A:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    return np.concatenate(A), np.concatenate(B)


def segment_graph(g, triangles, h, k=1, beta=0.6, d_free=0.03, min_cells=32, split=True,
                  internal_holes=True, explicit=(), plugs=(), verbose=False,
                  closed_centres=None, dims0=None):
    """Compartments of a node graph ``g`` (``octree.OctGraph``).

    closed_centres: centres of the closed level-0 cells; the distance field
    is then measured to them as on the uniform grid (``voxel_distance``);
    None: to the surface triangles (``wall_distance``).
    dims0: the octree's level-0 index space; markers are then the maxima of
    D over the 26-neighbourhood of the leaves (edge and corner neighbours of
    the same level added to the links), labelled 26-connected, as on the
    uniform grid; sub-cells are never markers. None: face links only.

    Returns (Compartments, D). ``label`` is per node (-1 inactive)."""
    import time
    from .model import _priority_flood
    from .par import label_bodies
    t0 = time.time()
    X, size, vol, active = g.X, g.size, g.vol, g.active
    ptr, idx = g.indptr, g.indices
    N = g.N
    s = h / max(int(k), 1)
    cap = 4.0 * float(size.max())
    if closed_centres is not None:
        D = voxel_distance(X, size, active, closed_centres, h, np.inf)   # uncapped, as EDT
    else:
        D = wall_distance(X, size, active, triangles, s, cap)
    u, v = links_of(ptr, idx)
    keep = active[u] & active[v]
    u, v = u[keep], v[keep]
    if not split:
        lab, n = label_bodies(active, np.zeros(N, np.int64), ptr, idx)
        lab = _exterior_zero(lab, g.boundary & active)
        n = int(lab.max()) + 1
        comp = Compartments(label=lab, n=n, dist=D)
        comp.throats = []
        comp.volume = np.bincount(lab[lab >= 0], weights=vol[lab >= 0], minlength=n)
        comp.touches_boundary = _touches_boundary(lab, n, g.boundary)
        return comp, D
    # markers: local maxima (plateaus) of D
    nmax = np.full(N, -np.inf)
    np.maximum.at(nmax, u, D[v])
    np.maximum.at(nmax, v, D[u])
    if dims0 is not None:
        a26, b26 = _links26(g, dims0)
        k26 = active[a26] & active[b26]
        a26, b26 = a26[k26], b26[k26]
        np.maximum.at(nmax, a26, D[b26])
        np.maximum.at(nmax, b26, D[a26])
        peak = active & ~g.fine & (D >= nmax)
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
        pa_ = np.r_[u, a26]
        pb_ = np.r_[v, b26]
        m_ = peak[pa_] & peak[pb_]
        G = coo_matrix((np.ones(m_.sum()), (pa_[m_], pb_[m_])), shape=(N, N))
        _, cl = connected_components(G, directed=False)
        pk = np.flatnonzero(peak)
        _, first = np.unique(cl[pk], return_index=True)
        rank = -np.ones(int(cl.max()) + 1, np.int64)
        rank[cl[pk[np.sort(first)]]] = np.arange(first.size)   # numbered by lowest node
        mk = np.where(peak, rank[cl], -1)
        nm = first.size
    else:
        peak = active & (D >= nmax)
        mk, nm = label_bodies(peak, np.zeros(N, np.int64), ptr, idx)
    if dims0 is not None:
        seeds = np.flatnonzero(mk >= 0)
        seeds = seeds[np.lexsort((seeds, -D[seeds]))]
        lab0 = _flood_fifo(mk.astype(np.int64), D, ptr, idx, active.astype(np.bool_),
                           seeds.astype(np.int64))
    else:
        lab0 = _priority_flood(mk.astype(np.int64), D, ptr, idx, active.astype(np.bool_))
    n0 = int(nm)
    r_free = np.inf if d_free is None else 0.5 * d_free
    min_volume = min_cells * h ** 3
    cands = []
    if internal_holes:
        cands = _hole_candidates(lab0, n0, D, X, size, u, v, active, beta, r_free, h,
                                 _Local(X, active, ptr, idx))
    if dims0 is not None:
        # as on the uniform grid: regions merge across leaf faces only, and
        # their size counts the leaves
        lf = ~g.fine[u] & ~g.fine[v]
        lab, n = _merge(lab0.copy(), n0, D, np.where(g.fine, 0.0, vol), u[lf], v[lf], beta,
                        r_free, min_volume)
    else:
        lab, n = _merge(lab0.copy(), n0, D, vol, u, v, beta, r_free, min_volume)
    lab = _exterior_zero(lab, g.boundary & active)
    lab[~active] = -1
    n = int(lab.max()) + 1
    comp = Compartments(label=lab, n=n, dist=D)
    comp.volume = np.bincount(lab[lab >= 0], weights=vol[lab >= 0], minlength=n)
    comp.touches_boundary = _touches_boundary(lab, n, g.boundary)
    # throats between compartments
    la, lb = lab[u], lab[v]
    skip = np.zeros(u.size, bool)
    fc_all, _, _, _ = face_geometry(X, size, u, v)
    for hh in list(explicit) + list(plugs):
        r = 0.5 * float(hh["diameter"]) + 2.0 * h
        skip |= np.linalg.norm(fc_all - np.asarray(hh["center"]), axis=1) <= r
    cross = np.flatnonzero((la != lb) & (la >= 0) & (lb >= 0) & ~skip)
    throats = []
    if cross.size:
        lo = np.minimum(la[cross], lb[cross]).astype(np.int64)
        hi = np.maximum(la[cross], lb[cross]).astype(np.int64)
        keys = lo * (n + 1) + hi
        rad = 1.5 * float(np.sqrt(np.minimum(size[u[cross]], size[v[cross]]) ** 2).max())
        for grp in _groups(fc_all[cross], keys, rad):
            e = cross[grp]
            a, b = int(min(la[e[0]], lb[e[0]])), int(max(la[e[0]], lb[e[0]]))
            a_first = la[e] == a
            ua = np.where(a_first, u[e], v[e])
            vb = np.where(a_first, v[e], u[e])
            throats.append(_throat(a, b, ua, vb, X, size, s))
    # internal throats (a == b)
    for ca, cb in cands:
        kk = lab[ca[0]]
        if np.all(lab[ca] == kk) and np.all(lab[cb] == kk):
            throats.append(_throat(kk, kk, ca, cb, X, size, s))
    # explicit holes
    if explicit:
        forced = hole_throats_graph(X, size, active, lab, u, v, explicit, h)
        used = np.zeros(N, bool)
        for t in forced:
            used[t.cells_a] = used[t.cells_b] = True
        kept = []
        for t in throats:
            near = any(np.linalg.norm(t.centroid - f.centroid) < f.diameter for f in forced)
            if not near and not (used[t.cells_a].any() or used[t.cells_b].any()):
                kept.append(t)
        throats = kept + forced
    comp.throats = throats
    if verbose:
        print(f"graph segmentation: {n} compartments, {len(throats)} throats "
              f"({len(cands)} internal candidates), {time.time()-t0:.0f} s", flush=True)
    return comp, D


def _hole_candidates(lab0, n0, D, X, size, u, v, active, beta, r_free, h, loc_q):
    valid = lab0 >= 0
    peak = np.zeros(max(n0, 1))
    np.maximum.at(peak, lab0[valid], D[valid])
    la, lb = lab0[u], lab0[v]
    cross = np.flatnonzero((la != lb) & (la >= 0) & (lb >= 0))
    out = []
    if cross.size == 0:
        return out
    fc, _, _, _ = face_geometry(X, size, u[cross], v[cross])
    lo = np.minimum(la[cross], lb[cross]).astype(np.int64)
    hi = np.maximum(la[cross], lb[cross]).astype(np.int64)
    keys = lo * (n0 + 1) + hi
    rad = 1.5 * float(np.minimum(size[u[cross]], size[v[cross]]).max())
    for grp in _groups(fc, keys, rad):
        e = cross[grp]
        pa, pb = int(min(la[e[0]], lb[e[0]])), int(max(la[e[0]], lb[e[0]]))
        rg = float(np.max(np.minimum(D[u[e]], D[v[e]])))
        if rg >= beta * min(peak[pa], peak[pb]) or rg >= r_free:
            continue
        a_first = la[e] == pa
        ca = np.where(a_first, u[e], v[e])
        cb = np.where(a_first, v[e], u[e])
        if not _locally_unique(ca, cb, lab0, pa, X, loc_q, rg, h):
            continue
        out.append((ca, cb))
    return out


def hole_throats_graph(X, size, active, lab, u, v, holes, h):
    """Throats of explicitly listed holes: the links crossing the hole's
    mid-plane within its radius (+ half a cell), with the true area and
    diameter (as ``compartments.hole_throats``)."""
    out = []
    for hh in holes:
        c, d, n = np.asarray(hh["center"]), float(hh["diameter"]), np.asarray(hh["axis"])
        R = 0.5 * d + 0.5 * h
        near = (np.linalg.norm(X[u] - c, axis=1) <= R + 2 * h) & active[u] & active[v]
        e = np.flatnonzero(near)
        sa = (X[u[e]] - c) @ n
        sb = (X[v[e]] - c) @ n
        mid = 0.5 * (X[u[e]] + X[v[e]]) - c
        rad = np.linalg.norm(mid - (mid @ n)[:, None] * n, axis=1)
        f = e[((sa < 0) != (sb < 0)) & (rad <= R)]
        if f.size == 0:
            raise ValueError(f"hole at {c} is closed in the grid")
        neg = ((X[u[f]] - c) @ n) < 0
        ca = np.where(neg, u[f], v[f])
        cb = np.where(neg, v[f], u[f])
        la_, lb_ = int(lab[ca[0]]), int(lab[cb[0]])
        swap = la_ > lb_
        fc, nrm, w, _ = face_geometry(X, size, ca, cb)
        t = Throat(a=min(la_, lb_), b=max(la_, lb_), cells_a=cb if swap else ca,
                   cells_b=ca if swap else cb, normals=-nrm if swap else nrm,
                   area=float(np.pi * d * d / 4.0), diameter=d, centroid=c.astype(float),
                   axis=n.astype(float), open_at=float(hh.get("open_at", -np.inf)))
        out.append(t)
    return out


# ------------------------------------------- uniform-grid compatible mode
def _adj_csr(N, a, b):
    """Symmetric CSR of the node pairs (a, b), sorted by source, target."""
    src = np.r_[a, b]
    dst = np.r_[b, a]
    o = np.lexsort((dst, src))
    src, dst = src[o], dst[o]
    ptr = np.zeros(N + 1, np.int64)
    np.cumsum(np.bincount(src, minlength=N), out=ptr[1:])
    return ptr, dst.astype(np.int64)


def _rows(ptr, idx, nodes):
    cnt = ptr[nodes + 1] - ptr[nodes]
    src = np.repeat(nodes, cnt)
    pos = np.repeat(ptr[nodes] - np.r_[0, np.cumsum(cnt)[:-1]], cnt) + np.arange(cnt.sum())
    return src, idx[pos]


def _side_groups(keys, ca, adj, loc, g, dims0):
    """Faces grouped per key (compartment pair), then by the 26-connected
    components of their side-a nodes (as ``compartments._face_groups``):
    face links (``adj``) plus same-level edge and corner neighbours.
    Yields (face indices) per group, groups ordered by key, then by their
    lowest side-a node."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    order = np.argsort(keys, kind="stable")
    ks = keys[order]
    brk = np.r_[0, np.flatnonzero(np.diff(ks)) + 1, len(ks)]
    ptr, idx = adj
    for s0, s1 in zip(brk[:-1], brk[1:]):
        sel = order[s0:s1]
        cells = np.unique(ca[sel])
        loc[cells] = np.arange(cells.size)
        try:
            s, d = _rows(ptr, idx, cells)
            ok = loc[d] >= 0
            a2, b2 = _pairs26(g, dims0, cells)
            G = coo_matrix((np.ones(ok.sum() + a2.size),
                            (np.r_[loc[s[ok]], loc[a2]], np.r_[loc[d[ok]], loc[b2]])),
                           shape=(cells.size,) * 2)
            ng, cl = connected_components(G, directed=False)
            gl = cl[loc[ca[sel]]]
        finally:
            loc[cells] = -1
        # number the groups by their lowest cell (raster order)
        first = np.full(ng, np.iinfo(np.int64).max)
        np.minimum.at(first, cl, cells)
        rank = np.argsort(np.argsort(first))
        gl = rank[gl]
        for gi in range(ng):
            yield sel[gl == gi]


def _whole_throat(a, b, ca, cb, X, size, D, h):
    """Throat of whole-cell faces, as ``compartments._make_throat``: net
    area (at least n / sqrt 3 faces), width 2 r + h/2 with r the largest
    distance to the wall over its faces."""
    d = X[cb] - X[ca]
    ax = np.argmax(np.abs(d), axis=1)
    nrm = np.zeros((len(ca), 3))
    nrm[np.arange(len(ca)), ax] = np.sign(d[np.arange(len(ca)), ax])
    w = np.minimum(size[ca], size[cb]) ** 2
    vs = np.linalg.norm((nrm * w[:, None]).sum(0))
    area = max(vs, w.sum() / np.sqrt(3.0))
    rt = float(np.max(np.minimum(D[ca], D[cb])))
    cen = 0.5 * (X[ca] + X[cb]).mean(0)
    return Throat(a=int(a), b=int(b), cells_a=ca.astype(np.int64), cells_b=cb.astype(np.int64),
                  normals=nrm, area=float(area), diameter=float(2 * rt + 0.5 * h),
                  centroid=cen, weights=None if np.all(w == h * h) else w)


def _locally_unique_box(ca, cb, lab0, pa, X, rg, h, tree, leaves, adj, loc):
    """``compartments._locally_unique`` on the graph: the leaves in the box
    around the opening (m cells beyond it), without the a-side faces'
    cells; are the b side and the rest of region pa connected there?"""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    P = np.r_[X[ca], X[cb]]
    m = max(3, int(np.ceil(2 * rg / h)) + 2)
    lo = P.min(0) - (m + 0.5) * h
    hi = P.max(0) + (m + 0.5) * h
    c = 0.5 * (lo + hi)
    cand = leaves[np.asarray(tree.query_ball_point(c, 0.5 * np.linalg.norm(hi - lo)), np.int64)]
    inb = np.all((X[cand] >= lo) & (X[cand] <= hi), axis=1)
    nodes = np.sort(np.setdiff1d(cand[inb], ca))
    if nodes.size == 0:
        return True
    loc[nodes] = np.arange(nodes.size)
    try:
        ptr, idx = adj
        s, d = _rows(ptr, idx, nodes)
        ok = loc[d] >= 0
        G = coo_matrix((np.ones(ok.sum()), (loc[s[ok]], loc[d[ok]])), shape=(nodes.size,) * 2)
        _, cl = connected_components(G, directed=False)
        cbl = loc[cb]
        lb = set(cl[cbl[cbl >= 0]].tolist())
        la = set(cl[loc[nodes[lab0[nodes] == pa]]].tolist())
    finally:
        loc[nodes] = -1
    return not (la & lb)


def segment_graph_voxel(g, grid, dims0, closed_centres, k=2, beta=0.6, d_free=0.03,
                        min_cells=None, split=True, internal_holes=True, explicit=(),
                        plugs=(), verbose=False):
    """Compartments of an octree node graph by the uniform grid's method,
    step by step (``compartments.segment`` on the whole cells, the fine
    nodes flooded from them, ``compartments.node_throats``). On an octree
    with no coarse levels the result is the uniform grid's.

    g: ``octree.OctGraph`` (plugged links cut); grid: the ``Octree`` (dx,
    face_area, triangles). Returns (Compartments, D) with labels for all
    nodes."""
    import time
    from scipy.spatial import cKDTree
    from .compartments import node_throats
    from .model import _node_wall_distance, _priority_flood
    t0 = time.time()
    X, size, vol, active, fine = g.X, g.size, g.vol, g.active, g.fine
    ptr, idx = g.indptr, g.indices
    N = g.N
    h = float(grid.dx)
    if min_cells is None:
        min_cells = 32
    leaf = active & ~fine
    D = voxel_distance(X, size, active, closed_centres, h, np.inf)
    u, v = links_of(ptr, idx)
    lf = leaf[u] & leaf[v]
    u, v = u[lf], v[lf]                                  # leaf faces only
    lptr, lidx = _adj_csr(N, u, v)
    loc = -np.ones(N, np.int64)
    if not split:
        from .par import label_bodies
        lab, n = label_bodies(active, np.zeros(N, np.int64), ptr, idx)
        lab = _exterior_zero(lab, g.boundary & active)
        comp = Compartments(label=lab, n=int(lab.max()) + 1, dist=D)
        comp.throats = []
        comp.volume = np.bincount(lab[lab >= 0], weights=vol[lab >= 0], minlength=comp.n)
        comp.touches_boundary = _touches_boundary(lab, comp.n, g.boundary)
        return comp, D
    # markers: maxima of D over the 26-neighbourhood, 26-connected. Face
    # links first; edge and corner neighbours only for the face maxima
    nmax = np.full(N, -np.inf)
    np.maximum.at(nmax, u, D[v])
    np.maximum.at(nmax, v, D[u])
    cand = np.flatnonzero(leaf & (D >= nmax))
    a26, b26 = _links26_to(g, dims0, cand)          # from candidates to any leaf
    k26 = leaf[b26]
    a26, b26 = a26[k26], b26[k26]
    np.maximum.at(nmax, a26, D[b26])
    peak = leaf & (D >= nmax)
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    pk = np.flatnonzero(peak)
    a2, b2 = _pairs26(g, dims0, pk)
    pa_ = np.r_[u, a2]
    pb_ = np.r_[v, b2]
    m_ = peak[pa_] & peak[pb_]
    Gm = coo_matrix((np.ones(m_.sum()), (pa_[m_], pb_[m_])), shape=(N, N))
    _, cl = connected_components(Gm, directed=False)
    pk = np.flatnonzero(peak)
    _, first = np.unique(cl[pk], return_index=True)
    rank = -np.ones(int(cl.max()) + 1, np.int64)
    rank[cl[pk[np.sort(first)]]] = np.arange(first.size)
    mk = np.where(peak, rank[cl], -1).astype(np.int64)
    n0 = int(first.size)
    seeds = pk[np.lexsort((pk, -D[pk]))]
    lab0 = _flood_fifo(mk, D, lptr, lidx, leaf.astype(np.bool_), seeds.astype(np.int64))
    r_free = np.inf if d_free is None else 0.5 * d_free
    tree = cKDTree(X[np.flatnonzero(leaf)])
    leaves = np.flatnonzero(leaf)
    cands = []
    if internal_holes:
        peakr = np.zeros(max(n0, 1))
        valid = lab0 >= 0
        np.maximum.at(peakr, lab0[valid], D[valid])
        la, lb = lab0[u], lab0[v]
        cross = np.flatnonzero((la != lb) & (la >= 0) & (lb >= 0))
        if cross.size:
            lo = np.minimum(la[cross], lb[cross]).astype(np.int64)
            hi = np.maximum(la[cross], lb[cross]).astype(np.int64)
            ca_all = np.where(la[cross] == lo, u[cross], v[cross])
            cb_all = np.where(la[cross] == lo, v[cross], u[cross])
            for grp in _side_groups(lo * (n0 + 1) + hi, ca_all, (lptr, lidx), loc, g, dims0):
                ca, cb = ca_all[grp], cb_all[grp]
                pa = int(lo[grp[0]])
                pb = int(hi[grp[0]])
                rg = float(np.max(np.minimum(D[ca], D[cb])))
                if rg >= beta * min(peakr[pa], peakr[pb]) or rg >= r_free:
                    continue
                if not _locally_unique_box(ca, cb, lab0, pa, X, rg, h, tree, leaves,
                                           (lptr, lidx), loc):
                    continue
                cands.append((ca, cb))
    lab, n = _merge(lab0.copy(), n0, D, np.where(fine, 0.0, size ** 3), u, v, beta, r_free,
                    min_cells * h ** 3)                    # sizes in whole cells
    lab = _exterior_zero(lab, g.boundary & leaf)
    lab[~leaf] = -1
    n = int(lab.max()) + 1
    comp = Compartments(label=lab, n=n, dist=D)
    # whole-cell throats
    la, lb = lab[u], lab[v]
    cross = np.flatnonzero((la != lb) & (la >= 0) & (lb >= 0))
    throats = []
    if cross.size:
        lo = np.minimum(la[cross], lb[cross]).astype(np.int64)
        hi = np.maximum(la[cross], lb[cross]).astype(np.int64)
        ca_all = np.where(la[cross] == lo, u[cross], v[cross])
        cb_all = np.where(la[cross] == lo, v[cross], u[cross])
        for grp in _side_groups(lo * (n + 1) + hi, ca_all, (lptr, lidx), loc, g, dims0):
            throats.append(_whole_throat(lo[grp[0]], hi[grp[0]], ca_all[grp], cb_all[grp],
                                         X, size, D, h))
    for ca, cb in cands:
        kk = lab[ca[0]]
        if np.all(lab[ca] == kk) and np.all(lab[cb] == kk):
            throats.append(_whole_throat(kk, kk, ca, cb, X, size, D, h))
    if explicit:
        forced = hole_throats_graph(X, size, leaf, lab, u, v, explicit, h)
        used = np.zeros(N, bool)
        for t in forced:
            used[t.cells_a] = used[t.cells_b] = True
        kept = []
        for t in throats:
            near = any(np.linalg.norm(t.centroid - f.centroid) < f.diameter for f in forced)
            if not near and not (used[t.cells_a].any() or used[t.cells_b].any()):
                kept.append(t)
        throats = kept + forced
    comp.throats = throats
    # fine nodes: flooded from the leaves, far from the wall first
    if fine.any():
        kmax = max(int(k), int(round(h / float(size[fine].min()))))
        pri = _node_wall_distance(grid, X, fine, kmax)
        pri[~fine] = D[~fine]
        lab = _priority_flood(lab.astype(np.int64), pri, ptr, idx, active.astype(np.bool_))
        lab = node_throats(grid, comp, lab, X, fine, (ptr, idx), k, explicit=explicit,
                           plugs=plugs, split=split, node_size=size)
    lab[~active] = -1
    comp.label = lab
    comp.n = int(lab.max()) + 1
    comp.volume = np.bincount(lab[lab >= 0], weights=vol[lab >= 0], minlength=comp.n)
    comp.touches_boundary = _touches_boundary(lab, comp.n, g.boundary)
    if verbose:
        print(f"graph segmentation (voxel): {comp.n} compartments, {len(comp.throats)} "
              f"throats ({len(cands)} internal), {time.time()-t0:.0f} s", flush=True)
    return comp, D
