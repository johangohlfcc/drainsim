"""Geometric share of the surface sub-cells (drainsim 6.3).

A closed ("surface") sub-cell holds a piece of the wall. Before 6.3 its
volume was shared equally among its fluid face neighbours: half to each
side of a sheet. That counts the metal of the sheet as fluid: for a panel of
thickness t, each side gains t/2 over its whole wetted area (the article
door's pan holds 0.07-0.1 l too much, about 1 %).

Here each surface sub-cell is sampled with m^3 points (cell midpoints). A
point belongs to the fluid neighbour it can see: the straight segment from
the point to that neighbour (to the nearest point of the neighbour's box,
just inside it) crosses no triangle; of several visible neighbours, the
nearest. If it sees none, the neighbour with the fewest crossings: an even
number means the point is fluid whose own side has no fluid face neighbour
here (it goes to that neighbour, as before 6.3), an odd number to every
neighbour means it lies in the metal between the two faces of a plate, and
it is dropped. This is agent A's cut-cell sampling (``dipsim.cutcells``)
applied to the sub-cells.

Used by ``octree._subcell_graph`` and ``volfrac.cut_cell_graph``.
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange

from .voxel import tri_box_overlap


@njit(cache=True, inline="always")
def _seg_hits_tri(px, py, pz, qx, qy, qz, t):
    """Does the segment p-q cross triangle t (3, 3)? (Moller-Trumbore with a
    small tolerance, so that a segment ending on a face counts as crossing.)"""
    eps = 1e-14
    d0, d1, d2 = qx - px, qy - py, qz - pz
    e1x, e1y, e1z = t[1, 0] - t[0, 0], t[1, 1] - t[0, 1], t[1, 2] - t[0, 2]
    e2x, e2y, e2z = t[2, 0] - t[0, 0], t[2, 1] - t[0, 1], t[2, 2] - t[0, 2]
    hx = d1 * e2z - d2 * e2y
    hy = d2 * e2x - d0 * e2z
    hz = d0 * e2y - d1 * e2x
    a = e1x * hx + e1y * hy + e1z * hz
    if abs(a) < eps:
        return False
    f = 1.0 / a
    sx, sy, sz = px - t[0, 0], py - t[0, 1], pz - t[0, 2]
    u = f * (sx * hx + sy * hy + sz * hz)
    if u < -1e-9 or u > 1.0 + 1e-9:
        return False
    qx_ = sy * e1z - sz * e1y
    qy_ = sz * e1x - sx * e1z
    qz_ = sx * e1y - sy * e1x
    v = f * (d0 * qx_ + d1 * qy_ + d2 * qz_)
    if v < -1e-9 or u + v > 1.0 + 1e-9:
        return False
    tt = f * (e2x * qx_ + e2y * qy_ + e2z * qz_)
    return -1e-9 <= tt <= 1.0 + 1e-9


# ------------------------------------------------ triangles per closed cell
@njit(cache=True)
def _bsearch(a, x):
    lo, hi = 0, a.shape[0]
    while lo < hi:
        mid = (lo + hi) >> 1
        if a[mid] < x:
            lo = mid + 1
        else:
            hi = mid
    return lo if lo < a.shape[0] and a[lo] == x else -1


@njit(cache=True, parallel=True)
def _pairs(tri, origin, dx, dims, keys, cnt, start, cell_out, tri_out, fill):
    h = 0.5 * dx * (1.0 + 1e-6)
    ny, nz = dims[1], dims[2]
    for t in prange(tri.shape[0]):
        lo = np.empty(3, np.int64)
        hi = np.empty(3, np.int64)
        ok = True
        for a in range(3):
            mn = min(tri[t, 0, a], min(tri[t, 1, a], tri[t, 2, a]))
            mx = max(tri[t, 0, a], max(tri[t, 1, a], tri[t, 2, a]))
            lo[a] = max(int(np.floor((mn - origin[a]) / dx - 1e-6)), 0)
            hi[a] = min(int(np.floor((mx - origin[a]) / dx + 1e-6)), dims[a] - 1)
            if lo[a] > hi[a]:
                ok = False
        n = 0
        w = start[t] if fill else 0
        if ok:
            c = np.empty(3)
            for i in range(lo[0], hi[0] + 1):
                c[0] = origin[0] + (i + 0.5) * dx
                for j in range(lo[1], hi[1] + 1):
                    c[1] = origin[1] + (j + 0.5) * dx
                    for k in range(lo[2], hi[2] + 1):
                        ci = _bsearch(keys, (i * ny + j) * nz + k)
                        if ci < 0:
                            continue
                        c[2] = origin[2] + (k + 0.5) * dx
                        if tri_box_overlap(c, h, tri[t, 0], tri[t, 1], tri[t, 2]):
                            if fill:
                                cell_out[w] = ci
                                tri_out[w] = t
                                w += 1
                            n += 1
        if not fill:
            cnt[t] = n


def triangles_per_cell(tri, origin, dx, dims, keys):
    """For the cells with sorted flat keys ``keys``: CSR (ptr, idx) of the
    triangles touching each cell (box enlarged by 1e-6)."""
    tri = np.ascontiguousarray(tri, np.float64)
    origin = np.asarray(origin, np.float64)
    dims = np.asarray(dims, np.int64)
    keys = np.ascontiguousarray(keys, np.int64)
    cnt = np.zeros(len(tri), np.int64)
    z = np.zeros(1, np.int64)
    _pairs(tri, origin, float(dx), dims, keys, cnt, z, z, z, False)
    start = np.zeros(len(tri) + 1, np.int64)
    np.cumsum(cnt, out=start[1:])
    cell = np.empty(int(start[-1]), np.int64)
    tid = np.empty(int(start[-1]), np.int64)
    _pairs(tri, origin, float(dx), dims, keys, cnt, start, cell, tid, True)
    o = np.argsort(cell, kind="stable")
    ptr = np.zeros(len(keys) + 1, np.int64)
    np.cumsum(np.bincount(cell, minlength=len(keys)), out=ptr[1:])
    return ptr, tid[o]


# ---------------------------------------------------------- the split
@njit(cache=True, parallel=True)
def _visible(lo, s, host, iptr, ivid, nc, ns, tri, tptr, tidx, m, best):
    """best[q, sample]: index (into item q's vote list) of the vote the
    sample belongs to (fewest crossings, then nearest); -1 = in the metal
    (odd crossings to every vote), -2 = no triangles (equal split)."""
    for q in prange(lo.shape[0]):
        v0 = iptr[q]
        nv = iptr[q + 1] - v0
        c = host[q]
        t0, t1 = tptr[c], tptr[c + 1]
        if t1 == t0:
            for a in range(m * m * m):
                best[q, a] = -2
            continue
        sq = s[q]
        a = 0
        for ia in range(m):
            px = lo[q, 0] + (ia + 0.5) / m * sq
            for ib in range(m):
                py = lo[q, 1] + (ib + 0.5) / m * sq
                for ic in range(m):
                    pz = lo[q, 2] + (ic + 0.5) / m * sq
                    bi = -1
                    bx = 1 << 30
                    bd = np.inf
                    for r in range(nv):
                        nid = ivid[v0 + r]
                        hs = 0.5 * ns[nid] * (1.0 - 1e-6)
                        qx = min(max(px, nc[nid, 0] - hs), nc[nid, 0] + hs)
                        qy = min(max(py, nc[nid, 1] - hs), nc[nid, 1] + hs)
                        qz = min(max(pz, nc[nid, 2] - hs), nc[nid, 2] + hs)
                        d = (qx - px) ** 2 + (qy - py) ** 2 + (qz - pz) ** 2
                        nx_ = 0
                        for x in range(t0, t1):
                            if _seg_hits_tri(px, py, pz, qx, qy, qz, tri[tidx[x]]):
                                nx_ += 1
                                if nx_ > bx:
                                    break
                        if nx_ < bx or (nx_ == bx and d < bd):
                            bi = r
                            bx = nx_
                            bd = d
                    best[q, a] = bi if bx % 2 == 0 else -1
                    a += 1


@njit(cache=True)
def _accumulate(iptr, ivid, best, own, credit, dropped):
    m3 = best.shape[1]
    for q in range(best.shape[0]):
        v0 = iptr[q]
        nv = iptr[q + 1] - v0
        w = own[q] / m3
        for a in range(m3):
            b = best[q, a]
            if b >= 0:
                credit[ivid[v0 + b]] += w
            elif b == -2:
                for r in range(nv):
                    credit[ivid[v0 + r]] += w / nv
            else:
                dropped[0] += w


def surface_items(cnt, vptr, vid, fine_solid, host_of, deep=True):
    """The surface sub-cells to share and their candidate nodes.

    Sub-cells with votes (fluid face neighbours) use those. With ``deep``,
    a closed sub-cell without any (inside a wall two sub-cells thick, where
    a sheet crosses a sub-cell corner) uses the votes of all sub-cells of
    its closed cell: the fluid it holds goes to the node it sees.
    Returns (items j, their host cell, iptr, ivid)."""
    voted = np.flatnonzero(cnt > 0)
    parts_j = [voted]
    ptr_parts = [cnt[voted]]
    vid_parts = [vid]
    if deep:
        un = np.flatnonzero(fine_solid & (cnt == 0))
        if un.size and voted.size:
            hv = np.repeat(host_of(voted), cnt[voted])
            pair = np.unique(np.stack([hv, vid], 1), axis=0)
            nh = int(max(hv.max(), host_of(un).max())) + 1
            uptr = np.zeros(nh + 1, np.int64)
            np.cumsum(np.bincount(pair[:, 0], minlength=nh), out=uptr[1:])
            hu = host_of(un)
            nu = uptr[hu + 1] - uptr[hu]
            un, hu, nu = un[nu > 0], hu[nu > 0], nu[nu > 0]
            if un.size:
                st = np.repeat(uptr[hu], nu) + (np.arange(nu.sum()) - np.repeat(np.cumsum(nu) - nu, nu))
                parts_j.append(un)
                ptr_parts.append(nu)
                vid_parts.append(pair[st, 1])
    items = np.concatenate(parts_j)
    iptr = np.zeros(items.size + 1, np.int64)
    np.cumsum(np.concatenate(ptr_parts), out=iptr[1:])
    return items, host_of(items), iptr, np.concatenate(vid_parts).astype(np.int64)


def split_by_sight(iptr, ivid, lo, s, host, own, nc, ns, tri, tptr, tidx, credit,
                   m=2, chunk=1_000_000):
    """Credit each surface item q (lower corner lo[q], edge s[q], closed cell
    host[q] into (tptr, tidx), volume own[q] in credit units, candidate
    nodes ivid[iptr[q]:iptr[q+1]]) to the nodes its m^3 sample points
    belong to (see the module docstring). nc, ns: centre and edge of every
    node. Returns the credit units dropped as metal."""
    dropped = np.zeros(1)
    tri = np.ascontiguousarray(tri, np.float64)
    n = lo.shape[0]
    for a in range(0, n, chunk):
        b = min(a + chunk, n)
        best = np.empty((b - a, m ** 3), np.int16)
        ip = iptr[a:b + 1] - iptr[a]
        iv = ivid[iptr[a]:iptr[b]]
        _visible(np.ascontiguousarray(lo[a:b]), np.ascontiguousarray(s[a:b]),
                 np.ascontiguousarray(host[a:b]), ip, iv, nc, ns, tri, tptr, tidx, int(m), best)
        _accumulate(ip, iv, best, np.ascontiguousarray(own[a:b]), credit, dropped)
    return float(dropped[0])
