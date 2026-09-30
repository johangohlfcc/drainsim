"""Exact (conservative) surface voxelisation: every cell whose closed box
touches a triangle.

The sampling approach used before (``grid.sample_triangles``: points on each
triangle at 0.45 dx spacing) marks a wall of cells that is leak-free for
face-connected fluid, but it can miss a cell that a sheet only grazes (a
corner or an edge of the cell). Such a cell stays a whole fluid cell: its
volume on the far side of the sheet is counted as fluid, and when it borders
the sub-cells of a cut cell it can link fluid on both sides of the sheet.
Here the separating-axis test of Akenine-Möller (triangle against box: the
3 box axes, the triangle normal and the 9 edge-axis cross products) decides
per cell, with the box enlarged by a relative 1e-6 so that a triangle lying
exactly on a cell face marks the cells on both sides.

Returns flat cell keys ``(i * ny + j) * nz + k`` (the octree's ``_key``
ordering), sorted and unique.

``cut_cells`` is what the grids use: the cells whose *interior* a triangle
enters (box shrunk by 1e-9), united with the point-sampled cells. A sheet
lying exactly on a cell face touches the closed boxes on both sides but the
interior of neither; the samples put it in the cell on the + side (floor),
as before, so it stays a one-cell wall instead of a two-cell one (a
two-cell wall would lose the fluid of corner cells, whose sub-cells are only
made next to a fluid cell).
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange


@njit(cache=True, inline="always")
def _sep(a0, a1, a2, v0, v1, v2, h):
    """True if axis a separates the triangle (v, relative to the box centre)
    from the cube of half size h."""
    q0 = a0 * v0[0] + a1 * v0[1] + a2 * v0[2]
    q1 = a0 * v1[0] + a1 * v1[1] + a2 * v1[2]
    q2 = a0 * v2[0] + a1 * v2[1] + a2 * v2[2]
    mn = min(q0, min(q1, q2))
    mx = max(q0, max(q1, q2))
    r = h * (abs(a0) + abs(a1) + abs(a2))
    return mn > r or mx < -r


@njit(cache=True)
def tri_box_overlap(c, h, t0, t1, t2):
    """Does triangle (t0, t1, t2) touch the cube of centre c and half size h?"""
    v0 = np.empty(3)
    v1 = np.empty(3)
    v2 = np.empty(3)
    for a in range(3):
        v0[a] = t0[a] - c[a]
        v1[a] = t1[a] - c[a]
        v2[a] = t2[a] - c[a]
    for a in range(3):                                   # box face normals
        if min(v0[a], min(v1[a], v2[a])) > h or max(v0[a], max(v1[a], v2[a])) < -h:
            return False
    e0x, e0y, e0z = v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]
    e1x, e1y, e1z = v2[0] - v1[0], v2[1] - v1[1], v2[2] - v1[2]
    e2x, e2y, e2z = v0[0] - v2[0], v0[1] - v2[1], v0[2] - v2[2]
    nx = e0y * e1z - e0z * e1y                           # triangle plane
    ny = e0z * e1x - e0x * e1z
    nz = e0x * e1y - e0y * e1x
    d = nx * v0[0] + ny * v0[1] + nz * v0[2]
    if abs(d) > h * (abs(nx) + abs(ny) + abs(nz)):
        return False
    # edge x box axis: x cross e = (0, -ez, ey), y cross e = (ez, 0, -ex),
    # z cross e = (-ey, ex, 0)
    if _sep(0.0, -e0z, e0y, v0, v1, v2, h) or _sep(e0z, 0.0, -e0x, v0, v1, v2, h) \
            or _sep(-e0y, e0x, 0.0, v0, v1, v2, h):
        return False
    if _sep(0.0, -e1z, e1y, v0, v1, v2, h) or _sep(e1z, 0.0, -e1x, v0, v1, v2, h) \
            or _sep(-e1y, e1x, 0.0, v0, v1, v2, h):
        return False
    if _sep(0.0, -e2z, e2y, v0, v1, v2, h) or _sep(e2z, 0.0, -e2x, v0, v1, v2, h) \
            or _sep(-e2y, e2x, 0.0, v0, v1, v2, h):
        return False
    return True


@njit(cache=True)
def _range(t, origin, dx, dims, lo, hi):
    ok = True
    for a in range(3):
        mn = min(t[0, a], min(t[1, a], t[2, a]))
        mx = max(t[0, a], max(t[1, a], t[2, a]))
        lo[a] = max(int(np.floor((mn - origin[a]) / dx - 1e-6)), 0)
        hi[a] = min(int(np.floor((mx - origin[a]) / dx + 1e-6)), dims[a] - 1)
        if lo[a] > hi[a]:
            ok = False
    return ok


@njit(cache=True, parallel=True)
def _mark_tri(tri, origin, dx, dims, grow, mask):
    """mask[key] = 1 for every cell whose box (enlarged by grow) touches a
    triangle. Several threads may write the same 1: the result does not
    depend on the order."""
    ny, nz = dims[1], dims[2]
    h = 0.5 * dx * (1.0 + grow)
    for t in prange(tri.shape[0]):
        lo = np.empty(3, np.int64)
        hi = np.empty(3, np.int64)
        if _range(tri[t], origin, dx, dims, lo, hi):
            c = np.empty(3)
            for i in range(lo[0], hi[0] + 1):
                c[0] = origin[0] + (i + 0.5) * dx
                for j in range(lo[1], hi[1] + 1):
                    c[1] = origin[1] + (j + 0.5) * dx
                    for k in range(lo[2], hi[2] + 1):
                        c[2] = origin[2] + (k + 0.5) * dx
                        if tri_box_overlap(c, h, tri[t, 0], tri[t, 1], tri[t, 2]):
                            mask[(i * ny + j) * nz + k] = 1


@njit(cache=True, parallel=True)
def _mark_samples(tri, nsub, origin, dx, dims, mask):
    """mask[key] = 1 for the cell of every sample_triangles point (the same
    lattice and arithmetic), in parallel over the triangles."""
    for t in prange(tri.shape[0]):
        k = nsub[t]
        for i in range(k + 1):
            w1 = i / k
            for j in range(k + 1 - i):
                w2 = j / k
                w0 = 1.0 - w1 - w2
                a = np.int64(np.floor(((w0 * tri[t, 0, 0] + w1 * tri[t, 1, 0]
                                        + w2 * tri[t, 2, 0]) - origin[0]) / dx))
                b = np.int64(np.floor(((w0 * tri[t, 0, 1] + w1 * tri[t, 1, 1]
                                        + w2 * tri[t, 2, 1]) - origin[1]) / dx))
                c = np.int64(np.floor(((w0 * tri[t, 0, 2] + w1 * tri[t, 1, 2]
                                        + w2 * tri[t, 2, 2]) - origin[2]) / dx))
                if 0 <= a < dims[0] and 0 <= b < dims[1] and 0 <= c < dims[2]:
                    mask[(a * dims[1] + b) * dims[2] + c] = 1


def surface_cells(tri, origin, dx, dims, grow=1e-6, block=2_000_000):
    """Sorted unique flat keys of the cells (grid ``origin``, spacing ``dx``,
    ``dims`` cells) whose box, enlarged by the relative ``grow``, touches any
    of the triangles ``tri`` (T, 3, 3). ``grow < 0`` shrinks the box: only
    cells whose interior the surface enters. The cells are marked in a byte
    mask over the grid (in parallel) and read out in order: sorted and unique
    without sorting. (``block`` is kept for compatibility.)"""
    from .par import compact
    tri = np.ascontiguousarray(tri, dtype=np.float64)
    origin = np.asarray(origin, np.float64)
    dims = np.asarray(dims, np.int64)
    mask = np.zeros(int(np.prod(dims)), np.uint8)
    if len(tri):
        _mark_tri(tri, origin, float(dx), dims, float(grow), mask)
    return compact(mask)


def cut_cells(tri, origin, dx, dims, spacing=0.45):
    """Cells whose interior the surface enters, plus the point-sampled cells
    (see the module docstring). Sorted unique flat keys. Both sets are marked
    in one byte mask (the samples with the lattice and arithmetic of
    ``grid.sample_triangles``) and read out in order."""
    from .par import compact
    tri = np.ascontiguousarray(tri, dtype=np.float64)
    origin = np.asarray(origin, np.float64)
    dims = np.asarray(dims, np.int64)
    mask = np.zeros(int(np.prod(dims)), np.uint8)
    if len(tri):
        _mark_tri(tri, origin, float(dx), dims, -1e-9, mask)
        emax = np.max(np.linalg.norm(tri[:, [1, 2, 0]] - tri, axis=2), axis=1)
        nsub = np.maximum(1, np.ceil(emax / (spacing * dx))).astype(np.int64)
        _mark_samples(tri, nsub, origin, float(dx), dims, mask)
    return compact(mask)
