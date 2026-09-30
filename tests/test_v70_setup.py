"""drainsim 7.0: the setup kernels give the same cells as 6.4 (the 6.4 code is
kept here as the reference). Run: python -m pytest -q tests/test_v70_setup.py"""
import numpy as np
import pytest
from numba import njit, prange

from drainsim.grid import sample_triangles
from drainsim.octree import (Octree, _coords, _dilate, _isin_sorted, _key, _triangles_near,
                             _unique_chunks)
from drainsim.voxel import _range, cut_cells, surface_cells, tri_box_overlap


# ------------------------------------------------------------ 6.4 reference
@njit(cache=True, parallel=True)
def _count64(tri, origin, dx, dims, cnt, grow):
    h = 0.5 * dx * (1.0 + grow)
    for t in prange(tri.shape[0]):
        lo = np.empty(3, np.int64)
        hi = np.empty(3, np.int64)
        n = 0
        if _range(tri[t], origin, dx, dims, lo, hi):
            c = np.empty(3)
            for i in range(lo[0], hi[0] + 1):
                c[0] = origin[0] + (i + 0.5) * dx
                for j in range(lo[1], hi[1] + 1):
                    c[1] = origin[1] + (j + 0.5) * dx
                    for k in range(lo[2], hi[2] + 1):
                        c[2] = origin[2] + (k + 0.5) * dx
                        if tri_box_overlap(c, h, tri[t, 0], tri[t, 1], tri[t, 2]):
                            n += 1
        cnt[t] = n


@njit(cache=True, parallel=True)
def _fill64(tri, origin, dx, dims, start, out, grow):
    h = 0.5 * dx * (1.0 + grow)
    ny, nz = dims[1], dims[2]
    for t in prange(tri.shape[0]):
        lo = np.empty(3, np.int64)
        hi = np.empty(3, np.int64)
        n = start[t]
        if _range(tri[t], origin, dx, dims, lo, hi):
            c = np.empty(3)
            for i in range(lo[0], hi[0] + 1):
                c[0] = origin[0] + (i + 0.5) * dx
                for j in range(lo[1], hi[1] + 1):
                    c[1] = origin[1] + (j + 0.5) * dx
                    for k in range(lo[2], hi[2] + 1):
                        c[2] = origin[2] + (k + 0.5) * dx
                        if tri_box_overlap(c, h, tri[t, 0], tri[t, 1], tri[t, 2]):
                            out[n] = (i * ny + j) * nz + k
                            n += 1


def surface_cells64(tri, origin, dx, dims, grow=1e-6):
    tri = np.ascontiguousarray(tri, dtype=np.float64)
    cnt = np.zeros(len(tri), np.int64)
    _count64(tri, origin, float(dx), dims, cnt, float(grow))
    start = np.zeros(len(tri) + 1, np.int64)
    np.cumsum(cnt, out=start[1:])
    out = np.empty(int(start[-1]), np.int64)
    _fill64(tri, origin, float(dx), dims, start, out, float(grow))
    return np.unique(out)


def cut_cells64(tri, origin, dx, dims, spacing=0.45):
    acc = [surface_cells64(tri, origin, dx, dims, grow=-1e-9)]
    for P in sample_triangles(np.asarray(tri, float), dx, spacing):
        idx = np.floor((P - origin) / dx).astype(np.int64)
        ok = np.all((idx >= 0) & (idx < dims), axis=1)
        idx = idx[ok]
        acc.append(np.unique((idx[:, 0] * dims[1] + idx[:, 1]) * dims[2] + idx[:, 2]))
    return np.unique(np.concatenate(acc))


def dilate64(keys, dims, r=1, chunk=2_000_000):
    offs = np.array([(a, b, c) for a in range(-r, r + 1) for b in range(-r, r + 1)
                     for c in range(-r, r + 1)], np.int64)

    def gen():
        for s0 in range(0, keys.size, chunk):
            C = _coords(keys[s0:s0 + chunk], dims)
            for o in offs:
                Q = C + o
                ok = np.all((Q >= 0) & (Q < dims), axis=1)
                yield _key(Q[ok], dims)
    return _unique_chunks(gen())


def triangles_near64(ot, S, mask):
    tri = ot.triangles
    d0 = ot.dims(0)
    lo = np.floor((tri.min(1) - ot.origin) / ot.h).astype(np.int64) - 1
    hi = np.floor((tri.max(1) - ot.origin) / ot.h).astype(np.int64) + 1
    sel = np.zeros(len(tri), bool)
    small = np.all(hi - lo <= 4, axis=1)
    ext = (hi - lo + 1)[small].max(0) if small.any() else np.zeros(3, np.int64)
    idx = np.flatnonzero(small)
    Ssel = S[mask]
    for a in range(int(ext[0])):
        for b in range(int(ext[1])):
            for c in range(int(ext[2])):
                C = lo[idx] + np.array([a, b, c])
                inb = np.all(C <= hi[idx], axis=1) & np.all((C >= 0) & (C < d0), axis=1)
                f, _ = _isin_sorted(_key(np.where(inb[:, None], C, 0), d0), Ssel)
                sel[idx[inb & f]] = True
    sel[~small] = True
    return tri[sel]


# ------------------------------------------------------------------ tests
def _mesh_triangles(rng):
    trimesh = pytest.importorskip("trimesh")
    parts = [trimesh.creation.icosphere(subdivisions=3, radius=0.07),
             trimesh.creation.box(extents=(0.12, 0.05, 0.004)),
             trimesh.creation.cylinder(radius=0.02, height=0.09, sections=40)]
    tri = [np.asarray(p.triangles, float) + rng.normal(size=3) * 0.02 for p in parts]
    # long thin triangles too
    a = rng.random((400, 3)) * 0.15 - 0.075
    d = rng.normal(size=(400, 3))
    d /= np.linalg.norm(d, axis=1)[:, None]
    thin = np.stack([a, a + 0.05 * d, a + 0.025 * d + rng.normal(size=(400, 3)) * 3e-4], 1)
    return np.concatenate(tri + [thin])


@pytest.mark.parametrize("dx", [0.004, 0.0063])
def test_surface_and_cut_cells_equal_6_4(dx):
    rng = np.random.default_rng(31)
    tri = _mesh_triangles(rng)
    origin = np.array([-0.13, -0.12, -0.11])
    dims = np.array([int(0.26 / dx), int(0.24 / dx), int(0.22 / dx)], np.int64)
    for grow in (1e-6, -1e-9, 0.3):
        assert np.array_equal(surface_cells(tri, origin, dx, dims, grow=grow),
                              surface_cells64(tri, origin, dx, dims, grow=grow))
    new = cut_cells(tri, origin, dx, dims)
    assert new.size > 1000 and np.array_equal(new, cut_cells64(tri, origin, dx, dims))


def test_dilate_equals_6_4():
    rng = np.random.default_rng(32)
    for dims in (np.array([37, 23, 41]), np.array([64, 64, 8])):
        n = int(np.prod(dims))
        keys = np.unique(rng.integers(0, n, n // 20))
        keys = np.concatenate([keys, [0, n - 1]])                # the corners too
        keys = np.unique(keys)
        for r in (1, 2):
            assert np.array_equal(_dilate(keys, dims, r), dilate64(keys, dims, r, chunk=997))


def test_triangles_near_equals_6_4():
    trimesh = pytest.importorskip("trimesh")
    m = trimesh.creation.box(extents=(0.1, 0.08, 0.06))
    m = trimesh.Trimesh(m.vertices, m.faces[m.face_normals[:, 2] < 0.5], process=False)
    m = m.subdivide().subdivide()
    lo = np.array([-0.08, -0.07, -0.05])
    ot = Octree.from_mesh(m, 0.004, levels=1, bounds=(lo, lo + np.array([40, 36, 32]) * 0.004))
    rng = np.random.default_rng(33)
    for frac in (0.02, 0.3, 1.0):
        mask = rng.random(ot.cut.size) < frac
        a = _triangles_near(ot, ot.cut, mask)
        b = triangles_near64(ot, ot.cut, mask)
        assert np.array_equal(a, b)
