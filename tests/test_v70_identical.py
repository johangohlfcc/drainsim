"""drainsim 7.0: the performance changes give the same numbers as 6.4.
Run: python -m pytest -q tests/test_v70_identical.py"""
import numpy as np
import pytest

from drainsim.grid import sample_triangles, surface_distance


def _single_tree(tri, h, Q, cap):
    """The 6.4 computation: one k-d tree over all samples."""
    from scipy.spatial import cKDTree
    P = np.concatenate(list(sample_triangles(tri, h, spacing=1.0)))
    d, _ = cKDTree(P).query(Q, distance_upper_bound=cap)
    return np.minimum(d, cap)


def _slivers(rng, n=4000):
    """Long thin triangles (as in CAD meshes) and a few large ones."""
    a = rng.random((n, 3)) * [0.4, 0.3, 0.2]
    d = rng.normal(size=(n, 3))
    d /= np.linalg.norm(d, axis=1)[:, None]
    L = rng.uniform(0.005, 0.08, n)[:, None]
    w = rng.normal(size=(n, 3)) * 0.0004
    tri = np.stack([a, a + L * d, a + 0.5 * L * d + w], 1)
    big = rng.random((20, 3, 3)) * [0.4, 0.3, 0.2]
    return np.concatenate([tri, big])


def test_lattice_kernel_gives_the_sample_triangles_points():
    """The numba lattice (surface_distance) reproduces sample_triangles bit
    for bit (same set of points)."""
    from drainsim.grid import _lattice_in_box
    rng = np.random.default_rng(5)
    tri = _slivers(rng, 3000)
    h = 0.002
    ref = np.concatenate(list(sample_triangles(tri, h, spacing=1.0)))
    emax = np.max(np.linalg.norm(tri[:, [1, 2, 0]] - tri, axis=2), axis=1)
    n = np.maximum(1, np.ceil(emax / h)).astype(np.int64)
    lo, hi = np.full(3, -np.inf), np.full(3, np.inf)
    sel = np.arange(len(tri))
    m = _lattice_in_box(tri, n, sel, lo, hi, np.zeros((0, 3)))
    got = np.empty((m, 3))
    _lattice_in_box(tri, n, sel, lo, hi, got)
    key = lambda P: np.sort(P.view(np.dtype((np.void, 24))).ravel())
    assert m == len(ref) and np.array_equal(key(got), key(ref))


@pytest.mark.parametrize("slab_points, workers", [(10**9, 1), (20_000, 1), (5_000, 4)])
def test_surface_distance_equals_one_tree(slab_points, workers):
    rng = np.random.default_rng(3)
    tri = _slivers(rng)
    h, cap = 0.002, 0.006
    Q = rng.random((30_000, 3)) * [0.44, 0.34, 0.24] - 0.02      # near and far from the surface
    ref = _single_tree(tri, h, Q, cap)
    got = surface_distance(tri, h, Q, cap, slab_points=slab_points, workers=workers)
    assert (ref < cap).sum() > 1000 and (ref == cap).sum() > 1000
    assert np.array_equal(got, ref)
    # a memory budget smaller than one slab: the slabs run one at a time
    tight = surface_distance(tri, h, Q, cap, slab_points=slab_points, workers=workers,
                             max_points=1000)
    assert np.array_equal(tight, ref)
