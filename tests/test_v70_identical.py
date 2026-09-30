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


def _random_sweep(rng, n=20000, deg=6):
    """A sweep problem like the film's: edges mostly downhill in the order,
    some uphill, some submerged elements and zero coefficients."""
    src = np.repeat(np.arange(n), deg)
    step = rng.integers(1, 400, src.size)
    dst = np.where(rng.random(src.size) < 0.15, src - step, src + step)   # 15 % uphill
    ok = (dst >= 0) & (dst < n) & (dst != src)
    src, dst = src[ok], dst[ok]
    o = np.argsort(src, kind="stable")
    src, dst = src[o], dst[o]
    indptr = np.r_[0, np.cumsum(np.bincount(src, minlength=n))].astype(np.int64)
    w = np.where(rng.random(dst.size) < 0.2, 0.0, rng.random(dst.size) * 1e-3)
    C = np.zeros(n)
    np.add.at(C, src, w)
    A = rng.uniform(1e-6, 4e-5, n)
    V = np.where(rng.random(n) < 0.3, 0.0, rng.random(n) * 1e-9)
    sub = rng.random(n) < 0.05
    return A, V, C, indptr, dst.astype(np.int64), w, sub


def test_film_sweep_by_levels_equals_the_serial_sweep():
    from drainsim.film import _sweep_levels, _sweep_plan, _sweep_sorted
    rng = np.random.default_rng(11)
    for superset in (False, True, False):
        A, V, C, indptr, recv, w, sub = _random_sweep(rng)
        # the plan may cover more edges than carry flow (FilmModel builds it on
        # every edge that can carry flow in the pose): here half the zero ones
        wp = np.where((w == 0.0) & (rng.random(w.size) < 0.5), 1.0, w) if superset else w
        lptr, lel, nin, isrc, iq, up_e, up_r, up_q = _sweep_plan(indptr, recv, wp)
        # the arrays laid out level by level, as FilmModel.update does
        A_l, C_l, sub_l = A[lel], C[lel], sub[lel]
        iw, iC, uw = w[iq], C_l[isrc], w[up_q]
        args = (lptr, nin, isrc, iw, iC, up_e, up_r, uw)
        Vs, As = V, np.zeros_like(V)
        Vp, Ap = V[lel], np.zeros_like(V)
        Vq = V[lel]
        for _ in range(4):                          # sub-steps chained, as in update()
            Vs, a1 = _sweep_sorted(A, Vs, C, indptr, recv, w, 0.05, sub)
            Vp, a2 = _sweep_levels.par(A_l, Vp, C_l, 0.05, sub_l, *args)
            Vq, a3 = _sweep_levels.serial(A_l, Vq, C_l, 0.05, sub_l, *args)
            As, Ap = As + a1, Ap + a2
            assert np.array_equal(a2, a1[lel]) and np.array_equal(a3, a1[lel])
        assert np.array_equal(Vp, Vs[lel]) and np.array_equal(Vq, Vs[lel])
        assert np.array_equal(Ap, As[lel])
        assert (Vs > 0).sum() > 1000 and As.sum() > 0


def test_film_renumbering_equals_numpy():
    from drainsim.film import _renumber
    rng = np.random.default_rng(12)
    A, V, C, indptr, recv, w, sub = _random_sweep(rng)
    n = A.size
    order = rng.permutation(n).astype(np.int64)
    inv = np.empty(n, np.int64)
    inv[order] = np.arange(n)
    indptr_s = np.r_[0, np.cumsum(np.diff(indptr)[order])].astype(np.int64)
    qmap = np.concatenate([np.arange(indptr[e], indptr[e + 1]) for e in order]).astype(np.int64)
    got = _renumber(indptr, recv, order)
    assert np.array_equal(got[0], indptr_s) and np.array_equal(got[1], qmap)
    assert np.array_equal(got[2], inv[recv[qmap]])


def test_film_model_same_with_1_and_16_threads():
    """The whole film model (level-parallel sweep with several threads, the
    serial sweep with one) gives the same state."""
    import numba
    from drainsim import cases
    from drainsim.model import Simulation
    from drainsim.motion import Keyframes
    nmax = numba.config.NUMBA_NUM_THREADS
    if nmax < 2:
        pytest.skip("needs NUMBA_NUM_THREADS >= 2")
    g, interior = cases.box3d(dx=0.02, hole=0.04, vent=0.04)
    m = Keyframes([0, 1.5, 3.0], np.array([[0, 0, 0], [0, 40, 0], [0, 40, 0]]),
                  np.array([[0, 0, 0], [0, 0, 0.6], [0, 0, 0.6]]), 3, bath_level=0.3)
    old = numba.get_num_threads()
    res = []
    for k in (1, nmax):
        numba.set_num_threads(k)
        sim = Simulation(g, m, initial_L=interior.astype(float), dt_max=0.25, film=True)
        sim.run(t_end=3.0)
        res.append((sim.film.s.h.copy(), sim.L.copy(), sim.film.n_drops))
    numba.set_num_threads(old)
    assert res[0][0].sum() > 0
    assert np.array_equal(res[0][0], res[1][0]) and np.array_equal(res[0][1], res[1][1])
    assert res[0][2] == res[1][2]
