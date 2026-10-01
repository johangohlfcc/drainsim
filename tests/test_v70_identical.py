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


def _terrain(rng, nx=90, ny=70):
    """A rough 2D terrain with nested basins (deep depression hierarchies)."""
    from drainsim.grid import Grid
    g = Grid.empty([0, 0], [nx * 0.01, ny * 0.01], 0.01)
    x, y = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    h = (np.sin(x / 7.0) * np.cos(y / 5.0) + 0.3 * np.sin(x / 2.3 + y / 3.1)
         + 0.2 * rng.random((nx, ny))).ravel()
    return g, h


def test_fill_cascade_with_euler_intervals_equals_the_serial_kernel():
    """fill_spill (split: the cascade tests 'inside the overflowing depression'
    on Euler-tour intervals) equals the serial kernel (which walks up the
    hierarchy)."""
    from drainsim.fsm import fill_spill
    rng = np.random.default_rng(21)
    for trial in range(4):
        g, h = _terrain(rng)
        N = h.size
        region = np.zeros(N, np.int64)
        sink = np.zeros(N, bool)
        sink[rng.choice(N, 3, replace=False)] = True
        src = np.where(rng.random(N) < 0.2, rng.random(N) * 3.0, 0.0)
        kw = dict(spill_routing=True, nregions=1, min_depth=0.0)
        a = fill_spill(h, region, g.neighbors(), sink, src, 1.0, method="split", **kw)
        b = fill_spill(h, region, g.neighbors(), sink, src, 1.0, method="serial", **kw)
        assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
        assert np.array_equal(a[2], b[2])
        assert (a[0] > 0).sum() > 100


def test_euler_intervals_match_the_ancestor_walk():
    from drainsim.fsm import _euler, fill_spill, prepare, to_csr
    rng = np.random.default_rng(22)
    g, h = _terrain(rng, 120, 90)
    N = h.size
    ptr, idx = to_csr(g.neighbors())
    sink = np.zeros(N, bool)
    sink[0] = True
    P = prepare(h, np.zeros(N, np.int64), ptr, idx, sink, np.ones(N), 0.0, np.zeros(N))
    par = P.node_parent
    assert P.nnodes > 200
    for m in rng.choice(P.nnodes, 300):
        for T in rng.choice(P.nnodes, 300):
            walk = False
            t = T
            while t > 0:
                if t == m:
                    walk = True
                    break
                t = par[t]
            fast = m > 0 and T > 0 and P.tin[m] <= P.tin[T] < P.tout[m]
            assert walk == fast


def test_int32_links_give_the_same_run():
    """After setup the links are stored in 32 bits and the octree's setup
    graph is dropped; a run gives the same state as on 64-bit links."""
    import trimesh
    from drainsim import cases
    from drainsim.model import Simulation
    from drainsim.motion import Keyframes
    from drainsim.octree import Octree
    m = trimesh.creation.box(extents=(0.1, 0.08, 0.06))
    m.apply_translation((0.0017, -0.0023, 0.0011))
    m = trimesh.Trimesh(m.vertices, m.faces[m.face_normals[:, 2] < 0.5], process=False)
    lo = np.array([-0.08, -0.07, -0.05])
    ot = Octree.from_mesh(m, 0.006, levels=1, bounds=(lo, lo + np.array([28, 24, 20]) * 0.006))
    mo = Keyframes([0, 1.0, 2.0], np.array([[0, 0, 0], [0, 30, 0], [0, 30, 0]]),
                   np.array([[0, 0, -0.1], [0, 0, 0.05], [0, 0, 0.05]]), 3, bath_level=0.0)
    states = []
    for wide in (False, True):
        sim = Simulation(ot, mo, subcells=2, dt_max=0.1, film=True)
        assert sim.nbr[1].dtype == np.int32 and sim.ograph is None
        assert "_host" in sim.__dict__ and sim.__dict__["_host"] is None
        assert np.array_equal(sim.host, np.arange(sim.N))
        assert sim.G is None and sim.P is None
        if wide:
            sim.nbr = (sim.nbr[0], sim.nbr[1].astype(np.int64))
        sim.run(t_end=2.0)
        states.append((sim.L.copy(), sim.B.copy(), sim.A.copy(), sim.film.s.h.copy()))
    for a, b in zip(*states):
        assert np.array_equal(a, b)
    assert states[0][0].sum() > 0


@pytest.mark.parametrize("n", [50, 5000, 300_000])
def test_height_order_of_floats_equals_the_stable_argsort(n):
    """height_order for heights that are not whole nanometres (cell floors
    h - ecell): the order of the stable float argsort, sinks first, ties by
    index, -0 equal to +0, with any number of threads; not finite heights
    take the argsort itself."""
    import numba
    from drainsim.fsm import height_order
    rng = np.random.default_rng(7)
    N = n + n // 3
    for case in range(4):
        h = rng.normal(size=N) * 0.4
        h[rng.random(N) < 0.2] = 0.25                           # ties
        h[rng.random(N) < 0.05] = 0.0
        h[rng.random(N) < 0.05] = -0.0
        h[: N // 10] = np.round(h[: N // 10], 3)                 # more ties
        sink = rng.random(N) < (0.0, 0.1, 0.5, 0.98)[case]
        active = np.sort(rng.choice(N, n, replace=False))
        ref = active[np.argsort(np.where(sink[active], -np.inf, h[active]), kind="stable")]
        for thr in (1, 4, numba.config.NUMBA_NUM_THREADS):
            numba.set_num_threads(thr)
            assert np.array_equal(height_order(h, sink, active), ref)
        numba.set_num_threads(numba.config.NUMBA_NUM_THREADS)
    h[active[3]] = np.nan
    ref = active[np.argsort(np.where(sink[active], -np.inf, h[active]), kind="stable")]
    assert np.array_equal(height_order(h, sink, active), ref)
