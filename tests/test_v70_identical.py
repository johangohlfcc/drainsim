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
        lptr, lel, nin, isrc, iq, up_e, up_r, up_q, uptr, ur = _sweep_plan(indptr, recv, wp)
        assert np.array_equal(np.repeat(ur, np.diff(uptr)), up_r)
        # the arrays laid out level by level, as FilmModel.update does
        A_l, C_l, sub_l = A[lel], C[lel], sub[lel]
        iw, iC, uw = w[iq], C_l[isrc], w[up_q]
        args = (lptr, nin, isrc, iw, iC, uptr, ur, up_e, uw)
        Vs, As = V, np.zeros_like(V)
        for k in range(1, 5):                       # sub-steps chained, as in update()
            Vs, a1 = _sweep_sorted(A, Vs, C, indptr, recv, w, 0.05, sub)
            As = As + a1
            Vp, Ap = _sweep_levels.par(A_l, V[lel], C_l, 0.05, k, sub_l, *args)
            Vq, Aq = _sweep_levels.serial(A_l, V[lel], C_l, 0.05, k, sub_l, *args)
            assert np.array_equal(Vp, Vs[lel]) and np.array_equal(Vq, Vs[lel])
            assert np.array_equal(Ap, As[lel]) and np.array_equal(Aq, As[lel])
        assert (Vs > 0).sum() > 1000 and As.sum() > 0 and up_r.size > 1000


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


# the plan as a serial kernel (v7.0 before the parallel plan): the reference
from numba import njit  # noqa: E402


@njit(cache=True, nogil=True)
def _sweep_plan_serial(indptr, recv, w):
    """Plan of a level-by-level sweep (elements in sweep order e):
    lptr    positions of each level (level L: lptr[L]..lptr[L+1]);
    lel     sweep element at each position;
    nin, isrc, iq   by receiver position: its senders' positions and the
            edges (sweep numbering), senders in ascending sweep order;
    up_e, up_r, up_q  the uphill edges (sender and receiver positions,
            edge), grouped by receiver, each receiver's in the serial order;
    uptr, ur  the receivers of uphill edges and their groups:
            up_*[uptr[g]:uptr[g+1]] go to position ur[g]."""
    n = indptr.shape[0] - 1
    lev = np.zeros(n, np.int64)
    cin = np.zeros(n, np.int64)
    nup = 0
    for e in range(n):
        for q in range(indptr[e], indptr[e + 1]):
            if w[q] <= 0.0:
                continue
            r = recv[q]
            if r > e:
                if lev[r] < lev[e] + 1:
                    lev[r] = lev[e] + 1
                cin[r] += 1
            else:
                nup += 1
    nlev = 0
    for e in range(n):
        if lev[e] + 1 > nlev:
            nlev = lev[e] + 1
    lptr = np.zeros(nlev + 1, np.int64)
    for e in range(n):
        lptr[lev[e] + 1] += 1
    for L in range(nlev):
        lptr[L + 1] += lptr[L]
    lel = np.empty(n, np.int64)
    pos = np.empty(n, np.int64)
    fill = lptr[:-1].copy()
    for e in range(n):
        p = fill[lev[e]]
        lel[p] = e
        pos[e] = p
        fill[lev[e]] += 1
    nin = np.zeros(n + 1, np.int64)
    for p in range(n):
        nin[p + 1] = nin[p] + cin[lel[p]]
    isrc = np.empty(nin[n], np.int64)
    iq = np.empty(nin[n], np.int64)
    at = nin[:-1].copy()
    up_e = np.empty(nup, np.int64)
    up_r = np.empty(nup, np.int64)
    up_q = np.empty(nup, np.int64)
    u = 0
    for e in range(n):
        for q in range(indptr[e], indptr[e + 1]):
            if w[q] <= 0.0:
                continue
            r = recv[q]
            if r > e:
                pr = pos[r]
                isrc[at[pr]] = pos[e]
                iq[at[pr]] = q
                at[pr] += 1
            else:
                up_e[u] = pos[e]
                up_r[u] = pos[r]
                up_q[u] = q
                u += 1
    # group by receiver, stable (counting sort over the positions)
    cnt = np.zeros(n + 1, np.int64)
    for j in range(nup):
        cnt[up_r[j] + 1] += 1
    nur = 0
    for p in range(n):
        if cnt[p + 1] > 0:
            nur += 1
        cnt[p + 1] += cnt[p]
    ge = np.empty(nup, np.int64)
    gr = np.empty(nup, np.int64)
    gq = np.empty(nup, np.int64)
    at2 = cnt[:-1].copy()
    for j in range(nup):
        k = at2[up_r[j]]
        ge[k] = up_e[j]
        gr[k] = up_r[j]
        gq[k] = up_q[j]
        at2[up_r[j]] += 1
    ur = np.empty(nur, np.int64)
    uptr = np.zeros(nur + 1, np.int64)
    g = 0
    for p in range(n):
        if cnt[p + 1] > cnt[p]:
            ur[g] = p
            uptr[g + 1] = cnt[p + 1]
            g += 1
    return lptr, lel, nin, isrc, iq, ge, gr, gq, uptr, ur


def test_film_plan_in_parallel_equals_the_serial_plan():
    """_sweep_plan (levels serial, the rest parallel) gives the arrays of
    the serial plan, on random sweep graphs and on a superset of edges."""
    import numba
    from drainsim.film import _sweep_plan
    rng = np.random.default_rng(12)
    for superset in (False, True):
        A, V, C, indptr, recv, w, sub = _random_sweep(rng, n=30000)
        wp = np.where((w == 0.0) & (rng.random(w.size) < 0.5), 1.0, w) if superset else w
        ref = _sweep_plan_serial(indptr, recv, wp)
        for thr in (1, 3, numba.config.NUMBA_NUM_THREADS):
            numba.set_num_threads(thr)
            got = _sweep_plan(indptr, recv, wp)
            assert len(got) == len(ref)
            for a, b in zip(got, ref):
                assert a.dtype == b.dtype and np.array_equal(a, b)
        numba.set_num_threads(numba.config.NUMBA_NUM_THREADS)


def test_film_sparse_sweep_equals_the_full_sweep():
    """The sweep of the elements with water only (heap in height order)
    equals the full sweep (_implicit_sweep in the stable order of neg),
    with few wet elements, ties in height, uphill flow and sub-steps."""
    from drainsim.film import _implicit_sweep, _sweep_sparse
    rng = np.random.default_rng(13)
    for wet in (0.001, 0.01, 0.2):
        A, V, C, indptr, recv, w, sub = _random_sweep(rng)
        n = A.size
        # element numbering: a random permutation of the sweep positions,
        # heights with ties
        perm = rng.permutation(n)
        neg = np.empty(n)
        neg[perm] = np.round(np.sort(rng.normal(size=n)), 2)
        inv = np.empty(n, np.int64)
        inv[perm] = np.arange(n)
        # the problem renumbered: element perm[i] is sweep position i
        ip = np.r_[0, np.cumsum(np.diff(indptr)[inv])].astype(np.int64)
        rows = [recv[indptr[inv[e]]:indptr[inv[e] + 1]] for e in range(n)]
        wr = [w[indptr[inv[e]]:indptr[inv[e] + 1]] for e in range(n)]
        rc = perm[np.concatenate(rows)]
        we = np.concatenate(wr)
        Ae, Ce, sube = A[inv], C[inv], sub[inv]
        h = np.where(rng.random(n) < wet, rng.random(n) * 1e-4, 0.0)
        order = np.argsort(neg, kind="stable")
        Vf, ab = Ae * h, np.zeros(n)
        for nsub in (1, 2, 3):
            Vf = Ae * h
            ab = np.zeros(n)
            for _ in range(nsub):
                Vf, a1 = _implicit_sweep(order, Ae, Vf, Ce, ip, rc, we, 0.05, sube)
                ab = ab + a1
            Vs, abs_ = _sweep_sparse(Ae, h, Ce, ip, rc, we, 0.05, nsub, sube, neg)
            assert np.array_equal(Vs, Vf) and np.array_equal(abs_, ab)
        assert (Vf > 0).sum() > 10


def test_film_element_passes_equal_numpy():
    """The per-step element passes of FilmModel.update give the values of
    the numpy expressions they replace."""
    from drainsim.film import (_clip0, _drain_submerged, _film_flags, _hang_flags,
                               _puddles, _scatter_add, _scatter_cells, _sub_ratio)
    rng = np.random.default_rng(14)
    n, N = 50000, 9000
    cell = rng.integers(0, N, n)
    Lc = np.where(rng.random(n) < 0.3, rng.choice([0.5, 0.99, 1.0], n), rng.random(n))
    h = np.where(rng.random(n) < 0.5, rng.random(n) * 1e-3, 0.0)
    h[:50] = -0.0
    sub, B = rng.random(n) < 0.4, rng.random(N) < 0.5
    area = rng.uniform(1e-6, 1e-5, n)
    t_full, fb = rng.random(n), rng.random(n) < 0.5
    inj = rng.random(N) * 1e-6
    # step 1
    now_r = Lc >= 0.5
    m = now_r & (h > 0)
    inj_r, h_r, tf_r, fb_r = inj.copy(), h.copy(), t_full.copy(), fb.copy()
    _scatter_add(inj_r, cell[m], h_r[m] * area[m])
    h_r[m] = 0.0
    tf_r[Lc >= 0.99] = 7.5
    fb_r[now_r] = B[cell[now_r]]
    now, em, cnt = _film_flags(Lc, h, sub, cell, B, 7.5, t_full, fb)
    _drain_submerged(inj, cell, h, area, now)
    assert np.array_equal(now, now_r) and np.array_equal(em, sub & ~now_r) and cnt == m.sum()
    for a, b in ((inj, inj_r), (h, h_r), (t_full, tf_r), (fb, fb_r)):
        assert np.array_equal(a, b)
    # scatter into the cells
    o = np.argsort(cell, kind="stable")
    cs = cell[o]
    st = np.flatnonzero(np.r_[True, cs[1:] != cs[:-1]])
    val = rng.random(n)
    a, b = inj.copy(), inj.copy()
    _scatter_cells(a, np.r_[st, n].astype(np.int64), o.astype(np.int64), cs[st], val)
    np.add.at(b, cell, val)
    assert np.array_equal(a, b)
    # steps 4-6
    up_n, dry = rng.uniform(-1, 1, n), ~now_r
    hb = 2e-4
    upward = up_n >= -0.2
    ex_r = np.where(upward & dry, np.maximum(h - hb, 0.0), 0.0) * area
    ex, upw, nex = _puddles(h, up_n, dry, area, -0.2, hb)
    assert np.array_equal(ex, ex_r) and np.array_equal(upw, upward) and nex == (ex_r != 0).sum()
    h2 = h.copy()
    h2 -= ex_r / area
    _sub_ratio(h, ex, area)
    assert np.array_equal(h, h2)
    hang_r = dry & ~upward & (h > 1e-4)
    hang, nh = _hang_flags(h, upw, dry, 1e-4)
    assert np.array_equal(hang, hang_r) and nh == hang_r.sum()
    h[:10] = -1e-9
    c0, c1 = _clip0(h), np.maximum(h, 0.0)
    assert np.array_equal(c0, c1) and np.array_equal(np.signbit(c0), np.signbit(c1))


def test_bath_atmosphere_from_one_labelling_equals_three():
    """connected_sides (both sides of the bath surface in one labelling) and
    connected_across (the submerged system: the side components joined
    across the surface) equal the three connected_to calls they replace,
    also with one-sided and cut links."""
    from drainsim.par import connected_across, connected_sides, connected_to
    rng = np.random.default_rng(15)
    N, W = 30000, 6
    for trial in range(4):
        tab = np.where(rng.random((N, W)) < 0.9, np.arange(N)[:, None] + rng.integers(-300, 300, (N, W)), -1)
        tab[(tab < 0) | (tab >= N)] = -1
        ptr = np.arange(0, N * W + 1, W, dtype=np.int64)
        idx = tab.ravel().astype((np.int64, np.int32)[trial % 2])
        comp = rng.integers(0, 2, N)
        ext = rng.random(N) < 0.95
        below = rng.random(N) < (0.3, 0.5, 0.7, 0.9)[trial]
        lo, hi = ext & below, ext & ~below
        seed = rng.random(N) < 0.01
        B0 = connected_to(lo, comp, ptr, idx, seed)
        A0 = connected_to(hi, comp, ptr, idx, seed)
        S0 = connected_to(ext & ~A0, comp, ptr, idx, B0)
        B, A, root = connected_sides(lo, hi, comp, ptr, idx, seed)
        S = connected_across(lo, hi, A, root, comp, ptr, idx, B)
        assert np.array_equal(B, B0) and np.array_equal(A, A0) and np.array_equal(S, S0)
        assert B0.sum() + A0.sum() > 1000 and (S0 & ~B0).sum() > 1000


def test_take_top_equals_the_python_loop():
    from drainsim.stepk import take_top
    rng = np.random.default_rng(16)
    for dV in (0.0, 1e-9, 0.3, 50.0):
        L = rng.random(200)
        v = rng.uniform(0.1, 1.0, 200)
        cells = rng.permutation(200)[:120]
        L1, L2 = L.copy(), L.copy()
        rem = dV
        for c in cells:
            have = L1[c] * v[c]
            take = min(have, rem)
            L1[c] -= take / v[c]
            rem -= take
            if rem <= 0:
                break
        assert take_top(cells, L2, v, dV) == rem and np.array_equal(L1, L2)


def test_hierarchy_in_rank_space_equals_node_space_and_serial():
    """hierarchy_rank (the graph renumbered into height order) gives every
    array of hierarchy_parallel (node space) and of the serial sweep: 2D
    terrains with nested basins, several regions, sinks, exact height ties
    (plateaus), inactive nodes and one-sided links; candidate filter on and
    off; 1, 3 and all threads."""
    import numba
    from drainsim.fsm import (_hierarchy_serial, height_order, hierarchy_parallel,
                              hierarchy_rank, to_csr)
    rng = np.random.default_rng(17)
    names = ("rank", "owner0", "dest", "node_parent", "node_spill", "node_region", "nnodes")
    for trial in range(6):
        g, h = _terrain(rng, 120, 90)
        N = h.size
        if trial % 2:
            h = np.round(h * 4) / 4                       # plateaus: many equal heights
        region = np.where(rng.random(N) < 0.97, (np.arange(N) // 3000) % 3, -1).astype(np.int64)
        sink = rng.random(N) < 0.002
        ptr, idx = to_csr(g.neighbors())
        idx = idx.copy()
        idx[rng.random(idx.size) < 0.03] = -1             # cut on one side only
        active = np.flatnonzero(region >= 0)
        order = height_order(h, sink, active)
        ref = _hierarchy_serial(order, h, region, ptr, idx, sink)
        for thr in (1, 3, numba.config.NUMBA_NUM_THREADS):
            numba.set_num_threads(thr)
            for filt in (True, False):
                a = hierarchy_parallel(order, h, region, ptr, idx, sink, filt=filt, full=True)
                b = hierarchy_rank(order, h, region, ptr, idx, sink, filt=filt, full=True)
                assert np.array_equal(a[7], b[7])         # node_cell
                for name, x, y, z in zip(names, ref, a, b):
                    if isinstance(x, np.ndarray):
                        if name.startswith("node_"):
                            x = x[:ref[6]]                # the serial sweep's arrays are N long
                        assert np.array_equal(y, z) and np.array_equal(x, z), (trial, thr, filt, name)
                    else:
                        assert x == y == z, (trial, thr, filt, name)
        numba.set_num_threads(numba.config.NUMBA_NUM_THREADS)
        assert ref[6] > 100


def test_flat_fill_by_node_equals_row_by_row():
    """_flat_fill (the partly filled depressions node by node, in parallel)
    equals _flat_fill_serial (row by row) on the same hierarchy and fill
    volumes: terrains with plateaus, box and linear cell shapes, several
    source volumes, 1 / 3 / all threads. (The split and serial fill_spill
    differ by an ulp in a few cells on plateaus with cell floors, as in 6.4;
    this compares the flat fill itself.)"""
    import numba
    from drainsim import fsm
    from drainsim.fsm import LINEAR, box_shape, to_csr
    rng = np.random.default_rng(18)
    shapes = (LINEAR, box_shape(np.array([0.3, 0.2, 0.93])))
    npart = 0
    for trial in range(6):
        g, h = _terrain(rng, 110, 80)
        N = h.size
        if trial % 3 == 2:
            h = np.round(h * 6) / 6
        region = np.where(rng.random(N) < 0.98, (np.arange(N) // 2500) % 2, -1).astype(np.int64)
        sink = rng.random(N) < 0.001
        src = np.where(rng.random(N) < 0.3, rng.random(N) * (0.02, 0.2, 1.0)[trial % 3], 0.0)
        ecell = np.where(rng.random(N) < 0.8, 0.005, 0.0025)
        vcell = rng.uniform(0.5, 1.0, N)
        es = shapes[trial % 2]
        ptr, idx = to_csr(g.neighbors())
        hf = h - ecell
        P = fsm.prepare(hf, region, ptr, idx, sink, vcell, 0.01, ecell, eshape=es)
        sd, sr, sw = fsm._par_sources(P.order, src, P.dest, P.region, 4)
        vol, _, _ = fsm._fill(sd, sr, sw, P.node_parent, P.node_region, P.nnodes, P.cap, True, 2,
                              P.tol, P.sptr, P.sL, P.sT)
        cells = P.order[P.rows]
        ref = fsm._flat_fill_serial(cells, hf, ecell, P.eshape, vcell, P.pn, P.pv, vol, P.cap,
                                    P.tol, P.nnodes, N)
        for thr in (1, 3, numba.config.NUMBA_NUM_THREADS):
            numba.set_num_threads(thr)
            got = fsm._flat_fill(cells, hf, ecell, P.eshape, vcell, P.pn, P.pv, vol, P.cap,
                                 P.tol, P.nnodes, N)
            assert np.array_equal(got, ref), (trial, thr)
        numba.set_num_threads(numba.config.NUMBA_NUM_THREADS)
        npart += int(fsm._part_nodes(vol, P.cap, P.tol, P.nnodes).sum())
    assert npart > 200                                    # partly filled depressions


# ------------------------------------------ forced thresholds (7.0.1)
def _film_run(dt_max):
    from drainsim import cases
    from drainsim.model import Simulation
    from drainsim.motion import Keyframes
    g, interior = cases.box3d(dx=0.02, hole=0.04, vent=0.04)
    m = Keyframes([0, 1.5, 3.0], np.array([[0, 0, 0], [0, 40, 0], [0, 40, 0]]),
                  np.array([[0, 0, 0], [0, 0, 0.6], [0, 0, 0.6]]), 3, bath_level=0.3)
    sim = Simulation(g, m, initial_L=interior.astype(float), dt_max=dt_max, film=True)
    sim.run(t_end=3.0)
    return sim.film.s.h.copy(), sim.L.copy(), sim.film.n_drops


@pytest.mark.parametrize("dt_max", [0.1, 0.25])
def test_film_sweeps_with_forced_thresholds_give_the_same_run(monkeypatch, dt_max):
    """The film's sweep is chosen by thresholds (SPARSE_SWEEP: the sparse
    sweep of a film with little water; WET_LEVELS: a level-parallel sweep
    of a wet film when the plan is not kept). Forced to each sweep, with 4
    or more threads, the run is the same. dt_max 0.1 gives 2 film sub-steps
    (a plan of this step's flow, or the serial sweep), 0.25 gives 5 (the
    kept plan)."""
    import numba
    from drainsim import film
    nmax = numba.config.NUMBA_NUM_THREADS
    if nmax < 4:
        pytest.skip("needs NUMBA_NUM_THREADS >= 4")
    used = {}

    def count(name):
        f = getattr(film, name)

        def g(*a):
            used[name] = used.get(name, 0) + 1
            return f(*a)
        monkeypatch.setattr(film, name, g)
    for name in ("_sweep_sparse", "_sweep_levels", "_sweep_sorted"):
        count(name)
    old = numba.get_num_threads()
    numba.set_num_threads(max(4, min(nmax, 8)))
    try:
        res = {}
        for sparse, wet in ((None, None), (0.0, 0.0), (0.0, 2.0), (1.01, 2.0)):
            if sparse is not None:
                monkeypatch.setattr(film, "SPARSE_SWEEP", sparse)
                monkeypatch.setattr(film, "WET_LEVELS", wet)
            used.clear()
            res[(sparse, wet)] = (_film_run(dt_max), dict(used))
    finally:
        numba.set_num_threads(old)
    ref, _ = res[(None, None)]
    assert ref[0].sum() > 0
    for key, (r, u) in res.items():
        assert np.array_equal(r[0], ref[0]) and np.array_equal(r[1], ref[1]), key
        assert r[2] == ref[2], key
    # each forced setting ran the sweep it forces
    assert set(res[(1.01, 2.0)][1]) == {"_sweep_sparse"}
    assert "_sweep_sparse" not in res[(0.0, 0.0)][1] and "_sweep_levels" in res[(0.0, 0.0)][1]
    if dt_max == 0.1:
        assert set(res[(0.0, 2.0)][1]) == {"_sweep_sorted"}
        u = res[(0.0, 0.0)][1]                     # serial only while the film is dry
        assert u["_sweep_levels"] > 10 * u.get("_sweep_sorted", 0)


def test_hierarchy_rank_above_the_rank_limit_computes_in_node_space(monkeypatch):
    """With RANK_LIMIT or more active nodes (32-bit ranks would overflow)
    hierarchy_rank computes in node space: the same arrays, in node and in
    rank numbering."""
    from drainsim import fsm
    from drainsim.fsm import height_order, hierarchy_rank, to_csr
    rng = np.random.default_rng(18)
    g, h = _terrain(rng, 120, 90)
    N = h.size
    region = np.where(rng.random(N) < 0.97, (np.arange(N) // 3000) % 3, -1).astype(np.int64)
    sink = rng.random(N) < 0.002
    ptr, idx = to_csr(g.neighbors())
    order = height_order(h, sink, np.flatnonzero(region >= 0))
    ref = [hierarchy_rank(order, h, region, ptr, idx, sink, full=True, rank_space=rs)
           for rs in (False, True)]
    calls = []
    par = fsm.hierarchy_parallel

    def spy(*a, **k):
        calls.append(1)
        return par(*a, **k)
    monkeypatch.setattr(fsm, "hierarchy_parallel", spy)
    monkeypatch.setattr(fsm, "RANK_LIMIT", order.size)
    for rs, r in zip((False, True), ref):
        got = hierarchy_rank(order, h, region, ptr, idx, sink, full=True, rank_space=rs)
        for x, y in zip(r, got):
            assert np.array_equal(x, y) if isinstance(x, np.ndarray) else x == y
    assert len(calls) == 2 and ref[0][6] > 100
