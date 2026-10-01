"""drainsim 7.0: the display fields of the movies (flat free surfaces) are the
same as in 6.4 (the 6.4 code is kept here as the reference).
Run: python -m pytest -q tests/test_v70_display.py"""
import numpy as np
import pytest
from numba import njit

from drainsim.par import _PAR_LOCK
from drainsim.worldviz import _node_e, planar_fraction, trapped_display


# ------------------------------------------------------------ 6.4 reference
@njit(cache=True, nogil=True)
def _grow_bodies64(body, lab, ptr, idx, iters):
    grow = body.copy()
    N = grow.shape[0]
    for _ in range(iters):
        new = grow.copy()
        for c in range(N):
            g = grow[c]
            if g < 0:
                continue
            for jj in range(ptr[c], ptr[c + 1]):
                n = idx[jj]
                if n >= 0 and lab[n] == lab[c] and grow[n] < 0:
                    new[n] = g
        grow = new
    return grow


def planar_fraction64(sim, frac, h, iters=2):
    from drainsim.fsm import to_csr
    from drainsim.model import _label_bodies
    e = _node_e(sim, float(sim._vis_e), frac.shape)
    w = np.asarray(sim.v, float)
    mask = sim.fl & (frac > 1e-6)
    body, nb = _label_bodies(mask, sim.lab, sim.nbr)
    out = np.zeros_like(frac)
    if nb == 0:
        return out
    ptr, nidx = to_csr(sim.nbr)
    grow = _grow_bodies64(body.astype(np.int64), np.asarray(sim.lab, np.int64), ptr, nidx,
                          int(iters))
    idx = np.flatnonzero(grow >= 0)
    order = np.argsort(grow[idx], kind="stable")
    idx = idx[order]
    b = grow[idx]
    starts = np.searchsorted(b, np.arange(nb + 1))
    wc = w[idx]
    hc = h[idx]
    ec = e[idx]
    V = np.bincount(b, weights=frac[idx] * wc, minlength=nb)
    lo = np.minimum.reduceat(hc - ec, starts[:-1])
    hi = np.maximum.reduceat(hc + ec, starts[:-1])
    for _ in range(40):
        H = 0.5 * (lo + hi)
        f = np.clip((H[b] - hc + ec) / (2 * ec), 0, 1)
        m = np.bincount(b, weights=f * wc, minlength=nb) > V
        hi = np.where(m, H, hi)
        lo = np.where(m, lo, H)
    out[idx] = np.clip((0.5 * (lo + hi)[b] - hc + ec) / (2 * ec), 0, 1)
    return out


def trapped_display64(sim, h, zb, L, B, A):
    en = _node_e(sim, float(sim._vis_e), h.shape)
    a = np.clip((h + en - zb) / (2.0 * en), 0.0, 1.0)
    liq = planar_fraction64(sim, np.where(sim.fl & ~B, L, 0.0), h) * a
    air = planar_fraction64(sim, np.where(sim.fl & ~A, 1.0 - L, 0.0), -h) * (1.0 - a)
    return liq, air


# ------------------------------------------------------------------ tests
def _box_run(octree):
    """A box open at the top, tilted while it is lowered into the bath and
    lifted out: liquid and air bodies, crossing the bath surface too."""
    trimesh = pytest.importorskip("trimesh")
    from drainsim.grid import Grid
    from drainsim.model import Simulation
    from drainsim.motion import Keyframes
    from drainsim.octree import Octree
    m = trimesh.creation.box(extents=(0.1, 0.08, 0.06))
    m.apply_translation((0.0017, -0.0023, 0.0011))
    m = trimesh.Trimesh(m.vertices, m.faces[m.face_normals[:, 2] < 0.5], process=False)
    inner = trimesh.creation.box(extents=(0.04, 0.03, 0.03))       # a cup inside
    inner.apply_translation((0.01, 0.0, -0.01))
    inner = trimesh.Trimesh(inner.vertices, inner.faces[inner.face_normals[:, 2] < 0.5],
                            process=False)
    m = trimesh.util.concatenate([m, inner])
    lo = np.array([-0.08, -0.07, -0.05])
    hi = lo + np.array([28, 24, 20]) * 0.006
    if octree:
        g = Octree.from_mesh(m, 0.006, levels=1, bounds=(lo, hi))
    else:
        g = Grid.from_mesh(m, 0.006, bounds=(lo, hi))
    mo = Keyframes([0, 1.0, 2.0, 3.0], np.array([[0, 0, 0], [0, 25, 10], [0, 40, 20], [0, 5, 0]]),
                   np.array([[0, 0, -0.1], [0, 0, 0.03], [0, 0, 0.03], [0, 0, -0.1]]), 3,
                   bath_level=0.0)
    return Simulation(g, mo, subcells=2, dt_max=0.1)


@pytest.mark.parametrize("octree", [True, False])
def test_planar_fraction_and_trapped_display_equal_6_4(octree):
    sim = _box_run(octree)
    rng = np.random.default_rng(41)
    checked = 0
    for t in (1.0, 1.6, 2.4, 3.0):
        sim.run(t_end=t)
        up, zb = sim.motion.frame(sim.t)
        h = sim.X @ up
        sim._vis_e = 0.5 * sim.grid.dx * np.abs(up).sum()
        a = trapped_display(sim, h, zb, sim.L, sim.B, sim.A)
        b = trapped_display64(sim, h, zb, sim.L, sim.B, sim.A)
        for x, y in zip(a, b):
            assert np.array_equal(x, y)
            checked += int((x > 0).sum())
        # many bodies: random blobs of fluid, in the compartments
        frac = np.where(sim.fl & (rng.random(sim.N) < 0.3), rng.random(sim.N), 0.0)
        for it in (0, 1, 2, 3):
            assert np.array_equal(planar_fraction(sim, frac, h, iters=it),
                                  planar_fraction64(sim, frac, h, iters=it))
        # the serial kernels (another thread holds the parallel ones)
        with _PAR_LOCK:
            assert np.array_equal(planar_fraction(sim, frac, h), planar_fraction64(sim, frac, h))
    assert checked > 100


def _state_fields64(sim, t_state, L, B, A):
    """car_movie._state_fields as in 6.4 (dense display fields)."""
    import os
    import sys
    from types import SimpleNamespace
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "examples"))
    from car_movie import grid_field
    view = SimpleNamespace(fl=sim.fl, lab=sim.lab, nbr=sim.nbr, v=sim.v, fine=sim.fine,
                           subcells=sim.subcells, host=sim.host, ncells=sim.ncells,
                           N=sim.N, grid=sim.grid, nsize=sim.nsize)
    up, zb = sim.motion.frame(t_state)
    h = sim.X @ up
    view._vis_e = 0.5 * sim.grid.dx * np.abs(up).sum()
    liq, air = trapped_display64(view, h, zb, L, B, A)
    if getattr(sim, "octree", False):
        from drainsim.octview import display_grid
        dg = display_grid(sim)
        liq, air = dg.field(sim, liq), dg.field(sim, air)
        ncells = dg.ncells
    else:
        liq, air = grid_field(view, liq), grid_field(view, air)
        ncells = sim.grid.ncells
    out = {}
    for k, f in (("liq", liq), ("air", air)):
        q = np.clip(np.rint(f * 255.0), 0, 255).astype(np.uint8)
        i = np.flatnonzero(q)
        out[k + "_i"] = i.astype(np.int64 if ncells >= 2 ** 31 else np.int32)
        out[k + "_v"] = q[i]
    return out


@pytest.mark.parametrize("octree", [True, False])
def test_recorded_fields_equal_6_4(octree):
    """car_movie --record: the fields written per step are the same (the
    display grid filled at its nonzero cells only)."""
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "examples"))
    from car_movie import _state_fields
    sim = _box_run(octree)
    rng = np.random.default_rng(42)
    n = 0
    for t in (1.0, 1.6, 2.4, 3.0):
        sim.run(t_end=t)
        a = _state_fields(sim, sim.t, sim.L, sim.B, sim.A)
        b = _state_fields64(sim, sim.t, sim.L, sim.B, sim.A)
        assert set(a) == set(b)
        for k in a:
            assert a[k].dtype == b[k].dtype and np.array_equal(a[k], b[k])
        n += a["liq_i"].size + a["air_i"].size
        if octree:
            from drainsim.octview import display_grid
            dg = display_grid(sim)
            f = np.where(rng.random(sim.N) < 0.5, rng.random(sim.N), 0.0)
            dense = dg.field(sim, f)
            i, v = dg.field_nonzero(sim, f)
            assert np.array_equal(i, np.flatnonzero(dense)) and np.array_equal(v, dense[i])
    assert n > 100


def test_label_masked_equals_label_bodies_on_one_sided_links():
    """The labelling on the masked nodes gives par.label_bodies' bodies and
    numbering, also when a link is cut on one side only."""
    from drainsim.par import label_bodies
    from drainsim.worldviz import _label_masked
    rng = np.random.default_rng(43)
    N, W = 20000, 6
    tab = rng.integers(0, N, (N, W))
    tab[rng.random((N, W)) < 0.3] = -1                      # cut, one side only
    ptr = np.arange(0, N * W + 1, W, dtype=np.int64)
    for idx in (tab.ravel().astype(np.int64), tab.ravel().astype(np.int32)):
        lab = rng.integers(0, 3, N)
        fl = rng.random(N) < 0.9
        frac = np.where(rng.random(N) < 0.4, rng.random(N), 0.0)
        body, act, nb = _label_masked(fl, frac, 1e-6, lab, ptr, idx)
        ref, nref = label_bodies(fl & (frac > 1e-6), lab, ptr, idx.astype(np.int64))
        assert nb == nref > 100 and np.array_equal(body, ref)
        assert np.array_equal(act, np.flatnonzero(ref >= 0))
