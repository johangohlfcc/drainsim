"""Octree grid, cell volume distribution, flat filling. Run:
python -m pytest -q tests/test_octree.py"""
import numpy as np
import pytest

from drainsim import cases
from drainsim.fsm import box_shape, cell_cdf, cell_cdf_vec
from drainsim.grid import Grid
from drainsim.model import Simulation
from drainsim.octree import Octree, build_graph

trimesh = pytest.importorskip("trimesh")


def _cup(ext=(0.1, 0.08, 0.06), shift=(0.0017, -0.0023, 0.0011)):
    m = trimesh.creation.box(extents=ext)
    m.apply_translation(shift)
    keep = m.face_normals[:, 2] < 0.5                 # open top
    return trimesh.Trimesh(m.vertices, m.faces[keep], process=False)


def _box(h, lo=(-0.08, -0.07, -0.05), n=(40, 36, 32)):
    lo = np.asarray(lo, float)
    return lo, lo + np.asarray(n) * h


def test_cell_cdf_matches_sampling_and_is_additive():
    """The cube's volume below a plane: Monte Carlo check, and the value of
    a cube is the mean of its eight children's (what makes an octree hold
    the same as the uniform grid)."""
    rng = np.random.default_rng(2)
    for up in ([0, 0, 1], [0.087, 0, 0.996], [0.3, -0.5, 0.8], [1, 1, 1]):
        u = np.asarray(up, float) / np.linalg.norm(up)
        w = box_shape(u)
        s = np.abs(u).sum()
        P = rng.random((200_000, 3)) - 0.5                   # unit cube at 0
        for H in (-0.4, -0.1, 0.0, 0.23, 0.45):
            t = (H + 0.5 * s) / s
            assert cell_cdf(t, w) == pytest.approx(np.mean(P @ u < H), abs=4e-3)
            kids = [np.array([a, b, c]) * 0.25 for a in (-1, 1) for b in (-1, 1)
                    for c in (-1, 1)]
            fk = np.mean([cell_cdf((H - k @ u + 0.25 * s) / (0.5 * s), w) for k in kids])
            assert cell_cdf(t, w) == pytest.approx(fk, abs=1e-12)


def test_octree_level0_is_the_uniform_grid():
    h = 0.006
    lo, hi = _box(h)
    m = _cup()
    g = Grid.from_mesh(m, h, bounds=(lo, hi))
    ot = Octree.from_mesh(m, h, levels=0, bounds=(lo, hi))
    assert ot.cut.size == int(g.solid.sum())
    su = Simulation(g, cases.static(ndim=3, t_end=0.1), subcells=4)
    so = Simulation(ot, cases.static(ndim=3, t_end=0.1), subcells=4)
    assert so.fl.sum() == su.fl.sum()
    assert so.v[so.fl].sum() == pytest.approx(su.v[su.fl].sum(), rel=1e-12)


def test_octree_is_graded_and_closes_the_volume():
    h = 0.004
    lo, hi = _box(h, n=(64, 56, 48))
    ot = Octree.from_mesh(_cup(), h, levels=3, bounds=(lo, hi))
    vol = sum(K.size * (h * 2 ** l) ** 3 for l, K in enumerate(ot.leaves)) + ot.cut.size * h ** 3
    assert vol == pytest.approx(np.prod(ot.dims0) * h ** 3, rel=1e-12)
    assert all(K.size for K in ot.leaves)                  # every level is used
    g = build_graph(ot, subcells=2)                        # raises if not 2:1
    assert g.N > sum(ot.nleaves())


def _tilted_cup_retained(grid):
    tilt = cases.static(ndim=3, t_end=0.5, angle=[4.0, 3.0, 0.0])
    sim = Simulation(grid, tilt, subcells=4, dt_max=0.1)
    sim.L = np.where(sim.fl, 1.0, 0.0)                     # flood, then let it settle
    sim.run()
    return sim.hist.liquid_retained[-1]


@pytest.mark.parametrize("levels", [2, 3])
def test_octree_cup_holds_what_the_uniform_grid_holds(levels):
    """A flooded open cup tilted 5 deg keeps the same volume on a graded
    octree (coarse cells in the pool and above it) as on the uniform grid:
    spill capacities are exact sums over the cells (box volume
    distribution, floors, flat filling)."""
    h = 0.004
    lo, hi = _box(h, n=(64, 56, 48))
    m = _cup()
    Vu = _tilted_cup_retained(Grid.from_mesh(m, h, bounds=(lo, hi)))
    Vo = _tilted_cup_retained(Octree.from_mesh(m, h, levels=levels, bounds=(lo, hi)))
    assert Vo == pytest.approx(Vu, rel=1e-6)


def test_octree_film_carrier_is_the_uniform_one():
    from drainsim.film import build_carrier
    h = 0.006
    lo, hi = _box(h)
    m = _cup()
    cu = build_carrier(Grid.from_mesh(m, h, bounds=(lo, hi)))
    co = build_carrier(Octree.from_mesh(m, h, levels=0, bounds=(lo, hi)))
    assert co.n == cu.n
    assert co.area.sum() == pytest.approx(cu.area.sum(), rel=1e-6)   # smoothing order


def test_display_grid_keeps_the_volume():
    from drainsim.octview import DisplayGrid
    h = 0.004
    lo, hi = _box(h, n=(64, 56, 48))
    ot = Octree.from_mesh(_cup(), h, levels=2, bounds=(lo, hi))
    sim = Simulation(ot, cases.static(ndim=3, t_end=0.1), subcells=2)
    f = np.random.default_rng(0).random(sim.N) * sim.fl
    for j in (0, 1, 2):
        dg = DisplayGrid(ot, j)
        d = dg.field(sim, f)
        assert (d * dg.fluid_volume(sim)).sum() == pytest.approx((f * sim.v).sum(), rel=1e-5)


def _slot_box_drain(grid, ext, shift, t_end):
    sim = Simulation(grid, cases.static(ndim=3, t_end=t_end), subcells=4, dt_max=0.1)
    inside = sim.fl & np.all(np.abs(sim.X - shift) < ext / 2, axis=1)
    sim.L = np.where(inside, 1.0, 0.0)
    sim.run()
    return sim, sim.hist.liquid_retained[-1] / (ext.prod() / 2)


@pytest.mark.parametrize("levels", [0, 2])
def test_octree_compartments_and_throats_as_uniform(levels):
    """The closed box of ``test_subcell_opening_between_compartments`` (5 mm
    drain slot and lid vent at dx = 8 mm, open only in the sub-cells): the
    octree finds the same compartments and throats and drains the same."""
    from test_drainsim import _closed_slot_box
    ext = np.array([0.1, 0.08, 0.06])
    shift = np.array([0.0017, -0.0023, 0.0011])
    m = _closed_slot_box(ext, shift, 0.005, 0.005)
    h = 0.008
    lo = np.asarray(m.vertices).min(0) - 0.024
    hi = lo + np.array([24, 20, 16]) * h
    su, ru = _slot_box_drain(Grid.from_mesh(m, h, bounds=(lo, hi)), ext, shift, 30.0)
    so, ro = _slot_box_drain(Octree.from_mesh(m, h, levels=levels, bounds=(lo, hi)), ext,
                             shift, 30.0)
    assert so.comp.n == su.comp.n
    assert len(so.comp.throats) == len(su.comp.throats) == 2
    for a, b in zip(sorted(su.comp.throats, key=lambda t: t.centroid[2]),
                    sorted(so.comp.throats, key=lambda t: t.centroid[2])):
        assert (a.a, a.b) == (b.a, b.b)
        assert b.area == pytest.approx(a.area, rel=1e-9)
        assert b.diameter == pytest.approx(a.diameter, rel=1e-9)
    assert ro == pytest.approx(ru, rel=0.02)


@pytest.mark.parametrize("levels", [0, 2])
def test_octree_explicit_hole_drains_as_uniform(levels):
    """An open cup drains through an explicit 12 mm hole in its bottom: the
    sub-cell links through the hole are cut as on the uniform grid (all flow
    through the orifice law), so the octree drains the same (identically
    with no coarse levels)."""
    h = 0.004
    lo, hi = _box(h, n=(64, 56, 48))
    m = _cup()
    hole = dict(center=(0.0217, -0.0023, -0.0289), diameter=0.012, axis=(0.0, 0.0, 1.0))
    res = []
    for grid in (Grid.from_mesh(m, h, bounds=(lo, hi)),
                 Octree.from_mesh(m, h, levels=levels, bounds=(lo, hi))):
        sim = Simulation(grid, cases.static(ndim=3, t_end=1.5), subcells=4, dt_max=0.05,
                         holes=[hole])
        inside = sim.fl & np.all(np.abs(sim.X - np.array([0.0017, -0.0023, 0.0011]))
                                 < np.array([0.05, 0.04, 0.03]), axis=1)
        sim.L = np.where(inside, 1.0, 0.0)
        res.append(np.array(sim.run().liquid_retained))
    u, o = res[0][1:], res[1][1:]                      # [0]: before the flooding
    assert u[-1] < 0.8 * u[0]                          # it drains
    tol = 1e-9 if levels == 0 else 0.01
    assert np.abs(o - u).max() <= tol * u[0]


@pytest.mark.parametrize("kind", ["uniform", "octree"])
def test_carve_holes_keeps_holes_inside_the_domain(kind):
    """A hole is skipped only if its disc leaves the grid (the margin is the
    radius, not the diameter): a 12 mm hole 8 mm from the boundary is kept,
    one 2 mm from it is skipped."""
    from drainsim.compartments import carve_holes
    h = 0.004
    lo, hi = _box(h, n=(64, 56, 48))
    m = _cup()
    grid = (Grid.from_mesh(m, h, bounds=(lo, hi)) if kind == "uniform"
            else Octree.from_mesh(m, h, levels=1, bounds=(lo, hi)))
    z = (0.0, 0.0, 1.0)
    near = dict(center=(lo[0] + 0.008, 0.0, 0.0), diameter=0.012, axis=(1.0, 0.0, 0.0))
    edge = dict(center=(lo[0] + 0.002, 0.0, 0.0), diameter=0.012, axis=(1.0, 0.0, 0.0))
    top = dict(center=(0.0, 0.0, hi[2] - 0.008), diameter=0.012, axis=z)
    out = (carve_holes(grid, [near, edge, top]) if kind == "uniform"
           else grid.carve_holes([near, edge, top]))
    assert [tuple(np.round(o["center"], 4)) for o in out] == \
        [tuple(np.round(near["center"], 4)), tuple(np.round(top["center"], 4))]


# ------------------------------------------------ narrow-passage refinement
def _lap_cup(ext, shift, z0, ov, g):
    """Open cup whose +x wall is two overlapping sheets with a gap g (a lap
    joint): the inner one up to z0, the outer one (at x + g) from z0 - ov
    to the rim; the channel between them is closed at its sides. Liquid
    above z0 can only leave through the channel."""
    hx, hy, hz = np.asarray(ext) / 2
    X = hx + g
    quads = [
        [(-hx, -hy, -hz), (hx, -hy, -hz), (hx, hy, -hz), (-hx, hy, -hz)],
        [(-hx, -hy, -hz), (-hx, hy, -hz), (-hx, hy, hz), (-hx, -hy, hz)],
        [(-hx, -hy, -hz), (X, -hy, -hz), (X, -hy, hz), (-hx, -hy, hz)],
        [(-hx, hy, -hz), (X, hy, -hz), (X, hy, hz), (-hx, hy, hz)],
        [(hx, -hy, -hz), (hx, hy, -hz), (hx, hy, z0), (hx, -hy, z0)],
        [(X, -hy, z0 - ov), (X, hy, z0 - ov), (X, hy, hz), (X, -hy, hz)],
    ]
    V, F = [], []
    for q in quads:
        b = len(V)
        V += list(q)
        F += [[b, b + 1, b + 2], [b, b + 2, b + 3]]
    m = trimesh.Trimesh(np.array(V, float), np.array(F), process=False)
    m.apply_translation(shift)
    return m


def _drain_level(m, h, ext, shift, z_ref, narrow):
    lo = np.asarray(m.vertices).min(0) - 0.024
    n = np.ceil((np.asarray(m.vertices).max(0) + 0.024 - lo) / (2 * h)).astype(int) * 2
    ot = Octree.from_mesh(m, h, levels=1, bounds=(lo, lo + n * h))
    sim = Simulation(ot, cases.static(ndim=3, t_end=3.0), subcells=2, dt_max=0.05,
                     narrow=narrow)
    inside = sim.fl & np.all(np.abs(sim.X - shift) < ext / 2, axis=1)
    sim.L = np.where(inside, 1.0, 0.0)
    sim.run()
    return sim, sim.hist.liquid_retained[-1] / (ext[0] * ext[1] * (z_ref + ext[2] / 2))


def test_narrow_refinement_opens_lap_joint_and_slot():
    """At h = 8 mm with 2 sub-cells (4 mm) a 3 mm lap-joint channel and a
    3 mm slot are shut; refining only the cells at them to 8 sub-cells
    (1 mm) opens both, and the cups drain to the channel's / slot's lower
    edge."""
    from test_drainsim import _slot_cup_mesh
    ext = np.array([0.1, 0.08, 0.06])
    shift = np.array([0.0017, -0.0023, 0.0011])
    z0 = ext[2] / 2 - 0.03
    lap = _lap_cup(ext, shift, z0, 0.02, 0.003)
    zs = ext[2] / 2 - 0.02
    slot = _slot_cup_mesh(ext, shift, zs, 0.003, 0.03)
    for m, zr in ((lap, z0), (slot, zs)):
        s0, r0 = _drain_level(m, 0.008, ext, shift, zr, None)
        s1, r1 = _drain_level(m, 0.008, ext, shift, zr, dict(k=8))
        assert r0 > 1.3                                  # shut: water to the rim
        assert r1 == pytest.approx(1.0, abs=0.04)        # open: drains to the edge
        nc = s1.volfrac_stats["narrow_cells"]
        assert 0 < nc < 0.2 * s1.volfrac_stats["closed_cells"]   # only there


def test_narrow_refinement_off_is_unchanged_and_keeps_volumes():
    """narrow=None gives the v6 graph; with refinement a closed cup keeps
    its volume (no leaks through the 2:1 sub-cell faces)."""
    ext = np.array([0.1, 0.08, 0.06])
    shift = np.array([0.0017, -0.0023, 0.0011])
    lap = _lap_cup(ext, shift, ext[2] / 2 - 0.03, 0.02, 0.003)
    lo = np.asarray(lap.vertices).min(0) - 0.024
    ot = Octree.from_mesh(lap, 0.008, levels=1, bounds=(lo, lo + np.array([20, 18, 14]) * 0.008))
    a = build_graph(ot, subcells=2)
    b = build_graph(ot, subcells=2, narrow=None)
    assert np.array_equal(a.indices, b.indices) and np.array_equal(a.vol, b.vol)
    c = build_graph(ot, subcells=2, narrow=dict(k=8))
    assert c.stats["narrow_cells"] > 0
    # total fluid volume of the domain is the same with and without
    assert c.vol[c.active].sum() == pytest.approx(a.vol[a.active].sum(), rel=0.01)
    # a cup without narrow passages: nothing refined, the same result
    cup = _cup()
    s0, r0 = _drain_level(cup, 0.008, ext, shift, ext[2] / 2, None)
    s1, r1 = _drain_level(cup, 0.008, ext, shift, ext[2] / 2, dict(k=8))
    assert s1.volfrac_stats["narrow_cells"] == 0 and r1 == r0


def test_narrow_refinement_skips_dead_end_crevice():
    """The lap joint closed at the bottom of its channel is a dead-end
    crevice: it connects no fluid regions, so its candidate cells are not
    refined (``connecting``, default) and the result is that of the base
    resolution; with connecting=False they are."""
    ext = np.array([0.1, 0.08, 0.06])
    shift = np.array([0.0017, -0.0023, 0.0011])
    z0, g, ov = ext[2] / 2 - 0.03, 0.003, 0.02
    hx, hy = ext[0] / 2, ext[1] / 2
    q = np.array([(hx, -hy, z0 - ov), (hx + g, -hy, z0 - ov), (hx + g, hy, z0 - ov),
                  (hx, hy, z0 - ov)]) + shift
    m = trimesh.util.concatenate([_lap_cup(ext, shift, z0, ov, g),
                                  trimesh.Trimesh(q, [[0, 1, 2], [0, 2, 3]], process=False)])
    s0, r0 = _drain_level(m, 0.008, ext, shift, z0, None)
    s1, r1 = _drain_level(m, 0.008, ext, shift, z0, dict(k=8))
    s2, _ = _drain_level(m, 0.008, ext, shift, z0, dict(k=8, connecting=False))
    assert s1.volfrac_stats["narrow_candidates"] > 0
    assert s1.volfrac_stats["narrow_cells"] == 0 and s1.N == s0.N and r1 == r0
    assert s2.volfrac_stats["narrow_cells"] == s2.volfrac_stats["narrow_candidates"]
    assert r0 > 1.3                                      # the cup stays full


def test_throat_axis_fallback_uses_the_axis_with_most_face_area(monkeypatch):
    """A throat whose net face area cancels (jagged opening) is projected
    along the axis with the largest total face area, not the axis of the one
    largest face: 6 faces along x and 2 along y, the first one along y."""
    from drainsim import gseg
    dirs = np.array([(0, 1, 0), (0, -1, 0)] + [(1, 0, 0)] * 3 + [(-1, 0, 0)] * 3, float)
    n = len(dirs)
    p = np.stack([10.0 * np.arange(n), np.zeros(n), np.zeros(n)], 1)
    X = np.concatenate([p, p + dirs])
    got = {}

    def fake_width(pos, nrm, fsize, nvec, s):
        got["nv"] = np.asarray(nvec)
        return 1.0
    monkeypatch.setattr(gseg, "raster_width", fake_width)
    gseg._throat(0, 1, np.arange(n), n + np.arange(n), X, np.ones(2 * n), 0.5)
    assert np.array_equal(got["nv"], [1.0, 0.0, 0.0])
