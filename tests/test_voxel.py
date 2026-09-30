"""Cut-cell detection (drainsim 6.3): exact triangle-box test.

Before 6.3 the cut cells were found by sampling points on the triangles.
That misses cells a sheet only grazes; such a cell stayed a whole fluid
cell (its volume beyond the sheet counted as fluid), and next to the
sub-cells of a cut cell it could join the fluid on both sides of the sheet
(the article door drained through its inner panel at 8 sub-cells per edge).
"""
import numpy as np
import pytest
import trimesh
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from drainsim import cases
from drainsim.grid import Grid, sample_triangles
from drainsim.model import Simulation
from drainsim.octree import Octree
from drainsim.voxel import cut_cells, surface_cells, tri_box_overlap


def _keys(idx, dims):
    return (idx[:, 0] * dims[1] + idx[:, 1]) * dims[2] + idx[:, 2]


def _sampled(tri, origin, dx, dims, spacing=0.45):
    acc = []
    for P in sample_triangles(tri, dx, spacing):
        idx = np.floor((P - origin) / dx).astype(np.int64)
        ok = np.all((idx >= 0) & (idx < dims), axis=1)
        acc.append(_keys(idx[ok], dims))
    return np.unique(np.concatenate(acc))


def test_surface_cells_match_brute_force():
    """The blockwise kernel gives exactly the cells whose (slightly
    enlarged) box the separating-axis test says a triangle touches."""
    rng = np.random.default_rng(1)
    dims = np.array([13, 11, 9])
    dx = 0.1
    origin = np.array([-0.05, 0.02, 0.01])
    tri = origin + rng.random((60, 3, 3)) * dims * dx
    tri[:5] *= [1.0, 1.0, 0.0]                          # some flat / clipped ones
    tri[5] = tri[5, [0, 0, 1]]                          # a degenerate one
    keys = surface_cells(tri, origin, dx, dims, block=17)
    bf = []
    for i in range(dims[0]):
        for j in range(dims[1]):
            for k in range(dims[2]):
                c = origin + (np.array([i, j, k]) + 0.5) * dx
                if any(tri_box_overlap(c, 0.5 * dx * (1 + 1e-6), *t) for t in tri):
                    bf.append((i * dims[1] + j) * dims[2] + k)
    assert np.array_equal(keys, np.array(bf))


def test_cut_cells_contain_samples_and_catch_grazes():
    """Every sampled cell is cut, and so is a cell that a sheet enters only
    near a corner, which the 0.45 dx samples miss."""
    dx = 0.01
    dims = np.array([6, 6, 6])
    origin = np.zeros(3)
    # the plane x + y + z = 2.8 dx cuts the corner (1, 1, 1) dx off the cell
    # (0, 0, 0), 0.2 dx / sqrt(3) deep
    s = 2.8 * dx
    tri = np.array([[[s, 0, 0], [0, s, 0], [0, 0, s]]], float)
    exact = cut_cells(tri, origin, dx, dims)
    samp = _sampled(tri, origin, dx, dims)
    assert np.isin(samp, exact).all()
    assert 0 in set(exact.tolist())                    # cell (0, 0, 0)
    # the interior test finds it by itself too
    assert 0 in set(surface_cells(tri, origin, dx, dims, grow=-1e-9).tolist())


def test_sheet_on_a_cell_face_adds_no_cells():
    """A sheet lying exactly on a cell face touches the closed boxes on
    both sides but the interior of neither, so the exact test adds nothing
    to the sampled cells (a wall on a face does not become two cells thick,
    which would lose the fluid of the corner cells)."""
    dx = 0.01
    dims = np.array([5, 5, 5])
    origin = np.zeros(3)
    z = 2 * dx                                          # the face between k = 1 and 2
    tri = np.array([[[0, 0, z], [5 * dx, 0, z], [5 * dx, 5 * dx, z]],
                    [[0, 0, z], [5 * dx, 5 * dx, z], [0, 5 * dx, z]]], float)
    assert surface_cells(tri, origin, dx, dims, grow=-1e-9).size == 0
    assert np.array_equal(cut_cells(tri, origin, dx, dims), _sampled(tri, origin, dx, dims))
    # the closed-box test does mark both layers
    assert set((surface_cells(tri, origin, dx, dims) % dims[2]).tolist()) == {1, 2}


def _divided_box(ang, off, t=0.0007):
    """A sealed box split by a plate (two faces t apart) tilted by ang about
    x through the centre shifted by off along its normal."""
    hx, hy, hz = 0.06, 0.05, 0.05
    x0, y0, z0, x1, y1, z1 = -hx, -hy, -hz, hx, hy, hz
    quads = [[(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0)],
             [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)],
             [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)],
             [(x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1)],
             [(x0, y0, z0), (x0, y1, z0), (x0, y1, z1), (x0, y0, z1)],
             [(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)]]
    a = np.radians(ang)
    n = np.array([0.0, -np.sin(a), np.cos(a)])
    for s in (-0.5 * t, 0.5 * t):
        ys = np.array([y0, y1])
        zs = (s + off - n[1] * ys) / n[2]
        quads.append([(x0, ys[0], zs[0]), (x1, ys[0], zs[0]), (x1, ys[1], zs[1]),
                      (x0, ys[1], zs[1])])
    V, F = [], []
    for q in quads:
        b = len(V)
        V += list(q)
        F += [[b, b + 1, b + 2], [b, b + 2, b + 3]]
    return trimesh.Trimesh(np.array(V, float), np.array(F), process=False), n


def _plate_leaks(cut, ang, off, k=8, levels=2, h=0.012):
    m, n = _divided_box(ang, off)
    lo = np.array([-0.06, -0.05, -0.05]) - 2 * h + 0.0013
    N = np.ceil((0.12 + 4 * h) / (h * 2 ** levels)) * 2 ** levels
    ot = Octree.from_mesh(m, h, levels=levels, bounds=(lo, lo + N * h), cut=cut)
    sim = Simulation(ot, cases.static(ndim=3, t_end=0.1), subcells=k, dt_max=0.05)
    inside = sim.fl & np.all(np.abs(sim.X) < [0.06, 0.05, 0.05], axis=1)
    side = sim.X @ n - off
    ptr, idx = sim.nbr
    rows = np.repeat(np.arange(sim.N), np.diff(ptr))
    ok = idx >= 0
    A = coo_matrix((np.ones(ok.sum()), (rows[ok], idx[ok])), shape=(sim.N, sim.N))
    _, lab = connected_components(A, directed=False)
    a = set(lab[inside & (side < -0.002)].tolist())
    b = set(lab[inside & (side > 0.002)].tolist())
    return bool(a & b)


@pytest.mark.parametrize("ang,off", [(55.0, -0.002), (60.0, 0.002)])
def test_tilted_plate_does_not_leak_with_fine_subcells(ang, off):
    """A 0.7 mm plate tilted like the door's inner panel splits a sealed box
    into two chambers. With 8 sub-cells per edge the sampled cut cells join
    the chambers (a grazed fluid cell next to far-side sub-cells); the exact
    cut cells keep them apart."""
    assert _plate_leaks("sample", ang, off)             # the pre-6.3 leak
    assert not _plate_leaks("exact", ang, off)


def test_uniform_grid_uses_exact_cut_cells():
    """Grid.from_mesh(bounds=...) marks the same cells as the octree's
    level 0."""
    m, _ = _divided_box(40.0, 0.001)
    h = 0.01
    lo = np.array([-0.08, -0.07, -0.07]) + 0.0013
    hi = lo + np.array([16, 14, 14]) * h
    g = Grid.from_mesh(m, h, bounds=(lo, hi))
    ot = Octree.from_mesh(m, h, levels=0, bounds=(lo, hi))
    assert np.array_equal(np.flatnonzero(g.solid.ravel()), ot.cut)


# ------------------------------------------------ surface sub-cell volumes
def _hollow_box(inner, t, seed):
    """Closed box with walls of thickness t (outer and inner surface),
    turned randomly; the fluid inside has the exact volume prod(inner)."""
    from scipy.spatial.transform import Rotation
    rng = np.random.default_rng(seed)
    a = trimesh.creation.box(extents=np.asarray(inner) + 2 * t)
    b = trimesh.creation.box(extents=inner)
    m = trimesh.util.concatenate([a, b])
    R = np.eye(4)
    R[:3, :3] = Rotation.random(random_state=seed).as_matrix()
    shift = rng.random(3) * 0.01
    m.apply_transform(R)
    m.apply_translation(shift)
    return m, shift


def _inner_volume(m, h, k, share, centre, octree=True):
    from drainsim.octree import build_graph
    from drainsim.volfrac import cut_cell_graph
    lo = np.asarray(m.vertices).min(0) - 2 * h
    n = np.ceil((np.asarray(m.vertices).max(0) + 2 * h - lo) / h).astype(int)
    if octree:
        ot = Octree.from_mesh(m, h, levels=0, bounds=(lo, lo + n * h))
        g = build_graph(ot, k, share=share)
        N, X, vol, act = g.X.shape[0], g.X, g.vol, g.active
        rows = np.repeat(np.arange(N), np.diff(g.indptr))
        cols = g.indices
    else:
        gr = Grid.from_mesh(m, h, bounds=(lo, lo + n * h))
        cg = cut_cell_graph(gr, k, share=share)
        nc = gr.ncells
        N, X, vol = cg["vol"].size, cg["X"], cg["vol"]
        act = cg["active"].copy()
        act[:nc] &= gr.fluid.ravel()
        base = gr.neighbors()
        bi = np.repeat(np.arange(nc), base.shape[1])
        bj = base.ravel()
        ok = bj >= 0
        rows = np.r_[bi[ok], cg["edges"][:, 0]]
        cols = np.r_[bj[ok], cg["edges"][:, 1]]
    A = coo_matrix((np.ones(rows.size), (rows, cols)), shape=(N, N))
    _, lab = connected_components(A, directed=False)
    ia = np.flatnonzero(act)
    c = lab[ia[np.argmin(np.linalg.norm(X[ia] - centre, axis=1))]]
    return vol[(lab == c) & act].sum()


@pytest.mark.parametrize("seed", [11, 12, 13])
def test_hollow_box_volume_by_sight(seed):
    """The fluid volume inside a turned box with 0.7 mm walls, 12 cells
    across at h = 8 mm with 4 sub-cells per edge: within 0.3 % by line of
    sight (the metal dropped, the corner cells refined), while the equal
    share counts part of the metal (over-estimate)."""
    inner = np.array([0.1, 0.08, 0.06])
    m, c = _hollow_box(inner, 0.0007, seed)
    V = inner.prod()
    sight = _inner_volume(m, 0.008, 4, "sight", c)
    equal = _inner_volume(m, 0.008, 4, "equal", c)
    assert sight == pytest.approx(V, rel=0.003)
    assert equal > sight
    # the uniform grid gives the same
    assert _inner_volume(m, 0.008, 4, "sight", c, octree=False) == pytest.approx(sight, rel=1e-9)


def test_corner_cells_are_refined():
    """Closed cells that touch the fluid only along an edge or a corner get
    sub-cells too: without them a turned box at 12 cells across loses 1-6 %
    of its volume in its corners and along its edges."""
    import drainsim.octree as O
    inner = np.array([0.1, 0.08, 0.06])
    m, c = _hollow_box(inner, 0.0007, 11)
    V = inner.prod()
    got = _inner_volume(m, 0.008, 4, "sight", c)
    O.REFINE_EDGE_NEIGHBOURS = False
    try:
        old = _inner_volume(m, 0.008, 4, "sight", c)
    finally:
        O.REFINE_EDGE_NEIGHBOURS = True
    assert got == pytest.approx(V, rel=0.003)
    assert old < got - 0.01 * V


def test_plate_metal_is_dropped():
    """A closed 2 mm plate across a sealed box: the fluid on each side is
    its geometric volume (the plate's volume goes to neither side)."""
    m, n = _divided_box(35.0, 0.004, t=0.002)
    h = 0.008
    lo = np.array([-0.06, -0.05, -0.05]) - 2 * h + 0.0013
    N = np.ceil((0.12 + 4 * h) / h)
    ot = Octree.from_mesh(m, h, levels=0, bounds=(lo, lo + N * h))
    from drainsim.octree import build_graph
    for share in ("sight", "equal"):
        g = build_graph(ot, 4, share=share)
        inside = np.all(np.abs(g.X) < [0.06, 0.05, 0.05], axis=1) & g.active
        tot = g.vol[inside].sum()
        Vbox = 0.12 * 0.10 * 0.10
        # the plate crosses the whole box: its volume is t * (its area in the box)
        a = np.radians(35.0)
        plate = 0.002 * 0.12 * 0.10 / np.cos(a)
        if share == "sight":
            assert tot == pytest.approx(Vbox - plate, rel=0.002)
            sight = tot
        else:
            assert tot > sight + 0.2 * plate          # part of the metal counted
