"""Fill-spill-merge on voxel scenes with exact answers (drainsim 6.3).

Ported from agent A's ``tests/test_spill.py`` (the compare_agents review):
gravity along a grid axis gives exact cell counts, oblique gravity is
checked for conservation. The answers are A's, recomputed by hand in the
comments; drainsim's ``fsm.fill_spill`` must give the same.
"""
import numpy as np
import pytest

from drainsim.fsm import fill_spill
from drainsim.grid import Grid

Z_DOWN = (0, 0, -1)


def _h(shape, g):
    """Heights h = -g . x (cell index coordinates)."""
    I, J, K = np.meshgrid(*[np.arange(n, dtype=float) for n in shape], indexing="ij")
    g = np.asarray(g, float)
    return np.round(-(g[0] * I + g[1] * J + g[2] * K).ravel(), 9)


def _settle(solid, f, g, spill_routing=True):
    """One fill-spill of the liquid fractions f (volumes in cells) with the
    domain boundary as the sink. Returns (retained field, retained, gone)."""
    shape = solid.shape
    grid = Grid(shape=shape, dx=1.0, origin=np.zeros(3), solid=solid.copy())
    fl = ~solid.ravel()
    region = np.where(fl, 0, -1)
    bnd = np.zeros(shape, bool)
    bnd[[0, -1], :, :] = bnd[:, [0, -1], :] = bnd[:, :, [0, -1]] = True
    sink = bnd.ravel() & fl
    src = np.where(fl & ~sink, np.asarray(f, float).ravel(), 0.0)
    ret, drained, lost, _ = fill_spill(_h(shape, g), region, grid.neighbors(), sink, src,
                                       1.0, spill_routing, nregions=1)
    assert lost.sum() == pytest.approx(0.0, abs=1e-9)
    return ret.reshape(shape), float(ret.sum()), float(drained.sum())


def _bottle(shape=(14, 14, 14)):
    """Closed 8x8x8 box (interior 3..10) with a 4x4 neck in its top wall."""
    solid = np.zeros(shape, bool)
    solid[2:12, 2:12, 2:12] = True
    solid[3:11, 3:11, 3:11] = False
    solid[5:9, 5:9, 11] = False
    return solid


@pytest.mark.parametrize("seed", range(4))
def test_random_geometry_conserves_and_is_a_fixed_point(seed):
    rng = np.random.default_rng(seed)
    shape = (18, 16, 20)
    solid = rng.random(shape) < 0.3
    f = np.where(~solid, rng.random(shape), 0.0)
    g = rng.normal(size=3)
    g /= np.linalg.norm(g)
    F, ret, gone = _settle(solid, f, g)
    inner = f.copy()
    inner[[0, -1], :, :] = inner[:, [0, -1], :] = inner[:, :, [0, -1]] = 0.0
    assert ret + gone == pytest.approx(inner.sum(), rel=1e-12)
    assert F.min() >= 0.0 and F.max() <= 1.0 + 1e-12
    F2, ret2, gone2 = _settle(solid, F, g)                 # settled: nothing moves
    assert gone2 == pytest.approx(0.0, abs=1e-9)
    np.testing.assert_allclose(F2, F, atol=1e-12)


def test_sealed_box_settles_into_its_lowest_cells_under_oblique_gravity():
    shape = (12, 12, 12)
    solid = np.ones(shape, bool)
    solid[1:11, 1:11, 1:11] = False
    f = np.zeros(shape)
    f[1:11, 1:11, 1:4] = 1.0                               # 300 cells
    g = (0.31, 0.53, -0.79)
    F, ret, gone = _settle(solid, f, g)
    assert gone == 0.0 and ret == pytest.approx(300)
    h = _h(shape, g)
    inside = np.flatnonzero(~solid.ravel())
    lowest = inside[np.argsort(h[inside], kind="stable")[:300]]
    assert F.ravel()[lowest].sum() == pytest.approx(300)
    # turned upside down, it all moves to the new bottom
    F, ret, _ = _settle(solid, f, (0, 0, 1))
    assert F[1:11, 1:11, 8:11].sum() == pytest.approx(300)


def test_bottle_on_its_side_keeps_the_liquid_below_the_neck():
    """Full: the neck spans x = 5..8, so 2 x 8 x 8 cells stay. A third full
    (192 cells): it settles first, then 128 stay and 64 go."""
    solid = _bottle()
    f = np.zeros(solid.shape)
    f[3:11, 3:11, 3:11] = 1.0
    _, ret, gone = _settle(solid, f, (-1, 0, 0))
    assert ret == pytest.approx(128) and gone == pytest.approx(384)
    f = np.zeros(solid.shape)
    f[3:11, 3:11, 3:6] = 1.0
    F, ret, gone = _settle(solid, f, (-1, 0, 0))
    assert ret == pytest.approx(128) and gone == pytest.approx(64)
    assert F[3:5, 3:11, 3:11].sum() == pytest.approx(128)


def test_flat_plate_holds_nothing_but_a_rimmed_tray_does():
    shape = (20, 20, 12)
    solid = np.zeros(shape, bool)
    solid[5:15, 5:15, 5] = True
    f = np.zeros(shape)
    f[6:14, 6:14, 6] = 1.0
    _, ret, gone = _settle(solid, f, Z_DOWN)
    assert ret == 0.0 and gone == pytest.approx(64)
    solid[5:15, 5:15, 6] = True
    solid[6:14, 6:14, 6] = False                           # one-cell rim
    _, ret, gone = _settle(solid, f, Z_DOWN)
    assert ret == pytest.approx(64) and gone == 0.0


def _cascade(cup_depth=5, second_cup=False):
    """A box on its side with a neck in its +x wall, an open cup under the
    neck, and optionally a notch in the cup's rim with a second cup below."""
    shape = (44, 12, 40)
    solid = np.zeros(shape, bool)
    solid[4:21, 1:11, 20:33] = True
    solid[5:20, 2:10, 21:32] = False
    solid[20, 4:8, 26:30] = False
    top = 12 + cup_depth
    solid[20:33, 1:11, 12:top + 1] = True
    solid[21:32, 2:10, 13:top + 1] = False
    if second_cup:
        solid[32, 4:8, top] = False
        solid[32:40, 1:11, 2:8] = True
        solid[33:39, 2:10, 3:8] = False
    f = np.zeros(shape)
    f[5:20, 2:10, 21:32] = 1.0                             # 1320 cells
    return solid, f


def test_spill_cascades_into_the_cup_below():
    solid, f = _cascade(5)
    F, ret, gone = _settle(solid, f, Z_DOWN)
    assert F[5:20, 2:10, 21:32].sum() == pytest.approx(15 * 8 * 5)     # below the neck
    assert F[21:32, 2:10, 13:18].sum() == pytest.approx(11 * 8 * 5)    # cup full
    assert gone == pytest.approx(1320 - 600 - 440)
    # without spill routing the cup stays empty
    F, ret, gone = _settle(solid, f, Z_DOWN, spill_routing=False)
    assert F[21:32, 2:10, 13:18].sum() == 0.0 and gone == pytest.approx(720)


def test_cascade_partly_fills_a_deep_cup_from_the_bottom():
    solid, f = _cascade(10)                                # room for 880 > 720
    F, ret, gone = _settle(solid, f, Z_DOWN)
    cup = F[21:32, 2:10, 13:23]
    assert gone == pytest.approx(0.0, abs=1e-9)
    assert cup.sum() == pytest.approx(720)
    assert cup[:, :, :8].sum() == pytest.approx(8 * 88)   # 8 full layers + 16
    assert cup[:, :, 9:].sum() == 0.0


def test_two_stage_cascade():
    solid, f = _cascade(5, second_cup=True)
    F, ret, gone = _settle(solid, f, Z_DOWN)
    assert F[21:32, 2:10, 13:18].sum() == pytest.approx(11 * 8 * 4)   # notch at z = 17
    assert F[33:39, 2:10, 3:8].sum() == pytest.approx(6 * 8 * 5)       # second cup full
    assert gone == pytest.approx(720 - 352 - 240)


def _tank(dividers):
    """Open tank, interior x 1..20, y 1..4, z 1..15; dividers (x0, x1, top)."""
    shape = (22, 6, 18)
    solid = np.zeros(shape, bool)
    solid[0:22, 0:6, 0:16] = True
    solid[1:21, 1:5, 1:16] = False
    for x0, x1, zt in dividers:
        solid[x0:x1 + 1, 1:5, 1:zt + 1] = True
    return solid


def _pour(solid, region, volume):
    f = np.zeros(solid.shape)
    cells = np.argwhere(region & ~solid)
    cells = cells[np.lexsort((cells[:, 1], cells[:, 0], cells[:, 2]))]
    n = int(volume)
    for c in cells[:n]:
        f[tuple(c)] = 1.0
    if volume > n:
        f[tuple(cells[n])] = volume - n
    return f


def test_overflow_fills_the_neighbour_then_the_merged_lake_rises():
    solid = _tank([(10, 11, 6)])
    A = np.zeros(solid.shape, bool)
    A[1:10] = True
    F, ret, gone = _settle(solid, _pour(solid, A, 216 + 216 + 80), Z_DOWN)
    assert gone == 0.0
    assert F[1:10, 1:5, 1:7].sum() == pytest.approx(216)
    assert F[12:21, 1:5, 1:7].sum() == pytest.approx(216)
    assert F[1:21, 1:5, 7].sum() == pytest.approx(80)
    assert F[:, :, 8:].sum() == 0.0


@pytest.mark.parametrize("overflow, b1, b2, above", [
    (40, 40, 0, 0),        # all lands in B1, next to the saddle
    (60, 48, 12, 0),       # B1 full, then over the low divider into B2
    (150, 48, 48, 54),     # both full, the merged B lake rises
])
def test_overflow_lands_next_to_the_saddle(overflow, b1, b2, above):
    solid = _tank([(10, 11, 6), (16, 16, 3)])
    A = np.zeros(solid.shape, bool)
    A[1:10] = True
    F, ret, gone = _settle(solid, _pour(solid, A, 216 + overflow), Z_DOWN)
    assert gone == 0.0
    assert F[1:10, 1:5, 1:7].sum() == pytest.approx(216)
    assert F[12:16, 1:5, 1:4].sum() == pytest.approx(b1)
    assert F[17:21, 1:5, 1:4].sum() == pytest.approx(b2)
    assert F[12:21, 1:5, 4:7].sum() == pytest.approx(above)


def test_sealed_chamber_never_feeds_its_neighbour():
    shape = (14, 6, 10)
    solid = np.ones(shape, bool)
    solid[1:6, 1:5, 1:9] = False
    solid[8:13, 1:5, 1:9] = False
    f = np.zeros(shape)
    f[1:6, 1:5, 1:9] = 1.0
    F, ret, gone = _settle(solid, f, (0.6, 0, -0.8))
    assert gone == 0.0 and ret == pytest.approx(160)
    assert F[8:13].sum() == 0.0


@pytest.mark.parametrize("g", [Z_DOWN, (0.3, 0.0, -0.95)])
def test_mirrored_scene_gives_the_mirrored_result(g):
    """Ties are broken by cell index, but the result may not depend on it:
    the cascade mirrored in y (and in x with gravity mirrored too) gives
    the mirrored field."""
    solid, f = _cascade(5, second_cup=True)
    F, ret, gone = _settle(solid, f, g)
    Fy, rety, goney = _settle(solid[:, ::-1], f[:, ::-1], g)
    np.testing.assert_allclose(Fy[:, ::-1], F, atol=1e-9)
    gx = (-g[0], g[1], g[2])
    Fx, retx, gonex = _settle(solid[::-1], f[::-1], gx)
    np.testing.assert_allclose(Fx[::-1], F, atol=1e-9)
