"""drainsim 7.1: gap channels (drainsim.channels): narrow gaps the grid
closes, as compartments of their own joined by slot throats.
Run: python -m pytest -q tests/test_channels.py"""
import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")
from drainsim import cases, openings as op                      # noqa: E402
from drainsim.model import Simulation                           # noqa: E402
from drainsim.octree import Octree                              # noqa: E402
from drainsim.physics import Fluid, ThroatModel                 # noqa: E402

T = 0.001                                                        # plate thickness


def _box(lo, hi):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    b = trimesh.creation.box(extents=hi - lo)
    b.apply_translation(0.5 * (lo + hi))
    return b


def _cup(gap=0.004, overlap=0.04):
    """An open cup 200 x 100 x 100 mm of 1 mm plates whose floor is a
    joggled seam: the lower plate (x < 80 + overlap mm) and the upper plate
    (x > 80 mm) overlap with a gap between them, a channel from the inside
    of the cup to the outside below it (closed at its ends by the walls)."""
    x1 = 0.08 + overlap
    return trimesh.util.concatenate([
        _box((0, 0, 0), (x1, 0.1, T)),
        _box((0.08, 0, T + gap), (0.2, 0.1, 2 * T + gap)),
        _box((-T, -T, 0), (0, 0.1 + T, 0.1)),
        _box((0.2, -T, T + gap), (0.2 + T, 0.1 + T, 0.1)),
        _box((-T, -T, 0), (0.2 + T, 0, 0.1)),
        _box((-T, 0.1, 0), (0.2 + T, 0.1 + T, 0.1)),
    ])


def _octree(m, h=0.01):
    lo = np.array([-0.05, -0.04, -0.06])
    n = np.ceil((np.array([0.25, 0.14, 0.17]) - lo) / h).astype(int)
    return Octree.from_mesh(m, h, levels=1, bounds=(lo, lo + n * h))


def _sim(m, channels, t_end=12.0, fill=True):
    ot = _octree(m)
    spec = None
    if channels:
        V, F = np.asarray(m.vertices, float), np.asarray(m.faces, np.int64)
        spec = dict(samples=op.gap_samples(V, F, +1, spacing=0.002), lo=0.003, hi=0.02,
                    verbose=False)
    sim = Simulation(ot, cases.static(ndim=3, t_end=t_end), subcells=2, dt_max=0.05,
                     channels=spec)
    if not fill:
        return sim
    inside = (sim.X[:, 0] > 0.001) & (sim.X[:, 0] < 0.199) & (sim.X[:, 1] > 0.001) & \
        (sim.X[:, 1] < 0.099) & (sim.X[:, 2] > 0.008) & (sim.X[:, 2] < 0.06) & sim.fl & \
        (np.arange(sim.N) < sim.ncells + (sim.N - sim.ncells - (sim.channels.v.size
                                                               if sim.channels else 0)))
    return Simulation(ot, cases.static(ndim=3, t_end=t_end), subcells=2, dt_max=0.05,
                      channels=spec, initial_L=inside.astype(float))


def _held(sim):
    return float((sim.L * sim.v)[sim.fl & ~sim.B].sum())


def test_the_seam_is_one_channel_with_two_mouths():
    sim = _sim(_cup(), True, fill=False)
    ch = sim.channels
    assert ch.n == 1
    assert ch.v.sum() == pytest.approx(0.04 * 0.1 * 0.004, rel=0.02)    # overlap x gap
    assert len(ch.mouths) == 2
    for mo in ch.mouths:
        assert mo["compartment"] == 0                          # both open to the outside
        assert np.sum(mo["weights"]) == pytest.approx(0.1 * 0.004, rel=0.1)   # 100 x 4 mm
        assert mo["width"] == pytest.approx(0.004, rel=0.01)
    xs = sorted(float(mo["centroid"][0]) for mo in ch.mouths)
    assert xs[0] == pytest.approx(0.08, abs=0.01) and xs[1] == pytest.approx(0.12, abs=0.01)
    # the channel is a compartment of its own, joined only by its mouths
    assert sim.comp.n == 2 and len(sim.comp.throats) == 2
    assert all(t.slot for t in sim.comp.throats)
    ptr, idx = sim.nbr
    N0 = sim.N - ch.v.size
    for c in range(N0, sim.N):
        nb = idx[ptr[c]:ptr[c + 1]]
        assert np.all((nb < 0) | (nb >= N0))


def test_without_channels_the_cup_holds():
    sim = _sim(_cup(), False, t_end=4.0)
    v0 = _held(sim)
    sim.run(t_end=4.0)
    assert v0 > 0.9e-3 and _held(sim) == pytest.approx(v0, rel=1e-9)


def test_through_the_channel_the_cup_drains_by_the_orifice_law():
    sim = _sim(_cup(), True)
    v0 = _held(sim)
    hist = [(0.0, v0)]
    while sim.t < 12.0 - 1e-9:
        sim.run(t_end=sim.t + 1.0)
        hist.append((sim.t, _held(sim)))
    t, v = np.array(hist).T
    assert v0 + 0.0 == pytest.approx(v[-1] + sim.drained_total, rel=1e-9)   # conserved
    dv = -np.diff(v)
    assert np.all(dv[:6] > 0) and dv[0] > 1.5 * dv[5]          # slows as the level falls
    # two orifices in series (inlet, outlet): a = A / sqrt(2); a prismatic
    # tank of 200 x 100 mm with its level h above the outlet:
    # sqrt(h) falls linearly, sqrt(h0) / (Cd a / A_tank sqrt(g / 2)) to empty
    tm, fl = ThroatModel(), Fluid()
    a = 0.1 * 0.004 / np.sqrt(2.0)
    A = 0.2 * 0.1
    # the level over the seam's mid-gap (z = 3 mm); the floor is 1 mm at
    # x < 80 mm and 6 mm beyond (8e-5 m^3 under a level above both)
    h0 = (v0 + 8e-5) / A - 0.003
    hold = tm.holdup_head(0.004, fl, 2)
    k = tm.Cd * a / A * np.sqrt(fl.g / 2.0)
    t_half = (np.sqrt(h0 - hold) - np.sqrt(h0 - 0.5 * v0 / A - hold)) / k
    assert np.interp(0.5 * v0, v[::-1], t[::-1]) == pytest.approx(t_half, rel=0.1)
    t_empty = np.sqrt(h0 - hold) / k                            # the prismatic tank
    assert np.interp(0.05 * v0, v[::-1], t[::-1]) == pytest.approx(t_empty, rel=0.2)


def test_gaps_on_both_sides_of_a_plate_are_two_channels():
    """Three stacked plates with 4 mm gaps: the two gaps are separate
    channels (not joined through the middle plate)."""
    m = trimesh.util.concatenate([
        _box((0, 0, 0), (0.1, 0.1, T)),
        _box((0, 0, T + 0.004), (0.1, 0.1, 2 * T + 0.004)),
        _box((0, 0, 2 * T + 0.008), (0.1, 0.1, 3 * T + 0.008)),
    ])
    ot = _octree(m)
    V, F = np.asarray(m.vertices, float), np.asarray(m.faces, np.int64)
    sim = Simulation(ot, cases.static(ndim=3, t_end=0.1), subcells=2,
                     channels=dict(samples=op.gap_samples(V, F, +1, spacing=0.002),
                                   lo=0.003, hi=0.02, verbose=False))
    ch = sim.channels
    assert ch.n == 2
    assert np.allclose(np.bincount(ch.channel, weights=ch.v), 0.1 * 0.1 * 0.004, rtol=0.02)
    for k in range(2):
        z = ch.X[ch.channel == k, 2]
        assert np.ptp(z) < 1e-6                                 # each in its own gap
    assert not np.any(ch.channel[ch.links[:, 0]] != ch.channel[ch.links[:, 1]])


def test_export_channels(tmp_path):
    from drainsim.channels import export_channels
    sim = _sim(_cup(), True, fill=False)
    pre = str(tmp_path / "cup")
    export_channels(pre, sim.channels, sim.comp.n - sim.channels.n)
    rows = open(pre + "_channels.csv").read().splitlines()
    assert len(rows) == 2 and rows[1].split(",")[7] == "0"         # joins the outside
    vol = float(rows[1].split(",")[3])
    assert vol == pytest.approx(16.0, rel=0.02)                    # ml
    import os
    assert os.path.getsize(pre + "_channels.vtp") > 0 and os.path.getsize(pre + "_mouths.vtp") > 0
