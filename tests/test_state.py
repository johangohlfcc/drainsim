"""drainsim 7.2: save and load the state of a simulation and go on exactly
as if it had not stopped.  Run: python -m pytest -q tests/test_state.py"""
import numpy as np
import pytest

from drainsim import cases
from drainsim.grid import Grid
from drainsim.model import Simulation


def _hist(sim):
    return {k: np.asarray(v) for k, v in vars(sim.hist).items()}


def _same(a, b):
    assert a.t == b.t
    assert np.array_equal(a.L, b.L) and np.array_equal(a.B, b.B) and np.array_equal(a.A, b.A)
    ha, hb = _hist(a), _hist(b)
    for k in ha:
        assert np.array_equal(ha[k], hb[k]), k
    if a.film is not None:
        assert np.array_equal(a.film.s.h, b.film.s.h)
        assert a.film.drip_volume == b.film.drip_volume


@pytest.mark.parametrize("film", [False, True])
def test_save_load_goes_on_exactly(tmp_path, film):
    """A box with a hole dipped and lifted: run to 3 s, save, load, run on
    to 8 s; the same as running to 8 s at once (liquid, bath, air, history,
    film), also through a pose that stays still (cached fill-spill)."""
    trimesh = pytest.importorskip("trimesh")
    from drainsim.motion import dip

    def box(lo, hi):
        b = trimesh.creation.box(extents=np.subtract(hi, lo))
        b.apply_translation(0.5 * (np.asarray(lo) + np.asarray(hi)))
        return b
    t = 0.003                                      # an open cup of 3 mm plates
    m = trimesh.util.concatenate([
        box((-0.05, -0.04, -0.03), (0.05, 0.04, -0.03 + t)),
        box((-0.05, -0.04, -0.03), (-0.05 + t, 0.04, 0.03)),
        box((0.05 - t, -0.04, -0.03), (0.05, 0.04, 0.03)),
        box((-0.05, -0.04, -0.03), (0.05, -0.04 + t, 0.03)),
        box((-0.05, 0.04 - t, -0.03), (0.05, 0.04, 0.03))])
    g = Grid.from_mesh(m, dx=0.006, pad=0.03)
    mo = dip(ndim=3, depth=0.12, start_height=0.05, t_down=2.0, t_hold=1.0, t_up=2.0,
             t_after=3.0, tilt=[12.0, 8.0, 0.0])
    kw = dict(dt_max=0.1, subcells=2, film=film)
    a = Simulation(g, mo, **kw)
    a.run(t_end=8.0)
    b = Simulation(g, mo, **kw)
    b.run(t_end=3.0)
    p = tmp_path / "state.pkl"
    b.save_state(p, extra=dict(step=30))
    c, extra = Simulation.load_state(p)
    assert extra == dict(step=30)
    c.run(t_end=8.0)
    _same(a, c)
    assert a.hist.liquid_retained[-1] > 1e-5                 # the cup holds liquid
    if film:
        assert a.film.volume > 0
