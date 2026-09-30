"""Suction (drainsim 6.3): the bath includes everything connected to it
without passing through the atmosphere (agent A's "submerged system"), so
an inverted glass lifted partly out of the bath keeps its water, and a gas
pocket above the bath level is under-pressured and expands."""
import numpy as np
import pytest

from drainsim import shapes as sh
from drainsim.grid import Grid
from drainsim.model import Simulation
from drainsim.motion import Keyframes
from drainsim.physics import Fluid

DX = 0.005


def _glass():
    """2D glass, open at the bottom: walls 1 cm, inside x in (-0.19, 0.19),
    y in (0.11, 0.44)."""
    g = Grid.empty([-0.5, -0.1], [0.5, 0.6], DX)
    sh.add(g, sh.shell_box(g, [-0.2, 0.1], [0.2, 0.45], 0.01, open_faces=[(1, -1)]))
    return g


def _in_glass(X):
    return (np.abs(X[:, 0]) < 0.19) & (X[:, 1] > 0.1) & (X[:, 1] < 0.44)


@pytest.mark.parametrize("comp", [False, True])
def test_inverted_glass_keeps_its_water_above_the_bath(comp):
    """The glass is full, its mouth under the bath, its upper part above the
    bath level: with suction it stays full; without (before 6.3) it drained
    as if air could get in."""
    g = _glass()
    res = {}
    for suction in (True, False):
        mo = Keyframes([0.0, 2.0], np.zeros(2), np.zeros((2, 2)), 2, bath_level=0.25)
        sim = Simulation(g, mo, dt_max=0.05, compressible_air=comp, suction=suction)
        ins = sim.fl & _in_glass(sim.X)
        sim.L = np.where(ins | sim.B, 1.0, 0.0)
        sim.G = np.where(sim.fl, (1.0 - sim.L) * sim.v, 0.0)
        sim.run()
        top = ins & (sim.X[:, 1] > 0.3)
        res[suction] = float((sim.L[top] * sim.v[top]).sum() / sim.v[top].sum())
    assert res[True] == pytest.approx(1.0, abs=1e-9)
    assert res[False] < 0.01


def test_full_glass_turned_under_water_and_lifted():
    """A glass is turned mouth-down under the bath (so it is full), then
    lifted until 60 % of it is above the bath level: it stays full. Lifted
    clear of the bath, it empties (air gets in through the mouth)."""
    g = _glass()
    # heights in the body frame; the bath rises and falls instead of the glass
    T = [0.0, 1.0, 2.0, 3.0, 4.0]
    zb = [0.8, 0.8, 0.30, 0.30, 0.0]
    # shifting the body down by s is a bath at +s in the body frame
    shifts = np.zeros((len(T), 2))
    shifts[:, 1] = -np.array(zb)
    mo = Keyframes(T, np.zeros(len(T)), shifts, 2, bath_level=0.0)
    sim = Simulation(g, mo, dt_max=0.02, compressible_air=True)
    ins = sim.fl & _in_glass(sim.X)
    top = ins & (sim.X[:, 1] > 0.35)
    # start: glass full (it went under mouth-up and was turned: its air is out)
    sim.L = np.where(ins | sim.B, 1.0, 0.0)
    sim.G = np.where(sim.fl, (1.0 - sim.L) * sim.v, 0.0)
    sim.run(t_end=3.0)
    assert sim.zb == pytest.approx(0.30, abs=1e-6)
    assert (sim.L[top] * sim.v[top]).sum() == pytest.approx(sim.v[top].sum(), rel=1e-9)
    sim.run(t_end=4.0)
    assert (sim.L[ins] * sim.v[ins]).sum() < 0.02 * sim.v[ins].sum()


def test_pocket_above_the_bath_is_under_pressured():
    """Gas at the top of an inverted glass whose upper part is above the
    bath: with compressible air its pressure is p_atm + rho g (z_b - z_i) <
    p_atm at its interface z_i, so it takes the volume G p_atm / p (Boyle).
    A dense liquid (rho = 20 000) makes the effect large."""
    g = _glass()
    fl = Fluid(rho=20000.0)
    zb = 0.15
    res = {}
    for comp in (False, True):
        mo = Keyframes([0.0, 1.0], np.zeros(2), np.zeros((2, 2)), 2, bath_level=zb)
        sim = Simulation(g, mo, dt_max=0.05, fluid=fl, compressible_air=comp)
        ins = sim.fl & _in_glass(sim.X)
        air = ins & (sim.X[:, 1] > 0.40)
        sim.L = np.where((ins & ~air) | (sim.B & ~ins), 1.0, 0.0)
        sim.G = np.where(sim.fl, (1.0 - sim.L) * sim.v, 0.0)
        G0 = float(sim.G[ins].sum())
        sim.run()
        Va = float(((1.0 - sim.L[ins]) * sim.v[ins]).sum())
        res[comp] = (G0, Va)
    G0, Va = res[False]
    assert Va == pytest.approx(G0, rel=1e-6)                 # incompressible
    G0, Va = res[True]
    # interface height from the pocket volume (glass inside 0.38 m wide)
    zi = 0.44 - Va / 0.38
    p = fl.p_atm + fl.rho * fl.g * (zb - zi)
    assert p < 0.8 * fl.p_atm
    assert Va == pytest.approx(G0 * fl.p_atm / p, rel=0.02)
    assert Va > 1.2 * G0
