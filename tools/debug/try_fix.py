"""Try a change on a saved state: load it, hold the pose of that moment (the
car still), apply the change, run on, and follow the biggest pockets.

    python tools/debug/try_fix.py STATE [--hold S] [--kind air|liquid] [--top 10]
           [--rt] [--press] [--necks]

--rt: ThroatModel.rt_orientation (the Rayleigh-Taylor cut-off for openings
facing up or down only, the capillary one for steep openings); --press:
ThroatModel.pressurised (the head of full compartments); --necks: grid
throats widened to their neck on the true geometry, necks of 3 mm and more
(drainsim.throat_size.neck_size_throats).
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))
sys.path.insert(0, HERE)
from drainsim.model import Simulation                           # noqa: E402
from pockets import pockets                                     # noqa: E402


class Held:
    """The pose of motion ``m`` at time ``t0``, for all times."""

    def __init__(self, m, t0):
        self.m, self.t0 = m, float(t0)
        self.t_end = 1e9
        self.bath_level = getattr(m, "bath_level", 0.0)

    def frame(self, t):
        return self.m.frame(self.t0)

    def speed_bound(self, *a, **k):
        return 0.0

    def __getattr__(self, k):
        return getattr(self.m, k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("state")
    ap.add_argument("--hold", type=float, default=8.0)
    ap.add_argument("--kind", choices=("air", "liquid"), default="air")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--rt", action="store_true")
    ap.add_argument("--press", action="store_true")
    ap.add_argument("--necks", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    sim, _ = Simulation.load_state(a.state)
    sim.motion = Held(sim.motion, sim.t)
    if a.rt:
        sim.tm.rt_orientation = True
    if a.press:
        sim.tm.pressurised = True
    if a.necks:
        from drainsim.throat_size import neck_size_throats
        t1_ = time.time()
        neck_size_throats(sim)
        print(f"   ({time.time() - t1_:.0f} s)", flush=True)
    P = pockets(sim, a.kind, a.top)
    print(f"t = {sim.t:.2f} s, rt {a.rt}, necks {a.necks}, pressurised {a.press} "
          f"({time.time() - t0:.0f} s)", flush=True)
    t1 = sim.t + a.hold
    sim.run(t_end=t1)
    liq_t, air_t = sim.trapped_fields()
    w2 = (air_t if a.kind == "air" else liq_t) * sim.v
    tot = w2.sum()
    print(f"after holding {a.hold:g} s: {a.kind} in all {tot*1e3:.3f} l ({time.time() - t0:.0f} s)")
    for n, (vol, nodes, _) in enumerate(P, 1):
        comps = np.unique(sim.lab[nodes])
        inner = comps[comps != 0]
        v2 = w2[np.isin(sim.lab, inner)].sum() if inner.size else 0.0
        if (comps == 0).any():
            v2 += w2[nodes[sim.lab[nodes] == 0]].sum()
        print(f"#{n:2d} {vol*1e3:7.3f} l -> {v2*1e3:7.3f} l")


if __name__ == "__main__":
    main()
