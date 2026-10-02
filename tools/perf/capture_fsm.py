"""Capture the fill_spill calls of real 8 mm car steps (inputs and results),
for kernel benchmarks without the setup:
  dip: the steps 12.0 -> 12.1 and 12.1 -> 12.2 s (pose changes every step)
  hang: 60.0 -> 60.5 s (pose still: cached hierarchies)
Writes runs/perf/captures/fsm/static.npz (graph, volumes, floors) and
<tag>_<k>.npz there."""
import os
import sys
import time
import warnings
from types import SimpleNamespace

import numpy as np

HOME = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
REPO = os.environ.get("DS_REPO", HOME)          # the drainsim to run (default: this one)
sys.path.insert(0, os.path.join(REPO, "examples"))
sys.path.insert(0, REPO)
import numba                                                       # noqa: E402
import car_movie                                                   # noqa: E402
from car_article import car_grid, car_motion, load_car             # noqa: E402
from drainsim import model as M                                    # noqa: E402
from drainsim.model import Simulation                              # noqa: E402
from drainsim.physics import ThroatModel                           # noqa: E402

OUT = os.path.join(HOME, "runs", "perf", "captures", "fsm")
os.makedirs(OUT, exist_ok=True)
warnings.filterwarnings("ignore", category=numba.NumbaWarning)
mesh, info = load_car(os.path.join(HOME, "xc90.stl"), "auto", False)
mo = car_motion(5.0, 120.0, "short", "pitch", 1)
lo = np.asarray(mesh.vertices).min(0) - 0.06
hi = np.asarray(mesh.vertices).max(0) + 0.06
grid = car_grid(mesh, 0.008, lo, hi, 3, octree=True)
del mesh
sim = Simulation(grid, mo, dt_max=0.1, cells_per_step=1e9, subcells=2, film=True,
                 throat_model=ThroatModel(Cd=0.65), threads=1,
                 narrow=car_movie.narrow_opt(SimpleNamespace(narrow=4)))
print("setup done", flush=True)
ptr, idx = sim.nbr
np.savez(os.path.join(OUT, "static.npz"), ptr=ptr, idx=idx, v=sim.v)
orig = M.fill_spill
state = dict(tag=None, k=0)


def spy(h, region, nbr, sink, src, vcell, spill_routing=True, nregions=None, min_depth=0.0,
        ecell=None, method=None, cache=None, key=None, prune=True, eshape=None):
    t = time.perf_counter()
    r = orig(h, region, nbr, sink, src, vcell, spill_routing, nregions, min_depth, ecell,
             method, cache, key, prune, eshape)
    dt = time.perf_counter() - t
    if state["tag"]:
        k = state["k"]
        state["k"] += 1
        np.savez(os.path.join(OUT, f"{state['tag']}_{k}.npz"), h=h, region=region, sink=sink,
                 src=src, spill_routing=spill_routing, nregions=nregions, min_depth=min_depth,
                 ecell=np.zeros(0) if ecell is None else ecell,
                 eshape=np.zeros(0) if eshape is None else np.asarray(eshape),
                 cached=cache is not None, vcell_is_v=vcell is sim.v,
                 retained=r[0], drained=r[1], lost=r[2], owner=r[3], seconds=dt)
        print(f"  {state['tag']}_{k}: {int((region >= 0).sum())} active, cache "
              f"{cache is not None}, {dt:.3f} s", flush=True)
    return r


M.fill_spill = spy
for tag, t_from, dt, n in (("dip", 12.0, 0.1, 2), ("hang", 60.0, 0.5, 2)):
    sim.dt_max = 0.1
    sim.run(t_end=t_from)
    sim.dt_max = dt
    state["tag"], state["k"] = tag, 0
    for _ in range(n):
        sim.run(t_end=sim.t + dt)
    state["tag"] = None
    print(tag, "captured", flush=True)
