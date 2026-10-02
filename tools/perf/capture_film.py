"""Capture film.update calls of the 8 mm car (the film's state with its
caches, the model arrays it reads, and the results) for replay without
the setup (bench_film.py). Steps: dip at t = 12 and 30 s, drip-off at
60 s (two in a row: the second reuses the level plan)."""
import os
import pickle
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
from drainsim.model import Simulation                              # noqa: E402
from drainsim.physics import ThroatModel                           # noqa: E402

OUT = os.path.join(HOME, "runs", "perf", "captures", "film")
os.makedirs(OUT, exist_ok=True)
SIMKEYS = ("B", "L", "N", "en", "fl", "fluid", "h", "lab", "motion", "t", "up", "v", "zb")
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
film = sim.film
orig = film.update
state = dict(tag=None, k=0)


def spy(dt):
    if not state["tag"]:
        return orig(dt)
    d = {k: v for k, v in film.__dict__.items() if k not in ("sim", "update")}
    stub = {k: getattr(sim, k) for k in SIMKEYS}
    stub["ncomp"] = sim.comp.n
    before = pickle.dumps(dict(film=d, sim=stub, dt=dt), protocol=5)
    t = time.perf_counter()
    inj = orig(dt)
    sec = time.perf_counter() - t
    name = f"{state['tag']}_{state['k']}"
    state["k"] += 1
    with open(os.path.join(OUT, name + ".pkl"), "wb") as f:
        f.write(before)
        pickle.dump(dict(inj=inj, h=film.s.h, sub=film.s.sub, L=sim.L, drips=len(film.s.drips),
                         drip_volume=film.drip_volume, seconds=sec, nsub=film.nsub), f,
                    protocol=5)
    print(f"  {name}: t = {sim.t:.1f} s, nsub {film.nsub}, {sec:.3f} s, wet "
          f"{int((film.s.h > 0).sum())}", flush=True)
    return inj


film.update = spy
for tag, t_from, dt, n in (("dip12", 12.0, 0.1, 2), ("dip30", 30.0, 0.1, 2),
                           ("hang60", 60.0, 0.5, 3)):
    sim.dt_max = 0.1
    sim.run(t_end=t_from)
    sim.dt_max = dt
    state["tag"], state["k"] = tag, 0
    for _ in range(n):
        sim.run(t_end=sim.t + dt)
    state["tag"] = None
print("done", flush=True)
