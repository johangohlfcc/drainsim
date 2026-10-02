"""Capture the connected-region calls of real 8 mm car steps (par.connected_to
from the bath / atmosphere, par.label_bodies from the throats), inputs and
results, for bench_label.py: dip steps at t = 12 and 30 s, drip-off at 60 s.
Writes runs/perf/captures/label/static.npz (graph, compartment labels) and
<tag>_<k>.npz."""
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
from drainsim import par                                           # noqa: E402
from drainsim.model import Simulation                              # noqa: E402
from drainsim.physics import ThroatModel                           # noqa: E402

OUT = os.path.join(HOME, "runs", "perf", "captures", "label")
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
np.savez(os.path.join(OUT, "static.npz"), ptr=ptr, idx=idx, lab=sim.lab, bnd=sim.bnd, X=sim.X)
if os.environ.get("STATIC_ONLY"):
    raise SystemExit("static only")
state = dict(tag=None, k=0)
orig_ct, orig_lb = par.connected_to, par.label_bodies


def save(kind, mask, comp, seed, res, sec):
    if not state["tag"]:
        return
    name = f"{state['tag']}_{state['k']}"
    state["k"] += 1
    same_comp = comp is sim.lab or np.array_equal(comp, sim.lab)
    np.savez(os.path.join(OUT, name + ".npz"), kind=kind, mask=mask,
             comp=np.zeros(0) if same_comp else comp,
             seed=np.zeros(0, bool) if seed is None else seed,
             res=res[0] if isinstance(res, tuple) else res,
             nb=res[1] if isinstance(res, tuple) else -1, seconds=sec)
    print(f"  {name}: {kind}, {int(mask.sum())} masked, {sec * 1e3:.1f} ms", flush=True)


def ct(mask, comp, ptr_, idx_, seed):
    t = time.perf_counter()
    r = orig_ct(mask, comp, ptr_, idx_, seed)
    save("connected_to", mask, comp, seed, r, time.perf_counter() - t)
    return r


def lb(mask, comp, ptr_, idx_):
    t = time.perf_counter()
    r = orig_lb(mask, comp, ptr_, idx_)
    save("label_bodies", mask, comp, None, r, time.perf_counter() - t)
    return r


par.connected_to, par.label_bodies = ct, lb
for tag, t_from, dt, n in (("dip12", 12.0, 0.1, 1), ("dip30", 30.0, 0.1, 1),
                           ("hang60", 60.0, 0.5, 1)):
    sim.dt_max = 0.1
    sim.run(t_end=t_from)
    sim.dt_max = dt
    state["tag"], state["k"] = tag, 0
    for _ in range(n):
        sim.run(t_end=sim.t + dt)
    state["tag"] = None
    print(tag, "captured", flush=True)
print("all done", flush=True)
