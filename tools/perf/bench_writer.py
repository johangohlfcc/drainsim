"""The movie field writer (car_movie._state_fields) on real states of the car
recording: the 6.4/v7.0 code (kept here) against the current one, same
fields required, time of each (serial kernels = as in the writer thread
while the model holds the parallel ones; and parallel).

    python bench_writer.py --dx 0.008 --every 30 [--until 400]
"""
import argparse
import os
import sys
import time
import warnings
from types import SimpleNamespace

import numpy as np

HOME = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
REPO = os.environ.get("DS_REPO", HOME)          # the drainsim to measure (default: this one)
sys.path.insert(0, os.path.join(REPO, "examples"))
sys.path.insert(0, os.path.join(REPO, "tests"))
sys.path.insert(0, REPO)
import numba                                                       # noqa: E402
import car_movie                                                   # noqa: E402
from car_article import car_grid, car_motion, load_car             # noqa: E402
from drainsim.model import Simulation                              # noqa: E402
from drainsim.par import _PAR_LOCK                                 # noqa: E402
from drainsim.physics import ThroatModel                           # noqa: E402
from test_v70_display import trapped_display64                     # noqa: E402


def old_state_fields(sim, t_state, L, B, A):
    """car_movie._state_fields as of 7aaaba4 (octree runs)."""
    view = SimpleNamespace(fl=sim.fl, lab=sim.lab, nbr=sim.nbr, v=sim.v, fine=sim.fine,
                           subcells=sim.subcells, host=sim.host, ncells=sim.ncells,
                           N=sim.N, grid=sim.grid, nsize=sim.nsize)
    up, zb = sim.motion.frame(t_state)
    h = sim.X @ up
    view._vis_e = 0.5 * sim.grid.dx * np.abs(up).sum()
    sv = SimpleNamespace(**vars(view))
    liq, air = trapped_display64(sv, h, zb, L, B, A)
    from drainsim.octview import display_grid
    dg = display_grid(sim)
    liq, air = dg.field(sim, liq), dg.field(sim, air)
    ncells = dg.ncells
    out = {}
    for k, f in (("liq", liq), ("air", air)):
        q = np.clip(np.rint(f * 255.0), 0, 255).astype(np.uint8)
        i = np.flatnonzero(q)
        out[k + "_i"] = i.astype(np.int64 if ncells >= 2 ** 31 else np.int32)
        out[k + "_v"] = q[i]
    return out


def timed(fn, *a, serial=False):
    t = time.perf_counter()
    if serial:
        with _PAR_LOCK:
            r = fn(*a)
    else:
        r = fn(*a)
    return r, time.perf_counter() - t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dx", type=float, default=0.008)
    ap.add_argument("--every", type=int, default=30)
    ap.add_argument("--until", type=int, default=10 ** 9)
    ap.add_argument("--no-old", action="store_true")
    a = ap.parse_args()
    warnings.filterwarnings("ignore", category=numba.NumbaWarning)
    t0 = time.time()
    mesh, info = load_car(os.path.join(HOME, "xc90.stl"), "auto", False)
    mo = car_motion(5.0, 120.0, "short", "pitch", 1)
    lo = np.asarray(mesh.vertices).min(0) - 0.06
    hi = np.asarray(mesh.vertices).max(0) + 0.06
    grid = car_grid(mesh, a.dx, lo, hi, 3, octree=True)
    del mesh
    narrow = car_movie.narrow_opt(SimpleNamespace(narrow=4))
    sim = Simulation(grid, mo, dt_max=0.1, cells_per_step=1e9, subcells=2, film=True,
                     throat_model=ThroatModel(Cd=0.65), threads=1, narrow=narrow)
    print(f"setup {time.time() - t0:.0f} s, N = {sim.N / 1e6:.2f} M", flush=True)
    import json
    # the step times of the 8 mm reference recording
    meta = json.load(open(os.path.join(HOME, "runs", "perf", "rec8_ref", "rec", "meta.json")))
    steps = meta["step_times"]
    dt_fine, dt_hang, t_slow = 0.1, 0.5, None
    tot = dict(old=0.0, old_par=0.0, new=0.0, new_par=0.0)
    n = 0
    for k, ts in enumerate(steps):
        if k > a.until:
            break
        if k > 0:
            sim.dt_max = 0.1 if ts - steps[k - 1] <= 0.1 + 1e-9 else 0.5
            sim.run(t_end=float(ts))
        if k % a.every:
            continue
        L, B, A = sim.L.copy(), sim.B.copy(), sim.A.copy()
        new, tn = timed(car_movie._state_fields, sim, sim.t, L, B, A, serial=True)
        _, tnp = timed(car_movie._state_fields, sim, sim.t, L, B, A)
        line = f"step {k:4d} t {sim.t:6.1f}  new {tn:6.2f} s (par {tnp:5.2f})"
        tot["new"] += tn
        tot["new_par"] += tnp
        if not a.no_old:
            old, to = timed(old_state_fields, sim, sim.t, L, B, A, serial=True)
            _, top = timed(old_state_fields, sim, sim.t, L, B, A)
            same = all(np.array_equal(old[x], new[x]) and old[x].dtype == new[x].dtype
                       for x in old) and set(old) == set(new)
            line += f"  old {to:6.2f} s (par {top:5.2f})  same {same}"
            tot["old"] += to
            tot["old_par"] += top
            if not same:
                print(line, flush=True)
                raise SystemExit("DIFFERENT")
        line += f"  liq {new['liq_i'].size} air {new['air_i'].size} cells"
        print(line, flush=True)
        n += 1
    print(f"{n} states: " + "  ".join(f"{k} {v / max(n, 1):.2f} s" for k, v in tot.items()),
          flush=True)


if __name__ == "__main__":
    main()
