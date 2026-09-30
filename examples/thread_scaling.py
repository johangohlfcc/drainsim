"""Thread scaling of the car movie model (drainsim v5).

Builds the car as ``car_movie.py --record`` does (film on), runs the motion
to ``--t-move`` and times ``--steps`` model steps from that same state with
each thread count in ``--threads``; then the same ``--hang-steps`` into the
still hang (where the hierarchies are reused). The state after the timed
steps must be identical for every thread count; this is checked.

    python examples/thread_scaling.py --stl xc90_nose_forward.stl --dx 0.01 \
        --subcells 4 --threads 1 2 4 8 16 --out scaling_10mm.json

NUMBA_NUM_THREADS must be at least the largest count (default: all cores).
"""
import argparse
import copy
import json
import os
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numba                                                    # noqa: E402

from drainsim import fsm                                        # noqa: E402
from drainsim.fsm import FSMCache                               # noqa: E402
from drainsim.grid import Grid                                  # noqa: E402
from drainsim.model import Simulation                           # noqa: E402
from drainsim.physics import ThroatModel                        # noqa: E402
from drainsim.par import fingerprint                            # noqa: E402
from car_article import add_car_args, car_grid, car_motion, load_car, narrow_opt  # noqa: E402

STATE = ("t", "L", "G", "P", "B", "A", "h", "up", "zb", "e", "en", "drained_total", "lost",
         "_pending", "_static", "_last_keys", "_pose_id")
PARTS = ("_throat_step", "_bath_atm", "_equilibrate", "_record")


def peak_gb():
    try:
        import psutil
        p = psutil.Process()
        mi = p.memory_info()
        return round(getattr(mi, "peak_wset", mi.rss) / 2 ** 30, 1)
    except Exception:
        return None


def snapshot(sim):
    s = {k: copy.deepcopy(getattr(sim, k)) for k in STATE if hasattr(sim, k)}
    if sim.film is not None:
        s["_film"] = {k: copy.deepcopy(v) for k, v in sim.film.__dict__.items()
                      if k not in ("sim", "c", "p", "_csr", "_order", "_sp", "update")}
    return s


def restore(sim, s):
    for k, v in s.items():
        if k == "_film":
            for a, b in v.items():
                setattr(sim.film, a, copy.deepcopy(b))
        else:
            setattr(sim, k, copy.deepcopy(v))
    sim._fc_l, sim._fc_a = FSMCache(), FSMCache()


def instrument(sim, acc):
    """Time the parts of a step (instance wrappers)."""
    def wrap(name, f):
        def g(*a, **k):
            t = time.perf_counter()
            try:
                return f(*a, **k)
            finally:
                acc[name] = acc.get(name, 0.0) + time.perf_counter() - t
        return g
    for n in PARTS:
        setattr(sim, n, wrap(n, getattr(sim, n)))
    if sim.film is not None:
        sim.film.update = wrap("film", sim.film.update)


def timed(sim, n, dt, thr, acc):
    numba.set_num_threads(thr)
    acc.clear()
    fsm.TIMING = {}
    sim.dt_max = dt
    T = []
    for _ in range(n):
        t = time.perf_counter()
        sim.run(t_end=sim.t + dt)
        T.append(time.perf_counter() - t)
    parts = {k: v / n for k, v in acc.items()}
    parts.update({"fsm " + k: v / n for k, v in fsm.TIMING.items()})
    fsm.TIMING = None
    return T, parts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stl", required=True)
    ap.add_argument("--dx", type=float, default=0.01)
    ap.add_argument("--subcells", type=int, default=4)
    ap.add_argument("--rock", type=float, default=5.0)
    ap.add_argument("--hang", type=float, default=120.0)
    ap.add_argument("--pad", type=float, default=0.06)
    ap.add_argument("--cd", type=float, default=0.65)
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--dt-hang", type=float, default=0.5)
    ap.add_argument("--t-move", type=float, default=8.0,
                    help="model time of the motion steps timed (car entering the bath)")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--hang-steps", type=int, default=8)
    ap.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    ap.add_argument("--out", default="thread_scaling.json")
    add_car_args(ap)
    a = ap.parse_args()
    warnings.filterwarnings("ignore", category=numba.NumbaWarning)

    nmax = numba.config.NUMBA_NUM_THREADS
    thr = [t for t in a.threads if t <= nmax]
    if len(thr) < len(a.threads):
        print(f"NUMBA_NUM_THREADS = {nmax}: skipping {sorted(set(a.threads) - set(thr))}",
              flush=True)
    t0 = time.time()
    mesh, info = load_car(a.stl, a.orient, a.reverse)
    mo = car_motion(a.rock, a.hang, "short", a.rotation, a.sense)
    lo = np.asarray(mesh.vertices).min(0) - a.pad
    hi = np.asarray(mesh.vertices).max(0) + a.pad
    grid = car_grid(mesh, a.dx, lo, hi, a.levels, octree=bool(a.narrow))
    del mesh
    numba.set_num_threads(max(thr))
    sim = Simulation(grid, mo, dt_max=a.dt, cells_per_step=1e9, subcells=a.subcells,
                     film=True, throat_model=ThroatModel(Cd=a.cd), narrow=narrow_opt(a))
    print(f"setup {time.time()-t0:.0f} s: grid {grid.shape}, {sim.N/1e6:.2f} M nodes, "
          f"peak {peak_gb()} GB", flush=True)
    t1 = time.time()
    sim.run(t_end=a.t_move)
    print(f"motion to t = {a.t_move:g} s: {time.time()-t1:.0f} s with {max(thr)} threads "
          f"(includes compiling), layer {numba.threading_layer()}", flush=True)
    acc = {}
    instrument(sim, acc)
    res = dict(dx=a.dx, subcells=a.subcells, nodes=int(sim.N), cells=int(grid.ncells),
               threading_layer=numba.threading_layer(), cpu_count=os.cpu_count(),
               numba_num_threads=nmax, runs=[])
    for phase, n, dt, t_start in (("motion", a.steps, a.dt, None),
                                  ("hang", a.hang_steps, a.dt_hang, mo.t_end - a.hang + 2.0)):
        if n <= 0:
            continue
        if t_start is not None:
            numba.set_num_threads(max(thr))
            sim.dt_max = a.dt
            sim.run(t_end=t_start)
        S = snapshot(sim)
        ref = None
        for k in thr:
            restore(sim, S)
            timed(sim, 1, dt, k, acc)                  # warm (compiles, fills caches)
            restore(sim, S)
            T, parts = timed(sim, n, dt, k, acc)
            fp = fingerprint(sim.L, sim.B, sim.A)
            same = ref is None or fp == ref
            ref = ref or fp
            r = dict(phase=phase, threads=k, s_per_step=float(np.mean(T)),
                     s_min=float(np.min(T)), identical=bool(same),
                     parts={p: round(v, 4) for p, v in sorted(parts.items())},
                     peak_gb=peak_gb())
            res["runs"].append(r)
            base = [x for x in res["runs"] if x["phase"] == phase][0]["s_per_step"]
            print(f"{phase:6s} {k:3d} threads: {r['s_per_step']:.3f} s/step "
                  f"(speed-up {base / r['s_per_step']:.2f}x), state identical: {same}, "
                  f"peak {r['peak_gb']} GB", flush=True)
            top = sorted(((v, p) for p, v in parts.items() if not p.startswith("fsm")),
                         reverse=True)
            print("        " + ", ".join(f"{p} {v:.3f}" for v, p in top), flush=True)
        restore(sim, S)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    print(f"wrote {a.out} ({(time.time()-t0)/60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
