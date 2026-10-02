"""Setup profile and memory inventory of the car model (as car_movie --record).

    python perf_setup.py --dx 0.008 --levels 3 --out perf_8mm.json [--steps 20]

Prints the wall time of the top-level setup parts, then every numpy array the
Simulation keeps (by attribute path), grouped, with bytes per node. With
--steps, runs that many model steps from t = --t0 and inventories again (the
caches filled by the time loop).
"""
import argparse
import gc
import json
import sys
import time
import warnings

import numpy as np

import os                                                          # noqa: E402
HOME = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
REPO = os.environ.get("DS_REPO", HOME)          # the drainsim to measure (default: this one)
CAR = os.path.join(HOME, "xc90.stl")
sys.path.insert(0, os.path.join(REPO, "examples"))
sys.path.insert(0, REPO)
import numba                                                       # noqa: E402
from car_article import car_grid, car_motion, load_car, peak_rss_bytes  # noqa: E402
from drainsim.model import Simulation                              # noqa: E402
from drainsim.physics import ThroatModel                           # noqa: E402


def arrays_of(root, name="sim", maxdepth=6):
    """(path, array) for every numpy array reachable from root (each array
    once, by the first path found; views counted by their base)."""
    seen, out = set(), []

    def walk(obj, path, d):
        if id(obj) in seen or d > maxdepth:
            return
        seen.add(id(obj))
        if isinstance(obj, np.ndarray):
            base = obj if obj.base is None else obj.base
            if isinstance(base, np.ndarray) and base is not obj:
                if id(base) in seen:
                    return
                seen.add(id(base))
                obj = base
            out.append((path, obj))
            return
        if isinstance(obj, (str, bytes, int, float, bool, type(None), type)):
            return
        if callable(obj) and not hasattr(obj, "__dict__"):
            return
        if isinstance(obj, dict):
            for k, v in list(obj.items()):
                walk(v, f"{path}[{k!r}]", d + 1)
            return
        if isinstance(obj, (list, tuple)):
            if len(obj) and all(isinstance(x, np.ndarray) for x in obj[:3]):
                tot = [x for x in obj if isinstance(x, np.ndarray)]
                for i, x in enumerate(obj):
                    walk(x, f"{path}[*]", d + 1)
                return
            for i, x in enumerate(obj[:100000]):
                walk(x, f"{path}[{i}]" if len(obj) < 50 else f"{path}[*]", d + 1)
            return
        dct = getattr(obj, "__dict__", None)
        if dct is not None and type(obj).__module__.split(".")[0] in ("drainsim", "__main__", "types"):
            for k, v in list(dct.items()):
                walk(v, f"{path}.{k}", d + 1)
        elif dct is not None and type(obj).__name__ in ("SimpleNamespace", "FSMCache"):
            for k, v in list(dct.items()):
                walk(v, f"{path}.{k}", d + 1)

    walk(root, name, 0)
    return out


def inventory(sim, label):
    arrs = arrays_of(sim)
    rows = {}
    for path, a in arrs:
        key = path
        r = rows.setdefault(key, dict(bytes=0, n=0, dtype=str(a.dtype), shape=str(a.shape)))
        r["bytes"] += a.nbytes
        r["n"] += 1
    tot = sum(r["bytes"] for r in rows.values())
    N = sim.N
    print(f"\n== {label}: {len(arrs)} arrays, {tot/2**30:.2f} GiB, {tot/N:.0f} B/node "
          f"(N = {N/1e6:.2f} M nodes)")
    for k, r in sorted(rows.items(), key=lambda kv: -kv[1]["bytes"])[:45]:
        print(f"  {r['bytes']/2**20:9.1f} MiB {r['bytes']/N:7.1f} B/node  {r['dtype']:8s} "
              f"x{r['n']:<5d} {k}  {r['shape'] if r['n'] == 1 else ''}")
    return dict(total=tot, per_node=tot / N,
                rows={k: dict(r) for k, r in rows.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dx", type=float, required=True)
    ap.add_argument("--levels", type=int, default=3)
    ap.add_argument("--subcells", type=int, default=2)
    ap.add_argument("--narrow", type=int, default=4)
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--t0", type=float, default=12.0)
    ap.add_argument("--dt-steps", type=float, default=0.1)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    warnings.filterwarnings("ignore", category=numba.NumbaWarning)
    T = {}
    t = time.time()
    mesh, info = load_car(CAR, "auto", False)
    T["load_car"] = time.time() - t
    mo = car_motion(5.0, 120.0, "short", "pitch", 1)
    lo = np.asarray(mesh.vertices).min(0) - 0.06
    hi = np.asarray(mesh.vertices).max(0) + 0.06
    t = time.time()
    grid = car_grid(mesh, a.dx, lo, hi, a.levels, octree=True)
    T["octree"] = time.time() - t
    del mesh
    gc.collect()
    t = time.time()
    sim = Simulation(grid, mo, dt_max=0.1, cells_per_step=1e9, subcells=a.subcells,
                     film=True, throat_model=ThroatModel(Cd=0.65),
                     narrow=dict(k=a.narrow) if a.narrow else None)
    T["simulation"] = time.time() - t
    print(f"N = {sim.N/1e6:.2f} M nodes, film {sim.film.c.n/1e6:.2f} M, "
          f"compartments {sim.comp.n}, throats {len(sim.comp.throats)}; times {T}; "
          f"peak {peak_rss_bytes()/2**30:.1f} GiB", flush=True)
    res = dict(dx=a.dx, levels=a.levels, N=int(sim.N), film=int(sim.film.c.n),
               ncomp=int(sim.comp.n), nthroats=len(sim.comp.throats), times=T,
               peak_setup=peak_rss_bytes(), stats=getattr(sim, "volfrac_stats", {}))
    from drainsim.par import fingerprint
    res["fp_setup"] = str(fingerprint(sim.comp.label, sim.v, sim.nsize))
    res["throat_sig"] = [round(float(sum(t.area for t in sim.comp.throats)), 12),
                         round(float(sum(t.diameter for t in sim.comp.throats)), 12)]
    print("fingerprint after setup", res["fp_setup"], "throats", res["throat_sig"], flush=True)
    res["inv_setup"] = inventory(sim, "after setup")
    if a.steps:
        t = time.time()
        sim.run(t_end=a.t0)
        T["run_to_t0"] = time.time() - t
        t = time.time()
        for _ in range(a.steps):
            sim.dt_max = a.dt_steps
            sim.run(t_end=sim.t + a.dt_steps)
        T["steps"] = (time.time() - t) / a.steps
        res["fp_run"] = str(fingerprint(sim.L, sim.B, sim.A))
        res["film_sum"] = float(sim.film.s.h.sum())
        res["fp_film"] = str(fingerprint(sim.film.s.h))
        print("fingerprint after the steps", res["fp_run"], "film", res["film_sum"], res["fp_film"], flush=True)
        res["inv_run"] = inventory(sim, f"after running to t = {sim.t:.1f} s")
        res["peak_run"] = peak_rss_bytes()
        print(f"s/step {T['steps']:.2f}; peak {peak_rss_bytes()/2**30:.1f} GiB")
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1, default=str)


if __name__ == "__main__":
    main()
