"""Fingerprints of a setup with explicit holes (links, labels, compartments,
throats, the cells near each hole) and its time, to check that a change of
the hole setup gives the same model. Run it with the old and the new tree
and compare the JSON files.

    python tools/openings/setup_fp.py --repo <tree> --out fp.json door [--holes door_holes.csv]
    python tools/openings/setup_fp.py --repo <tree> --out fp.json car --holes xc90_holes.csv [--dx 0.02]

(--data: the folder of door.stl and xc90.stl, if not the repo's.)

door: the door at 8 mm, uniform grid and octree, with the three hand-fitted
drain holes and, with --holes, the holes of a find_openings table (as they
are, kind "hole"). car: xc90.stl on the octree with the holes of the table
(3-80 mm), as car_article.py --holes sets them up.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from types import SimpleNamespace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def hsh(*arrs):
    m = hashlib.sha1()
    for a in arrs:
        a = np.ascontiguousarray(a)
        m.update(str(a.dtype).encode())
        m.update(a.tobytes())
    return m.hexdigest()[:16]


def fingerprint(sim):
    ptr, idx = sim.nbr
    th = sim.comp.throats
    return dict(N=int(sim.N), links=hsh(ptr, idx), label=hsh(sim.comp.label),
                ncomp=int(sim.comp.n), nthroat=len(th),
                throats=hsh(*[np.r_[t.a, t.b, t.area] for t in th],
                            *[t.cells_a for t in th], *[t.cells_b for t in th]),
                holes=len(sim.holes),
                near=hsh(np.array(sorted(sim._near), np.int64),
                         *[np.asarray(x, np.int64) for k in sorted(sim._near)
                           for x in sim._near[k]]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case", choices=("door", "car"))
    ap.add_argument("--repo", required=True, help="the drainsim tree to set up with")
    ap.add_argument("--out", required=True)
    ap.add_argument("--holes", help="a <prefix>_holes.csv of find_openings.py")
    ap.add_argument("--dx", type=float, default=0.02, help="car: the cell size (m)")
    ap.add_argument("--data", default=os.path.abspath(os.path.join(HERE, "..", "..")),
                    help="the folder of door.stl and xc90.stl (default: the repo)")
    a = ap.parse_args()
    sys.path.insert(0, a.repo)
    sys.path.insert(0, os.path.join(a.repo, "examples"))
    from drainsim.model import Simulation
    from drainsim.openings import read_holes_csv
    out = {}
    if a.case == "door":
        from door_article import DOOR_HOLES, door_setup
        sets = [("door3", DOOR_HOLES)]
        if a.holes:
            H = read_holes_csv(a.holes)
            sets.append(("table", [dict(center=tuple(c), diameter=float(d), axis=tuple(x))
                                   for c, d, x, k in zip(H["center"], H["diameter"], H["axis"],
                                                         H["kind"]) if k == 1]))
        for name, holes in sets:
            for levels in (None, 2):
                t = time.time()
                try:
                    mesh, grid, motion = door_setup(os.path.join(a.data, "door.stl"), 0.008, 22.5,
                                                    0.0, levels=levels)
                    fp = fingerprint(Simulation(grid, motion, dt_max=0.1, holes=holes, subcells=2))
                except Exception as ex:                     # a failure is a fingerprint too
                    fp = dict(error=str(ex)[:120])
                key = f"{name} {'octree' if levels else 'uniform'}"
                out[key] = fp
                print(key, fp, f"{time.time() - t:.0f} s", flush=True)
    else:
        from car_article import car_grid, car_motion, explicit_holes, load_car
        mesh, info = load_car(os.path.join(a.data, "xc90.stl"), "auto", False)
        mo = car_motion(5.0, 5.0, "short", "pitch", 1)
        lo = np.asarray(mesh.vertices).min(0) - 0.06
        hi = np.asarray(mesh.vertices).max(0) + 0.06
        grid = car_grid(mesh, a.dx, lo, hi, 2, octree=True)
        del mesh
        holes = explicit_holes(SimpleNamespace(holes=a.holes, holes_min=3.0, holes_max=80.0), info)
        t = time.time()
        sim = Simulation(grid, mo, dt_max=0.1, subcells=2, holes=holes)
        out["car"] = fingerprint(sim)
        print(f"setup {time.time() - t:.0f} s", out["car"], flush=True)
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
