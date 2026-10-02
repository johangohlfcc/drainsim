"""Holes (and optionally narrow gaps) found on a mesh, for review and as
explicit holes of a run.

    python examples/find_openings.py --stl xc90.stl --out runs/openings/xc90
    python examples/find_openings.py --stl door.stl --out runs/openings/door --gaps 3 8

Writes ``<out>_holes.csv`` (one row per hole, pin or unclear ring: kind,
diameter, centre, axis, depth, roundness; the table ``car_movie.py`` and
``car_article.py`` read with ``--holes``), ``<out>_holes.vtp`` (a disc per
ring, for ParaView over the mesh) and with ``--gaps LO HI``
``<out>_mesh.vtp`` (the mesh with the gap width per face; widths outside
LO-HI mm left out) and ``<out>_gaps.csv`` (the regions of such gaps).
Coordinates are those of the STL (metres). Which holes are open in reality
(not closed by a plug, a clip or a bolt) is for the engineer to decide: edit
the table, or set the kind of a row to "pin" to leave it out.
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from drainsim import openings as op                                    # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stl", required=True)
    ap.add_argument("--out", required=True, help="prefix of the output files")
    ap.add_argument("--min-d", type=float, default=3.0, help="smallest hole (mm)")
    ap.add_argument("--max-d", type=float, default=80.0, help="largest hole (mm)")
    ap.add_argument("--gaps", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                    help="also the gaps of LO-HI mm (ray casting; facing walls)")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    t0 = time.time()
    V, F = op.read_stl(a.stl)
    print(f"{a.stl}: {len(F)} faces ({time.time() - t0:.0f} s)", flush=True)
    H = op.find_holes(V, F, dmin=a.min_d * 1e-3, dmax=a.max_d * 1e-3, verbose=True)
    d = H["diameter"] * 1e3
    for lo, hi in ((3, 6), (6, 10), (10, 15), (15, 20), (20, 30), (30, 80)):
        n = ((H["kind"] == 1) & (d >= lo) & (d < hi)).sum()
        if n:
            print(f"   holes {lo:2d}-{hi:2d} mm: {n}")
    W = facing = regions = None
    if a.gaps:
        lo, hi = a.gaps[0] * 1e-3, a.gaps[1] * 1e-3
        W, _, facing = op.gap_map(V, F, +1, max_width=max(hi, 0.05), verbose=True)
        sel = facing & (W >= lo)
        W = np.where(sel, W, np.inf)
        regions = op.gap_regions(V, F, W, sel, hi)
        R = regions[1]
        print(f"gaps of {a.gaps[0]:g}-{a.gaps[1]:g} mm: {len(R['area'])} regions, "
              f"{R['area'].sum():.3f} m2", flush=True)
    op.export_openings(a.out, V, F, W, facing, H, regions)
    print(f"wrote {a.out}_* ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
