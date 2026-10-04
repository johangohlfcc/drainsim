"""How the pockets of a report change over a recording: the trapped air (or
held liquid) in each pocket's box, step by step, from the recorded display
fields (no state is loaded). Tells a pocket that is on its way out (it
shrinks) from one that stays.

    python tools/debug/pocket_history.py REC REPORT [--every 10] [--grow 0.01]
        [--kind air|liquid] [--from T] [--to T]

REPORT is a pockets.py report (pockets_t*.txt); each pocket's box is its
centre +- half its extent, grown by --grow m on every side. The volumes are
the display fields (8-bit fractions of a display cell), so they are close
to the model's, not equal; other air in the box counts too.
"""
import argparse
import glob
import json
import os
import re

import numpy as np

POCKET = re.compile(r"#\s*(\d+)\s+([\d.]+) l at \[([^\]]+)\].*extent \[([^\]]+)\]")


def pockets(report):
    out = []
    for line in open(report, encoding="utf-8", errors="replace"):
        m = POCKET.match(line.strip())
        if m:
            c = np.array(m.group(3).split(), float)
            e = np.array(m.group(4).split(), float)
            out.append((int(m.group(1)), float(m.group(2)), c, e))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rec")
    ap.add_argument("report")
    ap.add_argument("--kind", choices=("air", "liquid"), default="air")
    ap.add_argument("--every", type=int, default=10, help="every n-th recorded step")
    ap.add_argument("--grow", type=float, default=0.01, help="m added to each side of a box")
    ap.add_argument("--from", dest="t0", type=float, default=0.0)
    ap.add_argument("--to", dest="t1", type=float, default=np.inf)
    a = ap.parse_args()
    z = np.load(os.path.join(a.rec, "static.npz"))
    shape = np.array(z["shape"], np.int64)
    origin, dx = np.array(z["origin"], float), float(z["dx"])
    meta = json.load(open(os.path.join(a.rec, "meta.json")))
    times = meta.get("step_times")
    if isinstance(times, str):
        times = json.loads(times)
    P = pockets(a.report)
    # each box as index ranges of the display grid
    boxes = []
    for _, _, c, e in P:
        lo = np.floor((c - e / 2 - a.grow - origin) / dx).astype(np.int64)
        hi = np.floor((c + e / 2 + a.grow - origin) / dx).astype(np.int64)
        boxes.append((np.clip(lo, 0, shape - 1), np.clip(hi, 0, shape - 1)))
    name = "air" if a.kind == "air" else "liq"
    files = sorted(glob.glob(os.path.join(a.rec, "steps", "s*.npz")))
    print(f"{a.kind} in the box of each pocket of {os.path.basename(a.report)} "
          f"(l; display grid {dx * 1000:.0f} mm)")
    print("  t (s) " + "".join(f"{'#' + str(n):>8}" for n, _, _, _ in P))
    print(" report " + "".join(f"{v:8.3f}" for _, v, _, _ in P))
    for k in range(0, len(files), a.every):
        t = times[k] if times is not None and k < len(times) else k
        if not a.t0 <= t <= a.t1:
            continue
        s = np.load(files[k])
        i = s[name + "_i"].astype(np.int64)
        v = s[name + "_v"].astype(float) / 255.0 * dx ** 3 * 1000.0
        kz = i % shape[2]
        r = i // shape[2]
        kj = r % shape[1]
        ki = r // shape[1]
        row = []
        for lo, hi in boxes:
            m = ((ki >= lo[0]) & (ki <= hi[0]) & (kj >= lo[1]) & (kj <= hi[1])
                 & (kz >= lo[2]) & (kz <= hi[2]))
            row.append(v[m].sum())
        print(f"{t:7.1f} " + "".join(f"{x:8.3f}" for x in row), flush=True)


if __name__ == "__main__":
    main()
