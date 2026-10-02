"""Replay captured fill_spill calls (capture_fsm.py): same results as
captured (bit for bit), and the time of each phase (fsm.TIMING) per thread
count.   python bench_fsm.py dip_0 dip_1 --threads 16 1 --reps 3"""
import argparse
import os
import sys
import time

import numpy as np

HOME = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
REPO = os.environ.get("DS_REPO", HOME)          # the drainsim to measure (default: this one)
sys.path.insert(0, REPO)
import numba                                                       # noqa: E402
from drainsim import fsm                                           # noqa: E402

CAP = os.path.join(HOME, "runs", "perf", "captures", "fsm")


def load(tag, st):
    z = np.load(os.path.join(CAP, tag + ".npz"))
    a = {k: z[k] for k in z.files}
    a["nbr"] = (st["ptr"], st["idx"])
    a["vcell"] = st["v"]
    return a


def call(a):
    return fsm.fill_spill(a["h"], a["region"], a["nbr"], a["sink"], a["src"], a["vcell"],
                          bool(a["spill_routing"]), int(a["nregions"]), float(a["min_depth"]),
                          a["ecell"] if a["ecell"].size else None,
                          eshape=a["eshape"] if a["eshape"].size else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tags", nargs="+")
    ap.add_argument("--threads", type=int, nargs="+", default=[16, 1])
    ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args()
    z = np.load(os.path.join(CAP, "static.npz"))
    st = {k: z[k] for k in z.files}
    calls = [load(t, st) for t in a.tags]
    for thr in a.threads:
        numba.set_num_threads(thr)
        for c in calls:
            call(c)                                                  # warm
        fsm.TIMING = {}
        T = []
        for _ in range(a.reps):
            for c in calls:
                t = time.perf_counter()
                r = call(c)
                T.append(time.perf_counter() - t)
                for k, ref in zip(("retained", "drained", "lost", "owner"), r):
                    if not np.array_equal(ref, c[k]):
                        raise SystemExit(f"{k} DIFFERS")
        n = a.reps * len(calls)
        tim = {k: v / n for k, v in fsm.TIMING.items()}
        fsm.TIMING = None
        tot = sum(T) / n
        print(f"== {thr} threads: {tot:.3f} s per call (same results), phases "
              f"{sum(tim.values()):.3f} s, other {tot - sum(tim.values()):.3f} s")
        for k, v in sorted(tim.items(), key=lambda kv: -kv[1]):
            print(f"   {k:24s} {v * 1e3:7.1f} ms")


if __name__ == "__main__":
    main()
