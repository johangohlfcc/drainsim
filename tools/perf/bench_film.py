"""Replay captured film.update calls (capture_film.py): the same results
(film h, sub, injected volumes, L) bit for bit, and the time per call with
the time of the film module's kernels.
    python bench_film.py dip12_0 hang60_1 --threads 16 --reps 3"""
import argparse
import os
import pickle
import sys
import time
from types import SimpleNamespace

import numpy as np

HOME = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
REPO = os.environ.get("DS_REPO", HOME)          # the drainsim to measure (default: this one)
sys.path.insert(0, REPO)
import numba                                                       # noqa: E402
from drainsim import film as FM                                    # noqa: E402

CAP = os.path.join(HOME, "runs", "perf", "captures", "film")
ACC = {}
BC = {}
import inspect  # noqa: E402
NEW_PLAN = "uptr" in inspect.getsource(FM)


def timed(name, f):
    def g(*a, **k):
        t = time.perf_counter()
        try:
            return f(*a, **k)
        finally:
            ACC[name] = ACC.get(name, 0.0) + time.perf_counter() - t
    return g


for n in ("_film_geom", "_scatter_add", "_film_coeffs", "_height_keys", "_fix_ties",
          "_sweep_sorted", "_sweep_plan", "_sweep_levels", "_take", "_put", "_renumber",
          "_comp_liquid_pos", "_scale_comp", "sort_keys", "connected_components"):
    if hasattr(FM, n):
        setattr(FM, n, timed(n, getattr(FM, n)))


def load(tag):
    with open(os.path.join(CAP, tag + ".pkl"), "rb") as f:
        before = pickle.load(f)
        after = pickle.load(f)
    return before, after


def upgrade(fd):
    """A level plan captured before 7.0's grouped uphill edges: grouped the
    same way (stable by receiver)."""
    pl = fd.get("_plan")
    if pl is not None and len(pl[2]) == 8:
        lptr, lel, nin, isrc, iq, up_e, up_r, up_q = pl[2]
        o = np.argsort(up_r, kind="stable")
        ur, cnt = np.unique(up_r, return_counts=True)
        uptr = np.r_[0, np.cumsum(cnt)].astype(np.int64)
        fd["_plan"] = (pl[0], pl[1], (lptr, lel, nin, isrc, iq, up_e[o], up_r[o], up_q[o],
                                      uptr, ur.astype(np.int64)))


def replay(before):
    fm = FM.FilmModel.__new__(FM.FilmModel)
    if NEW_PLAN:
        upgrade(before["film"])
    fm.__dict__.update(before["film"])
    st = dict(before["sim"])
    ncomp = st.pop("ncomp")
    sim = SimpleNamespace(**st, comp=SimpleNamespace(n=ncomp))
    fm.sim = sim
    t = time.perf_counter()
    inj = fm.update(before["dt"])
    return fm, sim, inj, time.perf_counter() - t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tags", nargs="+")
    ap.add_argument("--threads", type=int, nargs="+", default=[16])
    ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args()
    raw = {}
    for t in a.tags:
        with open(os.path.join(CAP, t + ".pkl"), "rb") as f:
            raw[t] = f.read()
    for thr in a.threads:
        numba.set_num_threads(thr)
        for tag in a.tags:
            T = []
            ACC.clear()
            for r in range(a.reps + 1):
                before, after = pickle.loads(raw[tag]), None
                with open(os.path.join(CAP, tag + ".pkl"), "rb") as f:
                    pickle.load(f)
                    after = pickle.load(f)
                if r == 1:
                    ACC.clear()                       # the first one warms up
                if tag in BC:
                    before["film"]["_bc"] = BC[tag]  # static: built once in a run
                fm, sim, inj, sec = replay(before)
                if getattr(fm, "_bc", None) is not None:
                    BC[tag] = fm._bc
                if r >= 1:
                    T.append(sec)
                same = (np.array_equal(inj, after["inj"]) and np.array_equal(fm.s.h, after["h"])
                        and np.array_equal(fm.s.sub, after["sub"])
                        and np.array_equal(sim.L, after["L"])
                        and len(fm.s.drips) == after["drips"]
                        and fm.drip_volume == after["drip_volume"])
                if not same:
                    raise SystemExit(f"{tag}: DIFFERENT")
            print(f"== {tag} ({thr} threads, nsub {fm.nsub}): {np.mean(T):.3f} s per call "
                  f"(captured run {after['seconds']:.3f} s), same results")
            for k, v in sorted(ACC.items(), key=lambda kv: -kv[1])[:12]:
                print(f"   {k:22s} {v / a.reps * 1e3:8.1f} ms")


if __name__ == "__main__":
    main()
