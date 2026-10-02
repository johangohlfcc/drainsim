"""Replay captured connected-region calls (capture_label.py): the same
results bit for bit, the time per call, and the phases of par._roots
(chunk unions in parallel, the edges between chunks in sequence, the roots)
with the number of edges between chunks.
    python bench_label.py dip12_0 dip12_1 --threads 16 --reps 3"""
import argparse
import os
import sys
import time

import numpy as np

HOME = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
REPO = os.environ.get("DS_REPO", HOME)          # the drainsim to measure (default: this one)
sys.path.insert(0, REPO)
import numba                                                       # noqa: E402
from numba import njit, prange                                     # noqa: E402
from drainsim import par                                           # noqa: E402

CAP = os.path.join(HOME, "runs", "perf", "captures", "label")


@njit(parallel=True, cache=True)
def phase1(mask, comp, ptr, idx, C):
    N = mask.shape[0]
    uf = np.empty(N, np.int64)
    for c in prange(N):
        uf[c] = c
    ncross = np.zeros(C + 1, np.int64)
    for k in prange(C):
        lo = k * N // C
        hi = (k + 1) * N // C
        m = 0
        for c in range(lo, hi):
            if not mask[c]:
                continue
            for jj in range(ptr[c], ptr[c + 1]):
                n = idx[jj]
                if n < 0 or n < c or not mask[n] or comp[n] != comp[c]:
                    continue
                if n >= hi:
                    m += 1
                    continue
                a = c
                while uf[a] != a:
                    uf[a] = uf[uf[a]]
                    a = uf[a]
                b = n
                while uf[b] != b:
                    uf[b] = uf[uf[b]]
                    b = uf[b]
                if a != b:
                    if a < b:
                        uf[b] = a
                    else:
                        uf[a] = b
        ncross[k + 1] = m
    return uf, ncross


@njit(parallel=True, cache=True)
def edges_between(mask, comp, ptr, idx, C):
    """The edges from a chunk to a later one (as _roots collects them)."""
    N = mask.shape[0]
    cnt = np.zeros(C + 1, np.int64)
    for k in prange(C):
        lo = k * N // C
        hi = (k + 1) * N // C
        m = 0
        for c in range(lo, hi):
            if not mask[c]:
                continue
            for jj in range(ptr[c], ptr[c + 1]):
                n = idx[jj]
                if n >= hi and mask[n] and comp[n] == comp[c]:
                    m += 1
        cnt[k + 1] = m
    for k in range(C):
        cnt[k + 1] += cnt[k]
    ea = np.empty(cnt[C], np.int64)
    eb = np.empty(cnt[C], np.int64)
    for k in prange(C):
        lo = k * N // C
        hi = (k + 1) * N // C
        o = cnt[k]
        for c in range(lo, hi):
            if not mask[c]:
                continue
            for jj in range(ptr[c], ptr[c + 1]):
                n = idx[jj]
                if n >= hi and mask[n] and comp[n] == comp[c]:
                    ea[o] = c
                    eb[o] = n
                    o += 1
    return ea, eb


@njit(cache=True)
def phase2(ea, eb, uf):
    for e in range(ea.shape[0]):
        a = ea[e]
        while uf[a] != a:
            uf[a] = uf[uf[a]]
            a = uf[a]
        b = eb[e]
        while uf[b] != b:
            uf[b] = uf[uf[b]]
            b = uf[b]
        if a != b:
            if a < b:
                uf[b] = a
            else:
                uf[a] = b
    return uf


def load(tag, st):
    z = np.load(os.path.join(CAP, tag + ".npz"))
    a = {k: z[k] for k in z.files}
    a["comp"] = st["lab"] if a["comp"].size == 0 else a["comp"]
    return a


def call(a, st):
    if str(a["kind"]) == "connected_to":
        return par.connected_to(a["mask"], a["comp"], st["ptr"], st["idx"], a["seed"])
    return par.label_bodies(a["mask"], a["comp"], st["ptr"], st["idx"])


def tm(f, *x, reps=3):
    f(*x)
    T = []
    for _ in range(reps):
        t = time.perf_counter()
        r = f(*x)
        T.append(time.perf_counter() - t)
    return r, 1e3 * float(np.median(T))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tags", nargs="+")
    ap.add_argument("--threads", type=int, nargs="+", default=[16])
    ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args()
    z = np.load(os.path.join(CAP, "static.npz"))
    st = {k: z[k] for k in z.files}
    for thr in a.threads:
        numba.set_num_threads(thr)
        for tag in a.tags:
            c = load(tag, st)
            r, ms = tm(call, c, st, reps=a.reps)
            r0 = r[0] if isinstance(r, tuple) else r
            if not np.array_equal(r0, c["res"]) or (isinstance(r, tuple) and r[1] != c["nb"]):
                raise SystemExit(f"{tag}: DIFFERENT")
            C = par._nchunks(c["mask"].shape[0], thr)
            mask = np.ascontiguousarray(c["mask"], np.bool_)
            (uf, ncross), t1 = tm(phase1, mask, c["comp"], st["ptr"], st["idx"], C, reps=a.reps)
            (ea, eb), t1b = tm(edges_between, mask, c["comp"], st["ptr"], st["idx"], C,
                               reps=a.reps)
            t2 = []
            for _ in range(a.reps + 1):
                u = uf.copy()
                t = time.perf_counter()
                phase2(ea, eb, u)
                t2.append(time.perf_counter() - t)
            print(f"{tag:10s} {thr:2d} thr  {str(c['kind']):13s} masked {int(mask.sum()):9d}  "
                  f"call {ms:6.1f} ms (captured {1e3 * float(c['seconds']):6.1f})  "
                  f"chunk unions {t1:5.1f} ms, edges between chunks {ea.size:8d} (listed in "
                  f"{t1b:5.1f} ms), joined in sequence {1e3 * np.median(t2[1:]):6.1f} ms  same")


if __name__ == "__main__":
    main()
