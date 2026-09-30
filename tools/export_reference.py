"""Reference data for porting drainsim to another code (e.g. IBOFlow, C++).

Writes self-contained JSON files that a port can load in its unit tests and
compare against, without Python:

  fsm_cases.json      fill-spill-merge on random graphs (inputs + outputs)
  cdf_cases.json      the cell volume distribution (cube below a plane)
  throat_cases.json   throat law: head, hold-up, flow, venting limit
  film_cases.json     film deposition (LLD) and one implicit drainage sweep

    python tools/export_reference.py --out reference/ [--n-fsm 200]

Format of fsm_cases.json: {"cases": [case, ...]}, each case with
  n, indptr, indices (CSR; -1 entries = no link), h, region (-1 inactive),
  sink (0/1), src, vcell, ecell, eshape (3 numbers, see below), min_depth,
  spill_routing (0/1), nregions,
  and the expected results
  retained (per node), drained, lost (per region), owner (depression id,
  0 = ocean, -1 inactive), plus "order" (the height order the sweep uses).
Compare retained/drained/lost with a relative tolerance of 1e-12 (of the
largest retained value) and owner exactly.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from drainsim import fsm                                      # noqa: E402
from drainsim.grid import Grid                                # noqa: E402
from drainsim.physics import Fluid, ThroatModel               # noqa: E402


def _lst(a):
    a = np.asarray(a)
    if a.dtype == np.bool_:
        return a.astype(int).tolist()
    return a.tolist()


def fsm_cases(n, seed=7):
    rng = np.random.default_rng(seed)
    out = []
    for trial in range(n):
        nd = int(rng.integers(2, 4))
        shape = tuple(int(x) for x in rng.integers(3, 10 if nd == 3 else 16, size=nd))
        g = Grid(shape=shape, dx=1.0, origin=np.zeros(nd),
                 solid=rng.random(shape) < rng.uniform(0, 0.4))
        P, I = fsm.to_csr(g.neighbors())
        N = g.ncells
        h = np.round(rng.integers(0, rng.integers(2, 30), N) * 0.1
                     + (rng.random(N) * 0.05 if trial % 2 else 0), 9)
        nreg = 3 if trial % 3 == 0 else 1
        region = np.where(g.fluid.ravel(), rng.integers(0, nreg, N), -1).astype(np.int64)
        sink = (rng.random(N) < rng.uniform(0, 0.2)) & (region >= 0)
        src = np.where(region >= 0, rng.random(N) * rng.uniform(0, 3), 0.0)
        vc = rng.uniform(0.5, 1.0, N)
        ec = rng.uniform(0.0, 0.2, N) if trial % 4 else np.zeros(N)
        md = [0.0, 0.1, 0.3][trial % 3]
        sr = trial % 5 != 0
        es = fsm.LINEAR if trial % 2 == 0 else fsm.box_shape(rng.normal(size=3))
        ret, dr, lost, owner = fsm.fill_spill(h, region, (P, I), sink, src, vc, sr, nreg,
                                              md, ec, method="serial", eshape=es)
        # the sweep order: by cell floor (h - ecell) when ecell is given
        order = fsm.height_order(h - ec if np.any(ec > 0) else h, sink,
                                 np.flatnonzero(region >= 0))
        out.append(dict(n=int(N), indptr=_lst(P), indices=_lst(I), h=_lst(h),
                        region=_lst(region), sink=_lst(sink), src=_lst(src), vcell=_lst(vc),
                        ecell=_lst(ec), eshape=_lst(es), min_depth=md,
                        spill_routing=int(sr),
                        nregions=nreg, order=_lst(order), retained=_lst(ret),
                        drained=_lst(dr), lost=_lst(lost), owner=_lst(owner)))
    return out


def cdf_cases(seed=11):
    rng = np.random.default_rng(seed)
    rows = []
    shapes = [fsm.LINEAR, fsm.box_shape([0.087, 0, 0.996]), fsm.box_shape([1, 1, 1]),
              fsm.box_shape([0.3, -0.5, 0.8])] + [fsm.box_shape(rng.normal(size=3))
                                                  for _ in range(6)]
    for w in shapes:
        for t in np.r_[0.0, 1.0, np.linspace(0.01, 0.99, 25), rng.random(10)]:
            F = float(fsm.cell_cdf(float(t), w))
            rows.append([float(w[0]), float(w[1]), float(w[2]), float(t), F,
                         float(fsm.cell_cdf_inv(F, w))])
    return dict(columns=["a", "b", "c", "t", "F", "Finv_of_F"], rows=rows)


def throat_cases():
    fl = Fluid()
    tm = ThroatModel(Cd=0.65)
    rows = []
    for d in (0.002, 0.005, 0.008, 0.010, 0.019, 0.030):
        area = np.pi * d * d / 4
        for head in (0.0, 0.001, 0.005, 0.02, 0.1):
            hold = tm.holdup_head(d, fl, 3)
            q_free = tm.flow(area, head - hold, fl)
            q_sub = tm.flow(area, head, fl)
            rows.append(dict(diameter=d, area=area, head=head, holdup_head=hold,
                             q_free_outflow=q_free, q_submerged=q_sub,
                             d_crit=tm.d_crit(fl, 3), unvented_flows=bool(d >= tm.d_crit(fl, 3)),
                             counter_current_factor=tm.counter_current_factor))
    return dict(fluid=dict(rho=fl.rho, mu=fl.mu, sigma=fl.sigma, g=fl.g,
                           capillary_length=fl.capillary_length),
                Cd=tm.Cd, rows=rows)


def film_cases():
    from drainsim.film import _implicit_sweep
    fl = Fluid()
    lc = fl.capillary_length
    dep = []
    for U in (1e-4, 1e-3, 5e-3, 0.02, 0.1, 0.5):
        Ca = fl.mu * U / fl.sigma
        dep.append(dict(U=U, Ca=Ca, h0=lc * min(0.94 * Ca ** (2 / 3), Ca ** 0.5)))
    # one implicit sweep on a vertical strip of 20 elements (top first)
    n = 20
    A = np.full(n, 1e-4)                         # 1 cm x 1 cm elements
    h0 = np.full(n, 50e-6)
    k = fl.rho / (3 * fl.mu)
    gt = fl.g
    C = np.full(n, k * 0.01 * gt)                # edge length 1 cm, downhill neighbour
    C[-1] = 0.0                                  # bottom element: no outflow
    indptr = np.r_[np.arange(n), n - 1].astype(np.int64)
    recv = np.arange(1, n).astype(np.int64)
    w = C[:-1].copy()
    order = np.arange(n, dtype=np.int64)
    sub = np.zeros(n, np.bool_)
    V, ab = _implicit_sweep(order, A, h0 * A, C, indptr, recv, w, 0.05, sub)
    return dict(deposition=dep,
                sweep=dict(n=n, area=_lst(A), V_old=_lst(h0 * A), C=_lst(C), indptr=_lst(indptr),
                           recv=_lst(recv), w=_lst(w), dt=0.05, order=_lst(order),
                           submerged=_lst(sub), V_new=_lst(V), absorbed=_lst(ab)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reference")
    ap.add_argument("--n-fsm", type=int, default=200)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for name, obj in (("fsm_cases.json", dict(cases=fsm_cases(a.n_fsm))),
                      ("cdf_cases.json", cdf_cases()),
                      ("throat_cases.json", throat_cases()),
                      ("film_cases.json", film_cases())):
        p = os.path.join(a.out, name)
        with open(p, "w") as f:
            json.dump(obj, f)
        print("wrote", p, f"({os.path.getsize(p)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
