"""Work counters of fsm._fill on captured calls: leaves with water, spills,
neighbour scans, child-of-P walk steps; and the time of _fill itself."""
import sys, time
import numpy as np
sys.argv += [] 
import bench_fsm as B
from numba import njit
from drainsim import fsm
from drainsim.fsm import OCEAN

@njit(cache=False)
def counts(sdest, sreg, sw, h, region, nptr, nidx, rank, dest, owner0, node_parent,
           node_spill, node_region, nnodes, cap, spill_routing, nregions, tol, nn, tin, tout):
    cnt = np.zeros(6, np.int64)
    vol = np.zeros(nnodes); full = np.zeros(nnodes, np.bool_)
    cand = np.empty(nn, np.int64); wleaf = np.zeros(nnodes)
    for j in range(sw.shape[0]):
        if sdest[j] != OCEAN:
            wleaf[sdest[j]] += sw[j]
    for leaf in range(1, nnodes):
        w = wleaf[leaf]
        if w <= 0.0:
            continue
        cnt[0] += 1
        node = leaf; mode = 0
        while True:
            if mode == 0:
                cnt[5] += 1
                room = cap[node] - vol[node]
                if room > 0.0:
                    put = w if w < room else room
                    vol[node] += put; w -= put
                if cap[node] - vol[node] <= tol:
                    full[node] = True
                if w <= tol:
                    break
                mode = 1
            else:
                cnt[1] += 1
                P = node_parent[node]
                if P < 0:
                    break
                s = node_spill[node]
                nc = 0
                for jj in range(nptr[s], nptr[s + 1]):
                    n = nidx[jj]
                    cnt[2] += 1
                    if n >= 0 and region[n] == region[s] and rank[n] >= 0 and rank[n] < rank[s]:
                        cand[nc] = n; nc += 1
                for a in range(1, nc):
                    x = cand[a]; b = a - 1
                    while b >= 0 and h[cand[b]] > h[x]:
                        cand[b + 1] = cand[b]; b -= 1
                    cand[b + 1] = x
                found = False; to_ocean = False
                for a in range(nc):
                    n = cand[a]
                    T = owner0[n]
                    if node > 0 and T > 0 and tin[node] <= tin[T] and tin[T] < tout[node]:
                        continue
                    L = dest[n]
                    if L == OCEAN:
                        to_ocean = True; break
                    T = L
                    while T >= 0 and node_parent[T] != P:
                        T = node_parent[T]; cnt[3] += 1
                    if T < 0:
                        cnt[4] += 1
                        continue
                    if full[T]:
                        if P == OCEAN:
                            to_ocean = True; break
                        continue
                    node = L; mode = 0; found = True; break
                if to_ocean:
                    break
                if not found:
                    if P == OCEAN:
                        break
                    node = P; mode = 0
    return cnt

z = np.load(B.CAP + r"\static.npz"); st = {k: z[k] for k in z.files}
for tag in sys.argv[1:]:
    c = B.load(tag, st)
    h = c["h"] - c["ecell"]
    region, extra = fsm._prune_sinks(c["region"], c["sink"], c["src"], st["ptr"], st["idx"], int(c["nregions"]))
    P = fsm.prepare(h, region, st["ptr"], st["idx"], c["sink"], st["v"], float(c["min_depth"]), c["ecell"], eshape=c["eshape"])
    sd, sr, sw = fsm._par_sources(P.order, c["src"], P.dest, P.region, 16)
    args = (sd, sr, sw, P.h, P.region, P.nptr, P.nidx, P.rank, P.dest, P.owner0, P.node_parent,
            P.node_spill, P.node_region, P.nnodes, P.cap, bool(c["spill_routing"]), int(c["nregions"]), P.tol, P.nn, P.tin, P.tout)
    k = counts(*args)
    fsm._fill(*args)
    t = time.perf_counter(); fsm._fill(*args); dt = time.perf_counter() - t
    print(f"{tag}: nodes {P.nnodes}, sources {sw.size}, leaves with water {k[0]}, fills {k[5]}, spills {k[1]}, "
          f"nbr scans {k[2]}, child-walk steps {k[3]}, not under P {k[4]}; _fill {dt*1e3:.1f} ms")
