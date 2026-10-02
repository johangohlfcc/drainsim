"""Fill-Spill-Merge (FSM) on a voxel graph.

A 3D generalisation of the depression-hierarchy / fill-spill-merge idea used
for surface-water routing in hydrology (Barnes, Callaghan & Wickert, Earth
Surf. Dynam. 2020-21), here on face-connected fluid cells of a voxel grid and
with an arbitrary gravity direction.

Given a "height" per cell (projection of the cell centre on the up-vector),
a set of sink cells (bath / domain boundary: the "ocean"), and a source volume
per cell, the kernel returns the equilibrium volume retained in every cell:

1. **Depression hierarchy** - cells are swept in increasing height and merged
   with union-find. A cell with no lower neighbour starts a new *leaf*
   depression. A cell that joins two or more components is a *spill point*
   (saddle): the components become children of a new *meta-depression*
   (or of the ocean if one of them already drains to a sink).
2. **Routing** - every source parcel follows steepest descent to the leaf it
   drains into.
3. **Fill & spill** - a leaf fills its own cells; the excess spills over its
   saddle into the sibling the saddle's steepest-descent path leads to. When
   all siblings are full the meta-depression's own cells fill, and so on up
   the tree. Anything reaching the ocean is drained.

Liquid is therefore conserved: what escapes one cavity is routed to the next
cavity, or to a sink - never simply deleted (unless ``spill_routing=False``,
which reproduces the "escaped liquid is gone" behaviour for comparison).

The same kernel is used for air with the height negated (air "falls" up).
"""
from __future__ import annotations

import time

import numpy as np
from numba import njit, prange, get_num_threads

from .par import sort_keys, compact, fingerprint, dual

OCEAN = 0
TIMING = None             # dict: accumulated seconds per part (for benchmarks)
CAND_FILTER = None        # drop redundant saddle candidates (None: with >= 4 threads)

LINEAR = np.array([1.0, 0.0, 0.0])      # eshape of the linear portion rule


def box_shape(up):
    """``eshape`` of an axis-aligned cube for the up-vector ``up``: the
    widths of the cube's projection on ``up`` per axis, normalised to sum 1
    (sorted, largest first; widths below 1e-4 are dropped)."""
    w = np.abs(np.asarray(up, float).ravel())
    w = np.sort(np.r_[w, np.zeros(3)][:3])[::-1]
    w = w / w.sum()
    w[w < 1e-4] = 0.0
    return w / w.sum()


@njit(cache=True, nogil=True)
def _p(x, n):
    return x ** n if x > 0.0 else 0.0


@njit(cache=True, nogil=True)
def _cdf_raw(t, a, b, c):
    if b <= 0.0:
        return t
    if c <= 0.0:
        return (_p(t, 2) - _p(t - a, 2) - _p(t - b, 2) + _p(t - a - b, 2)) / (2.0 * a * b)
    return (_p(t, 3) - _p(t - a, 3) - _p(t - b, 3) - _p(t - c, 3) + _p(t - a - b, 3)
            + _p(t - a - c, 3) + _p(t - b - c, 3)) / (6.0 * a * b * c)


@njit(cache=True, nogil=True)
def cell_cdf(t, w):
    """Volume fraction of a cell below the normalised height t (0 = cell
    bottom, 1 = top). ``w`` = (a, b, c), a + b + c = 1: the cell is the sum
    of three uniform segments of these lengths along the up-vector (an
    axis-aligned cube: ``box_shape``); (1, 0, 0) is the linear rule."""
    if t <= 0.0:
        return 0.0
    if t >= 1.0:
        return 1.0
    if t <= 0.5:
        f = _cdf_raw(t, w[0], w[1], w[2])
    else:
        f = 1.0 - _cdf_raw(1.0 - t, w[0], w[1], w[2])
    return min(max(f, 0.0), 1.0)


@njit(cache=True, nogil=True)
def cell_cdf_vec(t, w):
    """``cell_cdf`` of an array of normalised heights."""
    out = np.empty(t.shape[0])
    for i in range(t.shape[0]):
        out[i] = cell_cdf(t[i], w)
    return out


@njit(cache=True, nogil=True)
def cell_cdf_inv(f, w):
    """Normalised height below which the fraction f of the cell lies."""
    if f <= 0.0:
        return 0.0
    if f >= 1.0:
        return 1.0
    if w[1] <= 0.0:
        return f
    lo = 0.0
    hi = 1.0
    for _ in range(50):
        m = 0.5 * (lo + hi)
        if cell_cdf(m, w) < f:
            lo = m
        else:
            hi = m
    return 0.5 * (lo + hi)



@njit(cache=True, nogil=True)
def _find(uf, x):
    r = x
    while uf[r] != r:
        r = uf[r]
    while uf[x] != r:          # path compression
        nx = uf[x]
        uf[x] = r
        x = nx
    return r


def to_csr(nbr):
    """Neighbour graph as (indptr, indices). Accepts a (N, W) table with -1
    for no neighbour (e.g. ``Grid.neighbors()``) or an (indptr, indices)
    pair, which is returned unchanged. Negative entries are ignored by all
    kernels, so a link can be cut by setting it to -1."""
    if isinstance(nbr, tuple):
        return nbr
    nbr = np.asarray(nbr)
    ok = nbr >= 0
    ptr = np.zeros(nbr.shape[0] + 1, np.int64)
    np.cumsum(ok.sum(1), out=ptr[1:])
    return ptr, nbr[ok].astype(np.int64)


@njit(cache=True, nogil=True)
def _hierarchy_serial(order, h, region, nptr, nidx, sink):
    """Depression hierarchy by a union-find sweep in height order (serial).
    Returns rank, owner0, dest, node_parent, node_spill, node_region, nnodes."""
    N = h.shape[0]
    nn = 1
    for c in range(N):
        if nptr[c + 1] - nptr[c] > nn:
            nn = nptr[c + 1] - nptr[c]
    rank = -np.ones(N, np.int64)
    for i in range(order.shape[0]):
        rank[order[i]] = i

    uf = np.arange(N)
    node_of_root = -np.ones(N, np.int64)
    owner0 = -np.ones(N, np.int64)
    dest = -np.ones(N, np.int64)

    maxnodes = order.shape[0] + 1
    node_parent = -np.ones(maxnodes, np.int64)
    node_spill = -np.ones(maxnodes, np.int64)
    node_region = -np.ones(maxnodes, np.int64)
    nnodes = 1                                   # node 0 = ocean

    roots = np.empty(nn, np.int64)
    nodes = np.empty(nn, np.int64)

    # ------------------------------------------------ 1. depression hierarchy
    for i in range(order.shape[0]):
        c = order[i]
        if sink[c]:
            node_of_root[c] = OCEAN
            owner0[c] = OCEAN
            dest[c] = OCEAN
            continue
        rc = region[c]
        k = 0
        ptr = -1
        for jj in range(nptr[c], nptr[c + 1]):
            n = nidx[jj]
            if n < 0 or region[n] != rc or rank[n] < 0 or rank[n] > i:
                continue
            if ptr < 0 or rank[n] < rank[ptr]:
                ptr = n
            r = _find(uf, n)
            nd = node_of_root[r]
            dup = False
            for q in range(k):
                if nodes[q] == nd:
                    dup = True
                    if roots[q] != r:        # same node, other root: merge roots
                        uf[r] = roots[q]
                    break
            if not dup:
                roots[k] = r
                nodes[k] = nd
                k += 1
        if k == 0:                                # new leaf depression
            m = nnodes
            nnodes += 1
            node_region[m] = rc
            node_of_root[c] = m
            owner0[c] = m
            dest[c] = m
        elif k == 1:
            uf[c] = roots[0]
            owner0[c] = nodes[0]
            dest[c] = dest[ptr]
        else:                                     # saddle: merge
            P = -1
            for q in range(k):
                if nodes[q] == OCEAN:
                    P = OCEAN
            if P < 0:
                P = nnodes
                nnodes += 1
                node_region[P] = rc
            for q in range(k):
                if nodes[q] != P:
                    node_parent[nodes[q]] = P
                    node_spill[nodes[q]] = c
            for q in range(k):
                uf[roots[q]] = c
            uf[c] = c
            node_of_root[c] = P
            owner0[c] = P
            dest[c] = dest[ptr]

    return rank, owner0, dest, node_parent, node_spill, node_region, nnodes


# ------------------------------------------------ parallel hierarchy
# The union-find sweep builds the join tree of the height function on the
# graph: at the time of node c, the components are the connected parts of the
# nodes at or below c. Every node is connected to the local minimum its
# steepest-descent path ends in (its basin), through nodes that are lower
# still. So the components at any time are unions of basins, and two basins
# join exactly when a node has lower neighbours in both. The tree can then be
# built as:
#   1. (parallel) steepest lower neighbour of every node;
#   2. (serial, one pass, cheap) basin of every node, in height order;
#   3. (parallel) candidate saddles: nodes with a lower neighbour in another
#      basin (only these can join components);
#   4. (serial, candidates only) union-find over basins at the candidates, in
#      height order: a candidate that sees two or more components creates the
#      meta node (or joins the ocean), as in the sweep;
#   5. nodes numbered in the order the sweep creates them (by the height rank
#      of the creating node), so the result is identical, node ids included;
#   6. (parallel) owner0 and dest of every node: the top of its basin's
#      branch at its own time.

@dual
def _par_rank_ptr(order, N, region, nptr, nidx, sink):
    n = order.shape[0]
    # node-sized arrays are filled in parallel (np.full would run on one
    # thread inside the kernel)
    rank = np.empty(N, np.int64)
    ptr = np.empty(N, np.int64)
    for c in prange(N):
        rank[c] = -1
        ptr[c] = -1
    for i in prange(n):
        rank[order[i]] = i
    for i in prange(n):
        c = order[i]
        if sink[c]:
            continue
        rc = region[c]
        best = -1
        bestr = n
        for jj in range(nptr[c], nptr[c + 1]):
            q = nidx[jj]
            if q < 0 or region[q] != rc:
                continue
            rq = rank[q]
            if rq < 0 or rq >= i:
                continue
            if best < 0 or rq < bestr:
                best = q
                bestr = rq
        ptr[c] = best
    return rank, ptr


@njit(cache=True, nogil=True)
def _basins(order, N, sink, ptr):
    """Basin index of every node (1.. = leaf, in creation order; 0 = ocean)
    and the creating cell of every leaf."""
    basin = np.full(N, -1, np.int64)
    nl = 0
    for i in range(order.shape[0]):
        c = order[i]
        if sink[c]:
            basin[c] = 0
        elif ptr[c] < 0:
            nl += 1
            basin[c] = nl
        else:
            basin[c] = basin[ptr[c]]
    leaf_cell = np.empty(nl + 1, np.int64)
    leaf_cell[0] = -1
    for i in range(order.shape[0]):
        c = order[i]
        if not sink[c] and ptr[c] < 0:
            leaf_cell[basin[c]] = c
    return basin, leaf_cell


@dual
def _par_candidates(order, rank, region, nptr, nidx, sink, basin):
    n = order.shape[0]
    flag = np.zeros(n, np.bool_)
    for i in prange(n):
        c = order[i]
        if sink[c]:
            continue
        rc = region[c]
        b = basin[c]
        for jj in range(nptr[c], nptr[c + 1]):
            q = nidx[jj]
            if q < 0 or region[q] != rc:
                continue
            rq = rank[q]
            if rq < 0 or rq >= i:
                continue
            if basin[q] != b:
                flag[i] = True
                break
    return flag


@dual
def _par_basins(order, N, sink, ptr, rank, nt):
    """``_basins`` in parallel: chunks of the height order are swept in
    parallel (a pointer into an earlier chunk is kept as it is), then every
    node follows those pointers back to its root (at most one hop per
    earlier chunk). Leaves are numbered in height order as in the sweep."""
    n = order.shape[0]
    C = 4 * nt
    if C > n:
        C = max(n, 1)
    # A and lnum are only read where they were written (the nodes of the
    # order, the leaves): no fill needed
    A = np.empty(N, np.int64)
    nleaf = np.zeros(C + 1, np.int64)
    for k in prange(C):
        lo = k * n // C
        hi = (k + 1) * n // C
        m = 0
        for i in range(lo, hi):
            c = order[i]
            p = ptr[c]
            if sink[c] or p < 0:
                A[c] = c
                if not sink[c]:
                    m += 1
            elif rank[p] >= lo:
                A[c] = A[p]
            else:
                A[c] = p
        nleaf[k + 1] = m
    for k in range(C):
        nleaf[k + 1] += nleaf[k]
    nl = nleaf[C]
    leaf_cell = np.empty(nl + 1, np.int64)
    leaf_cell[0] = -1
    lnum = np.empty(N, np.int64)
    for k in prange(C):
        lo = k * n // C
        hi = (k + 1) * n // C
        o = nleaf[k]
        for i in range(lo, hi):
            c = order[i]
            if not sink[c] and ptr[c] < 0:
                o += 1
                lnum[c] = o
                leaf_cell[o] = c
    basin = np.empty(N, np.int64)
    for c in prange(N):
        basin[c] = -1
    for i in prange(n):
        c = order[i]
        r = A[c]
        while A[r] != r:
            r = A[r]
        b = np.int64(0)
        if not sink[r]:
            b = lnum[r]
        basin[c] = b
    return basin, leaf_cell


@dual
def _par_cand_sets(order, rank, region, nptr, nidx, basin, cand):
    """Up to four distinct basins below every candidate (-2 in the first
    column: more than four)."""
    nc = cand.shape[0]
    S = np.full((nc, 4), -1, np.int64)
    for k in prange(nc):
        i = cand[k]
        c = order[i]
        rc = region[c]
        m = 0
        for jj in range(nptr[c], nptr[c + 1]):
            q = nidx[jj]
            if q < 0 or region[q] != rc:
                continue
            rq = rank[q]
            if rq < 0 or rq >= i:
                continue
            x = basin[q]
            dup = False
            for t in range(min(m, 4)):
                if S[k, t] == x:
                    dup = True
                    break
            if dup:
                continue
            if m < 4:
                S[k, m] = x
            m += 1
        if m > 4:
            S[k, 0] = -2
    return S


@dual
def _par_cand_filter(order, rank, region, nptr, nidx, cand, cidx, S):
    """Drop candidates that cannot create a node. After a candidate q is
    swept, the basins below it are in one component. So a candidate whose
    basins are already joined through lower neighbours that are candidates
    joins nothing when it is swept, and the sweep result is unchanged
    without it. (Tested locally: basins linked through the sets of the
    lower candidate neighbours.)"""
    nc = cand.shape[0]
    keep = np.zeros(nc, np.bool_)
    for k in prange(nc):
        if S[k, 0] == -2:
            keep[k] = True
            continue
        m = 0
        while m < 4 and S[k, m] >= 0:
            m += 1
        full = (1 << m) - 1
        # adjacency of the m basins as bit masks
        a0 = 1
        a1 = 2
        a2 = 4
        a3 = 8
        i = cand[k]
        c = order[i]
        rc = region[c]
        for jj in range(nptr[c], nptr[c + 1]):
            q = nidx[jj]
            if q < 0 or region[q] != rc:
                continue
            rq = rank[q]
            if rq < 0 or rq >= i:
                continue
            kq = cidx[rq]
            if kq < 0 or S[kq, 0] == -2:
                continue
            qm = 0
            for t in range(4):
                x = S[kq, t]
                if x < 0:
                    break
                for u in range(m):
                    if S[k, u] == x:
                        qm |= 1 << u
                        break
            if qm & (qm - 1):                 # two or more of ours: linked
                if qm & 1:
                    a0 |= qm
                if qm & 2:
                    a1 |= qm
                if qm & 4:
                    a2 |= qm
                if qm & 8:
                    a3 |= qm
        reach = 1
        for _ in range(4):
            r2 = reach
            if reach & 1:
                r2 |= a0
            if reach & 2:
                r2 |= a1
            if reach & 4:
                r2 |= a2
            if reach & 8:
                r2 |= a3
            reach = r2
        keep[k] = (reach & full) != full
    return keep


@dual
def _index_of(n, cand):
    cidx = np.empty(n, np.int64)
    for i in prange(n):
        cidx[i] = -1
    for k in prange(cand.shape[0]):
        cidx[cand[k]] = k
    return cidx


@njit(cache=True, nogil=True)
def _saddle_events(order, rank, region, nptr, nidx, basin, nl, cand):
    """Union-find over basins at the candidate saddles, in height order.
    Node handles: ocean 0, leaf k (1..nl), meta -(j+1)."""
    NONE = -(1 << 62)
    nn = 1
    for i in range(cand.shape[0]):
        c = order[cand[i]]
        if nptr[c + 1] - nptr[c] > nn:
            nn = nptr[c + 1] - nptr[c]
    uf = np.arange(nl + 1)
    top = np.arange(nl + 1)
    lpar = np.full(nl + 1, NONE, np.int64)
    lsp = np.full(nl + 1, -1, np.int64)
    ncand = cand.shape[0]
    mpar = np.full(ncand + 1, NONE, np.int64)
    msp = np.full(ncand + 1, -1, np.int64)
    mcell = np.full(ncand + 1, -1, np.int64)
    nm = 0
    evh = np.full(ncand, NONE, np.int64)
    tops = np.empty(nn, np.int64)
    rts = np.empty(nn, np.int64)
    for q in range(ncand):
        i = cand[q]
        c = order[i]
        rc = region[c]
        k = 0
        for jj in range(nptr[c], nptr[c + 1]):
            nb = nidx[jj]
            if nb < 0 or region[nb] != rc:
                continue
            rq = rank[nb]
            if rq < 0 or rq >= i:
                continue
            r = _find(uf, basin[nb])
            t = top[r]
            dup = False
            for x in range(k):
                if tops[x] == t:
                    dup = True
                    if rts[x] != r:              # same node, other root: merge
                        uf[r] = rts[x]
                    break
            if not dup:
                tops[k] = t
                rts[k] = r
                k += 1
        if k < 2:
            continue
        P = NONE
        for x in range(k):
            if tops[x] == 0:
                P = 0
        if P == NONE:
            P = -(nm + 1)
            mcell[nm] = c
            nm += 1
        for x in range(k):
            t = tops[x]
            if t == P:
                continue
            if t > 0:
                lpar[t] = P
                lsp[t] = c
            else:
                mpar[-t - 1] = P
                msp[-t - 1] = c
        R = rts[0]
        for x in range(1, k):
            r2 = _find(uf, rts[x])
            if r2 != R:
                uf[r2] = R
        top[R] = P
        evh[q] = P
    return lpar, lsp, mpar[:nm], msp[:nm], mcell[:nm], evh


@njit(cache=True, nogil=True)
def _number_nodes(rank, region, leaf_cell, lpar, lsp, mpar, msp, mcell):
    """Node ids in the order the sweep creates them (by the rank of the
    creating node); parent, spill cell and region per id."""
    NONE = -(1 << 62)
    nl = leaf_cell.shape[0] - 1
    nm = mcell.shape[0]
    nnodes = 1 + nl + nm
    lid = np.zeros(nl + 1, np.int64)
    mid = np.zeros(nm, np.int64)
    a = 1
    b = 0
    nid = 1
    while a <= nl or b < nm:
        if b >= nm or (a <= nl and rank[leaf_cell[a]] < rank[mcell[b]]):
            lid[a] = nid
            a += 1
        else:
            mid[b] = nid
            b += 1
        nid += 1
    node_parent = np.full(nnodes, -1, np.int64)
    node_spill = np.full(nnodes, -1, np.int64)
    node_region = np.full(nnodes, -1, np.int64)
    for k in range(1, nl + 1):
        m = lid[k]
        p = lpar[k]
        node_parent[m] = -1 if p == NONE else (0 if p == 0 else (lid[p] if p > 0 else mid[-p - 1]))
        node_spill[m] = lsp[k]
        node_region[m] = region[leaf_cell[k]]
    for j in range(nm):
        m = mid[j]
        p = mpar[j]
        node_parent[m] = -1 if p == NONE else (0 if p == 0 else (lid[p] if p > 0 else mid[-p - 1]))
        node_spill[m] = msp[j]
        node_region[m] = region[mcell[j]]
    return lid, mid, node_parent, node_spill, node_region, nnodes


@dual
def _event_ids(n, cand, evh, lid, mid):
    NONE = -(1 << 62)
    evid = np.empty(n, np.int64)
    for i in prange(n):
        evid[i] = -1
    for q in prange(cand.shape[0]):
        e = evh[q]
        if e == NONE:
            continue
        evid[cand[q]] = 0 if e == 0 else (lid[e] if e > 0 else mid[-e - 1])
    return evid


@dual
def _par_owner(order, N, rank, sink, ptr, basin, lid, evid, node_parent, node_spill):
    n = order.shape[0]
    owner0 = np.empty(N, np.int64)
    dest = np.empty(N, np.int64)
    for c in prange(N):
        owner0[c] = -1
        dest[c] = -1
    for i in prange(n):
        c = order[i]
        if sink[c]:
            owner0[c] = 0
            dest[c] = 0
            continue
        b = basin[c]
        m = 0 if b == 0 else lid[b]
        dest[c] = m
        if evid[i] >= 0:                          # a saddle: its new node
            owner0[c] = evid[i]
            continue
        if ptr[c] < 0:                            # a leaf's own minimum
            owner0[c] = m
            continue
        while m > 0 and node_parent[m] >= 0 and rank[node_spill[m]] < i:
            m = node_parent[m]
        owner0[c] = m
    return owner0, dest


def _tick(name, t0):
    """Add the time since t0 to TIMING[name]; returns the current time."""
    t = time.perf_counter()
    if TIMING is not None:
        TIMING[name] = TIMING.get(name, 0.0) + t - t0
    return t


def hierarchy_parallel(order, h, region, nptr, nidx, sink, filt=None, full=False):
    """Same result as ``_hierarchy_serial``, with the per-node work in
    parallel (numba prange) and only the candidate saddles in sequence.
    ``filt`` (default: with 4 or more threads) drops the candidates that
    cannot create a node first; the result is the same either way. With
    ``full``, the lowest cell of every leaf node (-1 for others) is also
    returned."""
    N = h.shape[0]
    t = time.perf_counter()
    rank, ptr = _par_rank_ptr(order, N, region, nptr, nidx, sink)
    t = _tick("P rank+ptr", t)
    if get_num_threads() > 1:
        basin, leaf_cell = _par_basins(order, N, sink, ptr, rank, get_num_threads())
    else:
        basin, leaf_cell = _basins(order, N, sink, ptr)
    t = _tick("P basins", t)
    flag = _par_candidates(order, rank, region, nptr, nidx, sink, basin)
    cand = compact(flag)
    del flag
    t = _tick("P candidates", t)
    if filt is None:
        filt = CAND_FILTER if CAND_FILTER is not None else get_num_threads() >= 4
    if filt and cand.size:
        S = _par_cand_sets(order, rank, region, nptr, nidx, basin, cand)
        cidx = _index_of(order.shape[0], cand)
        cand = cand[_par_cand_filter(order, rank, region, nptr, nidx, cand, cidx, S)]
        del S, cidx
        t = _tick("P candidate filter", t)
    nl = leaf_cell.shape[0] - 1
    lpar, lsp, mpar, msp, mcell, evh = _saddle_events(order, rank, region, nptr, nidx,
                                                      basin, nl, cand)
    lid, mid, node_parent, node_spill, node_region, nnodes = _number_nodes(
        rank, region, leaf_cell, lpar, lsp, mpar, msp, mcell)
    evid = _event_ids(order.shape[0], cand, evh, lid, mid)
    t = _tick("S saddles", t)
    owner0, dest = _par_owner(order, N, rank, sink, ptr, basin, lid, evid,
                              node_parent, node_spill)
    t = _tick("P owner", t)
    H = (rank, owner0, dest, node_parent, node_spill, node_region, nnodes)
    if full:
        node_cell = np.full(nnodes, -1, np.int64)
        node_cell[lid[1:]] = leaf_cell[1:]
        return H + (node_cell,)
    return H



# ------------------------------------------------ the hierarchy in rank space
# The kernels above visit the nodes in height order but read node-indexed
# arrays (and the neighbours' entries) all over memory. Renumbered once into
# height order (rank space: row i is the node order[i], its links the ranks
# of its active neighbours in the same region, in link order), the same
# steps read nearby entries: a node's neighbours are within about a cell in
# height, so within a slab of nearby ranks. The steps are the same, in the
# same order; node ids, spill cells and leaf cells are node indices as
# before, and owner0 and dest are written back in node numbering.

def hierarchy_rank(order, h, region, nptr, nidx, sink, filt=None, full=False):
    """``hierarchy_parallel`` (same arguments, same results) computed in
    rank space."""
    N = h.shape[0]
    n = order.shape[0]
    t = time.perf_counter()
    rank, rptr, ridx = _rank_graph(order, N, region, nptr, nidx)
    sink_s = _gather_b(sink, order)
    ptr_s = _rank_ptr(rptr, ridx, sink_s)
    t = _tick("P rank+ptr", t)
    basin, leaf_cell = _rank_basins(ptr_s, sink_s, order)
    t = _tick("S basins", t)
    cand = compact(_rank_candidates(rptr, ridx, sink_s, basin))
    t = _tick("P candidates", t)
    if filt is None:
        filt = CAND_FILTER if CAND_FILTER is not None else get_num_threads() >= 4
    if filt and cand.size:
        S = _rank_cand_sets(rptr, ridx, basin, cand)
        cidx = _index_of(n, cand)
        cand = cand[_rank_cand_filter(rptr, ridx, cand, cidx, S)]
        del S, cidx
        t = _tick("P candidate filter", t)
    nl = leaf_cell.shape[0] - 1
    lpar, lsp, mpar, msp, mcell, evh = _rank_saddle_events(order, rptr, ridx, basin, nl, cand)
    lid, mid, node_parent, node_spill, node_region, nnodes = _number_nodes(
        rank, region, leaf_cell, lpar, lsp, mpar, msp, mcell)
    evid = _event_ids(n, cand, evh, lid, mid)
    t = _tick("S saddles", t)
    owner0, dest = _rank_owner(order, N, sink_s, ptr_s, basin, lid, evid, node_parent,
                               _spill_values(rank, node_spill, node_parent))
    t = _tick("P owner", t)
    H = (rank, owner0, dest, node_parent, node_spill, node_region, nnodes)
    if full:
        node_cell = np.full(nnodes, -1, np.int64)
        node_cell[lid[1:]] = leaf_cell[1:]
        return H + (node_cell,)
    return H


@dual
def _rank_graph(order, N, region, nptr, nidx):
    """rank (node -> rank, -1 if not active) and the graph in rank space:
    row i (node order[i]) holds the ranks of its active neighbours in the
    same region, in link order (rptr, ridx)."""
    n = order.shape[0]
    rank = np.empty(N, np.int64)
    for c in prange(N):
        rank[c] = -1
    for i in prange(n):
        rank[order[i]] = i
    cap = np.empty(n + 1, np.int64)
    cap[0] = 0
    for i in prange(n):
        c = order[i]
        cap[i + 1] = nptr[c + 1] - nptr[c]
    for i in range(n):
        cap[i + 1] += cap[i]
    tmp = np.empty(cap[n], np.int32)
    rptr = np.empty(n + 1, np.int64)
    rptr[0] = 0
    for i in prange(n):
        c = order[i]
        rc = region[c]
        o = cap[i]
        m = 0
        for jj in range(nptr[c], nptr[c + 1]):
            q = nidx[jj]
            if q < 0 or region[q] != rc:
                continue
            rq = rank[q]
            if rq < 0:
                continue
            tmp[o + m] = rq
            m += 1
        rptr[i + 1] = m
    for i in range(n):
        rptr[i + 1] += rptr[i]
    ridx = np.empty(rptr[n], np.int32)
    for i in prange(n):
        o = cap[i]
        for k in range(rptr[i + 1] - rptr[i]):
            ridx[rptr[i] + k] = tmp[o + k]
    return rank, rptr, ridx


@dual
def _gather_b(a, idx):
    out = np.empty(idx.shape[0], np.bool_)
    for i in prange(idx.shape[0]):
        out[i] = a[idx[i]]
    return out


@dual
def _rank_ptr(rptr, ridx, sink_s):
    """Steepest lower neighbour of every rank (the lowest rank below it),
    -1 for sinks and local minima."""
    n = rptr.shape[0] - 1
    ptr = np.empty(n, np.int64)
    for i in prange(n):
        best = -1
        if not sink_s[i]:
            for jj in range(rptr[i], rptr[i + 1]):
                j = ridx[jj]
                if j < i and (best < 0 or j < best):
                    best = j
        ptr[i] = best
    return ptr


@njit(cache=True, nogil=True)
def _rank_basins(ptr_s, sink_s, order):
    """``_basins`` in rank space: basin of every rank (1.. = leaf, in
    creation order; 0 = ocean), and the creating cell (node) of every
    leaf."""
    n = ptr_s.shape[0]
    basin = np.empty(n, np.int64)
    nl = 0
    for i in range(n):
        if sink_s[i]:
            basin[i] = 0
        elif ptr_s[i] < 0:
            nl += 1
            basin[i] = nl
        else:
            basin[i] = basin[ptr_s[i]]
    leaf_cell = np.empty(nl + 1, np.int64)
    leaf_cell[0] = -1
    for i in range(n):
        if not sink_s[i] and ptr_s[i] < 0:
            leaf_cell[basin[i]] = order[i]
    return basin, leaf_cell


@dual
def _rank_candidates(rptr, ridx, sink_s, basin):
    n = rptr.shape[0] - 1
    flag = np.zeros(n, np.bool_)
    for i in prange(n):
        if sink_s[i]:
            continue
        b = basin[i]
        for jj in range(rptr[i], rptr[i + 1]):
            j = ridx[jj]
            if j < i and basin[j] != b:
                flag[i] = True
                break
    return flag


@dual
def _rank_cand_sets(rptr, ridx, basin, cand):
    """``_par_cand_sets`` in rank space."""
    nc = cand.shape[0]
    S = np.full((nc, 4), -1, np.int64)
    for k in prange(nc):
        i = cand[k]
        m = 0
        for jj in range(rptr[i], rptr[i + 1]):
            j = ridx[jj]
            if j >= i:
                continue
            x = basin[j]
            dup = False
            for t in range(min(m, 4)):
                if S[k, t] == x:
                    dup = True
                    break
            if dup:
                continue
            if m < 4:
                S[k, m] = x
            m += 1
        if m > 4:
            S[k, 0] = -2
    return S


@dual
def _rank_cand_filter(rptr, ridx, cand, cidx, S):
    """``_par_cand_filter`` in rank space."""
    nc = cand.shape[0]
    keep = np.zeros(nc, np.bool_)
    for k in prange(nc):
        if S[k, 0] == -2:
            keep[k] = True
            continue
        m = 0
        while m < 4 and S[k, m] >= 0:
            m += 1
        full = (1 << m) - 1
        a0 = 1
        a1 = 2
        a2 = 4
        a3 = 8
        i = cand[k]
        for jj in range(rptr[i], rptr[i + 1]):
            j = ridx[jj]
            if j >= i:
                continue
            kq = cidx[j]
            if kq < 0 or S[kq, 0] == -2:
                continue
            qm = 0
            for t in range(4):
                x = S[kq, t]
                if x < 0:
                    break
                for u in range(m):
                    if S[k, u] == x:
                        qm |= 1 << u
                        break
            if qm & (qm - 1):
                if qm & 1:
                    a0 |= qm
                if qm & 2:
                    a1 |= qm
                if qm & 4:
                    a2 |= qm
                if qm & 8:
                    a3 |= qm
        reach = 1
        for _ in range(4):
            r2 = reach
            if reach & 1:
                r2 |= a0
            if reach & 2:
                r2 |= a1
            if reach & 4:
                r2 |= a2
            if reach & 8:
                r2 |= a3
            reach = r2
        keep[k] = (reach & full) != full
    return keep


@njit(cache=True, nogil=True)
def _rank_saddle_events(order, rptr, ridx, basin, nl, cand):
    """``_saddle_events`` in rank space (the cells recorded are nodes)."""
    NONE = -(1 << 62)
    nn = 1
    for i in range(cand.shape[0]):
        d = rptr[cand[i] + 1] - rptr[cand[i]]
        if d > nn:
            nn = d
    uf = np.arange(nl + 1)
    top = np.arange(nl + 1)
    lpar = np.full(nl + 1, NONE, np.int64)
    lsp = np.full(nl + 1, -1, np.int64)
    ncand = cand.shape[0]
    mpar = np.full(ncand + 1, NONE, np.int64)
    msp = np.full(ncand + 1, -1, np.int64)
    mcell = np.full(ncand + 1, -1, np.int64)
    nm = 0
    evh = np.full(ncand, NONE, np.int64)
    tops = np.empty(nn, np.int64)
    rts = np.empty(nn, np.int64)
    for q in range(ncand):
        i = cand[q]
        c = order[i]
        k = 0
        for jj in range(rptr[i], rptr[i + 1]):
            j = ridx[jj]
            if j >= i:
                continue
            r = _find(uf, basin[j])
            t = top[r]
            dup = False
            for x in range(k):
                if tops[x] == t:
                    dup = True
                    if rts[x] != r:              # same node, other root: merge
                        uf[r] = rts[x]
                    break
            if not dup:
                tops[k] = t
                rts[k] = r
                k += 1
        if k < 2:
            continue
        P = NONE
        for x in range(k):
            if tops[x] == 0:
                P = 0
        if P == NONE:
            P = -(nm + 1)
            mcell[nm] = c
            nm += 1
        for x in range(k):
            t = tops[x]
            if t == P:
                continue
            if t > 0:
                lpar[t] = P
                lsp[t] = c
            else:
                mpar[-t - 1] = P
                msp[-t - 1] = c
        R = rts[0]
        for x in range(1, k):
            r2 = _find(uf, rts[x])
            if r2 != R:
                uf[r2] = R
        top[R] = P
        evh[q] = P
    return lpar, lsp, mpar[:nm], msp[:nm], mcell[:nm], evh


@dual
def _rank_owner(order, N, sink_s, ptr_s, basin, lid, evid, node_parent, srank):
    """``_par_owner`` from rank-space inputs (``srank``: the rank of each
    node's spill cell); owner0 and dest by node."""
    n = order.shape[0]
    owner0 = np.empty(N, np.int64)
    dest = np.empty(N, np.int64)
    for c in prange(N):
        owner0[c] = -1
        dest[c] = -1
    for i in prange(n):
        c = order[i]
        if sink_s[i]:
            owner0[c] = 0
            dest[c] = 0
            continue
        b = basin[i]
        m = 0 if b == 0 else lid[b]
        dest[c] = m
        if evid[i] >= 0:                          # a saddle: its new node
            owner0[c] = evid[i]
            continue
        if ptr_s[i] < 0:                          # a leaf's own minimum
            owner0[c] = m
            continue
        while m > 0 and node_parent[m] >= 0 and srank[m] < i:
            m = node_parent[m]
        owner0[c] = m
    return owner0, dest


@njit(cache=True, nogil=True)
def _spill_values(a, node_spill, node_parent):
    """a[node_spill[m]] for every node with a parent (0 for the others): a
    table as small as the hierarchy, read instead of a node-sized array."""
    nn = node_spill.shape[0]
    out = np.zeros(nn, a.dtype)
    for m in range(nn):
        if node_parent[m] >= 0 and node_spill[m] >= 0:
            out[m] = a[node_spill[m]]
    return out

@njit(cache=True, nogil=True)
def _fsm_rest(order, h, region, nptr, nidx, sink, src, vcell, spill_routing,
              nregions, min_depth, ecell, eshape, flat, rank, owner0, dest, node_parent, node_spill,
              node_region, nnodes):
    """Everything after the hierarchy: shallow depressions, ownership and
    portions, fill & spill, volumes back to the cells."""
    N = h.shape[0]
    nn = 1
    for c in range(N):
        if nptr[c + 1] - nptr[c] > nn:
            nn = nptr[c + 1] - nptr[c]
    vmax = 0.0
    for i in range(order.shape[0]):
        if vcell[order[i]] > vmax:
            vmax = vcell[order[i]]
    tol = 1e-12 * vmax

    # ------------------ shallow depressions (sub-grid roughness) hold nothing
    # A depression whose depth (spill level minus its lowest cell) is below
    # ``min_depth`` is merged into the first ancestor that is deep enough
    # (or the ocean): its cells only fill if that ancestor's pool covers them.
    tgt = np.arange(nnodes)
    if min_depth > 0.0:
        minh = np.full(nnodes, np.inf)
        for i in range(order.shape[0]):
            c = order[i]
            m = owner0[c]
            if m > 0 and h[c] < minh[m]:
                minh[m] = h[c]
        for m in range(1, nnodes):              # children have smaller ids
            p = node_parent[m]
            if p > 0 and minh[m] < minh[p]:
                minh[p] = minh[m]
        for m in range(nnodes - 1, 0, -1):      # parents have larger ids
            p = node_parent[m]
            if p >= 0 and h[node_spill[m]] - minh[m] < min_depth:
                tgt[m] = OCEAN if p == OCEAN else tgt[p]

    # ------------------------------ ownership for filling (below the spill)
    # A depression holds the cells below its spill point. With ecell given,
    # ``h`` here is the bottom of every cell (``fill_spill`` passes h - e):
    # cells are swept, and depressions merge, when the water reaches their
    # floor, so the spill level is the floor of the saddle cell. A cell
    # whose extent [h, h + 2e] straddles
    # the spill levels of the depression holding it and of its ancestors is
    # cut into portions: [bottom, H0] to that depression, [H0, H1] to its
    # parent, and so on (at most MAXP portions; the last one takes the rest
    # of the cell to the first ancestor whose spill level is above its top).
    # With ecell = 0 every cell is one portion, the plain "cells strictly
    # below the saddle" rule.
    MAXP = 4
    owner = owner0.copy()
    pn = -np.ones((N, MAXP), np.int32)          # portion node (after tgt)
    pv = np.zeros((N, MAXP))                   # portion volume
    cap = np.zeros(nnodes)
    for i in range(order.shape[0]):
        c = order[i]
        m = owner[c]
        if m == OCEAN:
            continue
        while m != OCEAN and node_parent[m] >= 0 and h[c] >= h[node_spill[m]]:
            m = node_parent[m]
        mt = tgt[m]
        owner[c] = mt
        if ecell[c] <= 0.0 or m == OCEAN or node_parent[m] < 0:
            pn[c, 0] = mt
            pv[c, 0] = vcell[c]
            if mt != OCEAN:
                cap[mt] += vcell[c]
            continue
        bot = h[c]
        span = 2.0 * ecell[c]
        lo = bot
        k = 0
        node = m
        while True:
            if k == MAXP - 2:
                # two slots left: the cell may straddle more spill levels
                # (nested shallow depressions). Up to the highest of them the
                # cell goes to that depression (the ancestor of the others,
                # so the capacity of the tree is kept), the rest above it
                # to its parent (last slot).
                while node != OCEAN and node_parent[node] >= 0:
                    p_ = node_parent[node]
                    if p_ == OCEAN or node_parent[p_] < 0 or h[node_spill[p_]] >= bot + span:
                        break
                    node = p_
            if k == MAXP - 1:
                # last slot: the rest of the cell goes to the first ancestor
                # whose spill level is above the cell top (or the ocean)
                while node != OCEAN and node_parent[node] >= 0:
                    s_ = node_spill[node]
                    if h[s_] >= bot + span:
                        break
                    node = node_parent[node]
            last = node == OCEAN or node_parent[node] < 0 or k == MAXP - 1
            if last:
                hi = bot + span
            else:
                s_ = node_spill[node]
                hi = min(h[s_], bot + span)
            if hi > lo:
                vv = vcell[c] * (cell_cdf((hi - bot) / span, eshape) - cell_cdf((lo - bot) / span, eshape))
                nt = tgt[node]
                if k > 0 and pn[c, k - 1] == nt:
                    pv[c, k - 1] += vv
                else:
                    pn[c, k] = nt
                    pv[c, k] = vv
                    k += 1
                if nt != OCEAN:
                    cap[nt] += vv
                lo = hi
            if last or lo >= bot + span - 1e-15 * span:
                break
            node = node_parent[node]

    # ------------------------------------------------------ 2-3. fill & spill
    vol = np.zeros(nnodes)
    full = np.zeros(nnodes, np.bool_)
    drained = np.zeros(nregions)
    lost = np.zeros(nregions)
    cand = np.empty(nn, np.int64)

    # route every source parcel to its leaf first (one cascade per leaf)
    wleaf = np.zeros(nnodes)
    for i in range(order.shape[0]):
        c = order[i]
        w = src[c]
        if w <= 0.0:
            continue
        if dest[c] == OCEAN:
            drained[region[c]] += w
        else:
            wleaf[dest[c]] += w

    for leaf in range(1, nnodes):
        w = wleaf[leaf]
        if w <= 0.0:
            continue
        rc = node_region[leaf]
        node = leaf
        mode = 0                      # 0 = fill own layer of node, 1 = spill node
        while True:
            if mode == 0:
                room = cap[node] - vol[node]
                if room > 0.0:
                    put = w if w < room else room
                    vol[node] += put
                    w -= put
                if cap[node] - vol[node] <= tol:
                    full[node] = True
                if w <= tol:
                    break
                mode = 1
            else:
                if not spill_routing:
                    drained[rc] += w
                    break
                P = node_parent[node]
                if P < 0:                         # closed region is overfull
                    lost[rc] += w
                    break
                s = node_spill[node]
                # candidate neighbours of the saddle, lowest first
                nc = 0
                for jj in range(nptr[s], nptr[s + 1]):
                    n = nidx[jj]
                    if n >= 0 and region[n] == region[s] and rank[n] >= 0 \
                            and rank[n] < rank[s]:
                        cand[nc] = n
                        nc += 1
                # insertion sort by height
                for a in range(1, nc):
                    x = cand[a]
                    b = a - 1
                    while b >= 0 and h[cand[b]] > h[x]:
                        cand[b + 1] = cand[b]
                        b -= 1
                    cand[b + 1] = x
                found = False
                to_ocean = False
                for a in range(nc):
                    n = cand[a]
                    # skip neighbours inside the overflowing depression itself
                    T = owner0[n]
                    inside = False
                    while T > 0:
                        if T == node:
                            inside = True
                            break
                        T = node_parent[T]
                    if inside:
                        continue
                    # follow steepest descent from the far side of the saddle
                    L = dest[n]
                    if L == OCEAN:
                        to_ocean = True
                        break
                    T = L
                    while T >= 0 and node_parent[T] != P:
                        T = node_parent[T]
                    if T < 0:
                        continue
                    if full[T]:
                        if P == OCEAN:       # passes over a full cavity -> sink
                            to_ocean = True
                            break
                        continue
                    node = L
                    mode = 0
                    found = True
                    break
                if to_ocean:
                    drained[rc] += w
                    break
                if not found:
                    if P == OCEAN:
                        drained[rc] += w
                        break
                    node = P
                    mode = 0

    # --------------------------------------------- 4. volumes back to cells
    if flat:
        retained = _flat_fill_serial(order, h, ecell, eshape, vcell, pn[order], pv[order],
                                     vol, cap, tol, nnodes, N)
        return retained, drained, lost, owner
    # Fill each node's own cells lowest first. Cells at exactly the same height
    # (flat floors when gravity is grid aligned) share the partial layer
    # equally instead of filling in index order.
    retained = np.zeros(N)
    INF = np.inf
    hp = np.full(nnodes, INF)                 # height of the partial layer
    vfull = np.zeros(nnodes)
    for i in range(order.shape[0]):
        c = order[i]
        for q in range(MAXP):
            m = pn[c, q]
            if m < 0:
                break
            vv = pv[c, q]
            if m == OCEAN or vv <= 0.0 or hp[m] < INF:
                continue
            if vfull[m] + vv <= vol[m] + tol:
                vfull[m] += vv
            else:
                hp[m] = h[c]
    vbelow = np.zeros(nnodes)
    veq = np.zeros(nnodes)
    for i in range(order.shape[0]):
        c = order[i]
        for q in range(MAXP):
            m = pn[c, q]
            if m < 0:
                break
            vv = pv[c, q]
            if m == OCEAN or vv <= 0.0:
                continue
            if h[c] < hp[m]:
                vbelow[m] += vv
            elif h[c] == hp[m]:
                veq[m] += vv
    for i in range(order.shape[0]):
        c = order[i]
        for q in range(MAXP):
            m = pn[c, q]
            if m < 0:
                break
            vv = pv[c, q]
            if m == OCEAN or vv <= 0.0:
                continue
            if h[c] < hp[m]:
                retained[c] += vv
            elif h[c] == hp[m] and veq[m] > 0:
                f = (vol[m] - vbelow[m]) / veq[m]
                f = min(max(f, 0.0), 1.0)
                retained[c] += f * vv
    return retained, drained, lost, owner


def _flat_fill(cells, h, ecell, eshape, vcell, pn, pv, vol, cap, tol, nnodes, N):
    """Volumes back to the cells with a flat level per partly filled node.

    ``cells`` in ascending ``h`` (the cell floors), ``pn``/``pv`` their
    portions (row r = cells[r]). A full node fills all its portions; a partly
    filled node m fills its portions below one level H_m with
    sum(portion volume below H_m) = vol[m], each cell's volume spread over its
    extent [h, h + 2e] by ``cell_cdf``. H_m lies in [hp, hp + S], hp being
    the floor at which the node's whole cells (in floor order) exceed vol[m]
    and S its largest cell extent, so only the cells in that band are
    bisected. The band's share is scaled to give vol[m] exactly.
    (In parallel: the rows that touch a partly filled node are picked first
    and the full nodes are filled row by row; the passes over the partly
    filled nodes, in row order, see only those rows.)"""
    part = _part_nodes(vol, cap, tol, nnodes)
    rsel = compact(_flat_rows(pn, part))
    retained = _flat_full(cells, pn, pv, part, vol, N)
    _flat_core(cells, rsel, h, ecell, eshape, vcell, pn, pv, vol, part, tol, nnodes,
               retained)
    return retained


@njit(cache=True, nogil=True)
def _flat_fill_serial(cells, h, ecell, eshape, vcell, pn, pv, vol, cap, tol, nnodes, N):
    """``_flat_fill`` in one thread (for the serial kernel)."""
    part = _part_nodes(vol, cap, tol, nnodes)
    flag = _flat_rows_1(pn, part)
    rsel = np.flatnonzero(flag)
    retained = _flat_full_1(cells, pn, pv, part, vol, N)
    _flat_core(cells, rsel, h, ecell, eshape, vcell, pn, pv, vol, part, tol, nnodes,
               retained)
    return retained


@njit(cache=True, nogil=True)
def _part_nodes(vol, cap, tol, nnodes):
    """The partly filled nodes."""
    part = np.zeros(nnodes, np.bool_)
    for m in range(nnodes):
        part[m] = m != OCEAN and vol[m] > 0.0 and vol[m] < cap[m] - tol
    return part


@dual
def _flat_rows(pn, part):
    """Rows with a portion of a partly filled node."""
    n = pn.shape[0]
    f = np.empty(n, np.bool_)
    for r in prange(n):
        hit = False
        for q in range(pn.shape[1]):
            m = pn[r, q]
            if m < 0:
                break
            if part[m]:
                hit = True
                break
        f[r] = hit
    return f


@dual
def _flat_full(cells, pn, pv, part, vol, N):
    """retained: the portions of full nodes, row by row (each cell has one
    row, so every cell sums its portions in the same order)."""
    retained = np.empty(N)
    for c in prange(N):
        retained[c] = 0.0
    for r in prange(cells.shape[0]):
        c = cells[r]
        for q in range(pn.shape[1]):
            m = pn[r, q]
            if m < 0:
                break
            if m == OCEAN or part[m] or vol[m] <= 0.0:
                continue
            retained[c] += pv[r, q]                     # full node
    return retained


_flat_rows_1 = _flat_rows.serial
_flat_full_1 = _flat_full.serial


@njit(cache=True, nogil=True)
def _flat_core(cells, rsel, h, ecell, eshape, vcell, pn, pv, vol, part, tol, nnodes,
               retained):
    """The partly filled nodes of ``_flat_fill``: their levels, and their
    volumes added to ``retained``; ``rsel`` the rows that touch them, in
    order."""
    MAXP = pn.shape[1]
    INF = np.inf
    hp = np.full(nnodes, INF)
    vfull = np.zeros(nnodes)
    S = np.zeros(nnodes)
    for r in rsel:
        c = cells[r]
        sp = 2.0 * ecell[c]
        for q in range(MAXP):
            m = pn[r, q]
            if m < 0:
                break
            if not part[m]:
                continue
            if sp > S[m]:
                S[m] = sp
            if hp[m] < INF:
                continue
            vv = pv[r, q]
            if vfull[m] + vv <= vol[m] + tol:
                vfull[m] += vv
            else:
                hp[m] = h[c]
    # entries fully below hp (full), in the band, or above hp + S (empty)
    base = np.zeros(nnodes)
    nb = 0
    for r in rsel:
        c = cells[r]
        for q in range(MAXP):
            m = pn[r, q]
            if m < 0:
                break
            if part[m] and h[c] + 2.0 * ecell[c] > hp[m] and h[c] < hp[m] + S[m]:
                nb += 1
    br = np.empty(nb, np.int64)
    bm = np.empty(nb, np.int64)
    bF = np.empty(nb)
    bw = np.empty(nb)
    k = 0
    for r in rsel:
        c = cells[r]
        F = 0.0
        for q in range(MAXP):
            m = pn[r, q]
            if m < 0:
                break
            w = pv[r, q] / vcell[c] if vcell[c] > 0.0 else 0.0
            if part[m]:
                if h[c] + 2.0 * ecell[c] <= hp[m]:
                    base[m] += pv[r, q]
                elif h[c] < hp[m] + S[m]:
                    br[k] = r
                    bm[k] = m
                    bF[k] = F
                    bw[k] = w
                    k += 1
            F += w
    lo = hp.copy()
    hi = hp + S
    acc = np.zeros(nnodes)
    npart = 0
    for m in range(nnodes):
        if part[m]:
            npart += 1
    plist = np.empty(npart, np.int64)
    k = 0
    for m in range(nnodes):
        if part[m]:
            plist[k] = m
            k += 1
    for it in range(48):                         # bracket / 2**48
        for m in plist:
            acc[m] = base[m]
        for j in range(nb):
            m = bm[j]
            c = cells[br[j]]
            H = 0.5 * (lo[m] + hi[m])
            sp = 2.0 * ecell[c]
            if sp > 0.0:
                f = cell_cdf((H - h[c]) / sp, eshape) - bF[j]
            else:
                f = bw[j] if h[c] < H else 0.0
            f = min(max(f, 0.0), bw[j])
            acc[m] += f * vcell[c]
        for m in plist:
            H = 0.5 * (lo[m] + hi[m])
            if acc[m] > vol[m]:
                hi[m] = H
            else:
                lo[m] = H
    # final level, band share scaled to vol exactly
    band = np.zeros(nnodes)
    fb = np.empty(nb)
    for j in range(nb):
        m = bm[j]
        c = cells[br[j]]
        H = 0.5 * (lo[m] + hi[m])
        sp = 2.0 * ecell[c]
        if sp > 0.0:
            f = cell_cdf((H - h[c]) / sp, eshape) - bF[j]
        else:
            f = bw[j] if h[c] < H else 0.0
        f = min(max(f, 0.0), bw[j]) * vcell[c]
        fb[j] = f
        band[m] += f
    scale = np.zeros(nnodes)
    for m in range(nnodes):
        if part[m] and band[m] > 0.0:
            scale[m] = max(vol[m] - base[m], 0.0) / band[m]
    for r in rsel:
        c = cells[r]
        for q in range(MAXP):
            m = pn[r, q]
            if m < 0:
                break
            if part[m] and h[c] + 2.0 * ecell[c] <= hp[m]:
                retained[c] += pv[r, q]
    for j in range(nb):
        retained[cells[br[j]]] += fb[j] * scale[bm[j]]
    return retained


@njit(cache=True, nogil=True)
def _fsm_kernel(order, h, region, nptr, nidx, sink, src, vcell, spill_routing,
                nregions, min_depth, ecell, eshape, flat):
    rank, owner0, dest, node_parent, node_spill, node_region, nnodes = \
        _hierarchy_serial(order, h, region, nptr, nidx, sink)
    return _fsm_rest(order, h, region, nptr, nidx, sink, src, vcell, spill_routing,
                     nregions, min_depth, ecell, eshape, flat, rank, owner0, dest, node_parent,
                     node_spill, node_region, nnodes)



# ------------------------------------------------ parallel remaining passes
@njit(cache=True, nogil=True)
def _max_degree(nptr):
    nn = 1
    for c in range(nptr.shape[0] - 1):
        d = nptr[c + 1] - nptr[c]
        if d > nn:
            nn = d
    return nn


_DEG = [None, 0]


def _degree(nptr):
    """Largest node degree (remembered for the last graph)."""
    key = (id(nptr), nptr.shape[0], int(nptr[-1]))
    if _DEG[0] != key:
        _DEG[0], _DEG[1] = key, int(_max_degree(nptr))
    return _DEG[1]


@dual
def _par_vmax(order, vcell):
    vmax = 0.0
    for i in prange(order.shape[0]):
        vmax = max(vmax, vcell[order[i]])
    return vmax


@njit(cache=True, nogil=True)
def _targets(h, node_parent, node_spill, node_cell, nnodes, min_depth):
    """Shallow depressions (sub-grid roughness) hold nothing: a depression
    whose depth (spill level minus its lowest cell) is below ``min_depth`` is
    merged into the first ancestor that is deep enough (or the ocean): its
    cells only fill if that ancestor's pool covers them. The lowest cell of
    a leaf is the cell that creates it, and of a meta-depression the lowest
    of its children's (the serial kernel finds the same by a pass over all
    cells)."""
    tgt = np.arange(nnodes)
    if min_depth > 0.0:
        minh = np.full(nnodes, np.inf)
        for m in range(1, nnodes):
            if node_cell[m] >= 0:
                minh[m] = h[node_cell[m]]
        for m in range(1, nnodes):              # children have smaller ids
            p = node_parent[m]
            if p > 0 and minh[m] < minh[p]:
                minh[p] = minh[m]
        for m in range(nnodes - 1, 0, -1):      # parents have larger ids
            p = node_parent[m]
            if p >= 0 and h[node_spill[m]] - minh[m] < min_depth:
                tgt[m] = OCEAN if p == OCEAN else tgt[p]
    return tgt


@dual
def _par_portions(order, N, h, ecell, eshape, vcell, owner0, node_parent, sh, tgt):
    """Ownership and portions of every active cell (independent per cell).
    pn and pv are indexed by height rank (row i = cell order[i]); ``use``
    (``_par_row_flag``) marks the rows with a portion in a depression.
    ``sh``: the height of each node's spill cell (``_spill_values``)."""
    MAXP = 4
    n = order.shape[0]
    owner = np.empty(N, np.int64)
    for c in prange(N):
        owner[c] = owner0[c]
    pn = np.empty((n, MAXP), np.int32)
    pv = np.empty((n, MAXP))
    for i in prange(n):
        for q in range(MAXP):
            pn[i, q] = -1
            pv[i, q] = 0.0
        c = order[i]
        m = owner[c]
        if m == OCEAN:
            continue
        while m != OCEAN and node_parent[m] >= 0 and h[c] >= sh[m]:
            m = node_parent[m]
        mt = tgt[m]
        owner[c] = mt
        if ecell[c] <= 0.0 or m == OCEAN or node_parent[m] < 0:
            pn[i, 0] = mt
            pv[i, 0] = vcell[c]
            continue
        bot = h[c]
        span = 2.0 * ecell[c]
        lo = bot
        k = 0
        node = m
        while True:
            if k == MAXP - 2:
                # see _fsm_rest: up to the highest spill level in the cell
                while node != OCEAN and node_parent[node] >= 0:
                    p_ = node_parent[node]
                    if p_ == OCEAN or node_parent[p_] < 0 or sh[p_] >= bot + span:
                        break
                    node = p_
            if k == MAXP - 1:
                while node != OCEAN and node_parent[node] >= 0:
                    if sh[node] >= bot + span:
                        break
                    node = node_parent[node]
            last = node == OCEAN or node_parent[node] < 0 or k == MAXP - 1
            if last:
                hi = bot + span
            else:
                hi = min(sh[node], bot + span)
            if hi > lo:
                vv = vcell[c] * (cell_cdf((hi - bot) / span, eshape) - cell_cdf((lo - bot) / span, eshape))
                nt = tgt[node]
                if k > 0 and pn[i, k - 1] == nt:
                    pv[i, k - 1] += vv
                else:
                    pn[i, k] = nt
                    pv[i, k] = vv
                    k += 1
                lo = hi
            if last or lo >= bot + span - 1e-15 * span:
                break
            node = node_parent[node]
    return owner, pn, pv


@njit(cache=True, nogil=True)
def _cap_of(pn, pv, nnodes):
    """Capacity of every node (rows in height order)."""
    cap = np.zeros(nnodes)
    for i in range(pn.shape[0]):
        for q in range(pn.shape[1]):
            m = pn[i, q]
            if m < 0:
                break
            if m != OCEAN:
                cap[m] += pv[i, q]
    return cap


@dual
def _par_sources(order, src, dest, region, nt):
    """Destination leaf, region and volume of the cells with src > 0, in
    height order."""
    n = order.shape[0]
    C = 4 * nt
    if C > n:
        C = max(n, 1)
    cnt = np.zeros(C + 1, np.int64)
    for k in prange(C):
        lo = k * n // C
        hi = (k + 1) * n // C
        m = 0
        for i in range(lo, hi):
            if src[order[i]] > 0.0:
                m += 1
        cnt[k + 1] = m
    for k in range(C):
        cnt[k + 1] += cnt[k]
    K = cnt[C]
    sd = np.empty(K, np.int64)
    sr = np.empty(K, np.int64)
    sw = np.empty(K)
    for k in prange(C):
        lo = k * n // C
        hi = (k + 1) * n // C
        o = cnt[k]
        for i in range(lo, hi):
            c = order[i]
            w = src[c]
            if w > 0.0:
                sd[o] = dest[c]
                sr[o] = region[c]
                sw[o] = w
                o += 1
    return sd, sr, sw


@njit(cache=True, nogil=True)
def _fill(sdest, sreg, sw, node_parent, node_region, nnodes, cap, spill_routing, nregions,
          tol, sptr, sL, sT):
    """Fill & spill cascade. The sources are given as the destination leaf,
    region and volume of every cell with src > 0, in height order
    (``_par_sources``), which is the order the serial kernel adds them in.
    Where a full node spills to is read from its spill table
    (``_spill_table``: the saddle's candidates, in the order and with the
    tests of the serial kernel); only the full flags change here."""
    # ------------------------------------------------------ 2-3. fill & spill
    vol = np.zeros(nnodes)
    full = np.zeros(nnodes, np.bool_)
    drained = np.zeros(nregions)
    lost = np.zeros(nregions)

    # route every source parcel to its leaf first (one cascade per leaf)
    wleaf = np.zeros(nnodes)
    for j in range(sw.shape[0]):
        if sdest[j] == OCEAN:
            drained[sreg[j]] += sw[j]
        else:
            wleaf[sdest[j]] += sw[j]

    for leaf in range(1, nnodes):
        w = wleaf[leaf]
        if w <= 0.0:
            continue
        rc = node_region[leaf]
        node = leaf
        mode = 0                      # 0 = fill own layer of node, 1 = spill node
        while True:
            if mode == 0:
                room = cap[node] - vol[node]
                if room > 0.0:
                    put = w if w < room else room
                    vol[node] += put
                    w -= put
                if cap[node] - vol[node] <= tol:
                    full[node] = True
                if w <= tol:
                    break
                mode = 1
            else:
                if not spill_routing:
                    drained[rc] += w
                    break
                P = node_parent[node]
                if P < 0:                         # closed region is overfull
                    lost[rc] += w
                    break
                found = False
                to_ocean = False
                for e in range(sptr[node], sptr[node + 1]):
                    L = sL[e]
                    if L == OCEAN:
                        to_ocean = True
                        break
                    if full[sT[e]]:
                        if P == OCEAN:       # passes over a full cavity -> sink
                            to_ocean = True
                            break
                        continue
                    node = L
                    mode = 0
                    found = True
                    break
                if to_ocean:
                    drained[rc] += w
                    break
                if not found:
                    if P == OCEAN:
                        drained[rc] += w
                        break
                    node = P
                    mode = 0
    return vol, drained, lost


@njit(cache=True, nogil=True)
def _spill_list(m, h, region, nptr, nidx, rank, dest, owner0, node_parent, node_spill, tin,
                tout, cand, oL, oT, o):
    """The spill candidates of node m as ``_fill`` walks them (written at
    oL[o:], oT[o:] unless oL is empty; returns their number): the saddle's
    lower neighbours by height (insertion sort: ties in link order), without
    those inside m (Euler-tour test), each as its leaf L and the child T of
    m's parent above L; candidates with no such child are left out, and a
    candidate that drains to the ocean (L = OCEAN) ends the list."""
    P = node_parent[m]
    if P < 0:
        return 0
    s = node_spill[m]
    nc = 0
    for jj in range(nptr[s], nptr[s + 1]):
        n = nidx[jj]
        if n >= 0 and region[n] == region[s] and rank[n] >= 0 and rank[n] < rank[s]:
            cand[nc] = n
            nc += 1
    for a in range(1, nc):
        x = cand[a]
        b = a - 1
        while b >= 0 and h[cand[b]] > h[x]:
            cand[b + 1] = cand[b]
            b -= 1
        cand[b + 1] = x
    k = 0
    write = oL.shape[0] > 0
    for a in range(nc):
        n = cand[a]
        T = owner0[n]
        if m > 0 and T > 0 and tin[m] <= tin[T] and tin[T] < tout[m]:
            continue
        L = dest[n]
        if L == OCEAN:
            if write:
                oL[o + k] = OCEAN
                oT[o + k] = -1
            k += 1
            break
        T = L
        while T >= 0 and node_parent[T] != P:
            T = node_parent[T]
        if T < 0:
            continue
        if write:
            oL[o + k] = L
            oT[o + k] = T
        k += 1
    return k


@dual
def _spill_table(h, region, nptr, nidx, rank, dest, owner0, node_parent, node_spill, tin,
                 tout, nnodes, nn):
    """Spill table of all nodes (``_spill_list``) as CSR (sptr, sL, sT).
    Fixed for a hierarchy; each node on its own."""
    cnt = np.zeros(nnodes + 1, np.int64)
    none = np.zeros(0, np.int64)
    for m in prange(1, nnodes):
        cand = np.empty(nn, np.int64)
        cnt[m + 1] = _spill_list(m, h, region, nptr, nidx, rank, dest, owner0, node_parent,
                                 node_spill, tin, tout, cand, none, none, 0)
    for m in range(nnodes):
        cnt[m + 1] += cnt[m]
    sL = np.empty(cnt[nnodes], np.int64)
    sT = np.empty(cnt[nnodes], np.int64)
    for m in prange(1, nnodes):
        cand = np.empty(nn, np.int64)
        _spill_list(m, h, region, nptr, nidx, rank, dest, owner0, node_parent, node_spill,
                    tin, tout, cand, sL, sT, cnt[m])
    return cnt, sL, sT


@njit(cache=True, nogil=True)
def _euler(node_parent, nnodes):
    """Euler-tour intervals of the depression forest (roots: parent -1): the
    nodes below m (m included) have tin in [tin[m], tout[m])."""
    cnt = np.zeros(nnodes + 1, np.int64)
    for m in range(nnodes):
        p = node_parent[m]
        if p >= 0:
            cnt[p + 1] += 1
    for m in range(nnodes):
        cnt[m + 1] += cnt[m]
    kids = np.empty(cnt[nnodes], np.int64)
    fill = cnt[:-1].copy()
    for m in range(nnodes):
        p = node_parent[m]
        if p >= 0:
            kids[fill[p]] = m
            fill[p] += 1
    tin = np.empty(nnodes, np.int64)
    tout = np.empty(nnodes, np.int64)
    stack = np.empty(nnodes, np.int64)
    nxt = cnt[:-1].copy()                  # next child to visit
    clock = 0
    for r in range(nnodes):
        if node_parent[r] >= 0:
            continue
        top = 0
        stack[0] = r
        tin[r] = clock
        clock += 1
        while top >= 0:
            m = stack[top]
            if nxt[m] < cnt[m + 1]:
                c = kids[nxt[m]]
                nxt[m] += 1
                top += 1
                stack[top] = c
                tin[c] = clock
                clock += 1
            else:
                tout[m] = clock
                top -= 1
    return tin, tout


@njit(cache=True, nogil=True)
def _partial_layers(rows, order, h, pn, pv, vol, tol, nnodes):
    """Height of the partial layer of every node, and the volume of its cells
    below and at that height (rows in height order, as in the serial
    kernel; only rows with a portion in a depression)."""
    MAXP = pn.shape[1]
    INF = np.inf
    hp = np.full(nnodes, INF)
    vfull = np.zeros(nnodes)
    for r in range(rows.shape[0]):
        c = order[rows[r]]
        for q in range(MAXP):
            m = pn[r, q]
            if m < 0:
                break
            vv = pv[r, q]
            if m == OCEAN or vv <= 0.0 or hp[m] < INF:
                continue
            if vfull[m] + vv <= vol[m] + tol:
                vfull[m] += vv
            else:
                hp[m] = h[c]
    vbelow = np.zeros(nnodes)
    veq = np.zeros(nnodes)
    for r in range(rows.shape[0]):
        c = order[rows[r]]
        for q in range(MAXP):
            m = pn[r, q]
            if m < 0:
                break
            vv = pv[r, q]
            if m == OCEAN or vv <= 0.0:
                continue
            if h[c] < hp[m]:
                vbelow[m] += vv
            elif h[c] == hp[m]:
                veq[m] += vv
    return hp, vbelow, veq


@dual
def _par_retained(rows, order, N, h, pn, pv, vol, hp, vbelow, veq):
    MAXP = pn.shape[1]
    retained = np.empty(N)
    for c in prange(N):
        retained[c] = 0.0
    for r in prange(rows.shape[0]):
        c = order[rows[r]]
        acc = 0.0
        for q in range(MAXP):
            m = pn[r, q]
            if m < 0:
                break
            vv = pv[r, q]
            if m == OCEAN or vv <= 0.0:
                continue
            if h[c] < hp[m]:
                acc += vv
            elif h[c] == hp[m] and veq[m] > 0:
                f = (vol[m] - vbelow[m]) / veq[m]
                f = min(max(f, 0.0), 1.0)
                acc += f * vv
        retained[c] = acc
    return retained


@dual
def _par_row_flag(pn):
    n = pn.shape[0]
    use = np.zeros(n, np.bool_)
    for i in prange(n):
        for q in range(pn.shape[1]):
            if pn[i, q] > 0:
                use[i] = True
    return use


class Prepared:
    """Everything of a fill-spill solve that does not depend on the source
    volumes: height order, depression hierarchy, portions and capacities.
    Reused as long as heights, regions, sinks and volumes do not change
    (e.g. while a part hangs still)."""
    __slots__ = ("h", "region", "nptr", "nidx", "order", "rank", "owner0", "dest",
                 "node_parent", "node_spill", "node_region", "nnodes", "N", "nn", "tol",
                 "owner", "rows", "pn", "pv", "cap", "key", "ecell", "eshape", "vcell",
                 "flat", "tin", "tout", "sptr", "sL", "sT")


def prepare(h, region, nptr, nidx, sink, vcell, min_depth, ecell, order=None, eshape=LINEAR):
    """Source-independent part of ``fill_spill`` (arrays as the kernels want
    them: see ``fill_spill``)."""
    P = Prepared()
    t = time.perf_counter()
    if order is None:
        order = height_order(h, sink, compact(_par_ge0(region)))
        t = _tick("P height order", t)
    P.h, P.region, P.nptr, P.nidx, P.order = h, region, nptr, nidx, order
    # in rank space (32-bit ranks in its graph), else in node space; the same result
    hier = hierarchy_rank if order.shape[0] < 2 ** 31 else hierarchy_parallel
    (P.rank, P.owner0, P.dest, P.node_parent, P.node_spill, P.node_region, P.nnodes,
     node_cell) = hier(order, h, region, nptr, nidx, sink, full=True)
    t = time.perf_counter()
    P.N = h.shape[0]
    P.nn = _degree(nptr)
    P.tol = 1e-12 * _par_vmax(order, vcell)
    tgt = _targets(h, P.node_parent, P.node_spill, node_cell, P.nnodes, float(min_depth))
    t = _tick("S setup", t)
    sh = _spill_values(h, P.node_spill, P.node_parent)
    owner, pn, pv = _par_portions(order, P.N, h, ecell, np.asarray(eshape, np.float64), vcell, P.owner0,
                                  P.node_parent, sh, tgt)
    P.owner = owner
    use = _par_row_flag(pn)
    P.rows = compact(use)
    del use
    P.pn, P.pv = _par_take_rows(pn, pv, P.rows)
    del pn, pv
    t = _tick("P portions", t)
    P.cap = _cap_of(P.pn, P.pv, P.nnodes)
    P.tin, P.tout = _euler(P.node_parent, P.nnodes)
    t = _tick("S cap", t)
    P.sptr, P.sL, P.sT = _spill_table(h, region, nptr, nidx, P.rank, P.dest, P.owner0,
                                      P.node_parent, P.node_spill, P.tin, P.tout,
                                      P.nnodes, P.nn)
    t = _tick("P spill table", t)
    P.key = None
    P.ecell = ecell
    P.eshape = np.asarray(eshape, np.float64)
    P.vcell = vcell
    P.flat = bool(np.any(ecell > 0.0))
    return P


def solve(P, src, spill_routing, nregions):
    """Fill & spill of ``src`` on a prepared hierarchy: (retained, drained,
    lost, owner) as ``fill_spill``."""
    t = time.perf_counter()
    sd, sr, sw = _par_sources(P.order, src, P.dest, P.region, get_num_threads())
    t = _tick("P sources", t)
    vol, drained, lost = _fill(sd, sr, sw, P.node_parent, P.node_region, P.nnodes, P.cap,
                               bool(spill_routing), int(nregions), P.tol, P.sptr, P.sL,
                               P.sT)
    t = _tick("S fill", t)
    if P.flat:
        retained = _flat_fill(P.order[P.rows], P.h, P.ecell, P.eshape, P.vcell, P.pn, P.pv,
                              vol, P.cap, P.tol, P.nnodes, P.N)
        t = _tick("S flat", t)
        return retained, drained, lost, P.owner
    hp, vbelow, veq = _partial_layers(P.rows, P.order, P.h, P.pn, P.pv, vol, P.tol,
                                      P.nnodes)
    t = _tick("S layers", t)
    retained = _par_retained(P.rows, P.order, P.N, P.h, P.pn, P.pv, vol, hp, vbelow, veq)
    t = _tick("P retained", t)
    return retained, drained, lost, P.owner


def fsm_parallel(order, h, region, nptr, nidx, sink, src, vcell, spill_routing,
                 nregions, min_depth, ecell, eshape=LINEAR):
    """Parallel version of ``_fsm_kernel`` (same arguments and results)."""
    P = prepare(h, region, nptr, nidx, sink, vcell, min_depth, ecell, order=order,
                eshape=eshape)
    return solve(P, src, spill_routing, nregions)


@dual
def _par_sub(a, b):
    """a - b (a new array), in parallel."""
    out = np.empty(a.shape[0])
    for i in prange(a.shape[0]):
        out[i] = a[i] - b[i]
    return out


@dual
def _par_take_rows(pn, pv, rows):
    """pn[rows], pv[rows], in parallel."""
    W = pn.shape[1]
    n = rows.shape[0]
    on = np.empty((n, W), pn.dtype)
    ov = np.empty((n, W), pv.dtype)
    for j in prange(n):
        r = rows[j]
        for q in range(W):
            on[j, q] = pn[r, q]
            ov[j, q] = pv[r, q]
    return on, ov


# ------------------------------------------------------------ height order
@dual
def _par_ge0(region):
    n = region.shape[0]
    f = np.empty(n, np.bool_)
    for c in prange(n):
        f[c] = region[c] >= 0
    return f


@dual
def _hstats(h, active):
    """Lowest and highest height in nm, and the number of heights that are
    not finite or not whole nanometres."""
    lo = np.inf
    hi = -np.inf
    bad = 0
    for j in prange(active.shape[0]):
        x = h[active[j]]
        if not np.isfinite(x):
            bad += 1
            continue
        q = np.rint(x * 1e9)
        if abs(x * 1e9 - q) >= 1e-3:
            bad += 1
        lo = min(lo, q)
        hi = max(hi, q)
    return lo, hi, bad


@dual
def _hkeys(h, sink, active, lo, S):
    n = active.shape[0]
    k = np.empty(n, np.int64)
    for j in prange(n):
        c = active[j]
        if sink[c]:
            k[j] = c
        else:
            k[j] = ((np.int64(np.rint(h[c] * 1e9) - lo) + 1) << S) | c
    return k


@dual
def _unkey(k, S):
    m = (np.int64(1) << S) - 1
    out = np.empty(k.shape[0], np.int64)
    for j in prange(k.shape[0]):
        out[j] = k[j] & m
    return out


def height_order(h, sink, active):
    """``active`` sorted by height, sinks first; ties by index.

    Same order as ``active[np.argsort(where(sink, -inf, h), kind="stable")]``
    but much faster: heights are whole nanometres (``Simulation`` rounds
    them), so height and index pack into one unique int64 key, and an
    integer sort (in parallel) replaces the float argsort."""
    n = h.shape[0]
    active = np.ascontiguousarray(active, np.int64)
    if active.size == 0:
        return active
    S = max(int(n - 1).bit_length(), 1)
    lo, hi, bad = _hstats(h, active)
    if bad or hi - lo + 2 >= 2.0 ** (62 - S):
        return _float_order(h, sink, active)
    k = _hkeys(h, sink, active, lo, S)
    sort_keys(k)
    return _unkey(k, S)


def _float_order(h, sink, active):
    """``height_order`` for heights that are not whole nanometres (e.g. the
    cell floors h - ecell): the same order as the stable float argsort, in
    parallel. The sinks come first in index order; the other heights are
    mapped to unsigned integers in the same order (-0 as +0) and sorted
    stably by value ranges (``_usort``). With one thread, or heights that
    are not finite: the argsort itself."""
    if get_num_threads() == 1:
        ha = h[active]
        key = np.where(sink[active], -np.inf, ha)
        return active[np.argsort(key, kind="stable")].astype(np.int64)
    flag = _par_sink_of(sink, active)
    ps = compact(flag)
    flag = ~flag
    pn = compact(flag)
    del flag
    u, nonfinite = _ukeys(h, active, pn)
    if nonfinite:
        ha = h[active]
        key = np.where(sink[active], -np.inf, ha)
        return active[np.argsort(key, kind="stable")].astype(np.int64)
    perm = _usort(u, get_num_threads())
    return _gather_order(active, ps, pn, perm)


@dual
def _par_sink_of(sink, active):
    f = np.empty(active.shape[0], np.bool_)
    for j in prange(active.shape[0]):
        f[j] = sink[active[j]]
    return f


@dual
def _ukeys(h, active, pn):
    """Unsigned keys in the order of the heights h[active[pn]] (finite;
    -0 counts as +0, as in a float comparison), and the number of heights
    that are not finite."""
    n = pn.shape[0]
    f = np.empty(n)
    bad = 0
    for j in prange(n):
        x = h[active[pn[j]]]
        if not np.isfinite(x):
            bad += 1
        if x == 0.0:
            x = 0.0
        f[j] = x
    u = f.view(np.uint64)
    top = np.uint64(1) << np.uint64(63)
    for j in prange(n):
        b = u[j]
        if b & top:
            u[j] = ~b
        else:
            u[j] = b | top
    return u, bad


@dual
def _usort(u, nt):
    """Stable argsort of the uint64 keys ``u``: split into value ranges by
    sampled splitters (equal keys share a range), the positions of each
    range in input order, then each range sorted stably (merge sort)."""
    n = u.shape[0]
    B = 8 * nt                                   # ranges
    OV = 64                                      # samples per range
    ns = B * OV
    if n < 4 * ns:
        return np.argsort(u, kind="mergesort")
    step = n // ns
    s = np.empty(ns, np.uint64)
    for j in prange(ns):
        s[j] = u[j * step + (j * 7919) % step]
    s.sort()
    spl = np.empty(B - 1, np.uint64)
    for j in range(B - 1):
        spl[j] = s[(j + 1) * OV]
    C = 4 * nt                                   # chunks of the input
    cnt = np.zeros((C, B), np.int64)
    bk = np.empty(n, np.int32)
    for k in prange(C):
        lo = k * n // C
        hi = (k + 1) * n // C
        for j in range(lo, hi):
            b = np.searchsorted(spl, u[j])
            bk[j] = b
            cnt[k, b] += 1
    off = np.empty((C, B), np.int64)
    bst = np.empty(B + 1, np.int64)
    t = 0
    for b in range(B):
        bst[b] = t
        for k in range(C):
            off[k, b] = t
            t += cnt[k, b]
    bst[B] = n
    pos = np.empty(n, np.int64)
    for k in prange(C):
        lo = k * n // C
        hi = (k + 1) * n // C
        o = off[k].copy()
        for j in range(lo, hi):
            b = bk[j]
            pos[o[b]] = j
            o[b] += 1
    out = np.empty(n, np.int64)
    for b in prange(B):
        seg = pos[bst[b]:bst[b + 1]]
        o = np.argsort(u[seg], kind="mergesort")
        for q in range(seg.shape[0]):
            out[bst[b] + q] = seg[o[q]]
    return out


@dual
def _gather_order(active, ps, pn, perm):
    m = ps.shape[0]
    out = np.empty(m + perm.shape[0], np.int64)
    for j in prange(m):
        out[j] = active[ps[j]]
    for j in prange(perm.shape[0]):
        out[m + j] = active[pn[perm[j]]]
    return out


# ------------------------------------------------------------ sink pruning
@dual
def _prune_sinks(region, sink, src, nptr, nidx, nregions):
    """Sinks with no neighbour that is not a sink (in the same region) can
    be left out of a sweep: nothing flows through a sink, so no path and no
    depression involves them, and every retained volume is the same. Their
    sources go straight to ``drained`` (returned per region; added in fixed
    blocks, independent of the thread count)."""
    N = region.shape[0]
    NB = 64
    out = np.empty(N, np.int64)
    part = np.zeros((NB, nregions))
    for k in prange(NB):
        lo = k * N // NB
        hi = (k + 1) * N // NB
        for c in range(lo, hi):
            r = region[c]
            out[c] = r
            if r < 0 or not sink[c]:
                continue
            keep = False
            for jj in range(nptr[c], nptr[c + 1]):
                q = nidx[jj]
                if q >= 0 and region[q] == r and not sink[q]:
                    keep = True
                    break
            if not keep:
                out[c] = -1
                if src[c] > 0.0:
                    part[k, r] += src[c]
    extra = np.zeros(nregions)
    for k in range(NB):
        for j in range(nregions):
            extra[j] += part[k, j]
    return out, extra


# ------------------------------------------------------------------- API
DEFAULT_METHOD = "split"


def fill_spill(h, region, nbr, sink, src, vcell, spill_routing=True,
               nregions=None, min_depth=0.0, ecell=None, method=None, cache=None,
               key=None, prune=True, eshape=None):
    """Equilibrium distribution of the volumes ``src`` under gravity.

    Parameters
    ----------
    h : (N,) float     height of each cell centre along the *up* direction
                       (pass ``-h`` to route air upwards).
    region : (N,) int  region label per cell, < 0 = inactive. Cells only
                       connect to face neighbours with the same label.
    nbr : (N, W) neighbour table with -1 for none (e.g. ``Grid.neighbors()``)
          or an (indptr, indices) CSR pair (see ``to_csr``).
    sink : (N,) bool   cells that absorb everything that reaches them.
    src : (N,) float   volume released in each cell (may exceed vcell).
    vcell : float or (N,) float   cell volume(s). Per-cell volumes carry
                       the cut-cell volume fractions (see ``volfrac.py``).
    min_depth : float  depressions shallower than this (spill level minus
                       lowest cell) retain nothing: they are treated as
                       surface roughness of the voxel staircase.
    ecell : (N,) float or None   half vertical extent of each cell. When
                       given, cells are ordered by their floor (h - e): the
                       spill level is the floor of the saddle cell and cells
                       straddling it are split (sub-cell spill height),
                       also when the cells differ in size (octree); None
                       keeps whole cells.
    eshape : (3,) or None   how a cell's volume is spread over its extent
                       (``cell_cdf``): None / ``LINEAR`` uniformly (the
                       linear rule), ``box_shape(up)`` exactly as an
                       axis-aligned cube (additive: a cube's portions are the
                       sums of its children's, as on an octree).
    method : "split" (default; parallel over numba's threads) or "serial"
                       (the original single sweep). Same results.
    cache : FSMCache or None   reuse the hierarchy of the previous call when
                       everything but ``src`` is unchanged ("split" only).
    key : hashable or None   with ``cache``: identifies h, region, sink,
                       vcell and ecell (the caller guarantees equal keys mean
                       equal arrays). None: a hash of the arrays is used.
    prune : leave out sinks surrounded by sinks ("split" only; same
                       retained volumes, see ``_prune_sinks``).

    Returns
    -------
    retained : (N,) float  volume held in each cell after fill & spill.
    drained  : (nregions,) volume that reached a sink, per region.
    lost     : (nregions,) volume that did not fit in a closed region.
    owner    : (N,) int    depression-node id holding each cell (0 = ocean).
    """
    method = method or DEFAULT_METHOD
    h = np.ascontiguousarray(h, np.float64)
    N = h.shape[0]
    region = np.ascontiguousarray(region, np.int64)
    sink = np.ascontiguousarray(sink, np.bool_)
    src = np.ascontiguousarray(src, np.float64)
    vc = np.asarray(vcell, np.float64)
    vc = np.full(N, float(vc)) if vc.ndim == 0 else np.ascontiguousarray(vc)
    ec = np.zeros(N) if ecell is None else np.asarray(ecell, np.float64)
    ec = np.full(N, float(ec)) if ec.ndim == 0 else np.ascontiguousarray(ec)
    es = LINEAR if eshape is None else np.ascontiguousarray(eshape, np.float64)
    if ecell is not None:
        h = _par_sub(h, ec)     # sweep by cell floors (same order for equal cells)
    ptr, idx = to_csr(nbr)
    if nregions is None:
        nregions = int(region.max()) + 1 if N and region.max() >= 0 else 1
    if method == "serial":
        active = np.flatnonzero(region >= 0)
        order = height_order(h, sink, active)
        return _fsm_kernel(order, h, region, ptr, idx, sink, src, vc,
                           bool(spill_routing), int(nregions), float(min_depth), ec, es,
                           bool(np.any(ec > 0.0)))
    extra = None
    if prune:
        region, extra = _prune_sinks(region, sink, src, ptr, idx, int(nregions))
    if cache is not None:
        key = (id(ptr), id(idx), ptr.shape[0], idx.shape[0], float(min_depth), bool(prune),
               tuple(es)) + \
            ((key,) if key is not None else fingerprint(h, region, sink, vc, ec))
        P = cache.prepared
        if P is None or P.key != key:
            cache.prepared = None                  # free the old one first
            P = prepare(h, region, ptr, idx, sink, vc, min_depth, ec, eshape=es)
            P.key = key
            cache.prepared = P
            cache.builds += 1
        else:
            cache.hits += 1
    else:
        P = prepare(h, region, ptr, idx, sink, vc, min_depth, ec, eshape=es)
    retained, drained, lost, owner = solve(P, src, spill_routing, nregions)
    if extra is not None:
        drained = drained + extra
    return retained, drained, lost, owner


class FSMCache:
    """Holds the ``Prepared`` hierarchy of the last ``fill_spill`` call made
    with it; see ``fill_spill(cache=...)``."""

    def __init__(self):
        self.prepared = None
        self.builds = 0
        self.hits = 0

    def clear(self):
        self.prepared = None
