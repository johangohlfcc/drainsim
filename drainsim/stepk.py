"""Per-step array passes of ``Simulation`` as parallel kernels.

Each kernel replaces a handful of whole-array numpy operations (one pass over
memory instead of many, and over numba's threads). Results that feed the
state are the same as the numpy code's, bit for bit; sums per compartment
are added in fixed blocks (``NBLK``), so they do not depend on the number of
threads either (they may differ from ``np.bincount`` in the last digit).
"""

import numpy as np
from numba import njit, prange, get_num_threads

from .par import dual
from .fsm import cell_cdf, cell_cdf_inv

NBLK = 64          # blocks of the per-compartment sums (fixed: reproducible)


# ------------------------------------------------------------------ pose
@dual
def geom_h(X, up):
    """``np.round(X @ up, 9)`` with the products added in axis order, as
    ``Simulation._geom`` does."""
    N = X.shape[0]
    nd = X.shape[1]
    h = np.empty(N)
    for c in prange(N):
        s = X[c, 0] * up[0]
        for k in range(1, nd):
            s = s + X[c, k] * up[k]
        h[c] = np.rint(s * 1e9) / 1e9
    return h


@dual
def node_extent(fine, e, ef):
    N = fine.shape[0]
    en = np.empty(N)
    for c in prange(N):
        en[c] = ef if fine[c] else e
    return en


@dual
def node_half_extent(size, s):
    """Half vertical extent of every node: half its edge length times
    |u_x| + |u_y| + |u_z|."""
    N = size.shape[0]
    en = np.empty(N)
    for c in prange(N):
        en[c] = 0.5 * size[c] * s
    return en


@dual
def below_masks(ext, h, zb):
    """ext & (h < zb) and ext & ~(h < zb)."""
    N = h.shape[0]
    lo = np.empty(N, np.bool_)
    hi = np.empty(N, np.bool_)
    for c in prange(N):
        b = h[c] < zb
        lo[c] = ext[c] and b
        hi[c] = ext[c] and not b
    return lo, hi


# ---------------------------------------------------------- equilibrate
@dual
def liquid_inputs(fl, lab, B, bnd, ext, L, v, inj, h, en):
    """region, sink and source of the liquid sweep, and the top of the
    highest source node (-inf if none)."""
    N = h.shape[0]
    region = np.empty(N, np.int64)
    sink = np.empty(N, np.bool_)
    src = np.empty(N)
    top = -np.inf
    for c in prange(N):
        region[c] = lab[c] if fl[c] else -1
        sink[c] = B[c] or (bnd[c] and ext[c])
        s = 0.0
        if fl[c]:
            s = (0.0 if B[c] else L[c] * v[c]) + inj[c]
        src[c] = s
        if s > 0.0:
            top = max(top, h[c] + en[c])
    return region, sink, src, top


@dual
def air_inputs(ext, A, bnd, L, v, h, en):
    """region, sink and source of the air sweep, and the bottom of the
    lowest source node (+inf if none)."""
    N = h.shape[0]
    region = np.empty(N, np.int64)
    sink = np.empty(N, np.bool_)
    src = np.empty(N)
    bot = np.inf
    for c in prange(N):
        region[c] = 0 if ext[c] else -1
        sink[c] = A[c] or (bnd[c] and ext[c])
        s = 0.0
        if ext[c] and not A[c]:
            s = (1.0 - L[c]) * v[c]
        src[c] = s
        if s > 0.0:
            bot = min(bot, h[c] - en[c])
    return region, sink, src, bot


@dual
def band_region(region, h, en, sink, lim, up):
    """Height band: keep the nodes whose extent reaches ``lim`` from below
    (``up``: h - en <= lim, the liquid) or from above (h + en >= lim, the
    air); if ``lim`` is infinite (no source), keep only the sinks."""
    N = h.shape[0]
    out = np.empty(N, np.int64)
    has = np.isfinite(lim)
    for c in prange(N):
        if has:
            k = (h[c] - en[c] <= lim) if up else (h[c] + en[c] >= lim)
        else:
            k = sink[c]
        out[c] = region[c] if k else -1
    return out


@dual
def combine(RL, RA, vsafe, B, fl):
    """New liquid fraction: liquid sweep outside the bath, one minus the air
    sweep inside it, clipped to [0, 1]."""
    N = RL.shape[0]
    L = np.empty(N)
    for c in prange(N):
        x = RL[c] / vsafe[c]
        if B[c]:
            x = 1.0 - RA[c] / vsafe[c]
        if not fl[c]:
            x = 0.0
        L[c] = min(max(x, 0.0), 1.0)
    return L


# ----------------------------------------------------- compartment sums
@dual
def comp_liquid(fl, lab, L, v, B, skip_bath, ncomp):
    """Liquid volume per compartment (bath excluded if ``skip_bath``)."""
    N = fl.shape[0]
    part = np.zeros((NBLK, ncomp))
    for k in prange(NBLK):
        lo = k * N // NBLK
        hi = (k + 1) * N // NBLK
        for c in range(lo, hi):
            if fl[c] and not (skip_bath and B[c]):
                part[k, lab[c]] += L[c] * v[c]
    out = np.zeros(ncomp)
    for k in range(NBLK):
        for j in range(ncomp):
            out[j] += part[k, j]
    return out


@dual
def record_sums(fl, lab, L, v, B, A, h, en, zb, ncomp, eshape):
    """Liquid per compartment (bath excluded) and: liquid above the bath
    level, trapped liquid, trapped air, air in the bath, air below the bath
    level (see ``Simulation._record``)."""
    N = fl.shape[0]
    part = np.zeros((NBLK, ncomp))
    ps = np.zeros((NBLK, 5))
    for k in prange(NBLK):
        lo = k * N // NBLK
        hi = (k + 1) * N // NBLK
        s0 = 0.0
        s1 = 0.0
        s2 = 0.0
        s3 = 0.0
        s4 = 0.0
        for c in range(lo, hi):
            if not fl[c]:
                continue
            lv = L[c] * v[c]
            av = (1.0 - L[c]) * v[c]
            if not B[c]:
                part[k, lab[c]] += lv
            below = h[c] < zb
            if not below:
                s0 += lv
            a = 1.0 - cell_cdf((zb - h[c] + en[c]) / (2.0 * en[c]), eshape)
            if not B[c]:
                s1 += L[c] * a * v[c]
            if not A[c]:
                s2 += (1.0 - L[c]) * (1.0 - a) * v[c]
            if B[c]:
                s3 += av
            if below:
                s4 += av
        ps[k, 0] = s0
        ps[k, 1] = s1
        ps[k, 2] = s2
        ps[k, 3] = s3
        ps[k, 4] = s4
    out = np.zeros(ncomp)
    sums = np.zeros(5)
    for k in range(NBLK):
        for j in range(ncomp):
            out[j] += part[k, j]
        for j in range(5):
            sums[j] += ps[k, j]
    return out, sums


# --------------------------------------------------------------- bodies
@dual
def liquid_mask(fl, L):
    N = fl.shape[0]
    m = np.empty(N, np.bool_)
    for c in prange(N):
        m[c] = fl[c] and L[c] > 1e-9
    return m


@dual
def group_by(body, nb, C):
    """Nodes grouped by body, in index order within a body: (nodes, starts)."""
    N = body.shape[0]
    cnt = np.zeros((C, nb), np.int64)
    for k in prange(C):
        lo = k * N // C
        hi = (k + 1) * N // C
        for c in range(lo, hi):
            b = body[c]
            if b >= 0:
                cnt[k, b] += 1
    gst = np.empty(nb + 1, np.int64)
    off = np.empty((C, nb), np.int64)
    t = 0
    for b in range(nb):
        gst[b] = t
        for k in range(C):
            off[k, b] = t
            t += cnt[k, b]
    gst[nb] = t
    grp = np.empty(t, np.int64)
    for k in prange(C):
        lo = k * N // C
        hi = (k + 1) * N // C
        o = off[k].copy()
        for c in range(lo, hi):
            b = body[c]
            if b >= 0:
                grp[o[b]] = c
                o[b] += 1
    return grp, gst


@dual
def body_stats(grp, gst, L, v, h, en, B, zb, face_area, eshape):
    """Volume, level, free-surface area and bath flag of every body (as
    ``Simulation._bodies``: sums in index order, like ``np.bincount``)."""
    nb = gst.shape[0] - 1
    vol = np.zeros(nb)
    level = np.full(nb, -np.inf)
    area = np.zeros(nb)
    bath = np.zeros(nb, np.bool_)
    for b in prange(nb):
        a0 = gst[b]
        a1 = gst[b + 1]
        isb = False
        for j in range(a0, a1):
            if B[grp[j]]:
                isb = True
                break
        if isb:
            bath[b] = True
            level[b] = zb
            vol[b] = np.inf
            area[b] = np.inf
            continue
        s = 0.0
        lv = -np.inf
        for j in range(a0, a1):
            c = grp[j]
            s += L[c] * v[c]
            top = h[c] - en[c] + 2 * en[c] * cell_cdf_inv(L[c], eshape)
            if top > lv:
                lv = top
        vol[b] = s
        level[b] = lv
        ar = 0.0
        for j in range(a0, a1):
            c = grp[j]
            if (h[c] - en[c] <= lv + 1e-12) and (h[c] + en[c] >= lv - 1e-12):
                ar += v[c] / (2 * en[c])
        area[b] = max(ar, face_area)
    return vol, level, area, bath


# -------------------------------------------------------------- throats
@dual
def side_bodies(sptr, sidx, body, level):
    """For every throat side (CSR list of nodes): the body with the highest
    level among its nodes (lowest id on ties) and that level; -1 and -inf
    if none of its nodes is in a body."""
    ns = sptr.shape[0] - 1
    lev = np.full(ns, -np.inf)
    bod = np.full(ns, -1, np.int64)
    for s in prange(ns):
        best = -1
        bl = -np.inf
        for j in range(sptr[s], sptr[s + 1]):
            b = body[sidx[j]]
            if b < 0:
                continue
            lv = level[b]
            if best < 0 or lv > bl or (lv == bl and b < best):
                best = b
                bl = lv
        if best >= 0:
            bod[s] = best
            lev[s] = bl
    return lev, bod


@dual
def throat_air(tptr, ca, cb, L):
    """Per throat: air (L < 0.5) at some face cell on side a, on side b, and
    on both sides of the same face."""
    nt = tptr.shape[0] - 1
    aa = np.zeros(nt, np.bool_)
    ab = np.zeros(nt, np.bool_)
    both = np.zeros(nt, np.bool_)
    for t in prange(nt):
        for j in range(tptr[t], tptr[t + 1]):
            xa = L[ca[j]] < 0.5
            xb = L[cb[j]] < 0.5
            if xa:
                aa[t] = True
            if xb:
                ab[t] = True
            if xa and xb:
                both[t] = True
    return aa, ab, both


@njit(cache=True, nogil=True)
def take_top(cells, L, v, dV):
    """Remove the volume dV from the nodes ``cells`` in order (a body's
    nodes, top first), each emptied before the next is touched; L is
    changed in place. Returns what could not be taken (0 if all)."""
    rem = dV
    for j in range(cells.shape[0]):
        c = cells[j]
        have = L[c] * v[c]
        take = min(have, rem)
        L[c] -= take / v[c]
        rem -= take
        if rem <= 0:
            break
    return rem
