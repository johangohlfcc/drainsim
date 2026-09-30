"""Surface film: deposition, gravity drainage, dripping (no flow solution).

The film lives on a *carrier* surface: the fluid-facing boundary of the solid
voxels (marching squares in 2D, marching cubes in 3D, Taubin-smoothed).
Because it is the boundary of the solid *voxels*, the two sides of a
sheet-metal wall are separate surfaces, each facing its own fluid space:
inside and outside of a cavity carry independent films.

State: film thickness ``h`` per element (m). Volume = h * area.

Physics (per element, per step)
-------------------------------
* **Submerged** (adjacent fluid cell >= half full): the film is part of the
  pool / bath; any film volume is handed to the voxel liquid.
* **Deposition** when the liquid level leaves an element. The local
  withdrawal speed is the bath-surface speed relative to the element (from
  the motion), or for pools inside the object the time the adjacent cell
  took to empty, ``U_vert = 2e / (t_empty - t_full)``; it is converted to a
  speed along the surface,
  and the Landau-Levich-Derjaguin law gives the film:
  ``h0 = l_c * min(0.94 Ca^(2/3), Ca^(1/2))``, ``Ca = mu U / sigma``.
  Film deposited by the bath comes from the (infinite) bath; film deposited
  by an internal pool is taken from that compartment's liquid.
* **Drainage**: lubrication film driven by the tangential gravity
  ``g_t = g - (g.n) n``; flux per unit width ``q = rho g_t h^3 / (3 mu)``,
  first-order upwind finite volumes over the element edges, backward Euler
  swept from the highest element down (stable for any step, exactly
  conservative). Surface tension is neglected on the surfaces.
  Film running onto a submerged element joins that pool / the bath.
* **Dripping**: on downward-facing surfaces film collects at low points;
  connected groups of thick hanging film release drops of volume
  ``drop_volume`` (Tate's law, ~0.075 ml for water). Drops are handed to
  the voxel model in the air cell below and routed by fill-spill-merge
  (they fall into a cavity or back into the bath).
* **Bulk transfer**: on upward-facing surfaces, film thicker than ``h_bulk``
  (a puddle) is handed to the voxel model, which pools or routes it.

Analytic check: a vertical wall with initial film h0 drains as
``h = sqrt(mu x / (rho g t))`` (Jeffreys 1930) above ``x* = rho g h0^2 t / mu``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage, sparse
from scipy.sparse.csgraph import connected_components


# ------------------------------------------------------------------ carrier

@dataclass
class Carrier:
    ndim: int
    verts: np.ndarray          # (nv, ndim)
    elems: np.ndarray          # (ne, ndim) vertex indices (segments / triangles)
    x: np.ndarray              # (ne, ndim) centroids
    area: np.ndarray           # (ne,) m^2 (3D) or m per unit depth (2D)
    normal: np.ndarray         # (ne, ndim) unit, pointing into the fluid
    cell: np.ndarray           # (ne,) adjacent fluid cell (flat index)
    ei: np.ndarray             # edge pairs
    ej: np.ndarray
    elen: np.ndarray           # edge length (3D), 1 (2D)
    mi: np.ndarray             # in-plane outward normal of ei at the edge
    mj: np.ndarray             # in-plane outward normal of ej at the edge

    @property
    def n(self):
        return len(self.area)


def _taubin(P, nbr_mat, iters=6, lam=0.5, mu=-0.53):
    deg = np.asarray(nbr_mat.sum(1)).ravel()
    deg[deg == 0] = 1
    for _ in range(iters):
        for f in (lam, mu):
            avg = (nbr_mat @ P) / deg[:, None]
            P = P + f * (avg - P)
    return P


def _cell_of_points(grid, pts):
    idx = np.floor((pts - grid.origin) / grid.dx).astype(int)
    idx = np.clip(idx, 0, np.array(grid.shape) - 1)
    return np.ravel_multi_index(tuple(idx.T), grid.shape)


def _tri_carrier(verts, faces, smooth_iters):
    """Taubin-smoothed triangle carrier: elements, areas, normals (not yet
    oriented) and the element pairs sharing an edge with their in-plane
    edge normals. Non-manifold edges (more than two faces) are paired in
    sequence, as a chain."""
    e = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    nv = len(verts)
    A = sparse.coo_matrix((np.ones(2 * len(e)),
                           (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])),
                          shape=(nv, nv)).tocsr()
    A.data[:] = 1.0
    verts = _taubin(verts, A, smooth_iters)
    a, b, c = (verts[faces[:, k]] for k in range(3))
    cr = np.cross(b - a, c - a)
    area2 = np.linalg.norm(cr, axis=1)
    keep = area2 > 1e-14
    faces, a, b, c, cr, area2 = faces[keep], a[keep], b[keep], c[keep], cr[keep], area2[keep]
    area = 0.5 * area2
    nrm = cr / area2[:, None]
    x = (a + b + c) / 3.0
    ne = len(faces)
    E = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    opp = np.r_[faces[:, 2], faces[:, 0], faces[:, 1]]
    fid = np.tile(np.arange(ne), 3)
    Es = np.sort(E, 1)
    key = Es[:, 0].astype(np.int64) * len(verts) + Es[:, 1]
    order = np.argsort(key, kind="stable")
    ks = key[order]
    same = np.flatnonzero(ks[1:] == ks[:-1])       # consecutive in a group
    p, q = order[same], order[same + 1]
    ei, ej = fid[p], fid[q]
    va, vb = verts[E[p, 0]], verts[E[p, 1]]
    ed = vb - va
    elen = np.linalg.norm(ed, axis=1)

    def inplane(fidx, oppv):
        m = np.cross(ed, nrm[fidx])
        m /= np.maximum(np.linalg.norm(m, axis=1), 1e-30)[:, None]
        flip = np.einsum("ij,ij->i", verts[oppv] - va, m) > 0
        m[flip] *= -1
        return m
    mi = inplane(ei, opp[p])
    mj = inplane(ej, opp[q])
    return verts, faces, x, area, nrm, ei, ej, elen, mi, mj


def build_carrier_octree(ot, smooth_iters=6, block=96) -> Carrier:
    """Carrier of an octree (``octree.Octree``): the fluid-facing surface of
    its closed level-0 cells, by marching cubes in blocks (the dense level-0
    space is never allocated). Elements map to the fluid leaf next to them."""
    from skimage import measure
    from .octree import _coords, _key, _isin_sorted
    h = ot.h
    d0 = np.asarray(ot.dims0, np.int64)
    C = _coords(ot.cut, d0)
    B = int(block)
    # cubes between voxel centres i and i+1, i = -1 .. d0-1 (padding of one
    # empty voxel, as np.pad in build_carrier); block b holds cubes
    # [b*B - 1, (b+1)*B - 1), i.e. voxels b*B - 1 .. (b+1)*B - 1
    b0, b1 = np.floor_divide(C, B), np.floor_divide(C + 1, B)
    nbk = (d0 + 1 + B - 1) // B
    combos = []
    for sx in (b0, b1):
        for sy in (b0, b1):
            for sz in (b0, b1):
                bb = np.stack([sx[:, 0], sy[:, 1], sz[:, 2]], 1)
                combos.append(np.unique(_key(np.clip(bb, 0, nbk - 1), nbk)))
    bk = np.unique(np.concatenate(combos))
    Vs, Fs = [], []
    nv = 0
    ckey = ot.cut
    for bkey in bk:
        b = _coords(np.array([bkey]), nbk)[0]
        lo = b * B - 1
        hi = np.minimum(lo + B, d0)                  # last voxel index (inclusive)
        shp = hi - lo + 1
        ax = [np.arange(lo[d], hi[d] + 1) for d in range(3)]
        G = np.stack(np.meshgrid(*ax, indexing="ij"), -1).reshape(-1, 3)
        inside = np.all((G >= 0) & (G < d0), axis=1)
        f = np.zeros(len(G), bool)
        found, _ = _isin_sorted(_key(G[inside], d0), ckey)
        f[np.flatnonzero(inside)[found]] = True
        if not f.any() or f.all():
            continue
        f = f.reshape(shp).astype(np.float32)
        try:
            v, fc, _, _ = measure.marching_cubes(f, 0.5)
        except (ValueError, RuntimeError):
            continue
        Vs.append(v + lo)
        Fs.append(fc + nv)
        nv += len(v)
    V = np.vstack(Vs)
    F = np.vstack(Fs)
    # merge the vertices shared by neighbouring blocks (edge midpoints:
    # exact half-integers)
    q = np.rint(2 * V).astype(np.int64) + 4
    span = int(q.max()) + 1
    vk = (q[:, 0] * span + q[:, 1]) * span + q[:, 2]
    uk, inv = np.unique(vk, return_inverse=True)
    verts = np.zeros((uk.size, 3))
    verts[inv] = V
    F = inv.ravel()[F]
    verts = ot.origin + (verts + 0.5) * h
    verts, elems, x, area, nrm, ei, ej, elen, mi, mj = _tri_carrier(verts, F, smooth_iters)
    # orient normals into the fluid; element -> fluid leaf node
    lo_d, hi_d = ot.origin + 1e-9, ot.origin + d0 * h - 1e-9

    def closed(P):
        lev, _ = ot.locate(np.clip(P, lo_d, hi_d))
        return lev < 0
    flip = closed(x + 0.6 * h * nrm) & ~closed(x - 0.6 * h * nrm)
    nrm[flip] *= -1
    cell = ot.node_of(np.clip(x + 0.6 * h * nrm, lo_d, hi_d))
    miss = cell < 0
    if miss.any():
        from scipy.spatial import cKDTree
        cand = ot.node_of(np.clip(x[miss] + 1.2 * h * nrm[miss], lo_d, hi_d))
        m2 = cand < 0
        if m2.any():
            X0 = ot.leaf_centres(0)
            tree = cKDTree(X0)
            _, j = tree.query(x[miss][m2])
            cand[m2] = ot.offsets()[0] + j
        cell[miss] = cand
    return Carrier(3, verts, elems, x, area, nrm, cell, ei, ej, elen, mi, mj)


def build_carrier(grid, smooth_iters=6) -> Carrier:
    """Fluid-facing surface of the solid voxels."""
    from .octree import Octree
    if isinstance(grid, Octree):
        return build_carrier_octree(grid, smooth_iters)
    from skimage import measure
    solid = grid.solid
    f = np.pad(solid.astype(np.float32), 1)
    nd = grid.ndim
    # nearest fluid cell of every cell (for mapping elements to fluid cells)
    _, near = ndimage.distance_transform_edt(solid, return_indices=True)
    near_flat = np.ravel_multi_index(tuple(n.ravel() for n in near), grid.shape)

    if nd == 2:
        V, E = [], []
        n0 = 0
        for c in measure.find_contours(f, 0.5):
            if len(c) < 3:
                continue
            closed = np.allclose(c[0], c[-1])
            if closed:
                c = c[:-1]
            p = grid.origin + (c - 0.5) * grid.dx
            k = len(p)
            # cyclic smoothing along the loop
            for _ in range(smooth_iters):
                for fct in (0.5, -0.53):
                    avg = 0.5 * (np.roll(p, 1, 0) + np.roll(p, -1, 0))
                    if not closed:
                        avg[0], avg[-1] = p[0], p[-1]
                    p = p + fct * (avg - p)
            V.append(p)
            ids = np.arange(k) + n0
            seg = np.stack([ids, np.roll(ids, -1)], 1) if closed else \
                np.stack([ids[:-1], ids[1:]], 1)
            E.append(seg)
            n0 += k
        verts = np.vstack(V)
        elems = np.vstack(E)
        a, b = verts[elems[:, 0]], verts[elems[:, 1]]
        t = b - a
        area = np.linalg.norm(t, axis=1)
        keep = area > 1e-12
        elems, a, b, t, area = elems[keep], a[keep], b[keep], t[keep], area[keep]
        t = t / area[:, None]
        x = 0.5 * (a + b)
        nrm = np.stack([-t[:, 1], t[:, 0]], 1)
        # edges: consecutive segments sharing a vertex
        ne = len(elems)
        by_start = -np.ones(len(verts), int)
        by_start[elems[:, 0]] = np.arange(ne)
        nxt = by_start[elems[:, 1]]
        ok = nxt >= 0
        ei = np.flatnonzero(ok)
        ej = nxt[ok]
        elen = np.ones(len(ei))
        mi = t[ei]                  # leaving i through its end vertex
        mj = -t[ej]                 # leaving j through its start vertex
    else:
        verts, faces, _, _ = measure.marching_cubes(f, 0.5)
        verts = grid.origin + (verts - 0.5) * grid.dx
        verts, elems, x, area, nrm, ei, ej, elen, mi, mj = _tri_carrier(verts, faces,
                                                                       smooth_iters)

    # orient normals into the fluid and map each element to its fluid cell
    probe = x + 0.6 * grid.dx * nrm
    cprobe = _cell_of_points(grid, probe)
    cback = _cell_of_points(grid, x - 0.6 * grid.dx * nrm)
    sflat = solid.ravel()
    flip = sflat[cprobe] & ~sflat[cback]
    nrm[flip] *= -1
    if nd == 2:
        # an in-plane normal does not depend on the sheet normal sign
        pass
    cell = _cell_of_points(grid, x + 0.6 * grid.dx * nrm)
    cell = np.where(sflat[cell], near_flat[cell], cell)
    return Carrier(nd, verts, elems, x, area, nrm, cell, ei, ej, elen, mi, mj)


# ------------------------------------------------------------------ kernel

from numba import njit, prange

from .par import dual, sort_keys


@njit(cache=True, nogil=True)
def _implicit_sweep(order, A, V, C, indptr, recv, w, dt, sub):
    """Backward-Euler upwind film transport, swept from high to low.

    For each element (highest first) solve  A h + dt C h^3 = V_old + inflow
    (monotone cubic, Newton from an upper bound), then pass the outflow
    dt C h^3 to the receivers in proportion to the edge coefficients.
    Unconditionally stable; exactly conservative. Flow towards an element
    that was already processed (uphill, rare) is added to it directly.
    """
    n = A.shape[0]
    inflow = np.zeros(n)
    Vn = np.zeros(n)
    done = np.zeros(n, np.bool_)
    absorbed = np.zeros(n)
    for k in range(order.shape[0]):
        e = order[k]
        rhs = V[e] + inflow[e]
        done[e] = True
        if sub[e]:
            absorbed[e] += rhs
            continue
        if C[e] <= 0.0 or rhs <= 0.0:
            Vn[e] = rhs
            continue
        a = A[e]
        b = dt * C[e]
        h = min(rhs / a, (rhs / b) ** (1.0 / 3.0))
        for _ in range(30):
            f = a * h + b * h * h * h - rhs
            fp = a + 3.0 * b * h * h
            dh = f / fp
            h -= dh
            if h < 0.0:
                h = 0.0
            if abs(dh) < 1e-14 * (1.0 + h):
                break
        out = rhs - a * h
        Vn[e] = a * h
        if out <= 0.0:
            continue
        for q in range(indptr[e], indptr[e + 1]):
            r = recv[q]
            share = out * w[q] / C[e]
            if done[r]:
                if sub[r]:
                    absorbed[r] += share
                else:
                    Vn[r] += share
            else:
                inflow[r] += share
    return Vn, absorbed


# ----------------------------------------------------- per-step kernels
@dual
def _film_geom(cell, L, normal, gvec, up):
    """Per element: liquid fraction of its cell, tangential gravity, its
    magnitude and the normal's up component."""
    n = cell.shape[0]
    nd = normal.shape[1]
    Lc = np.empty(n)
    gt = np.empty((n, nd))
    gmag = np.empty(n)
    upn = np.empty(n)
    for e in prange(n):
        Lc[e] = L[cell[e]]
        gn = 0.0
        un = 0.0
        for k in range(nd):
            gn += normal[e, k] * gvec[k]
            un += normal[e, k] * up[k]
        m2 = 0.0
        for k in range(nd):
            x = gvec[k] - gn * normal[e, k]
            gt[e, k] = x
            m2 += x * x
        gmag[e] = np.sqrt(m2)
        upn[e] = un
    return Lc, gt, gmag, upn


@dual
def _film_coeffs(indptr, de, dd, ei, ej, mi, mj, elen, gt, dry, kf):
    """Edge coefficients in the (static) CSR of directed edges, source
    first: kf * elen * max(g_t . m, 0) from a dry source element, else 0;
    and their sum per source element."""
    n = indptr.shape[0] - 1
    nd = gt.shape[1]
    w = np.empty(de.shape[0])
    C = np.empty(n)
    for e in prange(n):
        acc = 0.0
        for q in range(indptr[e], indptr[e + 1]):
            x = 0.0
            if dry[e]:
                j = de[q]
                s = 0.0
                if dd[q] == 0:
                    for k in range(nd):
                        s += gt[e, k] * mi[j, k]
                else:
                    for k in range(nd):
                        s += gt[e, k] * mj[j, k]
                x = kf * elen[j] * max(s, 0.0)
            w[q] = x
            acc += x
        C[e] = acc
    return w, C


@njit(cache=True, nogil=True)
def _scatter_add(out, idx, val):
    """out[idx] += val in index order (as np.add.at)."""
    for j in range(idx.shape[0]):
        out[idx[j]] += val[j]


@dual
def _height_keys(neg, S):
    """Sort keys of the values ``neg`` (whole nm, then index)."""
    n = neg.shape[0]
    lo = np.inf
    for e in prange(n):
        lo = min(lo, np.rint(neg[e] * 1e9))
    key = np.empty(n, np.int64)
    for e in prange(n):
        key[e] = (np.int64(np.rint(neg[e] * 1e9) - lo) << S) | e
    return key


@dual
def _fix_ties(key, neg, S):
    """Order from keys sorted by whole nm; within a run of equal nm, the
    elements are put in exact order of ``neg`` (stable: ties by index). The
    result equals ``np.argsort(neg, kind="stable")``."""
    n = key.shape[0]
    m = (np.int64(1) << S) - 1
    order = np.empty(n, np.int64)
    for j in prange(n):
        order[j] = key[j] & m
    # run starts
    start = np.zeros(n + 1, np.bool_)
    start[0] = True
    start[n] = True
    for j in prange(1, n):
        start[j] = (key[j] >> S) != (key[j - 1] >> S)
    rs = np.flatnonzero(start)
    for r in prange(rs.shape[0] - 1):
        a = rs[r]
        b = rs[r + 1]
        if b - a < 2:
            continue
        srt = True
        for j in range(a + 1, b):
            if neg[order[j]] < neg[order[j - 1]]:
                srt = False
                break
        if srt:
            continue
        seg = order[a:b].copy()
        vals = np.empty(b - a)
        for j in range(b - a):
            vals[j] = neg[seg[j]]
        o = np.argsort(vals, kind="mergesort")
        for j in range(b - a):
            order[a + j] = seg[o[j]]
    return order


@dual
def _comp_liquid_pos(fl, B, lab, L, v, ncomp):
    """Liquid volume per compartment in nodes with liquid (not bath)."""
    N = fl.shape[0]
    NB = 64
    part = np.zeros((NB, ncomp))
    for k in prange(NB):
        lo = k * N // NB
        hi = (k + 1) * N // NB
        for c in range(lo, hi):
            if fl[c] and not B[c] and L[c] > 0:
                part[k, lab[c]] += L[c] * v[c]
    out = np.zeros(ncomp)
    for k in range(NB):
        for j in range(ncomp):
            out[j] += part[k, j]
    return out


@dual
def _scale_comp(fl, B, lab, L, f):
    for c in prange(fl.shape[0]):
        if fl[c] and not B[c] and L[c] > 0:
            L[c] *= f[lab[c]]


@njit(cache=True, nogil=True)
def _sweep_sorted(A, V, C, indptr, recv, w, dt, sub):
    """``_implicit_sweep`` with the elements renumbered in sweep order
    (highest first), so that the sweep walks the arrays in memory order.
    Same operations in the same order: same result."""
    n = A.shape[0]
    inflow = np.zeros(n)
    Vn = np.zeros(n)
    absorbed = np.zeros(n)
    for e in range(n):
        rhs = V[e] + inflow[e]
        if sub[e]:
            absorbed[e] += rhs
            continue
        if C[e] <= 0.0 or rhs <= 0.0:
            Vn[e] = rhs
            continue
        a = A[e]
        b = dt * C[e]
        h = min(rhs / a, (rhs / b) ** (1.0 / 3.0))
        for _ in range(30):
            f = a * h + b * h * h * h - rhs
            fp = a + 3.0 * b * h * h
            dh = f / fp
            h -= dh
            if h < 0.0:
                h = 0.0
            if abs(dh) < 1e-14 * (1.0 + h):
                break
        out = rhs - a * h
        Vn[e] = a * h
        if out <= 0.0:
            continue
        for q in range(indptr[e], indptr[e + 1]):
            r = recv[q]
            share = out * w[q] / C[e]
            if r <= e:                      # already swept (uphill, rare)
                if sub[r]:
                    absorbed[r] += share
                else:
                    Vn[r] += share
            else:
                inflow[r] += share
    return Vn, absorbed


@dual
def _gather_ranges(indptr, order, indptr_s):
    """Positions in the CSR of the edges of order[0], order[1], ..."""
    n = order.shape[0]
    qmap = np.empty(indptr_s[n], np.int64)
    for i in prange(n):
        e = order[i]
        o = indptr_s[i]
        for q in range(indptr[e], indptr[e + 1]):
            qmap[o] = q
            o += 1
    return qmap


# ------------------------------------------------------------------ physics

@dataclass
class FilmParams:
    h_bulk: float | None = None          # upward puddle limit (default min(1 mm, dx/2))
    drop_volume: float | None = None     # default: 3D 3.8 l_c^3, 2D 2 l_c^2
    hang_normal: float = -0.2            # n.up below this: film hangs (drips)
    hang_min: float = 2e-4               # hanging film thicker than this can form a drop
    dt_film: float = 0.05                 # max film sub-step (implicit, stable)
    deposit: bool = True
    U_min: float = 1e-4                  # m/s, floor for the withdrawal speed
    slope_floor: float = 0.3             # sin(alpha) floor for the along-slope speed


@dataclass
class FilmState:
    h: np.ndarray
    sub: np.ndarray
    t_full: np.ndarray
    from_bath: np.ndarray
    drips: list = field(default_factory=list)       # (t, x_obj, volume)


class FilmModel:
    def __init__(self, sim, params: FilmParams | None = None, carrier=None):
        self.p = params or FilmParams()
        self.sim = sim
        self.c = carrier or build_carrier(sim.grid)
        fl = sim.fluid
        lc = fl.capillary_length
        nd = sim.grid.ndim
        if self.p.h_bulk is None:
            self.p.h_bulk = min(1e-3, 0.5 * sim.grid.dx)
        if self.p.drop_volume is None:
            self.p.drop_volume = 3.8 * lc ** 3 if nd == 3 else 2.0 * lc ** 2
        n = self.c.n
        L = sim.L[self.c.cell]
        self.s = FilmState(h=np.zeros(n), sub=L >= 0.5,
                           t_full=np.where(L >= 0.99, sim.t, -np.inf),
                           from_bath=sim.B[self.c.cell].copy())
        self.drip_volume = 0.0
        self.n_drops = 0               # running sum of s.drips[:][3] (the history list)
        self.bulk_volume = 0.0
        self.from_bath_volume = 0.0
        self.unbalanced = 0.0

    # -- helpers
    @property
    def volume(self):
        return float((self.s.h * self.c.area).sum())

    def deposit_thickness(self, U):
        fl = self.sim.fluid
        Ca = fl.mu * np.maximum(U, self.p.U_min) / fl.sigma
        lc = fl.capillary_length
        return lc * np.minimum(0.94 * Ca ** (2.0 / 3.0), np.sqrt(Ca))

    # -- one model step (called after the voxel equilibration)
    def _static(self):
        """Directed edges (both ways) as a CSR by source element, built once;
        the per-step coefficients are written in this order (zero for the
        edges that carry nothing)."""
        if getattr(self, "_csr", None) is not None:
            return self._csr
        c = self.c
        E = c.ei.size
        src = np.r_[c.ei, c.ej].astype(np.int64)
        dst = np.r_[c.ej, c.ei].astype(np.int64)
        o = np.argsort(src, kind="stable")
        indptr = np.r_[0, np.cumsum(np.bincount(src, minlength=c.n))].astype(np.int64)
        de = (o % E).astype(np.int64)                  # edge index
        dd = (o >= E).astype(np.int8)                  # 0: ei -> ej, 1: ej -> ei
        self._csr = (indptr, dst[o], de, dd)
        self._order = (None, None)
        return self._csr

    def _sweep_order(self, up):
        """Elements from the highest down (reused while ``up`` is unchanged)."""
        key = tuple(np.asarray(up, float).tolist())
        if self._order[0] == key:
            return self._order[1]
        c = self.c
        S = max(int(c.n - 1).bit_length(), 1)
        neg = -(c.x @ np.asarray(up, float))
        order = _fix_ties(sort_keys(_height_keys(neg, S)), neg, S)
        self._order = (key, order)
        return order

    def _sorted_space(self, order):
        """The directed-edge CSR renumbered in sweep order (cached with it)."""
        sp = getattr(self, "_sp", None)
        if sp is not None and sp[0] is order:
            return sp
        indptr, recv, de, dd = self._static()
        n = self.c.n
        inv = np.empty(n, np.int64)
        inv[order] = np.arange(n)
        indptr_s = np.r_[0, np.cumsum(np.diff(indptr)[order])].astype(np.int64)
        qmap = _gather_ranges(indptr, order, indptr_s)
        recv_s = inv[recv[qmap]]
        self._sp = (order, indptr_s, qmap, recv_s, self.c.area[order])
        return self._sp

    def __getstate__(self):
        d = self.__dict__.copy()
        d["_csr"] = None
        d["_order"] = (None, None)
        d["_sp"] = None
        return d

    # -- one model step (called after the voxel equilibration)
    def update(self, dt):
        sim, c, s, p = self.sim, self.c, self.s, self.p
        fl = sim.fluid
        t = sim.t
        indptr, recv, de, dd = self._static()
        inj = np.zeros(sim.N)
        up = np.asarray(sim.up, float)
        gvec = -fl.g * up
        Lc, gt, gt_mag, up_n = _film_geom(c.cell, sim.L, c.normal, gvec, up)
        now_sub = Lc >= 0.5

        # 1. submerged: film joins the pool / bath
        m = now_sub & (s.h > 0)
        if m.any():
            _scatter_add(inj, c.cell[m], s.h[m] * c.area[m])
            s.h[m] = 0.0
        full = Lc >= 0.99
        s.t_full[full] = t
        s.from_bath[now_sub] = sim.B[c.cell[now_sub]]

        # 2. deposition on elements the liquid just left
        emerged = s.sub & ~now_sub
        if p.deposit and emerged.any():
            # vertical speed of the liquid level relative to the element:
            # bath -> exact from the motion; pools -> time the cell took
            # to empty (the pool level is resolved with partial cells)
            tau = np.maximum(t - s.t_full[emerged], dt)
            U_vert = 2.0 * sim.en[c.cell[emerged]] / tau     # the node's own height
            fb0 = s.from_bath[emerged]
            if fb0.any():
                up0, zb0 = sim.motion.frame(t - dt)
                xe = c.x[emerged][fb0]
                rel1 = sim.zb - xe @ up
                rel0 = zb0 - xe @ up0
                U_vert[fb0] = np.abs(rel1 - rel0) / dt
            sin_a = np.maximum(gt_mag[emerged] / fl.g, p.slope_floor)
            h0 = self.deposit_thickness(U_vert / sin_a)
            s.h[emerged] = h0
            dep = h0 * c.area[emerged]
            fb = s.from_bath[emerged]
            self.from_bath_volume += dep[fb].sum()
            if (~fb).any():
                # film deposited by an internal pool is taken from the liquid
                # of its compartment (all compartments in one pass)
                comp = sim.lab[c.cell[emerged][~fb]]
                need = np.bincount(comp, weights=dep[~fb], minlength=sim.comp.n)
                V = _comp_liquid_pos(sim.fl, sim.B, sim.lab, sim.L, sim.v, sim.comp.n)
                take = np.minimum(need, V)
                f = np.ones(sim.comp.n)
                sel = (need > 0) & (V > 0)
                f[sel] = (V[sel] - take[sel]) / V[sel]
                if sel.any():
                    _scale_comp(sim.fl, sim.B, sim.lab, sim.L, f)
                self.unbalanced += float((need - take)[need > 0].sum())
        s.sub = now_sub

        # 3. drainage of dry film (implicit upwind sweep, high to low)
        k = fl.rho / (3.0 * fl.mu)
        dry = ~now_sub
        w, C = _film_coeffs(indptr, de, dd, c.ei, c.ej, c.mi, c.mj, c.elen, gt, dry, k)
        A = c.area
        order = self._sweep_order(up)
        _, indptr_s, qmap, recv_s, A_s = self._sorted_space(order)
        w_s = w[qmap]
        C_s = C[order]
        sub_s = now_sub[order]
        V_s = s.h[order] * A_s
        nsub = max(1, int(np.ceil(dt / p.dt_film)))
        ab_s = np.zeros(c.n)
        for _ in range(nsub):
            V_s, ab = _sweep_sorted(A_s, V_s, C_s, indptr_s, recv_s, w_s, dt / nsub, sub_s)
            ab_s += ab
        self.nsub = nsub
        V = np.empty(c.n)
        V[order] = V_s
        absorbed = np.empty(c.n)
        absorbed[order] = ab_s
        _scatter_add(inj, c.cell, absorbed)
        s.h = V / A

        # 4. puddles on upward faces -> voxel liquid
        upward = up_n >= p.hang_normal
        ex = np.where(upward & dry, np.maximum(s.h - p.h_bulk, 0.0), 0.0) * A
        if ex.any():
            _scatter_add(inj, c.cell, ex)
            s.h -= ex / A
            self.bulk_volume += ex.sum()

        # 5. drops from hanging film
        hang = dry & ~upward & (s.h > p.hang_min)
        if hang.any():
            idx = np.flatnonzero(hang)
            sel = hang[c.ei] & hang[c.ej]
            G = sparse.coo_matrix((np.ones(sel.sum()), (c.ei[sel], c.ej[sel])),
                                  shape=(c.n, c.n))
            ncomp, lab = connected_components(G, directed=False)
            lab = lab[idx]
            vol = np.bincount(lab, weights=s.h[idx] * A[idx])
            for g in np.flatnonzero(vol >= p.drop_volume):
                members = idx[lab == g]
                nd_ = int(vol[g] // p.drop_volume)
                rel = nd_ * p.drop_volume
                # release from the lowest member, taking volume from the
                # thickest film first
                low = members[np.argmin(sim.h[c.cell[members]])]
                order = members[np.argsort(-s.h[members])]
                rem = rel
                for e in order:
                    take = min(s.h[e] * A[e], rem)
                    s.h[e] -= take / A[e]
                    rem -= take
                    if rem <= 0:
                        break
                inj[c.cell[low]] += rel
                self.drip_volume += rel
                s.drips.append((t, c.x[low].copy(), rel, nd_))
                self.n_drops += nd_
        s.h = np.maximum(s.h, 0.0)
        return inj
