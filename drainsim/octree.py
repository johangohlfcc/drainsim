"""Octree grid: fine cells at the surface, coarse cells in the bulk.

The finest level (level 0, cell size ``h``) is the uniform voxel grid of the
same spacing, but it only exists near the surface: cells touched by the
surface are *closed* (solid), as on the uniform grid, and around them there
is at least one layer of fluid cells at level 0. Away from the surface the
cells double in size per level, up to ``h * 2**L`` (the base cells). Grading
is 2:1 (face and edge neighbours differ by at most one level), and every
level has at least ``margin`` cells of its own around the finer ones.

The fluid leaves are the nodes of a graph (CSR) with the same meaning as on
the uniform grid: every algorithm (fill-spill-merge, labelling, throats)
needs only face neighbours, a height, a volume and a vertical extent per
node. With ``subcells`` (k >= 2) the closed level-0 cells are refined into
their fluid sub-cells, which become nodes of their own, as on the uniform
grid (``volfrac.cut_cell_graph``).

An octree with L = 0 is the uniform grid (same cells, same links).

Cells are identified per level by a key, the flat index in that level's
dense index space: ``key = (i * ny_l + j) * nz_l + k`` with
``n*_l = n*_0 >> l``. The dense space is never allocated; sets of cells are
sorted unique key arrays.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from numba import njit

from .grid import sample_triangles


# ------------------------------------------------------------------ keys
def _dims(dims0, l):
    return np.asarray(dims0, np.int64) >> l


def _key(c, dims):
    return (c[:, 0] * dims[1] + c[:, 1]) * dims[2] + c[:, 2]


def _coords(key, dims):
    k = key % dims[2]
    t = key // dims[2]
    return np.stack([t // dims[1], t % dims[1], k], 1)


# 6.3: closed cells with a fluid leaf among their 26 neighbours get
# sub-cells (before: face neighbours only)
REFINE_EDGE_NEIGHBOURS = True


def _isin_sorted(q, s):
    """Mask of q found in the sorted unique array s, and positions."""
    if s.size == 0:
        return np.zeros(q.shape, bool), np.zeros(q.shape, np.int64)
    pos = np.searchsorted(s, q)
    pos = np.minimum(pos, s.size - 1)
    return s[pos] == q, pos


def _unique_chunks(chunks):
    """Sorted unique keys of an iterable of int64 arrays, memory-light."""
    acc = []
    tot = 0
    for c in chunks:
        u = np.unique(c)
        acc.append(u)
        tot += u.size
        if len(acc) > 8 or tot > 50_000_000:
            acc = [np.unique(np.concatenate(acc))]
            tot = acc[0].size
    if not acc:
        return np.zeros(0, np.int64)
    return np.unique(np.concatenate(acc))


_OFF26 = np.array([(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1)],
                  np.int64)


def _dilate(keys, dims, r=1, chunk=2_000_000):
    """Keys of all cells within Chebyshev distance r of the given cells."""
    offs = np.array([(a, b, c) for a in range(-r, r + 1) for b in range(-r, r + 1)
                     for c in range(-r, r + 1)], np.int64)

    def gen():
        for s0 in range(0, keys.size, chunk):
            C = _coords(keys[s0:s0 + chunk], dims)
            for o in offs:
                Q = C + o
                ok = np.all((Q >= 0) & (Q < dims), axis=1)
                yield _key(Q[ok], dims)
    return _unique_chunks(gen())


def _parents(keys, dims_child, dims_parent):
    return np.unique(_key(_coords(keys, dims_child) >> 1, dims_parent))


def _children(keys, dims_parent, dims_child):
    C = _coords(keys, dims_parent) * 2
    out = []
    for o in ((a, b, c) for a in (0, 1) for b in (0, 1) for c in (0, 1)):
        out.append(_key(C + np.array(o, np.int64), dims_child))
    return np.unique(np.concatenate(out)) if out else np.zeros(0, np.int64)


# --------------------------------------------------------------- octree
@dataclass
class Octree:
    h: float                       # finest cell size
    L: int                         # levels above the finest (base cell h * 2**L)
    origin: np.ndarray
    dims0: np.ndarray              # finest-level index space (multiple of 2**L)
    cut: np.ndarray                # level-0 keys of closed (surface) cells
    leaves: list                   # per level: sorted keys of fluid leaves
    refined: list                  # per level: sorted keys of refined cells
    triangles: np.ndarray | None = None
    margin: int = 1
    stats: dict = field(default_factory=dict)

    # grid-like attributes used by the model
    ndim = 3

    @property
    def dx(self):
        return self.h

    @property
    def cell_volume(self):
        return self.h ** 3

    @property
    def face_area(self):
        return self.h ** 2

    @property
    def shape(self):
        return tuple(int(v) for v in self.dims0)

    @property
    def ncells(self):
        """Number of leaves (fluid) plus closed cells."""
        return int(sum(K.size for K in self.leaves) + self.cut.size)

    @property
    def extent(self):
        return np.asarray(self.dims0) * self.h

    def dims(self, l):
        return _dims(self.dims0, l)

    def nleaves(self):
        return [int(k.size) for k in self.leaves]

    # ------------------------------------------------------------ build
    @classmethod
    def from_mesh(cls, mesh, h, levels=3, margin=1, pad=None, bounds=None, holes=None,
                  verbose=False, cut="exact"):
        """Octree for a triangle mesh (path or trimesh).

        h       finest cell size (the uniform-grid dx it replaces);
        levels  number of coarser levels (base cells of h * 2**levels);
        margin  cells of each level kept around the finer level (>= 1);
        pad     free space around the mesh (default: one base cell);
        bounds  (lo, hi): only this box is gridded (like Grid.from_mesh);
        holes   explicit holes to carve open (see ``carve_holes``);
        cut     "exact": every level-0 cell whose box touches a triangle is
                cut (default); "sample": point sampling as before 6.3."""
        import trimesh
        if not isinstance(mesh, trimesh.Trimesh):
            mesh = trimesh.load(mesh, force="mesh")
        t0 = time.time()
        L = int(levels)
        B = h * 2 ** L
        tri = np.asarray(mesh.triangles, float)
        if bounds is not None:
            lo = np.asarray(bounds[0], float)
            hi = np.asarray(bounds[1], float)
            tlo, thi = tri.min(1), tri.max(1)
            keep = np.all((thi >= lo - 2 * h) & (tlo <= hi + 2 * h), axis=1)
            tri = tri[keep]
        else:
            keep = slice(None)
            pad = B if pad is None else pad
            lo = tri.reshape(-1, 3).min(0) - pad
            hi = tri.reshape(-1, 3).max(0) + pad
        nb = np.ceil((hi - lo) / B - 1e-9).astype(np.int64)
        dims0 = nb * 2 ** L
        origin = lo.copy()
        # 1. level-0 cells touched by the surface: exact triangle-box test
        # (voxel.surface_cells); cut="sample" is the pre-6.3 point sampling,
        # which can miss cells a sheet only grazes
        if cut == "sample":
            def gen():
                for P in sample_triangles(tri, h):
                    idx = np.floor((P - origin) / h).astype(np.int64)
                    ok = np.all((idx >= 0) & (idx < dims0), axis=1)
                    yield _key(idx[ok], dims0)
            cut = _unique_chunks(gen())
        else:
            from .voxel import cut_cells
            cut = cut_cells(tri, origin, h, dims0)
        ot = cls(h=float(h), L=L, origin=origin, dims0=dims0, cut=cut, leaves=[], refined=[],
                 triangles=tri, margin=int(margin))
        # faces of the kept triangles (vertex ids): the free edges of the
        # mesh for the narrow-passage test
        ot.faces = np.asarray(mesh.faces, np.int64)[keep]
        if holes:
            ot.holes = ot.carve_holes(holes)
        else:
            ot.holes = []
        ot._build(verbose)
        ot.stats["build_s"] = time.time() - t0
        return ot

    def _build(self, verbose=False):
        L, m = self.L, max(1, int(self.margin))
        d0 = self.dims(0)
        refined = [np.zeros(0, np.int64)] * (L + 1)
        # level 0 must exist within `margin` cells of every closed cell
        R = _dilate(self.cut, d0, m) if self.cut.size else np.zeros(0, np.int64)
        for l in range(1, L + 1):
            P = _parents(R, self.dims(l - 1), self.dims(l)) if R.size else R
            refined[l] = P
            if l < L:
                R = _dilate(P, self.dims(l), m) if P.size else P
        leaves = [None] * (L + 1)
        # base level: every base cell that is not refined
        nbase = int(np.prod(self.dims(L)))
        allb = np.arange(nbase, dtype=np.int64)
        leaves[L] = np.setdiff1d(allb, refined[L], assume_unique=True) if L > 0 else allb
        for l in range(L - 1, -1, -1):
            ch = _children(refined[l + 1], self.dims(l + 1), self.dims(l))
            if l > 0:
                leaves[l] = np.setdiff1d(ch, refined[l], assume_unique=True)
            else:
                leaves[l] = np.setdiff1d(ch, self.cut, assume_unique=True)
        if L == 0:
            leaves[0] = np.setdiff1d(np.arange(int(np.prod(d0)), dtype=np.int64), self.cut,
                                     assume_unique=True)
        self.leaves, self.refined = leaves, refined
        if verbose:
            print("octree leaves per level:", self.nleaves(), "closed:", self.cut.size,
                  flush=True)

    # --------------------------------------------------------- queries
    def locate(self, P):
        """Node-level lookup: (level, key) of the leaf containing each point;
        level -1 for closed cells or outside."""
        P = np.asarray(P, float)
        lev = np.full(len(P), -1, np.int64)
        key = np.zeros(len(P), np.int64)
        c0 = np.floor((P - self.origin) / self.h).astype(np.int64)
        inside = np.all((c0 >= 0) & (c0 < self.dims0), axis=1)
        for l in range(self.L + 1):
            todo = inside & (lev < 0)
            if not todo.any():
                break
            d = self.dims(l)
            q = _key(c0[todo] >> l, d)
            f, _ = _isin_sorted(q, self.leaves[l])
            ii = np.flatnonzero(todo)[f]
            lev[ii] = l
            key[ii] = q[f]
        return lev, key

    def offsets(self):
        """First node index of each level's leaves (graph node order)."""
        offs = np.zeros(self.L + 2, np.int64)
        np.cumsum([K.size for K in self.leaves], out=offs[1:])
        return offs

    def node_of(self, P):
        """Graph node (leaf) containing each point, -1 in closed cells or
        outside."""
        lev, key = self.locate(P)
        offs = self.offsets()
        out = -np.ones(len(lev), np.int64)
        for l in range(self.L + 1):
            m = lev == l
            if m.any():
                out[m] = offs[l] + np.searchsorted(self.leaves[l], key[m])
        return out

    def leaf_centres(self, l):
        s = self.h * 2 ** l
        return self.origin + (_coords(self.leaves[l], self.dims(l)) + 0.5) * s

    # ------------------------------------------------------------ holes
    def carve_holes(self, holes):
        """Open listed holes in the closed level-0 cells (as
        ``compartments.carve_holes`` does on the uniform grid). Returns the
        holes as dicts with the axis filled in."""
        out = []
        h = self.h
        lo = self.origin
        hi = self.origin + self.dims0 * h
        for hole in holes:
            if not isinstance(hole, dict):
                hole = dict(center=hole[0], diameter=hole[1])
            c = np.asarray(hole["center"], float)
            d = float(hole["diameter"])
            if np.any(c - 0.5 * d < lo) or np.any(c + 0.5 * d > hi):
                continue                       # the hole is not inside the grid
            n = hole.get("axis")
            n = np.asarray(n if n is not None else self.hole_axis(c, d), float)
            n = n / np.linalg.norm(n)
            ci, Xb = self._box(c, 0.5 * d + 2 * h)
            rel = Xb - c
            sv = rel @ n
            rad = np.linalg.norm(rel - sv[:, None] * n, axis=1)
            carve = (np.abs(sv) <= h + 1e-9) & (rad <= max(0.5 * d - 0.5 * h, 0.75 * h))
            kill = _key(ci[carve], self.dims0)
            self.cut = np.setdiff1d(self.cut, kill, assume_unique=True)
            out.append(dict(center=c, diameter=d, axis=n,
                            open_at=float(hole.get("open_at", -np.inf))))
        return out

    def _box(self, c, radius):
        """Level-0 cell coordinates and centres in a box around point c."""
        h = self.h
        ci = np.floor((np.asarray(c) - self.origin) / h).astype(np.int64)
        m = int(np.ceil(radius / h)) + 1
        lo = np.maximum(ci - m, 0)
        hi = np.minimum(ci + m + 1, self.dims0)
        ax = [np.arange(lo[d], hi[d]) for d in range(3)]
        C = np.stack(np.meshgrid(*ax, indexing="ij"), -1).reshape(-1, 3)
        return C, self.origin + (C + 0.5) * h

    def hole_axis(self, center, d):
        """Normal of the sheet around a hole: least-spread direction of the
        closed cells in an annulus (as ``compartments._hole_axis``)."""
        C, Xb = self._box(center, 0.5 * d + 3 * self.h)
        r = np.linalg.norm(Xb - center, axis=1)
        solid, _ = _isin_sorted(_key(C, self.dims0), self.cut)
        near = solid & (r > 0.5 * d) & (r < 0.5 * d + 2.5 * self.h)
        P = Xb[near]
        if len(P) < 3:
            raise ValueError(f"no sheet found around hole at {center}")
        P = P - P.mean(0)
        w, V = np.linalg.eigh(P.T @ P)
        return V[:, 0]


# ----------------------------------------------------------------- graph
@dataclass
class OctGraph:
    """Nodes (fluid leaves, then sub-cells) and their links."""
    N: int
    level: np.ndarray              # int8 per node (-1: sub-cell)
    key: np.ndarray                # level key (sub-cells: key of their closed cell)
    X: np.ndarray
    size: np.ndarray               # edge length per node
    vol: np.ndarray
    boundary: np.ndarray           # touches the domain boundary
    indptr: np.ndarray
    indices: np.ndarray
    offsets: np.ndarray            # first node of each level (L+2 entries)
    fine: np.ndarray               # node is a sub-cell of a closed cell
    active: np.ndarray             # connected to a leaf (sealed sub-cell voids are not)
    stats: dict = field(default_factory=dict)


def _links_level(ot, l, ids):
    """Links of the leaves of level l: to same-level leaves (+ directions)
    and to coarser leaves (both directions). Returns (a, b) node arrays."""
    d = ot.dims(l)
    K = ot.leaves[l]
    if K.size == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    C = _coords(K, d)
    A, Bs = [], []
    me = ids[l] + np.arange(K.size)
    for ax in range(3):
        for sg in (-1, 1):
            Q = C.copy()
            Q[:, ax] += sg
            ok = (Q[:, ax] >= 0) & (Q[:, ax] < d[ax])
            q = _key(Q[ok], d)
            src = me[ok]
            found, pos = _isin_sorted(q, K)
            if sg > 0:
                A.append(src[found])
                Bs.append(ids[l] + pos[found])
            rest = ~found
            if l == 0:
                solid, _ = _isin_sorted(q[rest], ot.cut)
                rest[np.flatnonzero(rest)[solid]] = False
            if l < ot.L:
                refd, _ = _isin_sorted(q[rest], ot.refined[l])
                rest[np.flatnonzero(rest)[refd]] = False      # finer: from their side
                if rest.any():
                    pq = _key(Q[ok][rest] >> 1, ot.dims(l + 1))
                    f2, p2 = _isin_sorted(pq, ot.leaves[l + 1])
                    if not f2.all():
                        raise RuntimeError("octree not graded 2:1")
                    A.append(src[rest])
                    Bs.append(ids[l + 1] + p2)
    return np.concatenate(A), np.concatenate(Bs)


def build_graph(ot: Octree, subcells: int = 2, verbose=False, narrow=None,
                share="sight") -> OctGraph:
    """Node graph of an octree: fluid leaves (by level) and, with
    subcells >= 2, the fluid sub-cells of the closed level-0 cells.
    narrow: refine the closed cells with a narrow passage deeper (see
    ``_subcell_graph``)."""
    t0 = time.time()
    L = ot.L
    sizes = [K.size for K in ot.leaves]
    offs = np.zeros(L + 2, np.int64)
    np.cumsum(sizes, out=offs[1:])
    nleaf = int(offs[-1])
    ids = offs[:-1]
    level = np.concatenate([np.full(s, l, np.int8) for l, s in enumerate(sizes)])
    key = np.concatenate(ot.leaves)
    X = np.empty((nleaf, 3))
    size = np.empty(nleaf)
    bnd = np.zeros(nleaf, bool)
    for l in range(L + 1):
        d = ot.dims(l)
        C = _coords(ot.leaves[l], d)
        s = ot.h * 2 ** l
        sl = slice(offs[l], offs[l + 1])
        X[sl] = ot.origin + (C + 0.5) * s
        size[sl] = s
        bnd[sl] = np.any((C == 0) | (C == d - 1), axis=1)
    A, Bl = [], []
    for l in range(L + 1):
        a, b = _links_level(ot, l, ids)
        A.append(a)
        Bl.append(b)
    ea = np.concatenate(A)
    eb = np.concatenate(Bl)
    vol = size ** 3
    fine = np.zeros(nleaf, bool)
    active = np.ones(nleaf, bool)
    N = nleaf
    stats = dict(leaves=sizes, closed=int(ot.cut.size))
    if subcells and subcells >= 2 and ot.triangles is not None and ot.cut.size:
        sg = _subcell_graph(ot, int(subcells), ids, narrow=narrow,
                            leaf_links=(ea, eb), leaf_X=X, share=share)
        nf = sg["nfine"]
        N = nleaf + nf
        level = np.r_[level, np.full(nf, -1, np.int8)]
        key = np.r_[key, sg["host_key"]]
        X = np.vstack([X, sg["X"]])
        size = np.r_[size, sg["size"]]
        bnd = np.r_[bnd, sg["boundary"]]
        vol = np.r_[vol, np.zeros(nf)] + sg["credit"] * (ot.h / subcells) ** 3
        ea = np.r_[ea, sg["ea"]]
        eb = np.r_[eb, sg["eb"]]
        fine = np.r_[fine, np.ones(nf, bool)]
        stats.update(sg["stats"])
    # symmetric CSR, sorted by source then target (deterministic)
    src = np.r_[ea, eb]
    dst = np.r_[eb, ea]
    o = np.lexsort((dst, src))
    src, dst = src[o], dst[o]
    ptr = np.zeros(N + 1, np.int64)
    np.cumsum(np.bincount(src, minlength=N), out=ptr[1:])
    # sub-cells sealed from every leaf are inactive
    active = np.ones(N, bool)
    if N > nleaf:
        from .par import connected_to
        reach = connected_to(np.ones(N, bool), np.zeros(N, np.int64), ptr, dst,
                             ~fine)
        active = reach
        vol = np.where(active, vol, 0.0)
        stats["fine_active"] = int(active[nleaf:].sum())
    stats["links"] = int(ea.size)
    stats["graph_s"] = time.time() - t0
    g = OctGraph(N=int(N), level=level, key=key, X=X, size=size, vol=vol, boundary=bnd,
                 indptr=ptr, indices=dst.astype(np.int64), offsets=offs, fine=fine,
                 active=active, stats=stats)
    if verbose:
        print(f"octree graph: {N/1e6:.2f} M nodes ({nleaf/1e6:.2f} M leaves), "
              f"{ea.size/1e6:.2f} M links, {stats['graph_s']:.0f} s", flush=True)
    return g


# ----------------------------------------------- sub-cells of closed cells
@njit(cache=True)
def _fine_links_tab(k, nbcut, nbnode, fid, ea, eb, count_only):
    """Links of the sub-cell nodes (see ``volfrac._fine_links``), with the
    neighbours of every closed cell from tables: ``nbcut[ci, dir]`` = index
    of the closed neighbour cell (or -1), ``nbnode[ci, dir]`` = node of the
    fluid neighbour leaf (or -1); dir = 2 * axis + (0: -, 1: +)."""
    k3 = k * k * k
    ne = 0
    for ci in range(nbcut.shape[0]):
        for s0 in range(k3):
            a0 = fid[ci * k3 + s0]
            if a0 < 0:
                continue
            a = s0 // (k * k)
            b = (s0 // k) % k
            d = s0 % k
            for dirn in range(6):
                ax = dirn // 2
                sg = -1 if dirn % 2 == 0 else 1
                na, nb_, nd = a, b, d
                cross = False
                if ax == 0:
                    na += sg
                    if na < 0 or na >= k:
                        na = k - 1 if na < 0 else 0
                        cross = True
                elif ax == 1:
                    nb_ += sg
                    if nb_ < 0 or nb_ >= k:
                        nb_ = k - 1 if nb_ < 0 else 0
                        cross = True
                else:
                    nd += sg
                    if nd < 0 or nd >= k:
                        nd = k - 1 if nd < 0 else 0
                        cross = True
                g = (na * k + nb_) * k + nd
                if not cross:
                    if sg < 0:
                        continue
                    b0 = fid[ci * k3 + g]
                    if b0 < 0:
                        continue
                else:
                    if nbnode[ci, dirn] >= 0:
                        b0 = nbnode[ci, dirn]
                    else:
                        nci = nbcut[ci, dirn]
                        if nci < 0 or sg < 0:
                            continue
                        b0 = fid[nci * k3 + g]
                        if b0 < 0:
                            continue
                if not count_only:
                    ea[ne] = a0
                    eb[ne] = b0
                ne += 1
    return ne


@njit(cache=True)
def _split_surface_tab(k, nbcut, nbnode, fine_solid, fid, credit):
    """Share each surface sub-cell among its fluid face neighbours (see
    ``volfrac._split_surface_fine``)."""
    k3 = k * k * k
    votes = np.empty(6, np.int64)
    for ci in range(nbcut.shape[0]):
        for s0 in range(k3):
            if not fine_solid[ci * k3 + s0]:
                continue
            a = s0 // (k * k)
            b = (s0 // k) % k
            d = s0 % k
            nv = 0
            for dirn in range(6):
                ax = dirn // 2
                sg = -1 if dirn % 2 == 0 else 1
                na, nb_, nd = a, b, d
                cross = False
                if ax == 0:
                    na += sg
                    if na < 0 or na >= k:
                        na = k - 1 if na < 0 else 0
                        cross = True
                elif ax == 1:
                    nb_ += sg
                    if nb_ < 0 or nb_ >= k:
                        nb_ = k - 1 if nb_ < 0 else 0
                        cross = True
                else:
                    nd += sg
                    if nd < 0 or nd >= k:
                        nd = k - 1 if nd < 0 else 0
                        cross = True
                g = (na * k + nb_) * k + nd
                if not cross:
                    b0 = fid[ci * k3 + g]
                elif nbnode[ci, dirn] >= 0:
                    b0 = nbnode[ci, dirn]
                else:
                    nci = nbcut[ci, dirn]
                    b0 = -1 if nci < 0 else fid[nci * k3 + g]
                if b0 >= 0:
                    votes[nv] = b0
                    nv += 1
            for q in range(nv):
                credit[votes[q]] += 1.0 / nv
    return 0


@njit(cache=True)
def _nb_sub(k, a, b, d, dirn):
    """Sub-cell index in the same cell one step in ``dirn`` and whether the
    step leaves the cell (then the index is that of the entry sub-cell of
    the neighbour cell at the same resolution)."""
    ax = dirn // 2
    sg = -1 if dirn % 2 == 0 else 1
    na, nb_, nd = a, b, d
    cross = False
    if ax == 0:
        na += sg
        if na < 0 or na >= k:
            na = k - 1 if na < 0 else 0
            cross = True
    elif ax == 1:
        nb_ += sg
        if nb_ < 0 or nb_ >= k:
            nb_ = k - 1 if nb_ < 0 else 0
            cross = True
    else:
        nd += sg
        if nd < 0 or nd >= k:
            nd = k - 1 if nd < 0 else 0
            cross = True
    return na, nb_, nd, cross


@njit(cache=True)
def _fine_links_var(kc, off, nbcut, nbnode, fid, ea, eb, count_only):
    """``_fine_links_tab`` with a resolution ``kc[ci]`` per closed cell
    (sub-cells ``off[ci] .. off[ci] + kc^3``). Across a face between two
    closed cells of different resolution, each sub-cell of the finer one
    links to the sub-cell of the coarser one it faces (hanging faces)."""
    ne = 0
    for ci in range(nbcut.shape[0]):
        k = kc[ci]
        k3 = k * k * k
        o = off[ci]
        for s0 in range(k3):
            a0 = fid[o + s0]
            if a0 < 0:
                continue
            a = s0 // (k * k)
            b = (s0 // k) % k
            d = s0 % k
            for dirn in range(6):
                sg = -1 if dirn % 2 == 0 else 1
                na, nb_, nd, cross = _nb_sub(k, a, b, d, dirn)
                if not cross:
                    if sg < 0:
                        continue
                    b0 = fid[o + (na * k + nb_) * k + nd]
                    if b0 < 0:
                        continue
                elif nbnode[ci, dirn] >= 0:
                    b0 = nbnode[ci, dirn]
                else:
                    nci = nbcut[ci, dirn]
                    if nci < 0:
                        continue
                    kn = kc[nci]
                    if kn == k:
                        if sg < 0:
                            continue
                        b0 = fid[off[nci] + (na * k + nb_) * k + nd]
                    elif kn < k:
                        r = k // kn
                        ax = dirn // 2
                        pa, pb, pd = na // r, nb_ // r, nd // r
                        if ax == 0:
                            pa = kn - 1 if sg < 0 else 0
                        elif ax == 1:
                            pb = kn - 1 if sg < 0 else 0
                        else:
                            pd = kn - 1 if sg < 0 else 0
                        b0 = fid[off[nci] + (pa * kn + pb) * kn + pd]
                    else:
                        continue                      # from the finer side
                    if b0 < 0:
                        continue
                if not count_only:
                    ea[ne] = a0
                    eb[ne] = b0
                ne += 1
    return ne


@njit(cache=True)
def _split_surface_var(kc, off, kbase, nbcut, nbnode, fine_solid, fid, credit,
                       mode, cnt, vptr, vid):
    """``_split_surface_tab`` with a resolution per closed cell. Credits are
    in units of a base sub-cell volume (h / kbase)^3.

    mode 0: share each surface sub-cell equally among its votes (its fluid
    face neighbours); 1: only count the votes per surface sub-cell into cnt
    (indexed like fine_solid); 2: write them to vid[vptr[j]:] (for
    ``sidesplit.split_by_sight``)."""
    votes = np.empty(6 * 64, np.int64)
    for ci in range(nbcut.shape[0]):
        k = kc[ci]
        k3 = k * k * k
        o = off[ci]
        own = (kbase / k) ** 3
        for s0 in range(k3):
            if not fine_solid[o + s0]:
                continue
            a = s0 // (k * k)
            b = (s0 // k) % k
            d = s0 % k
            nv = 0
            for dirn in range(6):
                sg = -1 if dirn % 2 == 0 else 1
                na, nb_, nd, cross = _nb_sub(k, a, b, d, dirn)
                if not cross:
                    b0 = fid[o + (na * k + nb_) * k + nd]
                    if b0 >= 0:
                        votes[nv] = b0
                        nv += 1
                elif nbnode[ci, dirn] >= 0:
                    votes[nv] = nbnode[ci, dirn]
                    nv += 1
                else:
                    nci = nbcut[ci, dirn]
                    if nci < 0:
                        continue
                    kn = kc[nci]
                    ax = dirn // 2
                    if kn == k:
                        b0 = fid[off[nci] + (na * k + nb_) * k + nd]
                        if b0 >= 0:
                            votes[nv] = b0
                            nv += 1
                    elif kn < k:
                        r = k // kn
                        pa, pb, pd = na // r, nb_ // r, nd // r
                        if ax == 0:
                            pa = kn - 1 if sg < 0 else 0
                        elif ax == 1:
                            pb = kn - 1 if sg < 0 else 0
                        else:
                            pd = kn - 1 if sg < 0 else 0
                        b0 = fid[off[nci] + (pa * kn + pb) * kn + pd]
                        if b0 >= 0:
                            votes[nv] = b0
                            nv += 1
                    else:
                        r = kn // k
                        for u in range(r):
                            for w in range(r):
                                if ax == 0:
                                    qa = kn - 1 if sg < 0 else 0
                                    qb, qd = nb_ * r + u, nd * r + w
                                elif ax == 1:
                                    qb = kn - 1 if sg < 0 else 0
                                    qa, qd = na * r + u, nd * r + w
                                else:
                                    qd = kn - 1 if sg < 0 else 0
                                    qa, qb = na * r + u, nb_ * r + w
                                b0 = fid[off[nci] + (qa * kn + qb) * kn + qd]
                                if b0 >= 0 and nv < votes.shape[0]:
                                    votes[nv] = b0
                                    nv += 1
            if mode == 0:
                for q in range(nv):
                    credit[votes[q]] += own / nv
            elif mode == 1:
                cnt[o + s0] = nv
            else:
                w0 = vptr[o + s0]
                for q in range(nv):
                    vid[w0 + q] = votes[q]
    return 0


def _thin(P, N, s):
    """One sample per cube of size s (the first), to bound the sample count
    whatever the triangle shapes."""
    if len(P) == 0:
        return P, N
    q = np.floor(P / s).astype(np.int64)
    q -= q.min(0)
    span = q.max(0) + 1
    key = (q[:, 0] * span[1] + q[:, 1]) * span[2] + q[:, 2]
    _, first = np.unique(key, return_index=True)
    return P[first], N[first]


def _samples_with_normals(tri, s, max_points=2_000_000):
    """Points on the triangles with spacing below s (at most one per cube of
    size s/2), and the unit normal of their triangle."""
    tri = np.asarray(tri, float)
    nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    ln = np.linalg.norm(nrm, axis=1)
    ok = ln > 0
    tri, nrm = tri[ok], nrm[ok] / ln[ok, None]
    emax = np.max(np.linalg.norm(tri[:, [1, 2, 0]] - tri, axis=2), axis=1)
    nsub = np.maximum(1, np.ceil(emax / s)).astype(int)
    P, N = [], []
    for n in np.unique(nsub):
        sel = np.flatnonzero(nsub == n)
        if n == 1:                         # small triangles: their centroid
            p_, n_ = _thin(tri[sel].mean(1), nrm[sel], 0.5 * s)
            P.append(p_)
            N.append(n_)
            continue
        i, j = np.meshgrid(np.arange(n + 1), np.arange(n + 1), indexing="ij")
        okk = i + j <= n
        w1 = (i[okk] / n)[:, None]
        w2 = (j[okk] / n)[:, None]
        w0 = 1.0 - w1 - w2
        ch = max(1, max_points // len(w0))
        for c0 in range(0, len(sel), ch):
            T = tri[sel[c0:c0 + ch]]
            pts = (w0[None] * T[:, None, 0] + w1[None] * T[:, None, 1]
                   + w2[None] * T[:, None, 2]).reshape(-1, 3)
            p_, n_ = _thin(pts, np.repeat(nrm[sel[c0:c0 + ch]], len(w0), axis=0), 0.5 * s)
            P.append(p_)
            N.append(n_)
    P = np.concatenate(P)
    N = np.concatenate(N)
    return _thin(P, N, 0.5 * s)


def _boundary_samples(tri, s, faces=None):
    """Points on the free edges of the triangle mesh (edges of one
    triangle only: rims of holes and slots, sheet edges), spacing below s,
    with the in-plane unit direction pointing away from the sheet.
    faces: vertex ids of the triangles (edges are identified by them);
    None: by their rounded end points."""
    tri = np.asarray(tri, float)
    nt = len(tri)
    if faces is not None:
        f = np.asarray(faces, np.int64)
        ea = np.concatenate([f[:, 0], f[:, 1], f[:, 2]])
        eb = np.concatenate([f[:, 1], f[:, 2], f[:, 0]])
        nv = int(f.max()) + 1
        key = np.minimum(ea, eb) * nv + np.maximum(ea, eb)
        del ea, eb
    else:
        E0 = np.concatenate([tri[:, 0], tri[:, 1], tri[:, 2]])
        E1 = np.concatenate([tri[:, 1], tri[:, 2], tri[:, 0]])
        qa = np.round(E0 * 1e9).astype(np.int64)
        qb = np.round(E1 * 1e9).astype(np.int64)
        del E0, E1
        first = (qa[:, 0] < qb[:, 0]) | ((qa[:, 0] == qb[:, 0]) & (
            (qa[:, 1] < qb[:, 1]) | ((qa[:, 1] == qb[:, 1]) & (qa[:, 2] <= qb[:, 2]))))
        lo = np.where(first[:, None], qa, qb)
        hi = np.where(first[:, None], qb, qa)
        _, key = np.unique(np.c_[lo, hi], axis=0, return_inverse=True)
        key = key.ravel()
    order = np.argsort(key, kind="stable")
    ks = key[order]
    single = np.ones(len(ks), bool)
    same = ks[1:] == ks[:-1]
    single[1:] &= ~same
    single[:-1] &= ~same
    free = np.sort(order[single])
    del key, order, ks, single, same
    ti, e = free % nt, free // nt                     # triangle, edge 0/1/2
    a = tri[ti, e]
    b = tri[ti, (e + 1) % 3]
    o = tri[ti, (e + 2) % 3]
    if len(a) == 0:
        return np.zeros((0, 3)), np.zeros((0, 3))
    t = b - a
    L = np.linalg.norm(t, axis=1)
    t = t / np.maximum(L, 1e-30)[:, None]
    nrm = np.cross(t, o - a)
    m = np.cross(nrm, t)
    m /= np.maximum(np.linalg.norm(m, axis=1), 1e-30)[:, None]
    m[np.einsum("ij,ij->i", m, o - a) > 0] *= -1           # away from the sheet
    n = np.maximum(1, np.ceil(L / s)).astype(int)
    P, Mo = [], []
    for c0 in range(0, len(a), 500_000):
        nn = n[c0:c0 + 500_000]
        rep = np.repeat(np.arange(c0, c0 + len(nn)), nn + 1)
        start = np.r_[0, np.cumsum(nn + 1)[:-1]]
        f = (np.arange(len(rep)) - np.repeat(start, nn + 1)) / np.repeat(nn, nn + 1)
        p_, m_ = _thin(a[rep] + f[:, None] * (b - a)[rep], m[rep], 0.5 * s)
        P.append(p_)
        Mo.append(m_)
    return _thin(np.concatenate(P), np.concatenate(Mo), 0.5 * s)


def narrow_cells(ot, S, width, eps, s, nnb=48, dilate=False):
    """Closed cells (mask over S) with a narrow passage in or next to them:
    two nearly parallel sheets less than ``width`` apart (but more than
    ``eps``), or two free edges (a slot, a hole, a gap between the edges of
    two sheets) facing each other less than ``width`` apart. Surface
    samples at spacing ``s``. The cells holding the two walls are marked (a
    passage narrower than a cell lies in them); ``dilate`` adds their closed
    face neighbours. For plates with thickness (oriented faces in
    back-to-back pairs, ``_plate_sign``) only faces facing each other across
    fluid count, and the free-edge test is skipped."""
    from scipy.spatial import cKDTree
    h = ot.h
    d0 = ot.dims(0)
    hit = np.zeros(S.size, bool)

    def mark(Q):
        ci = np.floor((Q - ot.origin) / h).astype(np.int64)
        ok = np.all((ci >= 0) & (ci < d0), axis=1)
        f, p = _isin_sorted(_key(ci[ok], d0), S)
        hit[p[f]] = True

    sign = 0
    for facing, (P, N) in ((False, _samples_with_normals(ot.triangles, s)),
                           (True, _boundary_samples(ot.triangles, s,
                                                    getattr(ot, "faces", None)))):
        if len(P) < 2 or (facing and sign != 0):
            # plates with thickness: their free edges are mostly where parts
            # meet (a mesh soup), and slots have side walls the face test sees
            continue
        tree = cKDTree(P)
        if not facing:
            sign = _plate_sign(P, N, tree, min(width, 0.003))
            ot.stats["narrow_plate_sign"] = int(sign)
        kq = min(nnb, len(P))
        ch = 250_000
        for s0 in range(0, len(P), ch):
            dist, nb = tree.query(P[s0:s0 + ch], k=kq, distance_upper_bound=width,
                                  workers=-1)
            j = _gap_rows(P, N, s0, nb, dist, eps, width, facing, sign)
            del dist, nb
            sel = j >= 0
            if sel.any():
                mark(P[s0:s0 + ch][sel])
                mark(P[j[sel]])
        del tree
    if not hit.any() or not dilate:
        return hit
    # and their closed face neighbours
    C = _coords(S[hit], d0)
    out = hit.copy()
    for ax in range(3):
        for sg in (-1, 1):
            Q = C.copy()
            Q[:, ax] += sg
            ok = (Q[:, ax] >= 0) & (Q[:, ax] < d0[ax])
            f, p = _isin_sorted(_key(Q[ok], d0), S)
            out[p[f]] = True
    return out


def _gap_rows(P, N, s0, nb, dist, eps, width, facing, sign=0):
    """``_gap_test_rows`` for the rows s0.. of the samples (neighbour ids
    are global)."""
    return _gap_test_rows(P, N, int(s0), nb.astype(np.int64), dist, float(eps),
                          float(width), bool(facing), int(sign))


def _plate_sign(P, N, tree, tmax, nq=20000, seed=0):
    """Are the triangles oriented faces of plates with thickness (a car
    body exported as thin solids)? Looks for back-to-back face pairs less
    than ``tmax`` apart (antiparallel normals, the partner straight across).
    Returns the sign of the fluid side for ``_gap_test_rows``: +1 if the
    plates' partners lie behind the normals (outward normals), -1 if in
    front (inward normals), 0 if the faces are not paired (sheets)."""
    rng = np.random.default_rng(seed)
    q = rng.choice(len(P), min(nq, len(P)), replace=False)
    d, j = tree.query(P[q], k=min(24, len(P)), distance_upper_bound=tmax, workers=-1)
    pos = neg = 0
    for r in range(len(q)):
        ok = np.isfinite(d[r]) & (j[r] < len(P))
        jj = j[r][ok][1:]
        if jj.size == 0:
            continue
        w = P[jj] - P[q[r]]
        along = w @ N[q[r]]
        c = N[jj] @ N[q[r]]
        m = (c < -0.7) & (np.abs(along) > 0.7071 * np.linalg.norm(w, axis=1)) & \
            (np.abs(along) > 1e-5)
        if m.any():
            a = along[m][np.argmin(np.abs(along[m]))]
            pos += a > 0
            neg += a < 0
    tot = pos + neg
    if tot < 0.2 * len(q):
        return 0
    if neg >= 0.8 * tot:
        return 1
    if pos >= 0.8 * tot:
        return -1
    return 0


@njit(cache=True)
def _gap_test_rows(P, N, s0, nb, dist, eps, width, facing, sign):
    """For samples s0.. : is there another sample within ``width`` across a
    gap? Returns the partner or -1.

    Sheets (facing = False): a nearly parallel face (|n.n'| > 0.7) offset
    along the normal by more than ``eps``, within 45 deg of the normal.
    sign = 0: unoriented normals (zero-thickness sheets); sign = +1 / -1:
    oriented faces of plates with thickness: only faces that face each
    other across fluid count (normals antiparallel, the partner on the
    +n / -n side), so the two faces of one plate do not.
    Free edges (facing = True): an edge in front of this one (within 60 deg
    of its outward direction) facing it."""
    n = nb.shape[0]
    npts = P.shape[0]
    out = -np.ones(n, np.int64)
    for r in range(n):
        i = s0 + r
        for jj in range(1, nb.shape[1]):
            d = dist[r, jj]
            if d > width:
                break
            j = nb[r, jj]
            if j >= npts or d <= eps:
                continue
            w0 = P[j, 0] - P[i, 0]
            w1 = P[j, 1] - P[i, 1]
            w2 = P[j, 2] - P[i, 2]
            c = N[i, 0] * N[j, 0] + N[i, 1] * N[j, 1] + N[i, 2] * N[j, 2]
            along = N[i, 0] * w0 + N[i, 1] * w1 + N[i, 2] * w2
            if facing:
                if along > 0.5 * d and c < -0.3:
                    out[r] = j
                    break
            elif sign == 0:
                if abs(c) > 0.7 and abs(along) > eps and abs(along) > 0.7071 * d:
                    out[r] = j
                    break
            else:
                if c < -0.7 and sign * along > eps and sign * along > 0.7071 * d:
                    out[r] = j
                    break
    return out


@njit(cache=True)
def _find_sorted(S, key):
    lo = 0
    hi = S.shape[0]
    while lo < hi:
        mid = (lo + hi) // 2
        if S[mid] < key:
            lo = mid + 1
        else:
            hi = mid
    if lo < S.shape[0] and S[lo] == key:
        return lo
    return -1


@njit(cache=True)
def _bin_point(x, y, z, o0, o1, o2, h, d0, S, kc, off, target, fine_solid):
    q0 = (x - o0) / h
    q1 = (y - o1) / h
    q2 = (z - o2) / h
    c0 = np.int64(np.floor(q0))
    c1 = np.int64(np.floor(q1))
    c2 = np.int64(np.floor(q2))
    if c0 < 0 or c1 < 0 or c2 < 0 or c0 >= d0[0] or c1 >= d0[1] or c2 >= d0[2]:
        return
    f0 = q0 - c0
    f1 = q1 - c1
    f2 = q2 - c2
    cc = _find_sorted(S, (c0 * d0[1] + c1) * d0[2] + c2)
    # a sample on a coarse face goes to the closed cell of the coarse
    # voxelisation (axes in order, the lower side first)
    for d in range(3):
        for side in range(2):
            if cc >= 0:
                break
            fd = f0 if d == 0 else (f1 if d == 1 else f2)
            if side == 0:
                if not fd < 1e-6:
                    continue
                step = -1
            else:
                if not fd > 1 - 1e-6:
                    continue
                step = 1
            n0, n1, n2 = c0, c1, c2
            if d == 0:
                n0 += step
            elif d == 1:
                n1 += step
            else:
                n2 += step
            nd = n0 if d == 0 else (n1 if d == 1 else n2)
            if nd < 0 or nd >= d0[d]:
                continue
            j = _find_sorted(S, (n0 * d0[1] + n1) * d0[2] + n2)
            if j >= 0:
                c0, c1, c2 = n0, n1, n2
                cc = j
                fnew = 1.0 - 1e-9 if step < 0 else 0.0
                if d == 0:
                    f0 = fnew
                elif d == 1:
                    f1 = fnew
                else:
                    f2 = fnew
    if cc < 0 or not target[cc]:
        return
    k = kc[cc]
    s0 = min(max(np.int64(np.floor(f0 * k)), 0), k - 1)
    s1 = min(max(np.int64(np.floor(f1 * k)), 0), k - 1)
    s2 = min(max(np.int64(np.floor(f2 * k)), 0), k - 1)
    fine_solid[off[cc] + (s0 * k + s1) * k + s2] = True


@njit(cache=True)
def _bin_kernel(P, o, h, d0, S, kc, off, target, fine_solid):
    for i in range(P.shape[0]):
        _bin_point(P[i, 0], P[i, 1], P[i, 2], o[0], o[1], o[2], h, d0, S, kc, off, target,
                   fine_solid)
    return 0


@njit(cache=True)
def _sample_bin_kernel(tri, hs, o, h, d0, S, kc, off, target, fine_solid):
    """Sample each triangle on a barycentric lattice with spacing below hs
    and bin the samples (``_bin_point``), without storing them."""
    for t in range(tri.shape[0]):
        e = 0.0
        for a in range(3):
            b = (a + 1) % 3
            dx = tri[t, b, 0] - tri[t, a, 0]
            dy = tri[t, b, 1] - tri[t, a, 1]
            dz = tri[t, b, 2] - tri[t, a, 2]
            e = max(e, np.sqrt(dx * dx + dy * dy + dz * dz))
        n = max(1, int(np.ceil(e / hs)))
        for i in range(n + 1):
            for j in range(n + 1 - i):
                w1 = i / n
                w2 = j / n
                w0 = 1.0 - w1 - w2
                x = w0 * tri[t, 0, 0] + w1 * tri[t, 1, 0] + w2 * tri[t, 2, 0]
                y = w0 * tri[t, 0, 1] + w1 * tri[t, 1, 1] + w2 * tri[t, 2, 1]
                z = w0 * tri[t, 0, 2] + w1 * tri[t, 1, 2] + w2 * tri[t, 2, 2]
                _bin_point(x, y, z, o[0], o[1], o[2], h, d0, S, kc, off, target, fine_solid)
    return 0


@njit(cache=True)
def _sample_bin_kernel_uv(tri, hs, o, h, d0, S, kc, off, target, fine_solid):
    """As ``_sample_bin_kernel``, with a sampling that stays proportional to
    the area for sliver triangles: apex a opposite the shortest edge (b, c),
    points a + u ((b + v (c - b)) - a), u and v on uniform grids with
    spacing below hs along both directions."""
    for t in range(tri.shape[0]):
        # shortest edge
        best = 1e300
        ia = 0
        for a in range(3):
            b = (a + 1) % 3
            c = (a + 2) % 3
            dx = tri[t, c, 0] - tri[t, b, 0]
            dy = tri[t, c, 1] - tri[t, b, 1]
            dz = tri[t, c, 2] - tri[t, b, 2]
            L = dx * dx + dy * dy + dz * dz
            if L < best:
                best = L
                ia = a
        ib = (ia + 1) % 3
        ic = (ia + 2) % 3
        lbc = np.sqrt(best)
        lab = 0.0
        lac = 0.0
        for d in range(3):
            lab += (tri[t, ib, d] - tri[t, ia, d]) ** 2
            lac += (tri[t, ic, d] - tri[t, ia, d]) ** 2
        nu = max(1, int(np.ceil(np.sqrt(max(lab, lac)) / hs)))
        nv = max(1, int(np.ceil(lbc / hs)))
        for j in range(nv + 1):
            v = j / nv
            ex = tri[t, ib, 0] + v * (tri[t, ic, 0] - tri[t, ib, 0])
            ey = tri[t, ib, 1] + v * (tri[t, ic, 1] - tri[t, ib, 1])
            ez = tri[t, ib, 2] + v * (tri[t, ic, 2] - tri[t, ib, 2])
            for i in range(nu + 1):
                u = i / nu
                x = tri[t, ia, 0] + u * (ex - tri[t, ia, 0])
                y = tri[t, ia, 1] + u * (ey - tri[t, ia, 1])
                z = tri[t, ia, 2] + u * (ez - tri[t, ia, 2])
                _bin_point(x, y, z, o[0], o[1], o[2], h, d0, S, kc, off, target, fine_solid)
    return 0


def _bin_samples(ot, S, d0, chunks, kc, off, target, fine_solid):
    """Mark the sub-cells touched by the surface samples: each sample goes
    to its closed cell (a sample on a coarse face goes to the closed cell of
    the coarse voxelisation) and, if that cell is a ``target``, to its
    sub-cell at the cell's resolution ``kc``."""
    o = np.asarray(ot.origin, np.float64)
    d0 = np.asarray(d0, np.int64)
    for P in chunks:
        _bin_kernel(np.ascontiguousarray(P, np.float64), o, float(ot.h), d0, S, kc, off,
                    target, fine_solid)


def _triangles_near(ot, S, mask):
    """Triangles that can touch the closed cells S[mask] (bounding box
    against the cells, one cell of margin)."""
    tri = ot.triangles
    d0 = ot.dims(0)
    lo = np.floor((tri.min(1) - ot.origin) / ot.h).astype(np.int64) - 1
    hi = np.floor((tri.max(1) - ot.origin) / ot.h).astype(np.int64) + 1
    sel = np.zeros(len(tri), bool)
    small = np.all(hi - lo <= 4, axis=1)
    # small triangles: test every cell of their box
    ext = (hi - lo + 1)[small].max(0) if small.any() else np.zeros(3, np.int64)
    idx = np.flatnonzero(small)
    Ssel = S[mask]
    for a in range(int(ext[0])):
        for b in range(int(ext[1])):
            for c in range(int(ext[2])):
                C = lo[idx] + np.array([a, b, c])
                inb = np.all(C <= hi[idx], axis=1) & np.all((C >= 0) & (C < d0), axis=1)
                f, _ = _isin_sorted(_key(np.where(inb[:, None], C, 0), d0), Ssel)
                sel[idx[inb & f]] = True
    sel[~small] = True                                   # large ones: keep
    return tri[sel]


def _connecting_cells(ot, k, kc, off, nbcut, nbnode, fine_solid, C, nar, leaf_links,
                      leaf_X, margin=1.0):
    """Of the narrow candidate cells (refined in ``kc``), those holding a
    passage that connects fluid regions: the fine fluid sub-cells of the
    candidate cells form components; a component connects if the nodes
    outside the candidates it touches (leaves, sub-cells of other closed
    cells: its mouths) belong to two or more components of the outside
    graph restricted to the nodes within ``margin`` cells of those contacts.
    A dead-end seam or crevice has one mouth, whose contacts are joined by
    the fluid in front of it, and is left at the base resolution. A
    passage with two mouths (through a sheet, between two spaces) is kept,
    also when a wider opening elsewhere joins the two spaces, since that
    one may lie higher."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree
    h = ot.h
    nleaf = leaf_X.shape[0]
    fid = np.where(fine_solid, -1, 0).astype(np.int64)
    fl = ~fine_solid
    nf = int(fl.sum())
    fid[fl] = nleaf + np.arange(nf)
    m = _fine_links_var(kc, off, nbcut, nbnode, fid, np.zeros(1, np.int64),
                        np.zeros(1, np.int64), True)
    fa = np.empty(m, np.int64)
    fb = np.empty(m, np.int64)
    _fine_links_var(kc, off, nbcut, nbnode, fid, fa, fb, False)
    flat = np.flatnonzero(fl)
    ci = np.searchsorted(off, flat, side="right") - 1
    sub = flat - off[ci]
    kk = kc[ci]
    abd = np.stack([sub // (kk * kk), (sub // kk) % kk, sub % kk], 1)
    Xf = ot.origin + C[ci] * h + (abd + 0.5) * (h / kk)[:, None]
    N = nleaf + nf
    X = np.vstack([leaf_X, Xf])
    host = np.r_[-np.ones(nleaf, np.int64), ci]
    cand = np.r_[np.zeros(nleaf, bool), nar[ci]]
    u = np.r_[leaf_links[0], fa]
    v = np.r_[leaf_links[1], fb]
    # passage components: candidate nodes linked among themselves
    pp = cand[u] & cand[v]
    G = coo_matrix((np.ones(pp.sum()), (u[pp], v[pp])), shape=(N, N))
    _, pc = connected_components(G, directed=False)
    # contacts: candidate node - outside node
    cu = cand[u] & ~cand[v]
    cv = ~cand[u] & cand[v]
    pcomp = np.r_[pc[u[cu]], pc[v[cv]]]
    onode = np.r_[v[cu], u[cv]]
    keep = np.zeros(nar.size, bool)
    if pcomp.size == 0:
        return keep
    # outside graph (CSR) and a k-d tree of the outside nodes
    oo = ~cand[u] & ~cand[v]
    src = np.r_[u[oo], v[oo]]
    dst = np.r_[v[oo], u[oo]]
    o = np.argsort(src, kind="stable")
    optr = np.zeros(N + 1, np.int64)
    np.cumsum(np.bincount(src, minlength=N), out=optr[1:])
    oidx = dst[o]
    outside = np.flatnonzero(~cand)
    tree = cKDTree(X[outside])
    loc = -np.ones(N, np.int64)
    # the fine nodes of each passage component, grouped
    cn = np.flatnonzero(cand)
    order = np.argsort(pc[cn], kind="stable")
    cn = cn[order]
    pcs = pc[cn]
    starts = np.r_[0, np.flatnonzero(np.diff(pcs)) + 1, cn.size]
    comp_of_start = pcs[starts[:-1]]
    corder = np.argsort(pcomp, kind="stable")
    pcomp, onode = pcomp[corder], onode[corder]
    cst = np.searchsorted(pcomp, comp_of_start)
    cen = np.searchsorted(pcomp, comp_of_start, side="right")
    R = margin * h
    for gi in range(len(comp_of_start)):
        cont = np.unique(onode[cst[gi]:cen[gi]])
        if cont.size < 2:
            continue
        mem = cn[starts[gi]:starts[gi + 1]]
        near = tree.query_ball_point(X[cont], R)
        inb = outside[np.unique(np.concatenate([np.asarray(q_, np.int64) for q_ in near]))]
        nodes = np.union1d(inb, cont)
        loc[nodes] = np.arange(nodes.size)
        try:
            cnt = optr[nodes + 1] - optr[nodes]
            s_ = np.repeat(nodes, cnt)
            pos = np.repeat(optr[nodes] - np.r_[0, np.cumsum(cnt)[:-1]], cnt) + np.arange(cnt.sum())
            d_ = oidx[pos]
            ok = loc[d_] >= 0
            G2 = coo_matrix((np.ones(ok.sum()), (loc[s_[ok]], loc[d_[ok]])),
                            shape=(nodes.size,) * 2)
            _, lc = connected_components(G2, directed=False)
            regions = np.unique(lc[loc[cont]])
        finally:
            loc[nodes] = -1
        if regions.size >= 2:
            keep[np.unique(host[mem])] = True
    return keep


def _share_by_sight(kc, off, k, nbcut, nbnode, fine_solid, fid, credit, C, S, ot, nc, ns,
                    deep=True):
    """Surface sub-cells shared by line of sight (``sidesplit``). Returns
    the volume (m^3) dropped as metal."""
    from .sidesplit import split_by_sight, surface_items, triangles_per_cell
    h = ot.h
    z = np.zeros(1, np.int64)
    cnt = np.zeros(fine_solid.size, np.int64)
    _split_surface_var(kc, off, k, nbcut, nbnode, fine_solid, fid, credit, 1, cnt, z, z)
    vptr = np.zeros(fine_solid.size + 1, np.int64)
    np.cumsum(cnt, out=vptr[1:])
    vid = np.empty(int(vptr[-1]), np.int64)
    _split_surface_var(kc, off, k, nbcut, nbnode, fine_solid, fid, credit, 2, cnt, vptr, vid)
    items, ci, iptr, ivid = surface_items(
        cnt, vid, fine_solid, lambda j: np.searchsorted(off, j, side="right") - 1, deep)
    sub = items - off[ci]
    kk = kc[ci]
    abd = np.stack([sub // (kk * kk), (sub // kk) % kk, sub % kk], 1)
    s = h / kk
    lo = ot.origin + C[ci] * h + abd * s[:, None]
    tptr, tidx = triangles_per_cell(ot.triangles, ot.origin, h, ot.dims(0), S)
    dropped = split_by_sight(iptr, ivid, lo, s.astype(np.float64), ci.astype(np.int64),
                             ((k / kk) ** 3).astype(np.float64),
                             np.ascontiguousarray(nc, np.float64),
                             np.ascontiguousarray(ns, np.float64),
                             ot.triangles, tptr, tidx, credit)
    return float(dropped * (h / k) ** 3)


def _subcell_graph(ot, k, ids, narrow=None, leaf_links=None, leaf_X=None, share="sight"):
    """Fluid sub-cells of the closed level-0 cells as nodes (the octree
    version of ``volfrac.cut_cell_graph``).

    narrow: None, True or dict(k=..., width=..., connecting=True,
    margin=3): closed cells with a
    narrow passage (``narrow_cells``: parallel sheets or facing free edges
    less than ``width`` base sub-cells apart, default 2, i.e. passages the
    base resolution opens at most with two sub-cells) and their closed
    neighbours get k_narrow sub-cells per edge (default 2 k) instead of k.
    With ``connecting`` (default) only candidates holding a passage that
    connects different fluid regions stay refined (``_connecting_cells``)."""
    h = ot.h
    d0 = ot.dims(0)
    S = ot.cut
    ncut = S.size
    C = _coords(S, d0)
    # neighbour tables
    nbcut = -np.ones((ncut, 6), np.int64)
    nbnode = -np.ones((ncut, 6), np.int64)
    for ax in range(3):
        for j, sg in enumerate((-1, 1)):
            Q = C.copy()
            Q[:, ax] += sg
            ok = (Q[:, ax] >= 0) & (Q[:, ax] < d0[ax])
            q = _key(Q[ok], d0)
            f, p = _isin_sorted(q, S)
            col = 2 * ax + j
            rows = np.flatnonzero(ok)
            nbcut[rows[f], col] = p[f]
            f2, p2 = _isin_sorted(q[~f], ot.leaves[0])
            nbnode[rows[~f][f2], col] = ids[0] + p2[f2]
            if not f2.all():
                # a closed cell next to a coarser leaf would break the
                # sub-cell links; margin >= 1 prevents it
                miss = rows[~f][~f2]
                if miss.size:
                    raise RuntimeError("closed cell without a level-0 neighbour")
    # only closed cells next to a fluid leaf are refined (as on the uniform
    # grid: deeper solid stays solid). "Next to" includes the edge and
    # corner neighbours (6.3): where a sheet runs obliquely, the closed
    # cells form a wall two cells thick, and a closed cell touching the
    # fluid only along an edge or a corner holds fluid too (in the corners
    # of a tilted box about 5 % of its volume at 12 cells across).
    near = (nbnode >= 0).any(1)
    if REFINE_EDGE_NEIGHBOURS:
        for da in (-1, 0, 1):
            for db in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if abs(da) + abs(db) + abs(dc) < 2:
                        continue
                    todo = np.flatnonzero(~near)
                    if todo.size == 0:
                        break
                    Q = C[todo] + np.array([da, db, dc])
                    ok = np.all((Q >= 0) & (Q < d0), axis=1)
                    f, _ = _isin_sorted(_key(Q[ok], d0), ot.leaves[0])
                    near[todo[ok][f]] = True
    sel = np.flatnonzero(near)
    remap = -np.ones(ncut + 1, np.int64)
    remap[sel] = np.arange(sel.size)
    nbcut = remap[nbcut[sel]]                 # -1 stays -1 (index ncut -> -1)
    nbnode = nbnode[sel]
    S = S[sel]
    C = C[sel]
    ncut = S.size
    # base resolution everywhere: the surface resampled at h/k
    kc = np.full(ncut, k, np.int64)
    off = np.arange(ncut + 1, dtype=np.int64) * k ** 3
    fine_solid = np.zeros(int(off[-1]), np.bool_)
    tri_all = np.ascontiguousarray(ot.triangles, np.float64)
    _sample_bin_kernel(tri_all, 0.45 * (h / k), np.asarray(ot.origin, np.float64), float(h),
                       np.asarray(d0, np.int64), S, kc, off, np.ones(ncut, np.bool_),
                       fine_solid)
    stats = {}
    if narrow:
        nw = dict(narrow) if isinstance(narrow, dict) else {}
        kn = int(nw.get("k", 2 * k))
        # passages narrower than `width` base sub-cells
        width = float(nw.get("width", 2.0)) * h / k
        # ... but wide enough to open once refined (1.5 fine sub-cells)
        nar = narrow_cells(ot, S, width, eps=1.5 * h / kn, s=h / k,
                           dilate=bool(nw.get("dilate", False)))
        base_fs = fine_solid
        if kn > k and nar.any():
            kc[nar] = kn
            off = np.r_[0, np.cumsum(kc ** 3)].astype(np.int64)
            fs = np.zeros(int(off[-1]), np.bool_)
            base = np.flatnonzero(~nar)
            # copy the base cells' sub-cells
            src = (base[:, None] * k ** 3 + np.arange(k ** 3)).ravel()
            dst = (off[base][:, None] + np.arange(k ** 3)).ravel()
            fs[dst] = fine_solid[src]
            fine_solid = fs
            tri = _triangles_near(ot, S, nar)
            _sample_bin_kernel_uv(np.ascontiguousarray(tri, np.float64), 0.45 * (h / kn),
                               np.asarray(ot.origin, np.float64), float(h),
                               np.asarray(d0, np.int64), S, kc, off, nar, fine_solid)
        stats = dict(narrow_candidates=int(nar.sum()), narrow_k=kn)
        if kn > k and nar.any() and nw.get("connecting", True) and leaf_links is not None:
            # keep the refinement only where a passage connects fluid that
            # is not connected nearby otherwise (``_connecting_cells``)
            keep = _connecting_cells(ot, k, kc, off, nbcut, nbnode, fine_solid, C, nar,
                                     leaf_links, leaf_X, float(nw.get("margin", 1.0)))
            drop = nar & ~keep
            if drop.any():
                kc2 = np.where(drop, k, kc)
                off2 = np.r_[0, np.cumsum(kc2 ** 3)].astype(np.int64)
                fs2 = np.zeros(int(off2[-1]), np.bool_)
                # refined cells keep their fine sub-cells, dropped ones get the
                # base ones back
                for sel_, src_fs, src_off, kk_ in ((np.flatnonzero(~drop), fine_solid, off, None),
                                                   (np.flatnonzero(drop), base_fs,
                                                    np.arange(ncut + 1) * k ** 3, k)):
                    if sel_.size == 0:
                        continue
                    n3 = kc2[sel_] ** 3
                    rep = np.repeat(sel_, n3)
                    loc = np.arange(rep.size) - np.repeat(np.r_[0, np.cumsum(n3)[:-1]], n3)
                    fs2[off2[rep] + loc] = src_fs[src_off[rep] + loc]
                kc, off, fine_solid = kc2, off2, fs2
                nar = nar & keep
        stats["narrow_cells"] = int(nar.sum())
    nleaf = int(sum(K.size for K in ot.leaves))
    fid = np.where(fine_solid, -1, 0).astype(np.int64)
    fl_sub = ~fine_solid
    nfine = int(fl_sub.sum())
    fid[fl_sub] = nleaf + np.arange(nfine)
    m = _fine_links_var(kc, off, nbcut, nbnode, fid, np.zeros(1, np.int64),
                        np.zeros(1, np.int64), True)
    ea = np.empty(m, np.int64)
    eb = np.empty(m, np.int64)
    _fine_links_var(kc, off, nbcut, nbnode, fid, ea, eb, False)
    flat = np.flatnonzero(fl_sub)
    ci = np.searchsorted(off, flat, side="right") - 1
    sub = flat - off[ci]
    kk = kc[ci]
    abd = np.stack([sub // (kk * kk), (sub // kk) % kk, sub % kk], 1)
    corner = ot.origin + C[ci] * h
    if (kk == k).all():
        X = corner + (abd + 0.5) * (h / k)
        size = np.full(nfine, h / k)
    else:
        size = h / kk
        X = corner + (abd + 0.5) * size[:, None]
    credit = np.zeros(nleaf + nfine)
    credit[nleaf:] = (k / kk) ** 3
    if share == "sight":
        # leaf centres and edges (leaves of level l have edge h 2^l)
        nlev = np.diff(np.r_[ids, nleaf])
        lsize = np.repeat(h * 2.0 ** np.arange(nlev.size), nlev)
        LX = leaf_X if leaf_X is not None else np.vstack(
            [ot.leaf_centres(l) for l in range(ot.L + 1)])
        stats["metal_share"] = _share_by_sight(
            kc, off, k, nbcut, nbnode, fine_solid, fid, credit, C, S, ot,
            np.vstack([LX, X]), np.r_[lsize, np.broadcast_to(size, (nfine,))])
    else:
        _split_surface_var(kc, off, k, nbcut, nbnode, fine_solid, fid, credit, 0,
                           np.zeros(1, np.int64), np.zeros(1, np.int64), np.zeros(1, np.int64))
    bnd = np.any((C[ci] == 0) | (C[ci] == d0 - 1), axis=1)
    stats.update(closed_cells=int(ncut), fine_nodes=nfine, fine_links=int(m))
    return dict(nfine=nfine, host_key=S[ci], X=X, size=size, credit=credit, ea=ea, eb=eb,
                boundary=bnd, stats=stats)
