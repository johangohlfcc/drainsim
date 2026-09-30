"""Split the fluid space into compartments connected by throats (openings).

Inside a compartment liquid and air move instantly (quasi-static fill-spill).
Between compartments, exchange goes through *throats* and is rate limited
(orifice law, see ``physics.py``). This is what turns the equilibrium method
into a transient one.

Segmentation (orientation independent, done once):

1. ``D`` = distance from each fluid cell centre to the nearest solid surface.
2. Markers = local maxima (plateaus) of ``D`` - the "pore centres".
3. Watershed of ``-D`` from the markers: region boundaries land on the
   narrowest cross-sections (the constrictions), as in pore-network
   extraction for porous media (SNOW, Gostick 2017).
4. Over-segmentation is removed by merging neighbouring regions whose shared
   boundary is *not* a real constriction:
   ``r_throat >= beta * min(r_peak_a, r_peak_b)``, or ``r_throat >= r_free``
   (wide enough to be treated as instantaneous).
5. Every region touching the domain boundary is merged into compartment 0,
   the *exterior* (bath + atmosphere side).

Narrow, locally unique openings between regions that end up in the *same*
compartment (because they are also connected by some wide path) are kept as
*internal throats* (``a == b``); the simulation removes their faces from the
instantaneous (fill-spill) connectivity.

Throats are the connected groups of faces between two compartments. For each
throat we store the face list, the effective area (vector sum of the staircase
face normals = true area of a planar cross-section), and the inscribed
diameter ``d = 2 * max(min(D_a, D_b))`` used for capillary criteria.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

from .grid import Grid


@dataclass
class Throat:
    a: int                      # compartment on side a (a < b)
    b: int
    cells_a: np.ndarray         # flat index of the a-side cell of each face
    cells_b: np.ndarray
    normals: np.ndarray         # (nfaces, ndim) unit normal a -> b per face
    area: float                 # effective cross-section area (m^2; m in 2D)
    diameter: float             # inscribed diameter (smallest gap width)
    centroid: np.ndarray        # object-frame position of throat centre
    axis: np.ndarray | None = None   # explicit holes: unit hole axis. The wet
                                     # area and sill then come from the true
                                     # circle (slot in 2D), not the faces
    open_at: float = -np.inf         # plugged (no exchange) before this time
    weights: np.ndarray | None = None  # face area per link (sub-cell graph);
                                       # None = all faces equal (one cell face)

    @property
    def nfaces(self) -> int:
        return len(self.cells_a)


@dataclass
class Compartments:
    label: np.ndarray           # (ncells,) int, -1 for solid; 0 = exterior
    n: int
    throats: list = field(default_factory=list)
    volume: np.ndarray | None = None      # fluid volume per compartment
    dist: np.ndarray | None = None        # distance field D (ncells,)
    touches_boundary: np.ndarray | None = None

    def throats_of(self, k: int):
        return [t for t in self.throats if t.a == k or t.b == k]


def _uf_find(p, x):
    while p[x] != x:
        p[x] = p[p[x]]
        x = p[x]
    return x


def _face_pairs(grid: Grid, label: np.ndarray):
    """All faces between two different non-negative labels."""
    lab = label.reshape(grid.shape)
    idx = np.arange(grid.ncells).reshape(grid.shape)
    A, B, AX, SG = [], [], [], []
    for d in range(grid.ndim):
        s0 = [slice(None)] * grid.ndim
        s1 = [slice(None)] * grid.ndim
        s0[d] = slice(None, -1)
        s1[d] = slice(1, None)
        la = lab[tuple(s0)].ravel()
        lb = lab[tuple(s1)].ravel()
        m = (la >= 0) & (lb >= 0) & (la != lb)
        A.append(idx[tuple(s0)].ravel()[m])
        B.append(idx[tuple(s1)].ravel()[m])
        AX.append(np.full(m.sum(), d))
    return np.concatenate(A), np.concatenate(B), np.concatenate(AX)


def distance_field(grid: Grid) -> np.ndarray:
    fluid = grid.fluid
    if not grid.solid.any():
        return np.full(grid.ncells, np.inf)
    D = ndimage.distance_transform_edt(fluid) * grid.dx - 0.5 * grid.dx
    return np.maximum(D, 0.5 * grid.dx).ravel()


def segment(grid: Grid, beta: float = 0.6, d_free: float | None = 0.03,
            min_cells: int | None = None, split: bool = True,
            internal_holes: bool = True, holes=None) -> Compartments:
    """Segment fluid space into compartments and throats.

    beta     : a boundary is a throat only if its inscribed radius is smaller
               than ``beta`` times the smaller of the two pore radii.
    d_free   : openings at least this wide [m] are always treated as open
               (instantaneous exchange). Default 30 mm. None = no limit.
    min_cells: regions smaller than this are merged into their widest
               neighbour (default 2**ndim * 4).
    split    : False gives connected components only (no throats), i.e. the
               pure equilibrium model in which every opening is instantaneous.
    holes    : explicitly listed holes (dicts from ``carve_holes``). They
               always become throats with their true area and diameter;
               automatically found throats overlapping them are dropped.
    internal_holes : also keep narrow holes between parts of the *same*
               compartment as throats (``a == b``), e.g. drain holes in the
               bottom of a door whose interior is open to the outside
               through a large access hole higher up.
    """
    from skimage.segmentation import watershed

    fluid = grid.fluid
    D = distance_field(grid)
    Dg = D.reshape(grid.shape)
    bnd = grid.boundary_mask()
    if min_cells is None:
        min_cells = 4 * 2 ** grid.ndim

    auto_holes = []
    if not split:
        lab, n = ndimage.label(fluid)
        lab = lab.ravel() - 1
        lab[~fluid.ravel()] = -1
    else:
        fp = np.ones((3,) * grid.ndim, bool)
        Dm = np.where(fluid, Dg, 0.0)
        peaks = (Dm >= ndimage.maximum_filter(Dm, footprint=fp)) & fluid
        markers, nm = ndimage.label(peaks, structure=fp)
        lab = watershed(-Dm, markers, mask=fluid, connectivity=1).ravel() - 1
        lab[~fluid.ravel()] = -1
        n = nm
        auto_holes = hole_candidates(grid, lab, D, beta, d_free) \
            if internal_holes else []
        lab = _merge_regions(grid, lab, n, D, bnd.ravel(), beta, d_free,
                             min_cells)
        n = lab.max() + 1

    # ---- exterior = everything touching the domain boundary -> label 0
    lab = _make_exterior_zero(lab, bnd.ravel())
    n = int(lab.max()) + 1
    comp = Compartments(label=lab, n=n, dist=D)
    comp.volume = np.bincount(lab[lab >= 0], minlength=n) * grid.cell_volume
    tb = np.zeros(n, bool)
    tb[np.unique(lab[bnd.ravel() & (lab >= 0)])] = True
    comp.touches_boundary = tb
    comp.throats = extract_throats(grid, lab, D) if split else []
    if split and auto_holes:
        X = grid.centers()
        for ca, cb, ax, sg in auto_holes:
            k = lab[ca[0]]
            if np.all(lab[ca] == k) and np.all(lab[cb] == k):
                comp.throats.append(_make_throat(grid, X, D, k, k, ca, cb, ax, sg))
    if split and holes:
        forced = hole_throats(grid, lab, holes)
        used = np.zeros(grid.ncells, bool)
        for t in forced:
            used[t.cells_a] = used[t.cells_b] = True
        keep = []
        for t in comp.throats:
            near = any(np.linalg.norm(t.centroid - f.centroid) < f.diameter
                       for f in forced)
            if not near and not (used[t.cells_a].any() or used[t.cells_b].any()):
                keep.append(t)
        comp.throats = keep + forced
    return comp


def _make_exterior_zero(lab, bnd):
    ext = np.unique(lab[bnd & (lab >= 0)])
    n = lab.max() + 1
    new = -np.ones(n, np.int64)
    new[ext] = 0
    k = 1
    for r in range(n):
        if new[r] < 0 and np.any(lab == r):
            new[r] = k
            k += 1
    out = np.where(lab >= 0, new[np.maximum(lab, 0)], -1)
    return out


def _merge_regions(grid, lab, n, D, bnd, beta, d_free, min_cells):
    """Greedy merging of over-segmented watershed regions."""
    r_free = np.inf if d_free is None else 0.5 * d_free
    lab, _ = merge_regions(lab, n, D, None, lambda l: _face_pairs(grid, l)[:2],
                           beta, r_free, min_cells)
    return lab


def merge_regions(lab, n, D, weight, pairs, beta, r_free, min_size,
                  sort_kind="quicksort"):
    """Greedy merging of over-segmented regions, on the uniform grid
    (``_merge_regions``) and on the node graph (``gseg._merge``).

    ``pairs(lab)`` gives the two ends (a, b) of every link between two
    different non-negative labels; ``weight``: size of each cell (None: one
    per cell). Two regions merge if the opening between them is wide
    relative to the smaller peak of the distance field (``beta``), wider than
    ``r_free``, or if one of them is smaller than ``min_size``, the most open
    first. ``sort_kind`` orders ties (the graph uses a stable sort).
    Returns (labels, number of regions)."""
    for _ in range(50):
        valid = lab >= 0
        peak = np.zeros(n)
        np.maximum.at(peak, lab[valid], D[valid])
        size = np.bincount(lab[valid], weights=None if weight is None else weight[valid],
                           minlength=n)
        a, b = pairs(lab)
        if a.size == 0:
            break
        la, lb = lab[a], lab[b]
        lo = np.minimum(la, lb).astype(np.int64)
        hi = np.maximum(la, lb).astype(np.int64)
        rf = np.minimum(D[a], D[b])
        ukey, inv = np.unique(lo * n + hi, return_inverse=True)
        rt = np.zeros(ukey.size)
        np.maximum.at(rt, inv, rf)
        pa = ukey // n
        pb = ukey % n
        # score: how "open" the connection is
        score = rt / np.maximum(np.minimum(peak[pa], peak[pb]), 1e-30)
        small = (size[pa] < min_size) | (size[pb] < min_size)
        order = np.argsort(-(score + 10.0 * small), kind=sort_kind)
        par = np.arange(n)
        gpeak = peak.copy()
        gsize = size.copy()
        merged = 0
        for e in order:
            x = _uf_find(par, pa[e])
            y = _uf_find(par, pb[e])
            if x == y:
                continue
            is_small = gsize[x] < min_size or gsize[y] < min_size
            if rt[e] >= beta * min(gpeak[x], gpeak[y]) or rt[e] >= r_free or is_small:
                par[y] = x
                gpeak[x] = max(gpeak[x], gpeak[y])
                gsize[x] += gsize[y]
                merged += 1
        if merged == 0:
            break
        roots = np.array([_uf_find(par, i) for i in range(n)])
        _, newid = np.unique(roots, return_inverse=True)
        lab = np.where(lab >= 0, newid[np.maximum(lab, 0)], -1)
        n = int(newid.max()) + 1
    return lab, n


def _face_groups(grid: Grid, lab: np.ndarray):
    """Faces between different labels, oriented low label -> high label and
    split into connected groups (separate openings between the same pair).

    Yields (pa, pb, ca, cb, ax, sign) per group, ca on the pa side."""
    a, b, ax = _face_pairs(grid, lab)
    if a.size == 0:
        return
    la, lb = lab[a], lab[b]
    swap = la > lb
    sign = np.where(swap, -1.0, 1.0)
    ca = np.where(swap, b, a)
    cb = np.where(swap, a, b)
    lo = np.minimum(la, lb).astype(np.int64)
    hi = np.maximum(la, lb).astype(np.int64)
    key = lo * (int(hi.max()) + 1) + hi
    order = np.argsort(key, kind="stable")
    ks = key[order]
    brk = np.r_[0, np.flatnonzero(np.diff(ks)) + 1, len(ks)]
    fp = np.ones((3,) * grid.ndim, bool)
    for s0, s1 in zip(brk[:-1], brk[1:]):
        sel = order[s0:s1]
        pa, pb = int(lo[sel[0]]), int(hi[sel[0]])
        cells = np.unique(ca[sel])
        mi = np.array(np.unravel_index(cells, grid.shape)).T
        mn = mi.min(0)
        mx = mi.max(0) + 1
        sub = np.zeros(tuple(mx - mn), bool)
        sub[tuple((mi - mn).T)] = True
        gl, ng = ndimage.label(sub, structure=fp)
        fmi = np.array(np.unravel_index(ca[sel], grid.shape)).T - mn
        glab = gl[tuple(fmi.T)]
        for g in range(1, ng + 1):
            f = sel[glab == g]
            yield pa, pb, ca[f], cb[f], ax[f], sign[f]


def _make_throat(grid, X, D, a, b, ca, cb, ax, sign):
    nrm = np.zeros((len(ca), grid.ndim))
    nrm[np.arange(len(ca)), ax] = sign
    vsum = np.linalg.norm(nrm.sum(0))
    area = max(vsum, len(ca) / np.sqrt(grid.ndim)) * grid.face_area
    rt = np.max(np.minimum(D[ca], D[cb]))
    cen = 0.5 * (X[ca] + X[cb]).mean(0)
    return Throat(a=int(a), b=int(b), cells_a=ca, cells_b=cb, normals=nrm,
                  area=float(area), diameter=float(2 * rt + 0.5 * grid.dx),
                  centroid=cen)


def extract_throats(grid: Grid, lab: np.ndarray, D: np.ndarray) -> list:
    X = grid.centers()
    return [_make_throat(grid, X, D, pa, pb, ca, cb, ax, sg)
            for pa, pb, ca, cb, ax, sg in _face_groups(grid, lab)]


def _locally_unique(grid, lab0, ca, cb, pa, rg):
    """True if, near the opening, the two sides are connected only through it
    (a real hole), not also around it (a piece of a wider boundary)."""
    cells = np.r_[ca, cb]
    mi = np.array(np.unravel_index(cells, grid.shape)).T
    m = max(3, int(np.ceil(2 * rg / grid.dx)) + 2)
    lo = np.maximum(mi.min(0) - m, 0)
    hi = np.minimum(mi.max(0) + m + 1, grid.shape)
    box = tuple(slice(l, h) for l, h in zip(lo, hi))
    sub = grid.fluid[box].copy()
    ia = np.array(np.unravel_index(ca, grid.shape)).T - lo
    sub[tuple(ia.T)] = False
    lbl, _ = ndimage.label(sub)
    ib = np.array(np.unravel_index(cb, grid.shape)).T - lo
    lb = set(np.unique(lbl[tuple(ib.T)])) - {0}
    la = set(np.unique(lbl[(lab0.reshape(grid.shape)[box] == pa) & sub])) - {0}
    return not (la & lb)


def hole_candidates(grid, lab0, D, beta, d_free):
    """Narrow, locally unique openings between (pre-merge) watershed regions.

    After merging, two regions can end up in one compartment because they
    are *also* connected by a wide path (e.g. the bottom of a door is open
    to the outside through a large access hole, and drains through small
    holes). Such holes are still rate limiting and are kept as throats
    inside the compartment."""
    r_free = np.inf if d_free is None else 0.5 * d_free
    valid = lab0 >= 0
    n = int(lab0.max()) + 1
    peak = np.zeros(n)
    np.maximum.at(peak, lab0[valid], D[valid])
    out = []
    for pa, pb, ca, cb, ax, sg in _face_groups(grid, lab0):
        rg = np.max(np.minimum(D[ca], D[cb]))
        if rg >= beta * min(peak[pa], peak[pb]) or rg >= r_free:
            continue
        if not _locally_unique(grid, lab0, ca, cb, pa, rg):
            continue
        out.append((ca, cb, ax, sg))
    return out


# ------------------------------------------------------------ explicit holes

def _local_box(grid, c, radius):
    """Index box (slices, lo) around point c and flat indices / centres of
    its cells. Keeps memory local on large grids."""
    ci = np.floor((np.asarray(c) - grid.origin) / grid.dx).astype(int)
    m = int(np.ceil(radius / grid.dx)) + 1
    lo = np.maximum(ci - m, 0)
    hi = np.minimum(ci + m + 1, grid.shape)
    box = tuple(slice(l, h) for l, h in zip(lo, hi))
    axes = [grid.origin[d] + (np.arange(lo[d], hi[d]) + 0.5) * grid.dx
            for d in range(grid.ndim)]
    Xb = np.stack(np.meshgrid(*axes, indexing="ij"), -1)
    idx = np.arange(grid.ncells).reshape(grid.shape)[box]
    return box, Xb, idx


def _hole_axis(grid, center, d):
    """Axis of a hole = normal of the sheet at its rim (PCA of the solid
    voxels in an annulus around the centre: direction of least spread).
    Give the axis explicitly (e.g. fitted to the mesh rim) when you can."""
    box, Xb, _ = _local_box(grid, center, 0.5 * d + 3 * grid.dx)
    r = np.linalg.norm(Xb - center, axis=-1)
    near = grid.solid[box] & (r > 0.5 * d) & (r < 0.5 * d + 2.5 * grid.dx)
    return sheet_normal(Xb[near], center)


def sheet_normal(P, center):
    """Normal of the sheet through the points ``P`` around a hole: the
    direction of least spread (shared by the uniform grid and the octree)."""
    if len(P) < 3:
        raise ValueError(f"no sheet found around hole at {center}")
    P = P - P.mean(0)
    w, V = np.linalg.eigh(P.T @ P)
    return V[:, 0]


def carve_holes(grid: Grid, holes):
    """Make sure every listed hole is open in the voxel geometry.

    ``holes``: iterable of (center, diameter) or dicts with keys
    ``center``, ``diameter`` and optional ``axis``. Cells of the sheet within
    the hole radius (minus half a cell) are made fluid. Holes outside the
    grid are skipped. Returns the list of holes (dicts, axis filled in).
    """
    out = []
    lo = grid.origin
    hi = grid.origin + np.array(grid.shape) * grid.dx
    for hole in holes:
        if not isinstance(hole, dict):
            hole = dict(center=hole[0], diameter=hole[1])
        c = np.asarray(hole["center"], float)
        d = float(hole["diameter"])
        if np.any(c - 0.5 * d < lo) or np.any(c + 0.5 * d > hi):
            continue                           # outside the (cropped) grid
        n = np.asarray(hole.get("axis") if hole.get("axis") is not None
                       else _hole_axis(grid, c, d), float)
        n = n / np.linalg.norm(n)
        box, Xb, _ = _local_box(grid, c, 0.5 * d + 2 * grid.dx)
        rel = Xb - c
        sv = rel @ n
        rad = np.linalg.norm(rel - sv[..., None] * n, axis=-1)
        carve = (np.abs(sv) <= grid.dx + 1e-9) & \
            (rad <= max(0.5 * d - 0.5 * grid.dx, 0.75 * grid.dx))
        sub = grid.solid[box]
        sub[carve] = False
        grid.solid[box] = sub
        grid._nbr = None
        out.append(dict(center=c, diameter=d, axis=n,
                        open_at=float(hole.get("open_at", -np.inf))))
    return out


def _cut_separates(grid, c, n, d):
    """Within a box of about 2 d around the hole, the cells just in front of
    and just behind the hole must not be connected once the cut is removed."""
    box, Xb, _ = _local_box(grid, c, d + 2 * grid.dx)
    sub = grid.fluid[box]
    sv = (Xb - c) @ n
    rel = Xb - c
    rad = np.linalg.norm(rel - sv[..., None] * n, axis=-1)
    R = 0.5 * d + 0.5 * grid.dx
    fence = (rad <= R) & (np.abs(sv) < 0.5 * grid.dx + 1e-12)
    lbl, _ = ndimage.label(sub & ~fence)
    # seeds just in front of / behind the hole (not further: a thin cavity
    # behind a hole may end within a few cells, beyond it is the outside)
    near = (rad <= 0.5 * d) & (np.abs(sv) > 0.5 * grid.dx) & \
        (np.abs(sv) <= 2.0 * grid.dx)
    la = set(np.unique(lbl[sub & ~fence & near & (sv < 0)])) - {0}
    lb = set(np.unique(lbl[sub & ~fence & near & (sv > 0)])) - {0}
    return not (la & lb)


def hole_throats(grid, lab, holes):
    """Throats for explicitly listed holes (after ``carve_holes``).

    The cut is the set of fluid-fluid faces crossing the hole's mid-plane
    within the hole radius (+ half a cell). Area and diameter are the given
    (true) ones, not the voxel estimates, so the orifice law and the
    capillary hold-up do not depend on the grid resolution.
    """
    throats = []
    for h in holes:
        c, d, n = h["center"], h["diameter"], h["axis"]
        box, Xb, idx = _local_box(grid, c, 0.5 * d + 2 * grid.dx)
        fl = grid.fluid[box]
        A, B, AX = [], [], []
        for dd in range(grid.ndim):
            s0 = [slice(None)] * grid.ndim
            s1 = [slice(None)] * grid.ndim
            s0[dd] = slice(None, -1)
            s1[dd] = slice(1, None)
            m = fl[tuple(s0)] & fl[tuple(s1)]
            A.append(idx[tuple(s0)][m])
            B.append(idx[tuple(s1)][m])
            AX.append(np.full(m.sum(), dd))
        A, B, AX = np.concatenate(A), np.concatenate(B), np.concatenate(AX)
        XA = grid.origin + (np.array(np.unravel_index(A, grid.shape)).T + 0.5) * grid.dx
        XB = grid.origin + (np.array(np.unravel_index(B, grid.shape)).T + 0.5) * grid.dx
        sa = (XA - c) @ n
        sb = (XB - c) @ n
        cross = (sa < 0) != (sb < 0)
        mid = 0.5 * (XA + XB) - c
        rad = np.linalg.norm(mid - (mid @ n)[:, None] * n, axis=1)
        f = np.flatnonzero(cross & (rad <= 0.5 * d + 0.5 * grid.dx))
        if f.size == 0:
            raise ValueError(f"hole at {c} is closed in the voxel grid")
        if not _cut_separates(grid, c, n, d):
            import warnings
            warnings.warn(f"hole at {np.round(c, 4)}: the cut does not close "
                          "the opening locally (sheet not resolved around it?)")
        neg = sa[f] < 0
        ca = np.where(neg, A[f], B[f])            # side a = negative side
        cb = np.where(neg, B[f], A[f])
        sign = np.where(neg, 1.0, -1.0)
        la, lb = int(lab[ca[0]]), int(lab[cb[0]])
        swap = la > lb
        t = Throat(a=min(la, lb), b=max(la, lb),
                   cells_a=cb if swap else ca, cells_b=ca if swap else cb,
                   normals=np.zeros((f.size, grid.ndim)),
                   area=float(np.pi * d * d / 4.0 if grid.ndim == 3 else d),
                   diameter=d,
                   centroid=np.asarray(c, float), axis=np.asarray(n, float),
                   open_at=float(h.get("open_at", -np.inf)))
        t.normals[np.arange(f.size), AX[f]] = -sign if swap else sign
        throats.append(t)
    return throats


# ------------------------------------------------------ sub-cell throats
# With the sub-cell graph (volfrac.cut_cell_graph) an opening can be open
# through sub-cells only, or be wider in sub-cells than in whole cells. The
# segmentation above runs on whole cells, so the throats are completed here on
# the node graph: every link that crosses between two compartments belongs to
# a throat, and a throat's width, area and sill include its sub-cell links.

def face_raster_diameter(pos, nrm, w, nvec, dx, k, ndim):
    """Inscribed width of an opening from its faces.

    pos: position of each face (the fine node's centre for links with a
    sub-cell end, the face midpoint otherwise); nrm: unit face normals; w:
    face areas (a whole face covers k x k sub-faces); nvec: the opening's
    net normal. The faces are projected on the plane normal to ``nvec`` and
    drawn on a raster of sub-cell size; the width is that of the largest
    inscribed disc (segment in 2D)."""
    s = dx / k
    nvec, basis = opening_frame(nvec, ndim)
    # only the faces across the opening (the sides of a jagged boundary
    # would widen its outline)
    cap = np.abs(nrm @ nvec) > 0.5
    if cap.any():
        pos, nrm, w = pos[cap], nrm[cap], w[cap]
    whole = w > 1.5 * s ** (ndim - 1)
    pts = [pos[~whole]]
    if whole.any():
        # sub-face centres of whole faces, in the face's own plane
        g = (np.arange(k) + 0.5) / k - 0.5
        for ax in range(ndim):
            m = whole & (np.abs(nrm[:, ax]) > 0.5)
            if not m.any():
                continue
            od = [d for d in range(ndim) if d != ax]
            offs = np.array(np.meshgrid(*[g] * len(od), indexing="ij")).reshape(len(od), -1).T
            O = np.zeros((offs.shape[0], ndim))
            O[:, od] = offs * dx
            pts.append((pos[m][:, None, :] + O[None]).reshape(-1, ndim))
    return raster_disc_width(np.concatenate(pts), basis, s)


def opening_frame(nvec, ndim):
    """Unit normal of an opening and the basis of its plane (one vector in
    2D, two in 3D); shared by ``face_raster_diameter`` and
    ``gseg.raster_width``."""
    nvec = nvec / np.linalg.norm(nvec)
    if np.max(np.abs(nvec)) > 0.9:
        # an opening in a grid-aligned sheet: project exactly along the axis,
        # or the depth of a jagged boundary smears into the width
        nvec = np.sign(nvec) * (np.abs(nvec) == np.max(np.abs(nvec)))
    if ndim == 3:
        a = np.array([1.0, 0, 0]) if abs(nvec[0]) < 0.9 else np.array([0, 1.0, 0])
        u = np.cross(nvec, a)
        u /= np.linalg.norm(u)
        basis = np.stack([u, np.cross(nvec, u)])
    else:
        basis = np.array([[-nvec[1], nvec[0]]])
    return nvec, basis


def raster_disc_width(P, basis, s):
    """Width of the largest disc (segment in 2D) inscribed in the points
    ``P`` projected on ``basis`` and drawn on a raster of spacing ``s``."""
    q = np.round((P @ basis.T) / s).astype(np.int64)
    q = np.unique(q, axis=0)
    lo = q.min(0) - 2
    shape = q.max(0) - lo + 3
    img = np.zeros(tuple(shape), bool)
    img[tuple((q - lo).T)] = True
    img = ndimage.binary_closing(img, iterations=1)
    img = ndimage.binary_fill_holes(img)
    e = ndimage.distance_transform_edt(img)
    # free sub-cells miss up to one partly open sub-cell at each edge
    return float(max(2.0 * e.max() - 0.5, 1.0) * s)


def node_throats(grid, comp, lab, X, fine, nbr, k, explicit=(), plugs=(),
                 split=True, node_size=None):
    """Complete the throats with the sub-cell links of the node graph.

    lab : compartment label per node (fine nodes included), < 0 = none.
    nbr : CSR (ptr, idx) of the node graph (plugged links already cut).

    * split=True: every link between two compartments is added to the
      whole-cell throat it touches, or else grouped into a new throat (an
      opening that only the sub-cells resolve). Sub-cell links that cross the
      plane of an internal throat (a == b) next to it are added to it, so
      nothing bypasses its rate limit. Widths, areas and centroids are
      recomputed from all the links.
    * split=False: compartments joined by sub-cell links are merged.

    node_size: edge length per node when the sub-cells differ in size
    (octree with narrow refinement): face areas are then min(size)^2 per
    link, and widths are drawn on a raster of the smallest sub-cell.

    Returns the (possibly relabelled) node labels; comp is updated in place.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree

    ptr, idx = nbr
    N = len(ptr) - 1
    dx = grid.dx
    s = dx / k
    fa_whole = grid.face_area
    fa_fine = grid.face_area / k ** (grid.ndim - 1)
    var = node_size is not None and np.any(np.asarray(node_size)[fine] != dx / k)
    if var:
        node_size = np.asarray(node_size, float)
        s = float(node_size[fine].min())

    def link_area(a, b):
        """Face area of links (a, b) with a sub-cell end."""
        if not var:
            return np.full(len(a), fa_fine)
        return np.minimum(node_size[a], node_size[b]) ** (grid.ndim - 1)
    src = np.repeat(np.arange(N), np.diff(ptr))
    dst = idx
    ok = (dst >= 0) & (src < dst)
    u, v = src[ok], dst[ok]
    anyfine = fine[u] | fine[v]
    u, v = u[anyfine], v[anyfine]
    lu, lv = lab[u], lab[v]
    good = (lu >= 0) & (lv >= 0)
    u, v, lu, lv = u[good], v[good], lu[good], lv[good]
    # face position: the fine end (its sub-face), else the midpoint
    pos = np.where(fine[u][:, None], X[u], X[v])
    d = X[v] - X[u]
    ax = np.argmax(np.abs(d), axis=1)
    sg = np.sign(d[np.arange(len(d)), ax])
    wlink = link_area(u, v)
    # openings handled elsewhere: explicit holes and plugs
    skip = np.zeros(len(u), bool)
    for h in list(explicit) + list(plugs):
        r = 0.5 * float(h["diameter"]) + 2.0 * dx
        skip |= np.linalg.norm(pos - np.asarray(h["center"]), axis=1) <= r

    cross = (lu != lv) & ~skip
    if not split:
        if cross.any():
            n = comp.n
            G = coo_matrix((np.ones(cross.sum()), (lu[cross], lv[cross])), shape=(n, n))
            nc, cc = connected_components(G, directed=False)
            # compartment 0 (exterior) keeps label 0
            order = np.r_[cc[0], np.setdiff1d(np.unique(cc), [cc[0]])]
            new = np.empty(nc, np.int64)
            new[order] = np.arange(nc)
            lab = np.where(lab >= 0, new[cc[np.maximum(lab, 0)]], -1)
            comp.n = int(nc)
        return lab

    th = comp.throats
    add = [[] for _ in th]                 # (u, v) node pairs, a side first
    # 1. links between two compartments: to the whole-cell throat they touch
    cu, cv, cl_u, cl_v = u[cross], v[cross], lu[cross], lv[cross]
    cpos = pos[cross]
    used = np.zeros(len(cu), bool)
    auto = [i for i, t in enumerate(th) if t.axis is None and t.a != t.b]
    if auto and len(cu):
        F = np.concatenate([0.5 * (X[th[i].cells_a] + X[th[i].cells_b]) for i in auto])
        fid = np.concatenate([np.full(th[i].nfaces, i) for i in auto])
        dist, j = cKDTree(F).query(cpos, distance_upper_bound=1.5 * dx)
        for e in np.flatnonzero(np.isfinite(dist)):
            t = th[fid[j[e]]]
            if {cl_u[e], cl_v[e]} == {t.a, t.b}:
                a_first = cl_u[e] == t.a
                add[fid[j[e]]].append((cu[e], cv[e]) if a_first else (cv[e], cu[e]))
                used[e] = True
    # 2. the rest: openings resolved by sub-cells only -> new throats
    rest = np.flatnonzero(~used)
    new_throats = []
    if rest.size:
        pa = np.minimum(cl_u[rest], cl_v[rest])
        pb = np.maximum(cl_u[rest], cl_v[rest])
        for key in np.unique(pa * (comp.n + 1) + pb):
            sel = rest[(pa * (comp.n + 1) + pb) == key]
            P = cpos[sel]
            pairs = cKDTree(P).query_pairs(max(2.5 * s, dx), output_type="ndarray")
            G = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])),
                           shape=(len(sel), len(sel))) if len(pairs) else \
                coo_matrix((len(sel), len(sel)))
            ng, gl = connected_components(G, directed=False)
            a, b = int(key // (comp.n + 1)), int(key % (comp.n + 1))
            for g in range(ng):
                e = sel[gl == g]
                a_first = cl_u[e] == a
                ca = np.where(a_first, cu[e], cv[e])
                cb = np.where(a_first, cv[e], cu[e])
                new_throats.append(Throat(a=a, b=b, cells_a=ca, cells_b=cb,
                                          normals=np.zeros((len(e), grid.ndim)),
                                          area=0.0, diameter=0.0,
                                          centroid=np.zeros(grid.ndim)))
    # 3. internal throats (a == b): sub-cell links across their plane
    same = (lu == lv) & ~skip
    if same.any():
        su, sv_, spos = u[same], v[same], pos[same]
        tree = cKDTree(spos)
        for i, t in enumerate(th):
            if t.axis is not None or t.a != t.b or t.nfaces == 0:
                continue
            nsum = t.normals.sum(0)
            if np.linalg.norm(nsum) < 0.5 * t.nfaces:
                continue
            n = nsum / np.linalg.norm(nsum)
            c = t.centroid
            Fm = 0.5 * (X[t.cells_a] + X[t.cells_b]) - c
            rin = np.max(np.linalg.norm(Fm - (Fm @ n)[:, None] * n, axis=1)) + dx
            cand = np.array(tree.query_ball_point(c, rin + 2 * dx), dtype=np.int64)
            if cand.size == 0:
                continue
            ra = (X[su[cand]] - c) @ n
            rb = (X[sv_[cand]] - c) @ n
            mid = 0.5 * (X[su[cand]] + X[sv_[cand]]) - c
            rad = np.linalg.norm(mid - (mid @ n)[:, None] * n, axis=1)
            hit = (np.sign(ra) != np.sign(rb)) & (rad <= rin) & (np.abs(mid @ n) <= 1.5 * dx)
            sa = np.sign(np.mean((X[t.cells_a] - c) @ n))
            for e in cand[hit]:
                pu, pv = su[e], sv_[e]
                add[i].append((pu, pv) if np.sign((X[pu] - c) @ n) == sa else (pv, pu))
    # 4. finish: all links, weights, area, width, centroid
    def finish(t, extra, fresh):
        if extra:
            e = np.array(extra, dtype=np.int64)
            ca = np.r_[t.cells_a, e[:, 0]]
            cb = np.r_[t.cells_b, e[:, 1]]
        else:
            ca, cb = t.cells_a, t.cells_b
        if not fresh and not extra:
            return t
        nw = 0 if fresh else t.nfaces
        isf = fine[ca] | fine[cb]
        w = np.where(isf, link_area(ca, cb), fa_whole)
        if not fresh:
            w[:nw] = fa_whole if t.weights is None else t.weights
        dd = X[cb] - X[ca]
        axx = np.argmax(np.abs(dd), axis=1)
        nrm = np.zeros((len(ca), grid.ndim))
        nrm[np.arange(len(ca)), axx] = np.sign(dd[np.arange(len(ca)), axx])
        p = np.where(fine[ca][:, None], X[ca], np.where(fine[cb][:, None], X[cb],
                                                        0.5 * (X[ca] + X[cb])))
        # net (projected) area: faces of a jagged boundary through the
        # opening cancel pairwise, so this is the opening's cross-section
        nv = (nrm * w[:, None]).sum(0)
        vs = np.linalg.norm(nv)
        area = vs if vs > 1e-3 * w.sum() else w.sum() / np.sqrt(grid.ndim)
        if vs <= 1e-3 * w.sum():
            nv = np.eye(grid.ndim)[np.argmax(np.bincount(axx, weights=w,
                                                         minlength=grid.ndim))]
        if var:
            from .gseg import raster_width
            dr = raster_width(p, nrm, np.sqrt(w), nv, s)
        else:
            dr = face_raster_diameter(p, nrm, w, nv, dx, k, grid.ndim)
        dia = dr if fresh else max(t.diameter, dr)
        cen = (p * w[:, None]).sum(0) / w.sum()
        return Throat(a=t.a, b=t.b, cells_a=ca, cells_b=cb, normals=nrm,
                      area=float(area), diameter=float(dia), centroid=cen,
                      axis=t.axis, open_at=t.open_at, weights=w)

    comp.throats = [finish(t, add[i], False) for i, t in enumerate(th)] + \
                   [finish(t, [], True) for t in new_throats]
    return lab
