"""Openings in the true geometry: holes and narrow gaps found on the mesh.

The grid sees an opening only if it is a couple of (sub-)cells wide; holes
and seams below that are shut or open only at some grid offsets. Here they
are found on the triangles themselves, independent of any grid, so they can
be listed, reviewed and given to the model as explicit openings
(``Simulation(holes=...)``) with their true size.

Car bodies and doors are exported as thin solids: every panel has two faces
0.5-2 mm apart with side walls, the normals pointing out of the metal
(``octree._plate_sign``). Then

* a **hole** is a short ring-shaped wall through the plate, one plate
  thickness deep: the mesh split into smooth patches at sharp edges gives
  each hole wall as its own small patch, which goes round all the way and
  whose facets lie at one radius from an axis (the normal of the ring's
  plane). Its wall normals point to the axis (into the opening); those of a
  pin or a boss point away from it.
* a **gap** is where the fluid side of a face meets another wall within a
  short distance: a ray from the face along its (outward) normal hits a
  wall facing back. The distance is the local gap width, measured on the
  triangles (Embree ray casting).

What is open in reality (a hole may be closed by a plug, a clip or a bolt,
a seam by sealant or a weld) is for the engineer to decide: these functions
list the candidates.

All lengths in metres, in the mesh's own frame (see ``to_model`` for the
frame of ``car_article.load_car``).
"""
from __future__ import annotations

import time

import numpy as np

HOLE_FIELDS = ("center", "axis", "diameter", "roundness", "gap_deg", "depth", "kind",
               "nfaces")


# ------------------------------------------------------------------ mesh
def read_stl(path, scale=None):
    """Vertices (float64) and faces (int64) of a binary STL, the repeated
    corner vertices merged (exactly equal coordinates), without trimesh's
    per-face overhead (a 34 M triangle car in a few GB). ``scale``: None =
    millimetres to metres if the extent exceeds 100, else as given."""
    with open(path, "rb") as f:
        f.seek(80)
        n = int(np.frombuffer(f.read(4), "<u4")[0])
    dt = np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    raw = np.memmap(path, dtype=dt, mode="r", offset=84, shape=(n,))
    corners = np.ascontiguousarray(raw["v"]).reshape(-1, 3)
    # exact duplicates: the bytes of the three float32 as one 12-byte key
    key = corners.view(np.dtype((np.void, 12))).ravel()
    uniq, inv = np.unique(key, return_inverse=True)
    V = uniq.view(np.float32).reshape(-1, 3).astype(np.float64)
    F = inv.reshape(-1, 3).astype(np.int64)
    if scale is None:
        scale = 1e-3 if (V.max(0) - V.min(0)).max() > 100.0 else 1.0
    return V * scale, F


def face_geometry(V, F):
    """Centres, unit normals (from the winding) and areas of the faces."""
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    cr = np.cross(b - a, c - a)
    ar = np.linalg.norm(cr, axis=1)
    N = cr / np.maximum(ar, 1e-300)[:, None]
    return (a + b + c) / 3.0, N, 0.5 * ar


def face_pairs(F, nv):
    """Pairs of faces that share an edge (on a non-manifold edge, the faces
    in a chain)."""
    E = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    E.sort(axis=1)
    key = E[:, 0] * np.int64(nv) + E[:, 1]
    face = np.tile(np.arange(F.shape[0], dtype=np.int64), 3)
    o = np.argsort(key, kind="stable")
    key, face = key[o], face[o]
    same = key[1:] == key[:-1]
    return np.stack([face[:-1][same], face[1:][same]], 1)


def smooth_patches(F, N, nv, sharp_deg=50.0, pairs=None):
    """Label of the smooth patch of every face: faces joined across edges
    where the normals turn by less than ``sharp_deg``. Returns (number of
    patches, labels)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    pr = face_pairs(F, nv) if pairs is None else pairs
    cosang = np.einsum("ij,ij->i", N[pr[:, 0]], N[pr[:, 1]])
    pr = pr[cosang > np.cos(np.radians(sharp_deg))]
    nf = F.shape[0]
    G = coo_matrix((np.ones(len(pr), np.int8), (pr[:, 0], pr[:, 1])), shape=(nf, nf))
    return connected_components(G, directed=False)


# ------------------------------------------------------------------ holes
def find_holes(V, F, dmin=0.003, dmax=0.08, sharp_deg=50.0, max_roundness=0.1,
               max_gap_deg=60.0, verbose=False):
    """Round holes (and pins / bosses) as ring-shaped wall patches.

    Returns a dict of arrays (one entry per ring, by diameter): ``center``,
    ``axis`` (unit; the ring plane's normal), ``diameter`` (twice the mean
    facet distance from the axis), ``roundness`` (std / mean of that
    distance), ``gap_deg`` (largest angle round the axis without facets),
    ``depth`` (extent along the axis: about the plate thickness), ``kind``
    (+1 hole: the faces next to the wall all lie outside its radius, the
    plate round the hole; -1 pin or boss: some lie inside, its cap; 0
    unclear), ``kind_normals`` (the same from the wall normals: towards the
    axis for a hole; needs outward normals) and ``nfaces``."""
    t0 = time.time()
    C, N, A = face_geometry(V, F)
    pairs = face_pairs(F, V.shape[0])
    npatch, lab = smooth_patches(F, N, V.shape[0], sharp_deg, pairs)
    cnt = np.bincount(lab, minlength=npatch)
    # small patches only (a hole wall has tens to a few thousand facets)
    lo_ = np.full((npatch, 3), np.inf)
    hi_ = np.full((npatch, 3), -np.inf)
    for k in range(3):
        np.minimum.at(lo_[:, k], lab, C[:, k])
        np.maximum.at(hi_[:, k], lab, C[:, k])
    ext = (hi_ - lo_).max(1)
    cand = (cnt >= 8) & (cnt <= 5000) & (ext < 1.5 * dmax) & (ext > 0.5 * dmin)
    pid = np.flatnonzero(cand)
    idx = np.full(npatch, -1, np.int64)
    idx[pid] = np.arange(pid.size)
    sel = np.flatnonzero(cand[lab])
    g = idx[lab[sel]]
    P, Nf, w = C[sel], N[sel], A[sel]
    k = pid.size
    W = np.bincount(g, weights=w, minlength=k)
    ctr = np.stack([np.bincount(g, weights=w * P[:, j], minlength=k) for j in range(3)], 1)
    ctr /= np.maximum(W, 1e-300)[:, None]
    R = P - ctr[g]
    # the axis: for a flat ring (a hole through a plate) the normal of the
    # plane the facets lie in; for a tall wall (a pin, a deep hole) the
    # direction the wall normals spread least in (they are all across it)
    covp = np.zeros((k, 3, 3))
    covn = np.zeros((k, 3, 3))
    for a_ in range(3):
        for b_ in range(a_, 3):
            v = np.bincount(g, weights=w * R[:, a_] * R[:, b_], minlength=k)
            covp[:, a_, b_] = v
            covp[:, b_, a_] = v
            v = np.bincount(g, weights=w * Nf[:, a_] * Nf[:, b_], minlength=k)
            covn[:, a_, b_] = v
            covn[:, b_, a_] = v
    evp, vecp = np.linalg.eigh(covp)                     # ascending
    evn, vecn = np.linalg.eigh(covn)
    flat = evp[:, 0] < 0.1 * evp[:, 1]
    ax = np.where(flat[:, None], vecp[:, :, 0], vecn[:, :, 0])
    e1 = np.cross(ax, np.where(np.abs(ax[:, :1]) < 0.9, [[1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]]))
    e1 /= np.linalg.norm(e1, axis=1)[:, None]
    e2 = np.cross(ax, e1)
    along = np.einsum("ij,ij->i", R, ax[g])
    rel = R - along[:, None] * ax[g]
    r = np.linalg.norm(rel, axis=1)
    nloc = np.bincount(g, minlength=k)
    rm = np.bincount(g, weights=r, minlength=k) / np.maximum(nloc, 1)
    rs = np.sqrt(np.maximum(np.bincount(g, weights=r * r, minlength=k) / np.maximum(nloc, 1)
                            - rm * rm, 0.0))
    rnd = rs / np.maximum(rm, 1e-300)
    # angular coverage: the largest gap between facet angles round the axis
    th = np.arctan2(np.einsum("ij,ij->i", rel, e2[g]), np.einsum("ij,ij->i", rel, e1[g]))
    o = np.lexsort((th, g))
    gs, ts = g[o], th[o]
    first = np.r_[True, gs[1:] != gs[:-1]]
    last = np.r_[gs[1:] != gs[:-1], True]
    dth = np.r_[0.0, np.diff(ts)]
    dth[first] = 0.0
    gap = np.zeros(k)
    np.maximum.at(gap, gs, dth)
    wrap = ts[first] + 2 * np.pi - ts[last]              # last facet back to the first
    gap = np.maximum(gap, wrap)
    # hole or pin: the wall normals towards (-1) or away from (+1) the axis
    radial = np.einsum("ij,ij->i", Nf, rel / np.maximum(r, 1e-300)[:, None])
    outw = np.bincount(g, weights=radial, minlength=k) / np.maximum(nloc, 1)
    # depth: the extent of the wall's vertices along the axis
    av = np.einsum("ijk,ik->ij", V[F[sel]] - ctr[g][:, None, :], ax[g])
    amin = np.full(k, np.inf)
    amax = np.full(k, -np.inf)
    np.minimum.at(amin, g, av.min(1))
    np.maximum.at(amax, g, av.max(1))
    d = 2.0 * rm
    # a wall: its normals are across the axis (a flat disc, e.g. a pin's cap
    # fanned from its centre, has its facets at one radius too)
    across = np.bincount(g, weights=np.abs(np.einsum("ij,ij->i", Nf, ax[g])), minlength=k)
    across /= np.maximum(nloc, 1)
    ok = (rnd < max_roundness) & (gap < np.radians(max_gap_deg)) & (d > dmin) & \
        (d < dmax) & (across < 0.5)
    # hole or pin from the geometry: the faces next to the ring (across its
    # edges, outside its patch) lie outside its radius round a hole (the
    # plate) and partly inside it on a pin or a boss (its cap)
    inner = np.zeros(k)
    tot = np.zeros(k)
    for x, y in ((pairs[:, 0], pairs[:, 1]), (pairs[:, 1], pairs[:, 0])):
        m = (lab[x] != lab[y]) & cand[lab[x]]
        x, y = x[m], y[m]
        gi = idx[lab[x]]
        q = C[y] - ctr[gi]
        al = np.einsum("ij,ij->i", q, ax[gi])
        ry = np.linalg.norm(q - al[:, None] * ax[gi], axis=1)
        inner += np.bincount(gi, weights=(ry < 0.8 * rm[gi]).astype(np.float64), minlength=k)
        tot += np.bincount(gi, minlength=k)
    fin = inner / np.maximum(tot, 1.0)
    kind = np.where(fin < 0.05, 1, np.where(fin > 0.25, -1, 0))
    kind_n = np.where(outw < -0.5, 1, np.where(outw > 0.5, -1, 0))
    o = np.flatnonzero(ok)
    o = o[np.argsort(d[o], kind="stable")]
    res = dict(center=ctr[o], axis=ax[o], diameter=d[o], roundness=rnd[o],
               gap_deg=np.degrees(gap[o]), depth=(amax - amin)[o], kind=kind[o],
               kind_normals=kind_n[o], inner_share=fin[o], nfaces=nloc[o])
    if verbose:
        print(f"find_holes: {F.shape[0]} faces, {npatch} smooth patches, {pid.size} small, "
              f"{o.size} rings: {(kind[o] == 1).sum()} holes, {(kind[o] == -1).sum()} pins, "
              f"{(kind[o] == 0).sum()} unclear; from the normals {(kind_n[o] == 1).sum()} / "
              f"{(kind_n[o] == -1).sum()} / {(kind_n[o] == 0).sum()} ({time.time() - t0:.0f} s)",
              flush=True)
    return res


# ------------------------------------------------------------------ gaps
def gap_rays(V, F, P, D, max_width=0.05, N=None, rmi=None):
    """Distance from points P along unit directions D to the first wall
    (Embree ray casting on the triangles). Returns (width, partner face,
    facing): width inf where no wall is within ``max_width``; facing: the
    wall hit faces back (its normal against the ray), as a wall across
    fluid does."""
    if rmi is None:
        import trimesh
        from trimesh.ray.ray_pyembree import RayMeshIntersector
        rmi = RayMeshIntersector(trimesh.Trimesh(V, F, process=False, validate=False))
    if N is None:
        N = face_geometry(V, F)[1]
    eps = 1e-6
    n = P.shape[0]
    hit = np.full(n, -1, np.int64)
    step = 4_000_000
    for s0 in range(0, n, step):
        hit[s0:s0 + step] = rmi.intersects_first(P[s0:s0 + step] + eps * D[s0:s0 + step],
                                                 D[s0:s0 + step])
    width = np.full(n, np.inf)
    facing = np.zeros(n, np.bool_)
    h = np.flatnonzero(hit >= 0)
    # distance to the plane of the hit face along the ray
    j = hit[h]
    Pj = V[F[j, 0]]
    den = np.einsum("ij,ij->i", N[j], D[h])
    t = np.einsum("ij,ij->i", N[j], Pj - P[h]) / np.where(np.abs(den) > 1e-12, den, 1e-12)
    t = np.where(np.abs(den) > 1e-12, t, np.linalg.norm(Pj - P[h], axis=1))
    near = (t > 0) & (t <= max_width)
    width[h[near]] = t[near]
    facing[h[near]] = den[near] < 0
    partner = np.where(np.isfinite(width), hit, -1)
    return width, partner, facing


def gap_map(V, F, sign=1, max_width=0.05, verbose=False):
    """Gap width at every face: the distance from the face centre, along its
    normal times ``sign`` (+1: outward normals, the fluid side; see
    ``octree._plate_sign``), to the first wall (``gap_rays``). Returns
    (width, partner face, facing). One ray per face: see ``gap_samples``
    for large faces."""
    t0 = time.time()
    C, N, A = face_geometry(V, F)
    width, partner, facing = gap_rays(V, F, C, N * float(sign), max_width, N=N)
    if verbose:
        fin = np.isfinite(width)
        print(f"gap_map: {F.shape[0]} rays, {fin.sum()} hit a wall within {max_width*1e3:g} mm, "
              f"{(fin & ~facing).sum()} of them the inside of a face (orientation?) "
              f"({time.time() - t0:.0f} s)", flush=True)
    return width, partner, facing


def face_samples(V, F, spacing=0.004, max_per_face=20000, thin=True):
    """Points spread over the faces about ``spacing`` apart: a face of area
    A and longest edge L gets n = max(ceil(2 A / spacing^2), ceil(L /
    spacing)) points (a long sliver at least one per spacing along it),
    placed by the R2 low-discrepancy sequence folded into the triangle; a
    face with n = 1 its centre. ``thin``: the samples are gathered, one per
    cube of half a spacing and normal direction, the largest of them with
    the area of all (a fine mesh, its small faces and fans of slivers,
    would sample much closer than the spacing).
    Returns (points, face of each, area of each)."""
    v0 = V[F[:, 0]]
    e1 = V[F[:, 1]] - v0
    e2 = V[F[:, 2]] - v0
    L = np.sqrt(np.maximum(np.maximum((e1 ** 2).sum(1), (e2 ** 2).sum(1)),
                           ((e2 - e1) ** 2).sum(1)))
    A = 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1)
    n = np.maximum(np.ceil(2.0 * A / spacing ** 2), np.ceil(L / spacing))
    n = np.clip(n, 1, max_per_face).astype(np.int64)
    faces = np.arange(F.shape[0], dtype=np.int64)
    n = n[faces]
    fid = np.repeat(faces, n)
    start = np.cumsum(n) - n
    i = np.arange(fid.size, dtype=np.int64) - np.repeat(start, n)
    nf = np.zeros(F.shape[0], np.int64)
    nf[faces] = n
    g = 1.32471795724474602596                       # the plastic number (R2)
    a = np.mod(0.5 + i / g, 1.0)
    b = np.mod(0.5 + i / g ** 2, 1.0)
    fold = a + b > 1.0
    a = np.where(fold, 1.0 - a, a)
    b = np.where(fold, 1.0 - b, b)
    one = nf[fid] == 1
    a[one] = b[one] = 1.0 / 3.0
    P = v0[fid] + a[:, None] * e1[fid] + b[:, None] * e2[fid]
    area = A[fid] / nf[fid]
    if thin:
        # one sample per cube of half a spacing and normal direction (the
        # two sides of a plate apart): fine meshes (small faces, fans of
        # slivers) sample much closer than the spacing; the largest sample
        # of a cube stays, with the area of all
        Nf = np.cross(e1, e2)
        ax = np.argmax(np.abs(Nf), axis=1)
        sg = Nf[np.arange(F.shape[0]), ax] > 0
        del Nf
        key = np.c_[np.floor(P / (0.5 * spacing)).astype(np.int64), (2 * ax + sg)[fid]]
        _, grp = np.unique(key, axis=0, return_inverse=True)
        grp = grp.ravel()
        del key
        o = np.lexsort((-area, grp))
        first = o[np.r_[True, grp[o][1:] != grp[o][:-1]]]
        tot = np.bincount(grp, weights=area)
        P, fid, area = P[first], fid[first], tot[grp[first]]
    return P, fid, area


def gap_samples(V, F, sign=1, spacing=0.004, max_width=0.05, verbose=False):
    """``gap_map`` at points spread over the faces (``face_samples``): a gap
    often covers a large face only in part, and one ray per face then gives
    its width at one point only. Returns dict of arrays per sample:
    ``point``, ``face``, ``area``, ``normal`` (times ``sign``), ``width``,
    ``partner``, ``facing``; and ``spacing``."""
    t0 = time.time()
    P, fid, area = face_samples(V, F, spacing)
    N = face_geometry(V, F)[1]
    D = N[fid] * float(sign)
    width, partner, facing = gap_rays(V, F, P, D, max_width, N=N)
    if verbose:
        fin = np.isfinite(width)
        print(f"gap_samples: {P.shape[0]} rays on {F.shape[0]} faces ({spacing*1e3:g} mm), "
              f"{fin.sum()} hit a wall within {max_width*1e3:g} mm, {(fin & ~facing).sum()} "
              f"the inside of a face ({time.time() - t0:.0f} s)", flush=True)
    return dict(point=P, face=fid, area=area, normal=D, width=width, partner=partner,
                facing=facing, spacing=float(spacing))


def _same_surface(P, Nn, a, b, spacing):
    """Samples a, b on the same surface: normals less than 60 degrees apart
    and b within half a spacing of a's tangent plane (not on a parallel
    face of another plate)."""
    d = P[b] - P[a]
    return (np.einsum("ij,ij->i", Nn[a], Nn[b]) > 0.5) &         (np.abs(np.einsum("ij,ij->i", d, Nn[a])) < 0.5 * spacing)


def sample_regions(S, lo, hi):
    """Connected regions of the samples of ``gap_samples`` with a facing
    wall at lo <= width < hi: samples within 1.5 spacings of each other on
    the same surface (``_same_surface``). Returns
    (label per sample, -1 for the others; dict of arrays per region:
    ``nsamples``, ``nfaces``, ``area``, ``wmin``, ``wmed``, ``center``
    (area-weighted), ``extent`` (bounding box diagonal), ``normal``
    (area-weighted mean)), the regions by decreasing area."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree
    W = S["width"]
    sel = np.flatnonzero(S["facing"] & (W >= lo) & (W < hi))
    n = sel.size
    P, Nn, A, w = S["point"][sel], S["normal"][sel], S["area"][sel], W[sel]
    pr = cKDTree(P).query_pairs(1.5 * S["spacing"], output_type="ndarray")
    pr = pr[_same_surface(P, Nn, pr[:, 0], pr[:, 1], S["spacing"])]
    G = coo_matrix((np.ones(len(pr), np.int8), (pr[:, 0], pr[:, 1])), shape=(n, n))
    k, g = connected_components(G, directed=False)
    area = np.bincount(g, weights=A, minlength=k)
    ctr = np.stack([np.bincount(g, weights=A * P[:, j], minlength=k) for j in range(3)], 1)
    ctr /= np.maximum(area, 1e-300)[:, None]
    nrm = np.stack([np.bincount(g, weights=A * Nn[:, j], minlength=k) for j in range(3)], 1)
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1), 1e-300)[:, None]
    lo_ = np.full((k, 3), np.inf)
    hi_ = np.full((k, 3), -np.inf)
    for j in range(3):
        np.minimum.at(lo_[:, j], g, P[:, j])
        np.maximum.at(hi_[:, j], g, P[:, j])
    wmin = np.full(k, np.inf)
    np.minimum.at(wmin, g, w)
    o = np.lexsort((w, g))
    cnt = np.bincount(g, minlength=k)
    st = np.searchsorted(g[o], np.arange(k))
    wmed = w[o][st + cnt // 2] if k else np.zeros(0)
    gf = np.unique(np.stack([g, S["face"][sel]], 1), axis=0)[:, 0]
    nfaces = np.bincount(gf, minlength=k)
    order = np.argsort(-area, kind="stable")
    rank = np.empty(k, np.int64)
    rank[order] = np.arange(k)
    label = np.full(W.size, -1, np.int64)
    label[sel] = rank[g]
    return label, dict(nsamples=cnt[order], nfaces=nfaces[order], area=area[order],
                       wmin=wmin[order], wmed=wmed[order], center=ctr[order],
                       extent=np.linalg.norm(hi_ - lo_, axis=1)[order], normal=nrm[order])


def facing_walls(S, label):
    """Pairs (a, b) of regions of ``sample_regions`` facing each other
    across their gap: the partner face of a sample of a carries samples of
    b. Sorted, unique, a != b."""
    on = np.flatnonzero(label >= 0)
    fl = np.unique(np.stack([S["face"][on], label[on]], 1), axis=0)   # (face, region), sorted
    pf = S["partner"][on]
    lo = np.searchsorted(fl[:, 0], pf, "left")
    hi = np.searchsorted(fl[:, 0], pf, "right")
    cnt = hi - lo
    a = np.repeat(label[on], cnt)
    b = fl[np.repeat(lo, cnt) + np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt), 1]
    pr = np.unique(np.sort(np.stack([a, b], 1), axis=1), axis=0)
    return pr[pr[:, 0] != pr[:, 1]]


def mouth_edges(S, label, k=64, chunk=500_000):
    """The edges of the regions of ``sample_regions`` (or any labelling of
    the samples): pairs of a sample a of a region and a sample next to it
    outside it (within 1.5 spacings, on the same surface; of its k nearest
    samples). Returns (a, u, lu): u the unit direction from a towards the
    neighbour in a's tangent plane, lu the distance; per sample one pair in
    each of 8 directions (the nearest). In chunks of ``chunk`` samples."""
    from scipy.spatial import cKDTree
    on = np.flatnonzero(label >= 0)
    P, Nn = S["point"], S["normal"]
    r = 1.5 * S["spacing"]
    empty = np.zeros(0, np.int64), np.zeros((0, 3)), np.zeros(0)
    if on.size == 0:
        return empty
    # only samples near a region can be neighbours
    d, _ = cKDTree(P[on]).query(P, distance_upper_bound=r)
    cand = np.flatnonzero(np.isfinite(d))
    del d
    tree = cKDTree(P[cand])
    k = min(k, cand.size)
    out = []
    for c0 in range(0, on.size, chunk):
        q = on[c0:c0 + chunk]
        dd, jj = tree.query(P[q], k=k, distance_upper_bound=r)
        dd, jj = dd.reshape(q.size, k), jj.reshape(q.size, k)
        ok = np.isfinite(dd)
        a = np.broadcast_to(q[:, None], ok.shape)[ok]
        b = cand[jj[ok]]
        del dd, jj, ok
        keep = (label[b] != label[a]) & _same_surface(P, Nn, a, b, S["spacing"])
        a, b = a[keep], b[keep]
        u = P[b] - P[a]
        u -= np.einsum("ij,ij->i", u, Nn[a])[:, None] * Nn[a]
        lu = np.linalg.norm(u, axis=1)
        good = lu > 1e-12
        a, u, lu = a[good], u[good], lu[good]
        u /= lu[:, None]
        # one pair per sample and direction (8 bins in the tangent plane)
        n = Nn[a]
        e = np.where(np.abs(n[:, :1]) < 0.9, [[1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]])
        t1 = np.cross(n, e)
        t1 /= np.linalg.norm(t1, axis=1)[:, None]
        t2 = np.cross(n, t1)
        ang = np.arctan2(np.einsum("ij,ij->i", u, t2), np.einsum("ij,ij->i", u, t1))
        bins = np.floor((ang + np.pi) / (np.pi / 4)).astype(np.int64) % 8
        o = np.lexsort((lu, bins, a))
        a, u, lu, bins = a[o], u[o], lu[o], bins[o]
        first = np.r_[True, (a[1:] != a[:-1]) | (bins[1:] != bins[:-1])] if a.size else             np.zeros(0, bool)
        out.append((a[first], u[first], lu[first]))
    return (np.concatenate([x[0] for x in out]), np.concatenate([x[1] for x in out]),
            np.concatenate([x[2] for x in out]))


def gap_mouths(S, label, step, samples=False):
    """Where the regions of ``sample_regions`` open: for each edge of a
    region (``mouth_edges``), a point beyond the neighbour by ``step`` in
    the plane of the gap, at mid-gap height. Returns (points, region of
    each), and the sample each comes from if ``samples``."""
    a, u, lu = mouth_edges(S, label)
    P, Nn = S["point"], S["normal"]
    pts = P[a] + u * (lu + step)[:, None] + Nn[a] * (0.5 * S["width"][a])[:, None]
    return (pts, label[a], a) if samples else (pts, label[a])


def read_seals_csv(path):
    """Sealed places from a CSV with columns x, y, z (the STL's frame,
    metres, as the holes table) and radius_mm (more columns, e.g. a note,
    are ignored). Returns (centres (n, 3), radii (n,) in metres)."""
    import csv
    rows = list(csv.DictReader(open(path)))
    C = np.array([[float(r["x"]), float(r["y"]), float(r["z"])] for r in rows]).reshape(-1, 3)
    return C, np.array([float(r["radius_mm"]) * 1e-3 for r in rows])


def seal_samples(S, centres, radii):
    """``gap_samples`` S with the samples inside any of the spheres (sealed
    seams, welds, sealant) taken out of the gaps (``facing`` False). Returns
    (S, number of samples sealed)."""
    from scipy.spatial import cKDTree
    out = dict(S)
    if len(radii) == 0:
        return out, 0
    tree = cKDTree(S["point"])
    hit = np.zeros(S["point"].shape[0], bool)
    for c, r in zip(centres, radii):
        hit[tree.query_ball_point(c, float(r))] = True
    out["facing"] = S["facing"] & ~hit
    return out, int((hit & S["facing"]).sum())


def samples_to_model(S, info):
    """``gap_samples`` in the frame of ``car_article.load_car`` (points and
    normals; widths and areas are unchanged)."""
    out = dict(S)
    out["point"] = to_model(S["point"], info)
    out["normal"] = to_model(S["normal"], info, vectors=True)
    return out


def from_model(points, info, vectors=False):
    """``to_model`` reversed: points (or vectors) of the frame of
    ``car_article.load_car`` in the mesh's frame (metres, as ``read_stl``)."""
    R = np.asarray(info["rotation_file_to_car"], float)
    p = np.asarray(points, float) @ R
    if vectors:
        return p
    return p + np.asarray(info["centre_file"], float) * float(info["scale_to_m"])


def to_model(points, info, vectors=False):
    """Points (or direction vectors) of the mesh's frame in the frame of
    ``car_article.load_car`` (``info`` its second result): centred, scaled
    to metres, rotated."""
    R = np.asarray(info["rotation_file_to_car"], float)
    p = np.asarray(points, float)
    if vectors:
        return p @ R.T
    c = np.asarray(info["centre_file"], float) * float(info["scale_to_m"])
    return (p - c) @ R.T


def gap_regions(V, F, width, facing, wmax):
    """Connected regions of faces with a facing wall closer than ``wmax``
    (faces joined across shared edges). Returns (label per face, -1 for the
    others; dict of arrays per region: ``nfaces``, ``area``, ``wmin``,
    ``wmed`` (median width), ``center`` (area-weighted), ``extent``
    (bounding box diagonal)), the regions by decreasing area."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    nf = F.shape[0]
    on = facing & (width < wmax)
    pr = face_pairs(F, V.shape[0])
    pr = pr[on[pr[:, 0]] & on[pr[:, 1]]]
    G = coo_matrix((np.ones(len(pr), np.int8), (pr[:, 0], pr[:, 1])), shape=(nf, nf))
    _, lab = connected_components(G, directed=False)
    sel = np.flatnonzero(on)
    u, g = np.unique(lab[sel], return_inverse=True)
    k = u.size
    C, _, A = face_geometry(V, F)
    C, A, w = C[sel], A[sel], width[sel]
    area = np.bincount(g, weights=A, minlength=k)
    ctr = np.stack([np.bincount(g, weights=A * C[:, j], minlength=k) for j in range(3)], 1)
    ctr /= np.maximum(area, 1e-300)[:, None]
    lo = np.full((k, 3), np.inf)
    hi = np.full((k, 3), -np.inf)
    for j in range(3):
        np.minimum.at(lo[:, j], g, C[:, j])
        np.maximum.at(hi[:, j], g, C[:, j])
    wmin = np.full(k, np.inf)
    np.minimum.at(wmin, g, w)
    o = np.lexsort((w, g))
    cnt = np.bincount(g, minlength=k)
    st = np.searchsorted(g[o], np.arange(k))
    wmed = w[o][st + cnt // 2] if k else np.zeros(0)
    order = np.argsort(-area, kind="stable")
    rank = np.empty(k, np.int64)
    rank[order] = np.arange(k)
    label = np.full(nf, -1, np.int64)
    label[sel] = rank[g]
    return label, dict(nfaces=cnt[order], area=area[order], wmin=wmin[order],
                       wmed=wmed[order], center=ctr[order],
                       extent=np.linalg.norm(hi - lo, axis=1)[order])


def hole_discs(holes, n=24):
    """Triangulated discs of the holes: (points, triangles, the hole of
    each triangle)."""
    k = len(holes["diameter"])
    P, T, cid = [], [], []
    th = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    base = 0
    for i in range(k):
        a = holes["axis"][i]
        e1 = np.cross(a, [1.0, 0.0, 0.0] if abs(a[0]) < 0.9 else [0.0, 1.0, 0.0])
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(a, e1)
        r = 0.5 * holes["diameter"][i]
        ring = holes["center"][i] + r * (np.cos(th)[:, None] * e1 + np.sin(th)[:, None] * e2)
        P.append(np.vstack([holes["center"][i], ring]))
        T.append(np.stack([np.zeros(n, np.int64), 1 + np.arange(n), 1 + (np.arange(n) + 1) % n],
                          1) + base)
        cid.append(np.full(n, i))
        base += n + 1
    if not k:
        return np.zeros((0, 3)), np.zeros((0, 3), np.int64), np.zeros(0, np.int64)
    return np.vstack(P), np.vstack(T), np.concatenate(cid)


KIND_NAMES = {1: "hole", -1: "pin", 0: "unclear"}


def export_openings(prefix, V, F, width=None, facing=None, holes=None, regions=None):
    """Files to review the openings (open the .vtp files in ParaView):
    ``<prefix>_mesh.vtp``: the mesh with cell data ``gap_mm`` (gap width,
    -1 where none), ``facing`` and ``gap_region``; ``<prefix>_holes.vtp``: a
    disc per hole (cell data ``diameter_mm``, ``kind``, ``hole``);
    ``<prefix>_holes.csv`` and ``<prefix>_gaps.csv``: the tables."""
    from .worldviz import write_vtp
    if width is not None:
        cd = dict(gap_mm=np.where(np.isfinite(width), width * 1e3, -1.0))
        if facing is not None:
            cd["facing"] = facing.astype(np.float32)
        if regions is not None:
            cd["gap_region"] = regions[0].astype(np.float32)
        write_vtp(prefix + "_mesh.vtp", V, polys=F, cell_data=cd)
    if holes is not None and len(holes["diameter"]):
        P, T, cid = hole_discs(holes)
        write_vtp(prefix + "_holes.vtp", P, polys=T,
                  cell_data=dict(diameter_mm=holes["diameter"][cid] * 1e3,
                                 kind=holes["kind"][cid].astype(np.float32),
                                 hole=cid.astype(np.float32)))
        with open(prefix + "_holes.csv", "w") as f:
            f.write("hole,kind,diameter_mm,x,y,z,axis_x,axis_y,axis_z,depth_mm,roundness,"
                    "gap_deg,nfaces\n")
            for i in range(len(holes["diameter"])):
                c, a = holes["center"][i], holes["axis"][i]
                f.write(f"{i},{KIND_NAMES[int(holes['kind'][i])]},{holes['diameter'][i]*1e3:.3f},"
                        f"{c[0]:.5f},{c[1]:.5f},{c[2]:.5f},{a[0]:.5f},{a[1]:.5f},{a[2]:.5f},"
                        f"{holes['depth'][i]*1e3:.3f},{holes['roundness'][i]:.4f},"
                        f"{holes['gap_deg'][i]:.1f},{holes['nfaces'][i]}\n")
    if regions is not None:
        R = regions[1]
        with open(prefix + "_gaps.csv", "w") as f:
            f.write("region,nfaces,area_cm2,width_min_mm,width_median_mm,x,y,z,extent_mm\n")
            for i in range(len(R["area"])):
                c = R["center"][i]
                f.write(f"{i},{R['nfaces'][i]},{R['area'][i]*1e4:.3f},{R['wmin'][i]*1e3:.3f},"
                        f"{R['wmed'][i]*1e3:.3f},{c[0]:.5f},{c[1]:.5f},{c[2]:.5f},"
                        f"{R['extent'][i]*1e3:.1f}\n")


def read_holes_csv(path):
    """The holes of a ``<prefix>_holes.csv`` (``export_openings``), as
    ``find_holes`` gives them (a reviewed, edited table may be read back)."""
    import csv
    rows = list(csv.DictReader(open(path)))
    names = {v: k for k, v in KIND_NAMES.items()}
    f = lambda k: np.array([float(r[k]) for r in rows])          # noqa: E731
    return dict(center=np.stack([f("x"), f("y"), f("z")], 1) if rows else np.zeros((0, 3)),
                axis=np.stack([f("axis_x"), f("axis_y"), f("axis_z")], 1) if rows
                else np.zeros((0, 3)),
                diameter=f("diameter_mm") * 1e-3, depth=f("depth_mm") * 1e-3,
                roundness=f("roundness"), gap_deg=f("gap_deg"),
                kind=np.array([names[r["kind"]] for r in rows], np.int64),
                nfaces=np.array([int(r["nfaces"]) for r in rows], np.int64))


def holes_for_model(holes, info, kinds=(1,), dmin=0.0, dmax=np.inf, keep=None):
    """Explicit holes for ``Simulation(holes=...)`` from ``find_holes``: the
    chosen ones (``kinds``, a diameter range, or ``keep``: their indices)
    in the frame of ``car_article.load_car`` (``info``)."""
    sel = np.isin(holes["kind"], kinds) & (holes["diameter"] >= dmin) & \
        (holes["diameter"] <= dmax)
    if keep is not None:
        m = np.zeros(sel.size, np.bool_)
        m[np.asarray(keep, np.int64)] = True
        sel &= m
    C = to_model(holes["center"][sel], info)
    A = to_model(holes["axis"][sel], info, vectors=True)
    # optional: a hole that the grid cannot open is left out (with a count)
    # instead of stopping the setup, as for a hole listed by hand
    return [dict(center=tuple(c), diameter=float(d), axis=tuple(a / np.linalg.norm(a)),
                 optional=True)
            for c, d, a in zip(C, holes["diameter"][sel], A)]
