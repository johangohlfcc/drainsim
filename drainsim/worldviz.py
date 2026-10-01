"""World-frame visualisation: the object moves, the bath stays still.

The simulation runs in the object frame (grid fixed to the object, bath
surface and gravity rotate). For visualisation this is inverted with the
motion's pose

    x_world = R(t) (x_obj - pivot) + pivot + T(t)

so that the bath is a fixed box and the object, with the liquid it carries,
moves through it - as in the real process.

Output (ParaView: File > Open the ``.pvd`` files, press Play):

* ``<prefix>_object.pvd``  object surface per frame (moving)
* ``<prefix>_liquid.pvd``  surface of liquid that is *not* bath: cavity
  contents, compartments still draining, liquid carried out (moving)
* ``<prefix>_air.pvd``     surface of air below the bath surface: trapped
  pockets and compartments not yet filled - the accessibility result (moving)
* ``<prefix>_volume.pvd``  the full cell grid, transformed to world
  coordinates, with cell data ``liquid``, ``retained``, ``bath``,
  ``air_below``, ``solid``, ``compartment`` (for slices / thresholds)
* ``<prefix>_film.pvd``    film carrier surface with cell data ``film_um``
  (film thickness in micrometres; only for runs with ``film=True``)
* ``<prefix>_bath.vtp``    static bath box (world frame)

Files are VTK XML with raw appended binary data (compact, fast to read).
In 2D the surfaces become contour lines and the volume is a quad grid in
the x-y plane.
"""
from __future__ import annotations

import os

import numpy as np
from numba import njit, prange
from scipy import ndimage

from .par import dual

# ----------------------------------------------------------------- VTK XML I/O

_VT = {np.dtype("float32"): "Float32", np.dtype("float64"): "Float64",
       np.dtype("int32"): "Int32", np.dtype("int64"): "Int64",
       np.dtype("uint8"): "UInt8"}


class _Appended:
    """Collects arrays for a VTK XML file with raw appended data
    (optionally zlib-compressed, one block per array)."""

    compress = True

    def __init__(self):
        self.blocks = []
        self.offset = 0

    def add(self, arr, name=None, ncomp=1):
        arr = np.ascontiguousarray(arr)
        if arr.dtype not in _VT:
            arr = arr.astype(np.float32)
        raw = arr.tobytes()
        tag = (f'<DataArray type="{_VT[arr.dtype]}"'
               + (f' Name="{name}"' if name else "")
               + f' NumberOfComponents="{ncomp}" format="appended"'
               f' offset="{self.offset}"/>\n')
        if self.compress:
            import zlib
            if raw:
                c = zlib.compress(raw, 6)
                # header: nblocks, block size, last block size, compressed sizes
                hdr = np.array([1, len(raw), len(raw), len(c)], np.uint64)
                block = hdr.tobytes() + c
            else:
                block = np.zeros(3, np.uint64).tobytes()
        else:
            block = np.uint64(len(raw)).tobytes() + raw
        self.blocks.append(block)
        self.offset += len(block)
        return tag

    def write(self, path, header_xml, vtk_type):
        with open(path, "wb") as f:
            f.write(b'<?xml version="1.0"?>\n')
            comp = (' compressor="vtkZLibDataCompressor"'
                    if self.compress else "")
            f.write(f'<VTKFile type="{vtk_type}" version="1.0" '
                    f'byte_order="LittleEndian" header_type="UInt64"{comp}>\n'
                    .encode())
            f.write(header_xml.encode())
            f.write(b'<AppendedData encoding="raw">\n_')
            for b in self.blocks:
                f.write(b)
            f.write(b"\n</AppendedData>\n</VTKFile>\n")


def write_vtp(path, points, polys=None, lines=None, point_data=None,
              cell_data=None):
    """PolyData: points (n,3); polys (m,k) triangles; lines = list of index
    arrays. Cell data are ordered polys first, then lines (VTK convention)."""
    points = np.asarray(points, np.float32).reshape(-1, 3)
    ap = _Appended()
    npoly = 0 if polys is None else len(polys)
    nline = 0 if not lines else len(lines)
    xml = ["<PolyData>\n",
           f'<Piece NumberOfPoints="{len(points)}" NumberOfVerts="0" '
           f'NumberOfLines="{nline}" NumberOfStrips="0" NumberOfPolys="{npoly}">\n']
    if point_data:
        xml.append("<PointData>\n")
        for k, v in point_data.items():
            xml.append(ap.add(np.asarray(v, np.float32), k))
        xml.append("</PointData>\n")
    if cell_data:
        xml.append("<CellData>\n")
        for k, v in cell_data.items():
            xml.append(ap.add(np.asarray(v, np.float32), k))
        xml.append("</CellData>\n")
    xml.append("<Points>\n" + ap.add(points.ravel(), None, 3) + "</Points>\n")
    if npoly:
        polys = np.asarray(polys, np.int64)
        xml.append("<Polys>\n")
        xml.append(ap.add(polys.ravel(), "connectivity"))
        xml.append(ap.add(np.arange(1, npoly + 1, dtype=np.int64) * polys.shape[1],
                          "offsets"))
        xml.append("</Polys>\n")
    if nline:
        conn = np.concatenate([np.asarray(l, np.int64) for l in lines])
        offs = np.cumsum([len(l) for l in lines]).astype(np.int64)
        xml.append("<Lines>\n" + ap.add(conn, "connectivity")
                   + ap.add(offs, "offsets") + "</Lines>\n")
    xml.append("</Piece>\n</PolyData>\n")
    ap.write(path, "".join(xml), "PolyData")


def write_vts(path, corner_points, dims, cell_data):
    """StructuredGrid. corner_points (np,3) in VTK order (x fastest);
    dims = number of points per direction (3 ints); cell data in VTK order."""
    ap = _Appended()
    ext = f"0 {dims[0]-1} 0 {dims[1]-1} 0 {dims[2]-1}"
    xml = [f'<StructuredGrid WholeExtent="{ext}">\n<Piece Extent="{ext}">\n',
           "<CellData>\n"]
    for k, v in cell_data.items():
        xml.append(ap.add(v, k))
    xml.append("</CellData>\n<Points>\n")
    xml.append(ap.add(np.asarray(corner_points, np.float32).ravel(), None, 3))
    xml.append("</Points>\n</Piece>\n</StructuredGrid>\n")
    ap.write(path, "".join(xml), "StructuredGrid")


def write_pvd(path, entries):
    """entries: list of (time, relative file name)."""
    with open(path, "w") as f:
        f.write('<?xml version="1.0"?>\n<VTKFile type="Collection" '
                'version="0.1" byte_order="LittleEndian">\n<Collection>\n')
        for t, fn in entries:
            f.write(f'<DataSet timestep="{t:.6g}" group="" part="0" file="{fn}"/>\n')
        f.write("</Collection>\n</VTKFile>\n")


# ------------------------------------------------------------------ geometry

def pose3(motion, t):
    """(R 3x3, T 3, pivot 3) of the object at time t, 2D embedded in z=0."""
    R, T = motion.pose(t)
    nd = motion.ndim
    R3 = np.eye(3)
    R3[:nd, :nd] = R
    T3 = np.zeros(3)
    T3[:nd] = T
    p3 = np.zeros(3)
    p3[:nd] = motion.pivot
    return R3, T3, p3


def to_world(pts_obj, motion, t):
    """Object-frame points (n, ndim or 3) -> world (n, 3)."""
    pts = np.asarray(pts_obj, float)
    if pts.shape[1] == 2:
        pts = np.column_stack([pts, np.zeros(len(pts))])
    R, T, p = pose3(motion, t)
    return (pts - p) @ R.T + p + T


def _corner_points_obj(grid):
    """Grid corner points in VTK order (x fastest), padded to 3D."""
    axes = [grid.origin[d] + np.arange(n + 1) * grid.dx
            for d, n in enumerate(grid.shape)]
    if grid.ndim == 2:
        axes.append(np.zeros(1))
    mesh = np.meshgrid(*axes, indexing="ij")
    return np.stack([m.ravel(order="F") for m in mesh], 1)


def _vtk_order(a, grid):
    return np.asarray(a)[:grid.ncells].reshape(grid.shape).ravel(order="F")


def _iso_obj(field, grid, level=0.5):
    """Iso-surface (3D: triangles) or iso-lines (2D) in object coordinates."""
    from skimage import measure
    f = np.pad(np.asarray(field)[:grid.ncells].reshape(grid.shape).astype(np.float32), 1)
    if f.max() < level:
        return np.zeros((0, 3)), None, None
    if grid.ndim == 3:
        verts, faces, _, _ = measure.marching_cubes(f, level)
        pts = grid.origin + (verts - 0.5) * grid.dx
        return pts, faces, None
    lines, pts, n = [], [], 0
    for c in measure.find_contours(f, level):
        xy = grid.origin + (c - 0.5) * grid.dx
        pts.append(xy)
        lines.append(np.arange(n, n + len(xy)))
        n += len(xy)
    pts = np.vstack(pts) if pts else np.zeros((0, 2))
    return np.column_stack([pts, np.zeros(len(pts))]), None, lines


def _dilate_into_solid(field, grid):
    """Extend a fluid field into neighbouring solid cells so that iso-surfaces
    touch the walls instead of stopping half a cell away."""
    f = np.asarray(field)[:grid.ncells].reshape(grid.shape)
    fp = np.ones((3,) * grid.ndim, bool)
    grown = ndimage.maximum_filter(f, footprint=fp)
    return np.where(grid.solid, grown, f).ravel()


def world_bounds(sim, times):
    """Bounding box of the object over all times (world frame, 3D)."""
    g = sim.grid
    lo, hi = g.origin, g.origin + np.array(g.shape) * g.dx
    corners = np.array(np.meshgrid(*[[l, h] for l, h in zip(lo, hi)],
                                   indexing="ij")).reshape(g.ndim, -1).T
    allp = np.vstack([to_world(corners, sim.motion, t) for t in times])
    return allp.min(0), allp.max(0)


def bath_geometry(sim, times, margin=0.1, depth=None):
    """Static bath box (3D) / rectangle (2D) in the world frame."""
    lo, hi = world_bounds(sim, times)
    zb = sim.motion.bath_level
    nd = sim.grid.ndim
    span = hi - lo
    lo = lo - margin * span.max()
    hi = hi + margin * span.max()
    bottom = min(lo[nd - 1], zb - 0.1) if depth is None else zb - depth
    if nd == 2:
        pts = np.array([[lo[0], bottom, 0], [hi[0], bottom, 0],
                        [hi[0], zb, 0], [lo[0], zb, 0]])
        return pts, np.array([[0, 1, 2, 3]])
    x0, x1, y0, y1 = lo[0], hi[0], lo[1], hi[1]
    z0, z1 = bottom, zb
    pts = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                    [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]])
    quads = np.array([[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4],
                      [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]])
    return pts, quads


# ------------------------------------------------------------------- export

def frame_fields(sim, hist, t):
    """Cell fields of one snapshot (object frame, flat C order)."""
    L = np.asarray(hist.snapshots[t], float)
    B = hist.snap_bath[t]
    A = hist.snap_atm[t]
    up, zb = sim.motion.frame(t)
    below = (sim.X @ up) < zb
    fl = sim.fl
    retained = np.where(fl & ~B, L, 0.0)
    air = np.where(fl & ~A & below, 1.0 - L, 0.0)
    return dict(liquid=L, retained=retained, bath=B.astype(float),
                air_below=air)


def _node_e(sim, e, shape):
    """Half vertical extent per node, from the grid cell's ``e``: scaled by
    the node size (octree nodes, sub-cells)."""
    ns = getattr(sim, "nsize", None)
    if ns is not None:
        return e * np.asarray(ns, float) / float(sim.grid.dx)
    en = np.full(shape, e)
    if getattr(sim, "fine", None) is not None and sim.fine.any():
        en = np.where(sim.fine, e / max(sim.subcells, 1), e)
    return en


def planar_fraction(sim, frac, h, iters=2, e=None):
    """Display-only reconstruction of flat free surfaces.

    The quasi-static state fills whole cells lowest-first, so a pool in a
    rotated grid has a staircase surface. For drawing, each connected body of
    ``frac`` (liquid, or air with ``h`` negated) is replaced by the fraction
    cut by a single horizontal plane holding the same volume:
    ``f_c = clamp((H - (h_c - e)) / 2e)``, searched over the body and its
    fluid neighbours (``iters`` face layers) in the same compartment. Each
    body is bisected on its own (in parallel over the bodies), and the
    dilation walks only the nodes of the bodies, so the cost follows the
    bodies, not the grid. ``e``: the half extents per node, if the caller
    has them (``_node_e``).
    """
    from .fsm import to_csr
    if e is None:
        e = _node_e(sim, float(sim._vis_e), frac.shape)
    frac = np.asarray(frac, float)
    out = np.zeros_like(frac)
    ptr, nidx = to_csr(sim.nbr)
    lab = np.asarray(sim.lab)
    body, act, nb = _label_masked(np.asarray(sim.fl, np.bool_), frac, 1e-6, lab, ptr, nidx)
    if nb == 0:
        return out
    nodes, starts = _grow_sorted(body, act, lab, ptr, nidx, int(iters), int(nb))
    _bisect_bodies(nodes, starts, frac, np.asarray(sim.v, float), np.asarray(h, float),
                   np.asarray(e, float), out)
    return out


def trapped_display(sim, h, zb, L=None, B=None, A=None, view=None):
    """Node fields to draw: trapped liquid above the bath surface and
    trapped air below it (the definitions of ``Simulation.trapped_fields``),
    each with flat free surfaces (``planar_fraction``) and cut at the bath
    surface. A compartment that crosses the surface is drawn as liquid above
    it and as air below it, as it is; its content is not changed.

    ``view``: object with the static graph attributes and ``_vis_e`` for
    ``planar_fraction`` (default: ``sim``, whose ``_vis_e`` is then set)."""
    L = sim.L if L is None else L
    B = sim.B if B is None else B
    A = sim.A if A is None else A
    view = sim if view is None else view
    en = _node_e(view, float(view._vis_e), h.shape)
    h = np.asarray(h, float)
    a, fq, fa, hn = (np.empty(h.shape[0]) for _ in range(4))
    _trapped_split(h, en, float(zb), np.asarray(L, float), np.asarray(B, np.bool_),
                   np.asarray(A, np.bool_), np.asarray(sim.fl, np.bool_), a, fq, fa, hn)
    liq = planar_fraction(view, fq, h, e=en)
    liq *= a
    del fq
    air = planar_fraction(view, fa, hn, e=en)
    np.subtract(1.0, a, out=a)
    air *= a
    return liq, air


@njit(cache=True, nogil=True)
def _trapped_split(h, en, zb, L, B, A, fl, a, liq, air, hn):
    """The inputs of ``trapped_display`` in one pass (the numpy expressions,
    element by element): the bath cut ``a``, the liquid not in the bath, the
    air not open to the atmosphere, and -h."""
    for i in range(h.shape[0]):
        x = (h[i] + en[i] - zb) / (2.0 * en[i])
        a[i] = min(max(x, 0.0), 1.0)
        liq[i] = L[i] if fl[i] and not B[i] else 0.0
        air[i] = 1.0 - L[i] if fl[i] and not A[i] else 0.0
        hn[i] = -h[i]


@njit(cache=True, nogil=True)
def _label_masked(fl, frac, thr, lab, ptr, idx):
    """Connected bodies of the nodes with ``fl`` and ``frac > thr`` within
    the same compartment, numbered by their lowest node (the result of
    ``par.label_bodies``: the same links, n > c of the rows of c, joined by
    union-find), on the masked nodes only. Returns (body per node or -1,
    the masked nodes in ascending order, number of bodies)."""
    N = fl.shape[0]
    body = np.full(N, -1, np.int64)
    m = 0
    for c in range(N):
        if fl[c] and frac[c] > thr:
            body[c] = m                       # position in act, for now
            m += 1
    act = np.empty(m, np.int64)
    for c in range(N):
        if body[c] >= 0:
            act[body[c]] = c
    uf = np.arange(m)
    for q in range(m):
        c = act[q]
        for jj in range(ptr[c], ptr[c + 1]):
            n = idx[jj]
            if n < 0 or n < c or body[n] < 0 or lab[n] != lab[c]:
                continue
            a = q
            while uf[a] != a:
                uf[a] = uf[uf[a]]
                a = uf[a]
            b = body[n]
            while uf[b] != b:
                uf[b] = uf[uf[b]]
                b = uf[b]
            if a != b:
                if a < b:
                    uf[b] = a
                else:
                    uf[a] = b
    # each set's root is its lowest position = its lowest node; number the
    # roots in node order
    nb = 0
    rid = np.empty(m, np.int64)
    for q in range(m):
        r = q
        while uf[r] != r:
            r = uf[r]
        if r == q:
            rid[q] = nb
            nb += 1
        body[act[q]] = rid[r]
    return body, act, nb


@njit(cache=True, nogil=True)
def _grow_sorted(body, act, lab, ptr, idx, iters, nb):
    """Graph dilation of the body labels by ``iters`` layers within the same
    compartment (a node reached by several bodies takes the last one in
    node order, as a vectorised assignment would), then the grown nodes of
    each body in ascending node order (a stable sort by body): returns
    (nodes, starts), body k = nodes[starts[k]:starts[k+1]]. ``act``: the
    labelled nodes, ascending. Walks only these nodes and their links;
    ``body`` is overwritten."""
    for _ in range(iters):
        m = 0
        for q in range(act.shape[0]):
            c = act[q]
            m += ptr[c + 1] - ptr[c]
        wn = np.empty(m, np.int64)
        wg = np.empty(m, np.int64)
        k = 0
        # every write reads the labels of the previous layer; applied in order
        for q in range(act.shape[0]):
            c = act[q]
            g = body[c]
            for jj in range(ptr[c], ptr[c + 1]):
                t = idx[jj]
                if t >= 0 and lab[t] == lab[c] and body[t] < 0:
                    wn[k] = t
                    wg[k] = g
                    k += 1
        for q in range(k):
            body[wn[q]] = wg[q]
        new = np.sort(wn[:k])
        u = 0
        for q in range(k):
            if q == 0 or new[q] != new[q - 1]:
                new[u] = new[q]
                u += 1
        # merge the two ascending, disjoint lists
        out = np.empty(act.shape[0] + u, np.int64)
        i = j = r = 0
        while i < act.shape[0] or j < u:
            if j >= u or (i < act.shape[0] and act[i] < new[j]):
                out[r] = act[i]
                i += 1
            else:
                out[r] = new[j]
                j += 1
            r += 1
        act = out
    starts = np.zeros(nb + 1, np.int64)
    for q in range(act.shape[0]):
        starts[body[act[q]] + 1] += 1
    for k in range(nb):
        starts[k + 1] += starts[k]
    fill = starts[:-1].copy()
    nodes = np.empty(act.shape[0], np.int64)
    for q in range(act.shape[0]):
        b = body[act[q]]
        nodes[fill[b]] = act[q]
        fill[b] += 1
    return nodes, starts


@dual
def _bisect_bodies(nodes, starts, frac, w, h, e, out):
    """The plane of each body (``planar_fraction``): 40 bisection steps on
    the height H at which the cut volume, summed over the body's nodes in
    ascending order, exceeds the body's volume. Same arithmetic and the
    same order of the sums as the vectorised numpy version (``bincount``),
    so the same numbers; the bodies are independent."""
    nb = starts.shape[0] - 1
    for k in prange(nb):
        s0 = starts[k]
        s1 = starts[k + 1]
        n = s1 - s0
        hc = np.empty(n)
        ec = np.empty(n)
        wc = np.empty(n)
        V = 0.0
        lo = np.inf
        hi = -np.inf
        for q in range(n):
            i = nodes[s0 + q]
            hc[q] = h[i]
            ec[q] = e[i]
            wc[q] = w[i]
            V += frac[i] * w[i]
            lo = min(lo, h[i] - e[i])
            hi = max(hi, h[i] + e[i])
        for _ in range(40):
            H = 0.5 * (lo + hi)
            if H == lo or H == hi:
                break                 # adjacent numbers: no step changes them any more
            S = 0.0
            for q in range(n):
                f = (H - hc[q] + ec[q]) / (2 * ec[q])
                if f > 0.0:
                    S += min(f, 1.0) * wc[q]
            if S > V:
                hi = H
            else:
                lo = H
        H = 0.5 * (lo + hi)
        for q in range(n):
            f = (H - hc[q] + ec[q]) / (2 * ec[q])
            out[nodes[s0 + q]] = min(max(f, 0.0), 1.0)


def export_world(sim, hist, outdir, prefix="dip", volume=True,
                 volume_every=10, surfaces=True, object_mesh=None, times=None,
                 flat_surfaces=True):
    """Write world-frame VTK series of the stored snapshots.

    The surfaces (object, liquid, air) are written for every snapshot and are
    small. The full cell grid is large (about 1 MB per 100 k cells even
    compressed), so it is written only every ``volume_every`` snapshots
    (1 = all, ``volume=False`` = never).

    sim, hist   : a finished ``Simulation`` and its ``History`` (run with
                  ``snapshot_every=...`` or ``snapshot_times=...``).
    flat_surfaces : draw liquid/air surfaces as flat planes holding the same
                  volume (display only, see ``planar_fraction``); False draws
                  the raw cell staircase.
    object_mesh : optional trimesh.Trimesh / path of the original surface
                  mesh (object coordinates) to draw instead of the voxel
                  surface.
    Returns a dict of the written file names.

    Uniform grids only: an octree run has no cell array to draw (the fields
    are per node). For those, resample the fields on a uniform display grid
    (``octview.DisplayGrid``), as ``examples/car_movie.py`` does.
    """
    if getattr(sim, "octree", False):
        raise NotImplementedError(
            "export_world needs a uniform grid; this simulation runs on an octree. "
            "Resample the node fields on a uniform display grid (octview.DisplayGrid, "
            "see examples/car_movie.py --record/--render).")
    os.makedirs(outdir, exist_ok=True)
    sub = f"{prefix}_frames"
    os.makedirs(os.path.join(outdir, sub), exist_ok=True)
    g = sim.grid
    times = sorted(hist.snapshots) if times is None else list(times)
    motion = sim.motion

    # static geometry in the object frame
    if object_mesh is not None:
        import trimesh
        if not isinstance(object_mesh, trimesh.Trimesh):
            object_mesh = trimesh.load(object_mesh, force="mesh")
        obj_pts, obj_faces, obj_lines = object_mesh.vertices, object_mesh.faces, None
    else:
        obj_pts, obj_faces, obj_lines = _iso_obj(g.solid.ravel().astype(float), g)
    corners = _corner_points_obj(g) if volume else None
    dims = [n + 1 for n in g.shape] + ([1] if g.ndim == 2 else [])
    static = {"solid": _vtk_order(g.solid.ravel().astype(np.float32), g),
              "compartment": _vtk_order(sim.comp.label.astype(np.int32), g)}

    bpts, bq = bath_geometry(sim, times)
    write_vtp(os.path.join(outdir, f"{prefix}_bath.vtp"), bpts, polys=bq)

    series = {k: [] for k in ("object", "liquid", "air", "film", "volume")}
    film = getattr(sim, "film", None)
    if film is not None and hist.snap_film:
        fc = film.c
        if g.ndim == 2:
            f_lines = [np.asarray(e) for e in fc.elems]
            f_polys = None
        else:
            f_lines, f_polys = None, fc.elems
    for i, t in enumerate(times):
        f = frame_fields(sim, hist, t)
        tag = f"{i:05d}"
        if surfaces:
            fn = f"{sub}/object_{tag}.vtp"
            write_vtp(os.path.join(outdir, fn), to_world(obj_pts, motion, t),
                      polys=obj_faces, lines=obj_lines)
            series["object"].append((t, fn))
            up, _ = motion.frame(t)
            h = sim.X @ up
            sim._vis_e = 0.5 * g.dx * np.abs(up).sum()
            e = sim._vis_e
            for name in ("liquid", "air"):
                if not flat_surfaces:
                    fld = f["retained" if name == "liquid" else "air_below"]
                elif name == "liquid":
                    fld = planar_fraction(sim, f["retained"], h)
                else:
                    # all enclosed air, flattened, then cut smoothly by the
                    # bath plane (only air below the bath surface is shown)
                    _, zb = motion.frame(t)
                    air = np.where(sim.fl & ~hist.snap_atm[t],
                                   1.0 - f["liquid"], 0.0)
                    fld = planar_fraction(sim, air, -h) * \
                        np.clip((zb - h + e) / (2 * e), 0, 1)
                pts, faces, lines = _iso_obj(_dilate_into_solid(fld, g), g)
                fn = f"{sub}/{name}_{tag}.vtp"
                write_vtp(os.path.join(outdir, fn), to_world(pts, motion, t),
                          polys=faces, lines=lines)
                series[name].append((t, fn))
        if film is not None and t in hist.snap_film:
            fn = f"{sub}/film_{tag}.vtp"
            write_vtp(os.path.join(outdir, fn), to_world(fc.verts, motion, t),
                      polys=f_polys, lines=f_lines,
                      cell_data={"film_um": hist.snap_film[t] * 1e6})
            series["film"].append((t, fn))
        if volume and i % max(1, volume_every) == 0:
            cd = {k: _vtk_order(v.astype(np.float32), g) for k, v in f.items()}
            cd.update(static)
            fn = f"{sub}/volume_{tag}.vts"
            write_vts(os.path.join(outdir, fn), to_world(corners, motion, t),
                      dims, cd)
            series["volume"].append((t, fn))
    files = {"bath": f"{prefix}_bath.vtp"}
    for k, entries in series.items():
        if entries:
            write_pvd(os.path.join(outdir, f"{prefix}_{k}.pvd"), entries)
            files[k] = f"{prefix}_{k}.pvd"
    _write_paraview_script(outdir, prefix, files)
    return files


def _write_paraview_script(outdir, prefix, files):
    """A pvpython/ParaView 'Run Script' helper that loads and styles all."""
    lines = [
        '"""Open in ParaView: View > Python Shell > Run Script (or pvpython)."""',
        "import os",
        "from paraview.simple import *",
        "d = os.path.dirname(os.path.abspath(__file__))",
        "view = GetActiveViewOrCreate('RenderView')",
        "def load(name, color, opacity=1.0):",
        "    fn = os.path.join(d, name)",
        "    r = PVDReader(FileName=fn) if fn.endswith('.pvd') else XMLPolyDataReader(FileName=[fn])",
        "    s = Show(r, view)",
        "    ColorBy(s, None)",
        "    s.AmbientColor = color",
        "    s.DiffuseColor = color",
        "    s.Opacity = opacity",
        "    return r",
    ]
    styles = {"bath": ("[0.55, 0.75, 0.95]", 0.25),
              "object": ("[0.75, 0.75, 0.78]", 0.45),
              "liquid": ("[0.10, 0.35, 0.80]", 1.0),
              "air": ("[1.00, 0.60, 0.10]", 1.0)}
    for k in ("bath", "object", "liquid", "air"):
        if k in files:
            c, o = styles[k]
            lines.append(f"load('{files[k]}', {c}, {o})")
    if "film" in files:
        lines += [f"fr = PVDReader(FileName=os.path.join(d, '{files['film']}'))",
                  "fs = Show(fr, view)",
                  "ColorBy(fs, ('CELLS', 'film_um'))",
                  "fs.LookupTable.RescaleTransferFunction(0.0, 60.0)",
                  "fs.SetScalarBarVisibility(view, True)"]
    lines += ["GetAnimationScene().UpdateAnimationUsingDataTimeSteps()",
              "view.ResetCamera()", "Render()"]
    with open(os.path.join(outdir, f"{prefix}_open_in_paraview.py"), "w") as f:
        f.write("\n".join(lines) + "\n")


# ------------------------------------------------------------- 2D animation

def animate_2d(sim, hist, path, fps=15, times=None, dpi=100, width=6.0,
               title=None, film_max_um=60.0):
    """GIF / MP4 of a 2D run in the world frame (object moves, bath fixed).

    Colours: bath light blue, liquid that is not bath (retained / carried /
    inside compartments) dark blue, air below the bath surface orange,
    object dark grey. With ``film=True`` runs the wall film is drawn as a
    coloured line along the walls (0..``film_max_um`` micrometres). One frame
    per stored snapshot; with snapshots every
    ``dt_s`` seconds, ``fps = 1/dt_s`` plays in real time.
    """
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib import animation
    from matplotlib.patches import Polygon

    g = sim.grid
    if g.ndim != 2:
        raise ValueError("animate_2d needs a 2D simulation; use render3d for 3D")
    times = sorted(hist.snapshots) if times is None else list(times)
    motion = sim.motion
    corners = _corner_points_obj(g)[:, :2]                 # VTK order
    nx, ny = g.shape
    col = {"air": np.array([1.0, 1.0, 1.0]),
           "bath": np.array([0.62, 0.79, 0.95]),
           "ret": np.array([0.10, 0.33, 0.75]),
           "trap": np.array([1.0, 0.62, 0.15]),
           "solid": np.array([0.2, 0.2, 0.22])}

    def colours(t):
        f = frame_fields(sim, hist, t)
        L, B = f["liquid"], f["bath"] > 0.5
        air_b = f["air_below"]
        c = np.empty((g.ncells, 4))
        c[:, 3] = 1.0
        rgb = (col["air"] * (1 - L)[:, None] + col["ret"] * L[:, None])
        rgb[B] = (col["bath"] * L[B, None] + col["trap"] * (1 - L[B])[:, None])
        m = (~B) & (air_b > 0)
        rgb[m] = (col["trap"] * air_b[m, None] + col["ret"] * (1 - air_b[m])[:, None])
        rgb[g.solid.ravel()] = col["solid"]
        c[:, :3] = rgb
        # outside the object and above the bath: transparent (show background)
        c[(~g.solid.ravel()) & (L < 1e-6) & (air_b <= 0), 3] = 0.0
        return c.reshape(nx, ny, 4).transpose(1, 0, 2)

    lo, hi = world_bounds(sim, times)
    pad = 0.05 * (hi - lo).max()
    bpts, _ = bath_geometry(sim, times)
    xl = (lo[0] - pad, hi[0] + pad)
    yl = (min(lo[1], bpts[:, 1].min()) - pad, hi[1] + pad)
    aspect = (yl[1] - yl[0]) / (xl[1] - xl[0])
    head = (1.0 if title else 0.7) + (0.2 if getattr(sim, "film", None) is not None else 0.0)
    fig = plt.figure(figsize=(width, width * aspect + head))
    ax = fig.add_axes([0.02, 0.02, 0.96, (width * aspect) / (width * aspect + head) - 0.03])
    ax.set_xlim(*xl)
    ax.set_ylim(*yl)
    ax.set_aspect("equal")
    ax.set_facecolor("white")
    ax.add_patch(Polygon(bpts[:, :2], closed=True, fc=col["bath"], ec="none",
                         zorder=0))
    ax.axhline(motion.bath_level, color="#3b78c2", lw=1, zorder=0.5)
    W = to_world(corners, motion, times[0])
    X = W[:, 0].reshape(nx + 1, ny + 1, order="F").T
    Y = W[:, 1].reshape(nx + 1, ny + 1, order="F").T
    mesh = ax.pcolormesh(X, Y, colours(times[0]), shading="flat", zorder=1,
                         rasterized=True)
    film = getattr(sim, "film", None)
    fcoll = None
    if film is not None and hist.snap_film:
        from matplotlib.collections import LineCollection
        from matplotlib.colors import Normalize
        cmap_f = plt.get_cmap("plasma")
        norm_f = Normalize(0.0, film_max_um)
        fverts = film.c.verts
        felems = film.c.elems

        def film_segments(t):
            W = to_world(fverts, motion, t)[:, :2]
            return W[felems]

        def film_colours(t):
            hh = np.asarray(hist.snap_film.get(t, np.zeros(len(felems)))) * 1e6
            col_ = cmap_f(norm_f(hh))
            col_[hh < 0.05, 3] = 0.0
            return col_

        fcoll = LineCollection(film_segments(times[0]), colors=film_colours(times[0]),
                               linewidths=2.2, zorder=3)
        ax.add_collection(fcoll)
        cax = fig.add_axes([0.70, 0.06, 0.25, 0.018])
        cb = fig.colorbar(plt.cm.ScalarMappable(norm_f, cmap_f), cax=cax,
                          orientation="horizontal")
        cb.set_label("wall film [µm]", fontsize=8)
        cb.ax.tick_params(labelsize=7)
    a = hist.arrays()
    top = 1.0 - 0.08 / (width * aspect + head)
    if title:
        fig.text(0.02, top, title, va="top", ha="left", fontsize=10,
                 weight="bold")
        top -= 0.3 / (width * aspect + head)
    txt = fig.text(0.02, top, "", va="top", fontsize=9, family="monospace")
    ax.set_xticks([])
    ax.set_yticks([])

    def update(i):
        t = times[i]
        W = to_world(corners, motion, t)
        X = W[:, 0].reshape(nx + 1, ny + 1, order="F").T
        Y = W[:, 1].reshape(nx + 1, ny + 1, order="F").T
        # pcolormesh coordinates cannot be moved in place for all backends:
        # replace the mesh
        nonlocal mesh
        mesh.remove()
        mesh = ax.pcolormesh(X, Y, colours(t), shading="flat", zorder=1,
                             rasterized=True)
        k = int(np.argmin(np.abs(a["t"] - t)))
        msg = (f"t = {t:6.2f} s   liquid above bath "
               f"{a['liquid_above_bath'][k]*1e3:7.2f} l/m\n"
               f"{'':13s}air below bath    "
               f"{a['air_below_bath'][k]*1e3:7.2f} l/m")
        if fcoll is not None:
            fcoll.set_segments(film_segments(t))
            fcoll.set_color(film_colours(t))
            msg += (f"\n{'':13s}wall film         "
                    f"{a['film_volume'][k]*1e3:7.2f} l/m   drops {a['n_drips'][k]}")
        txt.set_text(msg)
        return mesh, txt

    anim = animation.FuncAnimation(fig, update, frames=len(times), blit=False)
    if path.endswith(".gif"):
        anim.save(path, writer=animation.PillowWriter(fps=fps), dpi=dpi)
    else:
        anim.save(path, writer=animation.FFMpegWriter(fps=fps, bitrate=2400),
                  dpi=dpi)
    plt.close(fig)
    return path
