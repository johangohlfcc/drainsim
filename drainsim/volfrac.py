"""Side-aware cut-cell volume fractions.

In the voxel model a cell touched by the surface is *closed* (solid) and a
fluid cell counts as a full ``dx^3``. Every sheet therefore removes about one
cell of fluid volume along each of its faces, and the volume error only
decreases linearly with dx (for the door's corner pocket at 45°: -14 % at
2 mm, -9 % at 1.5 mm).

This module gives every fluid cell an *effective volume*:

    v_eff(c) = dx^3 + fluid volume inside neighbouring cut cells that lies
                      on c's side of the wall

1. Every cut cell (solid cell with a fluid face neighbour) is split into
   ``k^3`` sub-cells. The triangles are sampled again at spacing
   0.45 * dx/k, so sub-cells touched by the surface are solid at the fine
   level (leak-free for face connectivity, like the coarse voxelisation).
2. The fluid sub-cells are labelled by a breadth-first search that starts
   from the sub-cells touching a coarse fluid cell (label = that cell) and
   spreads through fluid sub-cells only - within the cut cell and into
   neighbouring cut cells. Each fluid sub-cell thus goes to the nearest
   coarse fluid cell *on the same side of the wall* (geodesic, not
   Euclidean, so it never jumps through a sheet).
3. Each sub-cell's volume (dx/k)^3 is credited to its label. Sub-cells that
   no fluid cell can reach (sealed slivers) are dropped.
4. The surface itself runs through the marked (solid) sub-cells, so each of
   them is shared equally among the labels of its fluid face neighbours
   (half to each side of a sheet). Sub-cells deeper inside a thick solid
   have no fluid neighbour and stay solid.

In this mode (``effective_volumes``, ``Simulation(subcell_connect=False)``)
connectivity is unchanged: cut cells stay closed, as in IBOFlow. Only the
volumes - and so the level/volume relation of every pool, air pocket and
compartment - become accurate to about the fine resolution dx/k.

The default since v4 is ``cut_cell_graph`` (end of this module): the cut
cells are refined into their fluid sub-cells, which become nodes of the
graph themselves, so connectivity (slots, gaps, sheet edges, sills) is
resolved to dx/k as well.

Requires the surface triangles (``Grid.triangles``, set by
``Grid.from_mesh``). Without them all fluid cells keep ``dx^3``.
"""
from __future__ import annotations

import numpy as np
from numba import njit

from .grid import Grid, sample_triangles


@njit(cache=True)
def _bfs_label(shape, k, cut_of, cut_cells, fine_solid, fluid, lab, queue, conv,
               piece):
    """Label fluid sub-cells of cut cells with the nearest coarse fluid cell
    reachable through fluid sub-cells. ``lab`` (preset to -1) receives the
    label per sub-cell; ``queue`` is work space of the same size. Cut cells
    with ``conv[ci]`` are nodes themselves: the sub-cells of their node piece
    (``piece``) get their own index and seed the search; their other fluid
    sub-cells are labelled like those of closed cut cells."""
    nx, ny, nz = shape
    k3 = k * k * k
    qh = 0
    qt = 0
    sx = ny * nz
    sy = nz
    for ci in range(cut_cells.shape[0]):
        if not conv[ci]:
            continue
        for s_ in range(k3):
            f = ci * k3 + s_
            if piece[f]:
                lab[f] = cut_cells[ci]
                queue[qt] = f
                qt += 1
    # seeds: fluid sub-cells on a face shared with a coarse fluid cell
    for ci in range(cut_cells.shape[0]):
        c = cut_cells[ci]
        cx = c // sx
        cy = (c // sy) % ny
        cz = c % nz
        for a in range(k):
            for b in range(k):
                for d in range(k):
                    f = ci * k3 + (a * k + b) * k + d
                    if fine_solid[f] or lab[f] >= 0:
                        continue
                    best = -1
                    # six directions, only when on the cell boundary
                    for dirn in range(6):
                        ax = dirn // 2
                        sg = -1 if dirn % 2 == 0 else 1
                        sub = a if ax == 0 else (b if ax == 1 else d)
                        if (sg < 0 and sub != 0) or (sg > 0 and sub != k - 1):
                            continue
                        ncx, ncy, ncz = cx, cy, cz
                        if ax == 0:
                            ncx += sg
                        elif ax == 1:
                            ncy += sg
                        else:
                            ncz += sg
                        if ncx < 0 or ncy < 0 or ncz < 0 or ncx >= nx \
                                or ncy >= ny or ncz >= nz:
                            continue
                        nc = ncx * sx + ncy * sy + ncz
                        if fluid[nc]:
                            best = nc
                            break
                    if best >= 0:
                        lab[f] = best
                        queue[qt] = f
                        qt += 1
    # breadth-first spread through fluid sub-cells
    while qh < qt:
        f = queue[qh]
        qh += 1
        ci = f // k3
        r = f % k3
        a = r // (k * k)
        b = (r // k) % k
        d = r % k
        c = cut_cells[ci]
        cx = c // sx
        cy = (c // sy) % ny
        cz = c % nz
        for dirn in range(6):
            ax = dirn // 2
            sg = -1 if dirn % 2 == 0 else 1
            na, nb, nd = a, b, d
            ncx, ncy, ncz = cx, cy, cz
            if ax == 0:
                na += sg
                if na < 0:
                    na = k - 1
                    ncx -= 1
                elif na >= k:
                    na = 0
                    ncx += 1
            elif ax == 1:
                nb += sg
                if nb < 0:
                    nb = k - 1
                    ncy -= 1
                elif nb >= k:
                    nb = 0
                    ncy += 1
            else:
                nd += sg
                if nd < 0:
                    nd = k - 1
                    ncz -= 1
                elif nd >= k:
                    nd = 0
                    ncz += 1
            if ncx < 0 or ncy < 0 or ncz < 0 or ncx >= nx or ncy >= ny \
                    or ncz >= nz:
                continue
            nc = ncx * sx + ncy * sy + ncz
            nci = cut_of[nc]
            if nci < 0:
                continue                     # fluid cell (seed) or deep solid
            g = nci * k3 + (na * k + nb) * k + nd
            if fine_solid[g] or lab[g] >= 0:
                continue
            lab[g] = lab[f]
            queue[qt] = g
            qt += 1
    return 0


@njit(cache=True)
def _split_solid(shape, k, cut_of, cut_cells, fine_solid, fluid, lab, credit):
    """Share each surface sub-cell (fine solid with a fluid face neighbour)
    equally among the labels of its fluid face neighbours: the surface runs
    through the marked sub-cell, so on average half of it is fluid on each
    side. ``credit`` (per coarse cell, in sub-cell units) is incremented."""
    nx, ny, nz = shape
    k3 = k * k * k
    sx = ny * nz
    sy = nz
    votes = np.empty(6, np.int64)
    for ci in range(cut_cells.shape[0]):
        c = cut_cells[ci]
        cx = c // sx
        cy = (c // sy) % ny
        cz = c % nz
        for a in range(k):
            for b in range(k):
                for d in range(k):
                    f = ci * k3 + (a * k + b) * k + d
                    if not fine_solid[f]:
                        continue
                    nv = 0
                    for dirn in range(6):
                        ax = dirn // 2
                        sg = -1 if dirn % 2 == 0 else 1
                        na, nb, nd = a, b, d
                        ncx, ncy, ncz = cx, cy, cz
                        if ax == 0:
                            na += sg
                            if na < 0:
                                na = k - 1
                                ncx -= 1
                            elif na >= k:
                                na = 0
                                ncx += 1
                        elif ax == 1:
                            nb += sg
                            if nb < 0:
                                nb = k - 1
                                ncy -= 1
                            elif nb >= k:
                                nb = 0
                                ncy += 1
                        else:
                            nd += sg
                            if nd < 0:
                                nd = k - 1
                                ncz -= 1
                            elif nd >= k:
                                nd = 0
                                ncz += 1
                        if ncx < 0 or ncy < 0 or ncz < 0 or ncx >= nx \
                                or ncy >= ny or ncz >= nz:
                            continue
                        nc = ncx * sx + ncy * sy + ncz
                        if fluid[nc]:
                            votes[nv] = nc
                            nv += 1
                            continue
                        nci = cut_of[nc]
                        if nci < 0:
                            continue
                        g = nci * k3 + (na * k + nb) * k + nd
                        if not fine_solid[g] and lab[g] >= 0:
                            votes[nv] = lab[g]
                            nv += 1
                    for q in range(nv):
                        credit[votes[q]] += 1.0 / nv
    return 0


def _cut_subcells(grid: Grid, k: int):
    """Cut cells (solid cells with a fluid face neighbour), their index map and
    the fine solid mask of their k^3 sub-cells (surface resampled at dx/k)."""
    n = grid.ncells
    fluid = grid.fluid
    solid = grid.solid
    cut = np.zeros(grid.shape, bool)
    for d in range(3):
        s0 = [slice(None)] * 3
        s1 = [slice(None)] * 3
        s0[d] = slice(None, -1)
        s1[d] = slice(1, None)
        cut[tuple(s0)] |= solid[tuple(s0)] & fluid[tuple(s1)]
        cut[tuple(s1)] |= solid[tuple(s1)] & fluid[tuple(s0)]
    from .octree import REFINE_EDGE_NEIGHBOURS
    if REFINE_EDGE_NEIGHBOURS:
        # also solid cells touching the fluid along an edge or a corner
        # (6.3, as in octree._subcell_graph)
        from scipy import ndimage
        cut |= solid & ndimage.binary_dilation(fluid, np.ones((3, 3, 3), bool))
    cut_cells = np.flatnonzero(cut.ravel()).astype(np.int64)
    cut_of = -np.ones(n, np.int64)
    cut_of[cut_cells] = np.arange(cut_cells.size)
    k3 = k ** 3
    fine_solid = np.zeros(cut_cells.size * k3, np.bool_)
    hfine = grid.dx / k
    shape = np.array(grid.shape)
    for P in sample_triangles(grid.triangles, hfine):
        # coarse cell exactly as in the coarse voxelisation (_mark_surface),
        # then the sub-cell inside it: a sample on a coarse face must mark a
        # sub-cell of the same cell it made solid, or the wall leaks
        q = (P - grid.origin) / grid.dx
        ci = np.floor(q).astype(np.int64)
        ok = np.all((ci >= 0) & (ci < shape), axis=1)
        q, ci = q[ok], ci[ok]
        c = (ci[:, 0] * shape[1] + ci[:, 1]) * shape[2] + ci[:, 2]
        cc = cut_of[c]
        # a sample (within rounding) on a coarse face may have gone to the
        # other cell in the coarse pass: give it to the cut cell across
        fr = q - ci
        for d in range(3):
            for side, step in ((fr[:, d] < 1e-6, -1), (fr[:, d] > 1 - 1e-6, 1)):
                m = (cc < 0) & side
                if not m.any():
                    continue
                cj = ci[m].copy()
                cj[:, d] += step
                okj = (cj[:, d] >= 0) & (cj[:, d] < shape[d])
                cn = (cj[:, 0] * shape[1] + cj[:, 1]) * shape[2] + cj[:, 2]
                cn = np.where(okj, cn, 0)
                hit = okj & (cut_of[cn] >= 0)
                mi = np.flatnonzero(m)[hit]
                ci[mi] = cj[hit]
                cc[mi] = cut_of[cn[hit]]
                fr[mi, d] = 1.0 - 1e-9 if step < 0 else 0.0
        keep = cc >= 0
        sub = np.clip(np.floor(fr[keep] * k).astype(np.int64), 0, k - 1)
        fine_solid[cc[keep] * k3 + (sub[:, 0] * k + sub[:, 1]) * k + sub[:, 2]] = True
    return cut_cells, cut_of, fine_solid


def subcell_labels(grid: Grid, k: int = 4, conv_fn=None):
    """Sub-cell data of the cut cells: which sub-cells are solid and which
    coarse fluid cell each fluid sub-cell belongs to (same side of the wall).

    Returns dict(k, cut_cells, cut_of, fine_solid, label) with
    ``cut_of[c]`` = index of cell c among the cut cells (-1 if not cut) and
    sub-cell (a, b, d) of cut cell ``ci`` at ``ci*k^3 + (a*k + b)*k + d``.
    ``label`` is -1 for solid or unreachable sub-cells. None if the grid has
    no surface triangles (or is 2D, or k < 2).
    """
    n = grid.ncells
    if grid.triangles is None or grid.ndim != 3 or k < 2:
        return None
    fluid = grid.fluid
    cut_cells, cut_of, fine_solid = _cut_subcells(grid, k)
    conv, piece = (None, None) if conv_fn is None else \
        conv_fn(cut_cells, cut_of, fine_solid)
    nf = fine_solid.size
    it = np.int32 if max(nf, n) < 2 ** 31 - 1 else np.int64   # halves memory
    lab = np.full(nf, -1, it)
    queue = np.empty(nf, it)
    _bfs_label(np.array(grid.shape, np.int64), int(k), cut_of, cut_cells,
               fine_solid, fluid.ravel(), lab, queue,
               np.zeros(cut_cells.size, np.bool_) if conv is None else conv,
               np.zeros(1, np.bool_) if piece is None else piece)
    del queue
    return dict(k=int(k), cut_cells=cut_cells, cut_of=cut_of,
                fine_solid=fine_solid, label=lab, conv=conv, piece=piece)


def effective_volumes(grid: Grid, k: int = 4, return_stats: bool = False,
                      split_surface: bool = True):
    """Per-cell effective fluid volume (0 for solid cells), see module doc.

    split_surface : share the surface sub-cells themselves between the sides
                    they touch (removes the bias of half a sub-cell per face).
    """
    n = grid.ncells
    vol = np.where(grid.fluid.ravel(), grid.cell_volume, 0.0)
    sd = subcell_labels(grid, k)
    if sd is None:
        return (vol, {}) if return_stats else vol
    cut_cells, cut_of = sd["cut_cells"], sd["cut_of"]
    fine_solid, lab = sd["fine_solid"], sd["label"]
    hfine = grid.dx / k
    got = lab >= 0
    vsub = hfine ** 3
    credit = np.bincount(lab[got], minlength=n).astype(np.float64)
    nfluid_sub = credit.sum()
    if split_surface:
        _split_solid(np.array(grid.shape, np.int64), int(k), cut_of, cut_cells,
                     fine_solid, grid.fluid.ravel(), lab, credit)
    vol += credit * vsub
    if not return_stats:
        return vol
    stats = dict(cut_cells=int(cut_cells.size), subcells=int(fine_solid.size),
                 fine_solid=float(fine_solid.mean()),
                 credited=float(nfluid_sub * vsub),
                 surface_split=float((credit.sum() - nfluid_sub) * vsub),
                 dropped=float((~fine_solid & ~got).sum() * vsub))
    return vol, stats


# ----------------------------------------------------- sub-cell connectivity
#
# Cut cells are closed in the coarse voxel model, so any opening narrower
# than about two cells (a slot, the gap beside a tab, the free edge of a
# sheet) is shut, or its sill is moved up to the next open cell.
# ``cut_cell_graph`` refines the cut cells instead: every fluid sub-cell of a
# cut cell becomes a node of its own (a two-level grid, fine only where the
# surface is). Sub-cells touched by the surface are solid at the fine level,
# so a sheet is still leak-free, but openings, sills and free sheet edges
# are resolved to dx/k. Fine nodes link to their fluid face neighbours: fine
# ones in the same or a neighbouring cut cell, and coarse fluid cells across
# a coarse face. They get indices ncells, ncells + 1, ...

@njit(cache=True)
def _fine_links(shape, k, cut_of, cut_cells, fid, fluid, ea, eb, count_only):
    """Links of the fine nodes: within a cut cell (+ directions), into the
    neighbouring cut cell (+ directions of the coarse face) and to a coarse
    fluid cell across a coarse face. Returns the number of links."""
    nx, ny, nz = shape
    k3 = k * k * k
    sx = ny * nz
    sy = nz
    ne = 0
    for ci in range(cut_cells.shape[0]):
        c = cut_cells[ci]
        cx = c // sx
        cy = (c // sy) % ny
        cz = c % nz
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
                ncx, ncy, ncz = cx, cy, cz
                cross = False
                if ax == 0:
                    na += sg
                    if na < 0 or na >= k:
                        na = k - 1 if na < 0 else 0
                        ncx += sg
                        cross = True
                elif ax == 1:
                    nb_ += sg
                    if nb_ < 0 or nb_ >= k:
                        nb_ = k - 1 if nb_ < 0 else 0
                        ncy += sg
                        cross = True
                else:
                    nd += sg
                    if nd < 0 or nd >= k:
                        nd = k - 1 if nd < 0 else 0
                        ncz += sg
                        cross = True
                g = (na * k + nb_) * k + nd
                if not cross:
                    if sg < 0:
                        continue                  # visited from the other one
                    b0 = fid[ci * k3 + g]
                    if b0 < 0:
                        continue
                else:
                    if ncx < 0 or ncy < 0 or ncz < 0 or ncx >= nx or ncy >= ny \
                            or ncz >= nz:
                        continue
                    nc = ncx * sx + ncy * sy + ncz
                    if fluid[nc]:
                        b0 = nc
                    else:
                        nci = cut_of[nc]
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
def _split_surface_fine(shape, k, cut_of, cut_cells, fine_solid, fid, fluid, credit,
                        mode, cnt, vptr, vid):
    """Share each surface sub-cell (fine solid with a fluid face neighbour)
    equally among its fluid face neighbours (fine nodes, or coarse fluid
    cells across a coarse face): half to each side of a sheet. ``credit`` is
    per node, in sub-cell units. mode 1 / 2: count / write the votes
    instead (for ``sidesplit.split_by_sight``), indexed like fine_solid."""
    nx, ny, nz = shape
    k3 = k * k * k
    sx = ny * nz
    sy = nz
    votes = np.empty(6, np.int64)
    for ci in range(cut_cells.shape[0]):
        c = cut_cells[ci]
        cx = c // sx
        cy = (c // sy) % ny
        cz = c % nz
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
                ncx, ncy, ncz = cx, cy, cz
                if ax == 0:
                    na += sg
                    if na < 0 or na >= k:
                        na = k - 1 if na < 0 else 0
                        ncx += sg
                elif ax == 1:
                    nb_ += sg
                    if nb_ < 0 or nb_ >= k:
                        nb_ = k - 1 if nb_ < 0 else 0
                        ncy += sg
                else:
                    nd += sg
                    if nd < 0 or nd >= k:
                        nd = k - 1 if nd < 0 else 0
                        ncz += sg
                if ncx < 0 or ncy < 0 or ncz < 0 or ncx >= nx or ncy >= ny or ncz >= nz:
                    continue
                nc = ncx * sx + ncy * sy + ncz
                g = (na * k + nb_) * k + nd
                if nc == c:
                    b0 = fid[ci * k3 + g]
                elif fluid[nc]:
                    b0 = nc
                else:
                    nci = cut_of[nc]
                    b0 = -1 if nci < 0 else fid[nci * k3 + g]
                if b0 >= 0:
                    votes[nv] = b0
                    nv += 1
            if mode == 0:
                for q in range(nv):
                    credit[votes[q]] += 1.0 / nv
            elif mode == 1:
                cnt[ci * k3 + s0] = nv
            else:
                w0 = vptr[ci * k3 + s0]
                for q in range(nv):
                    vid[w0 + q] = votes[q]
    return 0


def cut_cell_graph(grid: Grid, k: int = 4, split_surface=True, share="sight"):
    """Two-level node graph: coarse fluid cells plus the fluid sub-cells of
    the cut cells (see the comment above).

    Returns None (no triangles / 2D / k < 2) or a dict with
      nextra  number of fine nodes (indices ncells .. ncells+nextra-1),
      host    (N,) grid cell of every node,
      fine    bool (N,), node is a sub-cell,
      active  bool (N,), fluid node connected to a coarse fluid cell (sealed
              fine voids are inactive),
      vol     volume per node (sub-cell volume plus surface share),
      X       position per node,
      edges   (m, 2) links of the fine nodes (coarse faces not included),
      stats   counts.
    Does not modify the grid.
    """
    n = grid.ncells
    if grid.triangles is None or grid.ndim != 3 or k < 2:
        return None
    cut_cells, cut_of, fine_solid = _cut_subcells(grid, k)
    ncut = cut_cells.size
    k3 = k ** 3
    shape = np.array(grid.shape, np.int64)
    fluid = grid.fluid.ravel()
    fid = np.where(fine_solid, -1, 0).astype(np.int64)
    fl_sub = ~fine_solid
    nfine = int(fl_sub.sum())
    fid[fl_sub] = n + np.arange(nfine)
    N = n + nfine
    m = _fine_links(shape, int(k), cut_of, cut_cells, fid, fluid,
                    np.zeros(1, np.int64), np.zeros(1, np.int64), True)
    ea = np.empty(m, np.int64)
    eb = np.empty(m, np.int64)
    _fine_links(shape, int(k), cut_of, cut_cells, fid, fluid, ea, eb, False)
    edges = np.column_stack([ea, eb])
    # positions and volumes
    hf = grid.dx / k
    X = np.empty((N, 3))
    X[:n] = grid.centers()
    ci, sub = np.divmod(np.flatnonzero(fl_sub), k3)
    corner = grid.origin + np.stack(np.unravel_index(cut_cells[ci], grid.shape), 1) * grid.dx
    abd = np.stack([sub // (k * k), (sub // k) % k, sub % k], 1)
    X[n:] = corner + (abd + 0.5) * hf
    host = np.concatenate([np.arange(n), cut_cells[ci]])
    fine = np.zeros(N, bool)
    fine[n:] = True
    credit = np.zeros(N)
    credit[n:] = 1.0
    metal = 0.0
    z = np.zeros(1, np.int64)
    if split_surface and share == "sight":
        # surface sub-cells shared by line of sight: the metal is dropped
        from .sidesplit import split_by_sight, surface_items, triangles_per_cell
        cnt = np.zeros(fine_solid.size, np.int64)
        _split_surface_fine(shape, int(k), cut_of, cut_cells, fine_solid, fid, fluid, credit,
                            1, cnt, z, z)
        vptr = np.zeros(fine_solid.size + 1, np.int64)
        np.cumsum(cnt, out=vptr[1:])
        vid = np.empty(int(vptr[-1]), np.int64)
        _split_surface_fine(shape, int(k), cut_of, cut_cells, fine_solid, fid, fluid, credit,
                            2, cnt, vptr, vid)
        items, cj, iptr, ivid = surface_items(cnt, vid, fine_solid, lambda j: j // k3)
        sb = items - cj * k3
        cabd = np.stack([sb // (k * k), (sb // k) % k, sb % k], 1)
        clo = grid.origin + np.stack(np.unravel_index(cut_cells[cj], grid.shape), 1) * grid.dx \
            + cabd * hf
        tptr, tidx = triangles_per_cell(grid.triangles, grid.origin, grid.dx, shape, cut_cells)
        ns = np.full(N, hf)
        ns[:n] = grid.dx
        metal = split_by_sight(iptr, ivid, clo, np.full(items.size, hf), cj.astype(np.int64),
                               np.ones(items.size), X, ns, grid.triangles, tptr, tidx,
                               credit) * hf ** 3
    elif split_surface:
        _split_surface_fine(shape, int(k), cut_of, cut_cells, fine_solid, fid, fluid, credit,
                            0, z, z, z)
    vol = np.zeros(N)
    vol[:n][fluid] = grid.cell_volume
    vol += credit * hf ** 3
    # fine nodes sealed from every coarse fluid cell (voids between sheets at
    # the fine level) are inactive
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    base = grid.neighbors()
    bi = np.repeat(np.arange(n), base.shape[1])
    bj = base.ravel()
    okb = bj >= 0
    rows = np.concatenate([bi[okb], ea])
    cols = np.concatenate([bj[okb], eb])
    G = coo_matrix((np.ones(rows.size, np.int8), (rows, cols)), shape=(N, N))
    ncomp, comp = connected_components(G, directed=False)
    reach = np.zeros(ncomp, bool)
    reach[comp[:n][fluid]] = True
    active = np.zeros(N, bool)
    active[:n] = fluid
    active[n:] = reach[comp[n:]]
    vol[~active] = 0.0
    keep = active[ea] & active[eb]
    stats = dict(cut_cells=int(ncut), fine_nodes=nfine,
                 fine_active=int(active[n:].sum()), links=int(keep.sum()),
                 metal_share=float(metal))
    return dict(nextra=nfine, host=host, fine=fine, active=active, vol=vol, X=X,
                edges=edges[keep], stats=stats)
