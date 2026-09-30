"""Transient geometric drainage / accessibility simulation.

State: liquid volume fraction ``L`` in every fluid cell (object frame).

Each time step
--------------
1. **Pose** - new up-vector and bath level in the object frame.
2. **Throats** - liquid exchange between compartments through each throat
   with the orifice law (``physics.ThroatModel``), using the pool levels on
   both sides, capillary hold-up and venting. Outflow is taken from the top of
   the source pool; inflow is released at the throat on the receiving side.
3. **Equilibrate** every compartment for the new orientation:
   * exterior compartment (0): ``B`` = bath (cells below the bath level
     connected to the domain boundary below it); ``A`` = atmosphere
     (above, connected to the boundary above it).
     - liquid fill-spill-merge with sinks ``B`` + boundary: liquid that is
       not bath is routed downhill, cavity to cavity, until it rests or
       reaches the bath;
     - air fill-spill-merge (height reversed) with sinks ``A`` + boundary:
       air under the bath rises until it escapes or is trapped in a pocket.
   * internal compartments: liquid fill-spill-merge without sinks (volume
     conserved, only changed by throat flows).

Setting ``split=False`` (no throats: every opening instantaneous) gives the
quasi-static equilibrium model; additionally ``spill_routing=False`` makes
liquid that escapes a cavity vanish, as in the original connectivity method.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import numba
from numba import njit
from scipy import ndimage

from . import compartments as cp
from . import par, stepk
from .fsm import fill_spill, to_csr, FSMCache, box_shape, LINEAR, cell_cdf_vec
from .grid import Grid
from .octree import Octree
from .physics import Fluid, ThroatModel
from .volfrac import cut_cell_graph, effective_volumes


def _label_bodies(mask, comp, nbr):
    """Connected bodies of ``mask`` (see ``par.label_bodies``; same result
    as the serial ``_label_bodies_csr``)."""
    ptr, idx = to_csr(nbr)
    return par.label_bodies(mask, comp, ptr, idx)


@njit(cache=True, nogil=True)
def _label_bodies_csr(mask, comp, ptr, idx):
    N = mask.shape[0]
    uf = np.arange(N)
    for c in range(N):
        if not mask[c]:
            continue
        for jj in range(ptr[c], ptr[c + 1]):
            n = idx[jj]
            if n < 0 or n < c or not mask[n] or comp[n] != comp[c]:
                continue
            a = c
            while uf[a] != a:
                a = uf[a]
            b = n
            while uf[b] != b:
                b = uf[b]
            if a != b:
                uf[max(a, b)] = min(a, b)
    body = -np.ones(N, np.int64)
    nb = 0
    rootid = -np.ones(N, np.int64)
    for c in range(N):
        if not mask[c]:
            continue
        a = c
        while uf[a] != a:
            a = uf[a]
        if rootid[a] < 0:
            rootid[a] = nb
            nb += 1
        body[c] = rootid[a]
    return body, nb


@njit(cache=True)
def _priority_flood(lab, pri, ptr, idx, active):
    """Watershed by flooding on the node graph: unlabelled active nodes take
    the label of the neighbour that reaches them first, visiting nodes far
    from the walls (high ``pri``) first. Boundaries between labels then lie
    at the narrowest cross-sections, as in the whole-cell segmentation."""
    N = lab.shape[0]
    hk = np.empty(N, np.float64)
    hv = np.empty(N, np.int64)
    cand = -np.ones(N, np.int64)
    queued = np.zeros(N, np.bool_)
    size = 0
    for c in range(N):
        if lab[c] < 0:
            continue
        for jj in range(ptr[c], ptr[c + 1]):
            n = idx[jj]
            if n < 0 or lab[n] >= 0 or not active[n] or queued[n]:
                continue
            queued[n] = True
            cand[n] = lab[c]
            # push (-pri, n)
            i = size
            size += 1
            hk[i] = -pri[n]
            hv[i] = n
            while i > 0:
                q = (i - 1) // 2
                if hk[q] < hk[i] or (hk[q] == hk[i] and hv[q] < hv[i]):
                    break
                hk[q], hk[i] = hk[i], hk[q]
                hv[q], hv[i] = hv[i], hv[q]
                i = q
    while size > 0:
        c = hv[0]
        size -= 1
        hk[0] = hk[size]
        hv[0] = hv[size]
        i = 0
        while True:
            l = 2 * i + 1
            if l >= size:
                break
            r = l + 1
            m = l
            if r < size and (hk[r] < hk[l] or (hk[r] == hk[l] and hv[r] < hv[l])):
                m = r
            if hk[i] < hk[m] or (hk[i] == hk[m] and hv[i] < hv[m]):
                break
            hk[m], hk[i] = hk[i], hk[m]
            hv[m], hv[i] = hv[i], hv[m]
            i = m
        lab[c] = cand[c]
        for jj in range(ptr[c], ptr[c + 1]):
            n = idx[jj]
            if n < 0 or lab[n] >= 0 or not active[n] or queued[n]:
                continue
            queued[n] = True
            cand[n] = lab[c]
            i = size
            size += 1
            hk[i] = -pri[n]
            hv[i] = n
            while i > 0:
                q = (i - 1) // 2
                if hk[q] < hk[i] or (hk[q] == hk[i] and hv[q] < hv[i]):
                    break
                hk[q], hk[i] = hk[i], hk[q]
                hv[q], hv[i] = hv[i], hv[q]
                i = q
    return lab


def _node_wall_distance(grid, X, fine, k):
    """Distance from each fine node to the surface (sampled at about dx/k);
    whole cells keep the segmentation's distance field."""
    from .grid import surface_distance
    D = np.zeros(X.shape[0])
    f = np.flatnonzero(fine)
    D[f] = surface_distance(grid.triangles, grid.dx / k, X[f], 3.0 * grid.dx)
    return D


@dataclass
class History:
    t: list = field(default_factory=list)
    bath_level: list = field(default_factory=list)
    liquid_retained: list = field(default_factory=list)     # all non-bath liquid
    liquid_above_bath: list = field(default_factory=list)   # carried out of bath
    liquid_trapped: list = field(default_factory=list)      # held above the surface
    liquid_exterior: list = field(default_factory=list)     # pools in exterior
    liquid_comp: list = field(default_factory=list)         # per compartment
    air_trapped: list = field(default_factory=list)         # air held under the surface
    air_below_bath: list = field(default_factory=list)      # all air below bath
    air_gas: list = field(default_factory=list)   # trapped gas, m^3 at ambient p
    drained: list = field(default_factory=list)             # cumulative to bath
    throat_flow: list = field(default_factory=list)         # per throat a->b
    snapshots: dict = field(default_factory=dict)     # t -> liquid fraction
    snap_bath: dict = field(default_factory=dict)     # t -> bath mask
    snap_atm: dict = field(default_factory=dict)      # t -> atmosphere mask
    snap_film: dict = field(default_factory=dict)     # t -> film thickness
    film_volume: list = field(default_factory=list)   # film on surfaces
    film_pending: list = field(default_factory=list)  # film -> liquid in transit
    drip_volume: list = field(default_factory=list)   # cumulative dripped
    n_drips: list = field(default_factory=list)       # cumulative drop count

    def arrays(self):
        out = {}
        for k, v in self.__dict__.items():
            if not k.startswith("snap"):
                out[k] = np.asarray(v)
        return out


_GL = np.polynomial.legendre.leggauss(48)          # nodes, weights on [-1, 1]


class Simulation:
    def __init__(self, grid: Grid, motion, fluid: Fluid | None = None,
                 throat_model: ThroatModel | None = None,
                 split: bool = True, spill_routing: bool = True,
                 segment_kwargs: dict | None = None,
                 cells_per_step: float = 1.0, dt_max: float = 0.05,
                 dt_min: float = 1e-4, initial_L: np.ndarray | None = None,
                 film=False, min_depth_cells: float = 1.5, holes=None,
                 subcells: int = 4, compressible_air: bool = False,
                 subcell_connect: bool = True, spill_floor: bool | None = None,
                 plugs=None, threads: int = 2, portion: str = "box", narrow=None,
                 suction: bool = True, surface_share: str = "sight"):
        self.grid = grid
        # How the closed sub-cells at the surface are shared among the fluid
        # nodes next to them: "sight" (6.3) by sample points and line of
        # sight, dropping the metal of plates; "equal": equally among the
        # fluid face neighbours (before 6.3; counts the metal as fluid).
        if surface_share not in ("sight", "equal"):
            raise ValueError("surface_share must be 'sight' or 'equal'")
        self.surface_share = surface_share
        # Suction (6.3, from agent A's "submerged system"): the bath is the
        # exterior below the bath level connected to the domain boundary
        # *plus* every exterior node connected to it without passing through
        # the atmosphere. Such a node above the bath level (inside an inverted
        # glass lifted partly out) is held full of bath liquid by the lower
        # pressure (p_atm + rho g (z_bath - z) < p_atm), except where trapped
        # air sits. False: the bath ends at its level and that liquid drains
        # as if air could replace it (before 6.3).
        self.suction = bool(suction)
        # How a node's volume is spread over its vertical extent when a spill
        # level cuts it: "box" exactly as an axis-aligned cube (additive over
        # an octree's levels), "linear" uniformly.
        if portion not in ("box", "linear"):
            raise ValueError("portion must be 'box' or 'linear'")
        self.portion = portion
        # octree only: closed cells with a narrow passage (gap, slot, hole
        # narrower than narrow["width"] cells) get narrow["k"] sub-cells
        # per edge instead of ``subcells`` (True: the defaults)
        self.narrow = narrow
        # Parallelism. With more than one numba thread (NUMBA_NUM_THREADS,
        # default: all cores) every pass of a step runs in parallel inside
        # its kernel, and the passes run one after the other. With one numba
        # thread, ``threads`` = 2 instead runs independent passes (liquid
        # and air fill-spill, bath and atmosphere labelling) in two Python
        # threads (the kernels release the GIL); threads=1: all in sequence.
        # Results do not depend on either setting.
        self.threads = max(1, int(threads))
        self._pool = None
        self.height_bands = True
        # while the pose does not change (a part hanging still), the bath and
        # atmosphere and the fill-spill hierarchies are reused
        self.reuse_static = True
        self._static = False
        self._pose_id = 0              # changes with every new pose
        self._fc_l = FSMCache()
        self._fc_a = FSMCache()
        self.motion = motion
        self.fluid = fluid or Fluid()
        self.tm = throat_model or ThroatModel()
        self.spill_routing = spill_routing
        # depressions / air domes shallower than this many cell heights hold
        # nothing: they are the staircase roughness of the voxel surface
        self.min_depth_cells = float(min_depth_cells)
        self.cells_per_step = cells_per_step
        self.dt_max = dt_max
        self.dt_min = dt_min
        # octree (fine at the surface, coarse in the bulk) or uniform grid
        self.octree = isinstance(grid, Octree)
        if narrow and not self.octree:
            raise ValueError("narrow refinement needs an octree (Octree.from_mesh with "
                             "levels=0 is the uniform grid)")
        if self.octree:
            lab = self._init_octree(grid, split, segment_kwargs, subcells, spill_floor,
                                    holes, plugs)
        else:
            lab = self._init_uniform(grid, split, segment_kwargs, subcells,
                                     subcell_connect, spill_floor, holes, plugs)
        self.comp.label = lab
        self.vg = grid.cell_volume                      # geometric cell volume
        self._vsafe = np.where(self.v > 0, self.v, 1.0)
        # neighbour table for instantaneous (fill-spill) connectivity: faces
        # of throats are removed, so exchange through them is rate limited
        ptr, idx = self.nbr
        for t in self.comp.throats:
            for ca, cb in ((t.cells_a, t.cells_b), (t.cells_b, t.cells_a)):
                if ca.size == 0:
                    continue
                cnt = ptr[ca + 1] - ptr[ca]
                rows = np.repeat(ca, cnt)
                pos = np.repeat(ptr[ca], cnt) + (np.arange(cnt.sum()) -
                                                 np.repeat(np.cumsum(cnt) - cnt, cnt))
                pair = np.isin(idx[pos], cb)
                idx[pos[pair]] = -1
        self.bnd = (self.ograph.boundary if self.octree else
                    grid.boundary_mask().ravel()[self.host]) & self.fl
        # fluid cells the segmentation left unlabelled (tiny sealed voids,
        # e.g. next to a carved hole on a coarse grid) join the exterior
        stray = self.fl & (self.comp.label < 0)
        if stray.any():
            self.comp.label[stray] = 0
        fl0 = self.fl & (self.comp.label >= 0)
        self.comp.volume = np.bincount(self.comp.label[fl0], weights=self.v[fl0],
                                       minlength=self.comp.n)
        self.lab = self.comp.label
        self.ext = self.lab == 0
        self.extent = float(np.max(np.linalg.norm(
            self.X[self.fl] - self.X[self.fl].mean(0), axis=1)))
        # exterior cells at throats, and the cell across the opening
        ext_side, other, owner = [np.zeros(0, int)], [np.zeros(0, int)], [np.zeros(0, int)]
        for i, t in enumerate(self.comp.throats):
            if t.a == 0:
                ext_side.append(t.cells_a)
                other.append(t.cells_b)
                owner.append(np.full(t.cells_a.size, i))
            if t.b == 0:
                ext_side.append(t.cells_b)
                other.append(t.cells_a)
                owner.append(np.full(t.cells_b.size, i))
        self._tx_ext = np.concatenate(ext_side)
        self._tx_int = np.concatenate(other)
        self._tx_own = np.concatenate(owner)
        self._open_at = np.array([getattr(t, "open_at", -np.inf)
                                  for t in self.comp.throats], float)
        # throats the segmentation found at a plugged hole stay shut
        for pl in self.plugs:
            r = 0.5 * pl["diameter"] + grid.dx
            for i, t in enumerate(self.comp.throats):
                if np.linalg.norm(np.asarray(t.centroid) - pl["center"]) <= r:
                    self._open_at[i] = np.inf
        # explicit holes: fluid cells near the hole on each side, used to find
        # the pools on either side even when the level is below the face
        # cells (the true sill can lie below the voxel faces)
        self._near = {}
        for i, t in enumerate(self.comp.throats):
            if t.axis is None:
                continue
            rel = self.X - t.centroid
            r = np.sqrt((rel ** 2).sum(1))
            sv = rel @ t.axis
            sa = np.sign(np.mean((self.X[t.cells_a] - t.centroid) @ t.axis))
            near = self.fl & (r <= 0.5 * t.diameter + 1.5 * grid.dx)
            self._near[i] = tuple(
                self._grow(seed, near & (sv * sg > 0))
                for seed, sg in ((t.cells_a, sa), (t.cells_b, -sa)))
        self.X = np.ascontiguousarray(self.X, np.float64)
        self._throat_csr()
        self.t = 0.0
        self.dt = dt_max
        self.hist = History()
        self.drained_total = 0.0
        up, zb = motion.frame(0.0)
        self._frame(up, zb)
        if initial_L is None:
            self.L = np.zeros(self.N)
            self.L[self.B] = 1.0
        else:
            L0 = np.asarray(initial_L, float)
            self.L = np.zeros(self.N)
            self.L[:L0.size] = L0
            if L0.size < self.N:
                # fine nodes start like the cells they are linked to
                ptr, idx = self.nbr
                src = np.repeat(np.arange(self.N), np.diff(ptr))
                m = self.fine[src] & (idx >= 0)
                for _ in range(2 * self.subcells):
                    val = np.zeros(self.N)
                    np.maximum.at(val, src[m], self.L[idx[m]])
                    grow = self.fine & self.fl & (val > self.L)
                    if not grow.any():
                        break
                    self.L[grow] = val[grow]
        self._pending = np.zeros(self.N)
        # compressible air pockets (exterior): gas amount per cell, in m^3 at
        # ambient pressure, and the pressure each cell's gas is at
        self.compressible_air = bool(compressible_air)
        self.G = np.where(self.fl, (1.0 - self.L) * self.v, 0.0)
        self.P = self._hydrostatic()
        self.film = None
        if film:
            from .film import FilmModel, FilmParams
            self.film = FilmModel(self, film if isinstance(film, FilmParams)
                                  else None)
        self._record(np.zeros(len(self.comp.throats)))

    # ------------------------------------------------------------ geometry setup
    def _init_uniform(self, grid, split, segment_kwargs, subcells, subcell_connect,
                      spill_floor, holes, plugs):
        """Nodes, links and compartments on the uniform grid (with the
        sub-cell graph of the cut cells). Returns the node labels."""
        # explicitly listed holes: opened in the voxel grid (modifies
        # grid.solid) and always treated as throats with their true size
        self.holes = cp.carve_holes(grid, holes) if holes else []
        self.comp = cp.segment(grid, split=split, holes=self.holes,
                               **(segment_kwargs or {}))
        n = grid.ncells
        self.ncells = n
        # effective (side-aware cut-cell) volume of every fluid cell, see
        # volfrac.py; subcells < 2 or no surface triangles -> dx^3 per cell.
        # With subcell_connect, the cut cells are refined instead: each of
        # their fluid sub-cells is a node (indices n, n+1, ...), so openings,
        # sills and sheet edges are resolved to dx/k.
        self.subcells = int(subcells or 0)
        # spill level at the floor of the saddle cell, with straddling cells
        # split (fsm.py); default on together with the sub-cell graph
        self.spill_floor = (subcell_connect and self.subcells >= 2) \
            if spill_floor is None else bool(spill_floor)
        _t0 = time.time()
        cg = None
        if subcell_connect and self.subcells >= 2:
            cg = cut_cell_graph(grid, self.subcells, share=self.surface_share)
        base = grid.neighbors()
        lab = self.comp.label.copy()
        if cg is None:
            self.v, self.volfrac_stats = effective_volumes(grid, self.subcells,
                                                           return_stats=True)
            self.N = n
            self.host = np.arange(n)
            self.fine = np.zeros(n, bool)
            self.X = grid.centers()
            self.fl = grid.fluid.ravel().copy()
            self.nbr = to_csr(base)
        else:
            self.volfrac_stats = dict(cg["stats"])
            self.N = N = n + cg["nextra"]
            self.v = cg["vol"]
            self.host = cg["host"]
            self.fine = cg["fine"]
            self.X = cg["X"]
            self.fl = cg["active"]
            E = cg["edges"]
            # explicit holes: no fine link near the rim may cross the sheet
            # plane (exchange through the hole goes via its throat)
            for h in self.holes:
                c0, ax = np.asarray(h["center"]), np.asarray(h["axis"])
                rr = 0.5 * h["diameter"] + 2.0 * grid.dx
                near = (np.linalg.norm(self.X[E[:, 0]] - c0, axis=1) <= rr) | \
                       (np.linalg.norm(self.X[E[:, 1]] - c0, axis=1) <= rr)
                sa = np.sign((self.X[E[:, 0]] - c0) @ ax)
                sb = np.sign((self.X[E[:, 1]] - c0) @ ax)
                E = E[~(near & (sa * sb < 0))]
            bi = np.repeat(np.arange(n), base.shape[1])
            bj = base.ravel()
            ok = bj >= 0
            src = np.concatenate([bi[ok], E[:, 0], E[:, 1]])
            dst = np.concatenate([bj[ok], E[:, 1], E[:, 0]])
            order = np.argsort(src, kind="stable")
            ptr = np.zeros(N + 1, np.int64)
            np.cumsum(np.bincount(src, minlength=N), out=ptr[1:])
            self.nbr = (ptr, dst[order].astype(np.int64))
            # compartment of each fine node: flooded from the whole cells,
            # nodes far from the walls first, so that a boundary between two
            # compartments through an opening lies in its narrowest section
            lab = np.concatenate([lab, -np.ones(N - n, np.int64)])
            pri = _node_wall_distance(grid, self.X, self.fine, self.subcells)
            pri[:n] = self.comp.dist
            lab = _priority_flood(lab.astype(np.int64), pri, self.nbr[0], self.nbr[1],
                                  self.fl.astype(np.bool_))
        # plugged holes: every link crossing the sheet plane inside the hole
        # (radius + one cell) is cut, coarse and fine alike
        self.plugs = []
        if plugs:
            ptr, idx = self.nbr
            srcn = np.repeat(np.arange(self.N), np.diff(ptr))
            for pl in plugs:
                if not isinstance(pl, dict):
                    pl = dict(center=pl[0], diameter=pl[1])
                c0 = np.asarray(pl["center"], float)
                ax = pl.get("axis")
                if ax is None:
                    ax = cp._hole_axis(grid, c0, float(pl["diameter"]))
                ax = np.asarray(ax, float) / np.linalg.norm(ax)
                r = 0.5 * float(pl["diameter"]) + grid.dx
                near = np.flatnonzero(np.linalg.norm(self.X - c0, axis=1) <= r + 2 * grid.dx)
                sel = np.isin(srcn, near) & (idx >= 0)
                a_, b_ = srcn[sel], idx[sel]
                sa = (self.X[a_] - c0) @ ax
                sb = (self.X[b_] - c0) @ ax
                mid = 0.5 * (self.X[a_] + self.X[b_]) - c0
                rad = np.linalg.norm(mid - (mid @ ax)[:, None] * ax, axis=1)
                cut = (np.sign(sa) != np.sign(sb)) & (rad <= r)
                idx[np.flatnonzero(sel)[cut]] = -1
                # and the reverse direction of the same links
                for u, w in zip(b_[cut], a_[cut]):
                    row = idx[ptr[u]:ptr[u + 1]]
                    row[row == w] = -1
                self.plugs.append(dict(center=c0, diameter=float(pl["diameter"]), axis=ax))
        # throats on the node graph: links between compartments through
        # openings that only the sub-cells resolve become throats (or merge
        # the compartments when split=False), and every throat gets the
        # width, area and sill of its sub-cell links too
        if cg is not None:
            lab = cp.node_throats(grid, self.comp, lab, self.X, self.fine, self.nbr,
                                  self.subcells, explicit=self.holes, plugs=self.plugs,
                                  split=split)
        self.volfrac_stats["time_s"] = time.time() - _t0
        self.nsize = np.where(self.fine, grid.dx / max(self.subcells, 1), grid.dx)
        return lab

    def _init_octree(self, ot, split, segment_kwargs, subcells, spill_floor, holes, plugs):
        """Nodes, links and compartments on an octree (see octree.py). The
        explicit holes are the octree's own (carved when it was built)."""
        from . import gseg
        from .octree import build_graph
        _t0 = time.time()
        if holes:
            # carve them into the closed cells and rebuild the levels around
            # them (modifies the octree, as carving modifies a uniform grid)
            ot.holes = list(getattr(ot, "holes", [])) + ot.carve_holes(holes)
            ot._build()
        self.holes = list(getattr(ot, "holes", []))
        self.subcells = int(subcells or 0)
        self.spill_floor = True if spill_floor is None else bool(spill_floor)
        g = build_graph(ot, self.subcells, narrow=self.narrow, share=self.surface_share)
        self.ograph = g
        self.N = g.N
        self.ncells = int(g.offsets[-1])
        self.v = g.vol
        self.X = g.X
        self.fl = g.active.copy()
        self.fine = g.fine
        self.nsize = g.size
        self.host = np.arange(g.N)
        ptr, idx = g.indptr.copy(), g.indices.copy()
        # explicit holes: no link near the rim may cross the sheet plane
        # (exchange through the hole goes via its throat)
        if self.holes:
            src = np.repeat(np.arange(g.N), np.diff(ptr))
            for hh in self.holes:
                c0, ax = np.asarray(hh["center"]), np.asarray(hh["axis"])
                rr = 0.5 * hh["diameter"] + 2.0 * ot.h
                ok = idx >= 0
                j = np.maximum(idx, 0)
                near = (np.linalg.norm(self.X[src] - c0, axis=1) <= rr) | \
                       (ok & (np.linalg.norm(self.X[j] - c0, axis=1) <= rr))
                sa = np.sign((self.X[src] - c0) @ ax)
                sb = np.where(ok, np.sign((self.X[j] - c0) @ ax), 0)
                # as on the uniform grid: every sub-cell link near the hole
                # that crosses the sheet plane is cut, also through the hole
                # (whole-cell links through it are its throat)
                rim = near & ok & (sa * sb < 0) & (self.fine[src] | self.fine[j])
                idx[rim] = -1
        self.nbr = (ptr, idx)
        self.plugs = []
        if plugs:
            srcn = np.repeat(np.arange(self.N), np.diff(ptr))
            for pl in plugs:
                if not isinstance(pl, dict):
                    pl = dict(center=pl[0], diameter=pl[1])
                c0 = np.asarray(pl["center"], float)
                ax = pl.get("axis")
                if ax is None:
                    ax = ot.hole_axis(c0, float(pl["diameter"]))
                ax = np.asarray(ax, float) / np.linalg.norm(ax)
                r = 0.5 * float(pl["diameter"]) + ot.h
                ok = idx >= 0
                j = np.maximum(idx, 0)
                sa = (self.X[srcn] - c0) @ ax
                sb = (self.X[j] - c0) @ ax
                mid = 0.5 * (self.X[srcn] + self.X[j]) - c0
                rad = np.linalg.norm(mid - (mid @ ax)[:, None] * ax, axis=1)
                idx[ok & (np.sign(sa) != np.sign(sb)) & (rad <= r)] = -1
                self.plugs.append(dict(center=c0, diameter=float(pl["diameter"]), axis=ax))
        gcut = type(g)(**{**g.__dict__, "indices": idx})
        kw = dict(segment_kwargs or {})
        # "voxel" (default): the segmentation of the uniform grid, on the
        # graph (distance to the closed cells, 26-neighbourhood markers,
        # first-in first-out flooding, merging across leaf faces);
        # "surface": distance to the surface triangles, face links only
        if kw.pop("distance", "voxel") == "voxel":
            from .octree import _coords
            cc = ot.origin + (_coords(ot.cut, ot.dims(0)) + 0.5) * ot.h
            self.comp, _ = gseg.segment_graph_voxel(gcut, ot, ot.dims0, cc,
                                                    k=max(self.subcells, 1), split=split,
                                                    explicit=self.holes, plugs=self.plugs,
                                                    **kw)
        else:
            self.comp, _ = gseg.segment_graph(gcut, ot.triangles, ot.h,
                                              k=max(self.subcells, 1), split=split,
                                              explicit=self.holes, plugs=self.plugs, **kw)
        self.volfrac_stats = dict(g.stats)
        self.volfrac_stats["time_s"] = time.time() - _t0
        return self.comp.label.copy()

    # ------------------------------------------------------------ threads
    def _py_threads(self):
        """Overlap passes in Python threads (only with one numba thread)."""
        return self.threads > 1 and numba.get_num_threads() == 1

    def _both(self, f1, a1, f2, a2):
        """f1(*a1), f2(*a2), in two Python threads if ``_py_threads()``."""
        if not self._py_threads():
            return f1(*a1), f2(*a2)
        if self._pool is None:
            from concurrent.futures import ThreadPoolExecutor
            self._pool = ThreadPoolExecutor(max_workers=1)
        fut = self._pool.submit(f2, *a2)
        r1 = f1(*a1)
        return r1, fut.result()

    def __getstate__(self):
        d = self.__dict__.copy()
        d["_pool"] = None
        d["_fc_l"] = FSMCache()
        d["_fc_a"] = FSMCache()
        return d

    def __setstate__(self, d):
        d.setdefault("_fc_l", FSMCache())
        d.setdefault("_fc_a", FSMCache())
        d.setdefault("reuse_static", True)
        d.setdefault("_static", False)
        d.setdefault("_pose_id", 0)
        d.setdefault("octree", False)
        d.setdefault("portion", "linear")
        self.__dict__.update(d)
        if not hasattr(self, "nsize"):
            self.nsize = np.where(self.fine, self.grid.dx / max(self.subcells, 1), self.grid.dx)
        if not hasattr(self, "_tptr"):
            self._throat_csr()

    def _throat_csr(self):
        """Face cells of all throats (CSR by throat) and the nodes of both
        sides of every throat (CSR by 2 * throat + side), for the kernels."""
        th = self.comp.throats
        ca = [t.cells_a for t in th]
        cb = [t.cells_b for t in th]
        self._tptr = np.zeros(len(th) + 1, np.int64)
        np.cumsum([c.size for c in ca], out=self._tptr[1:])
        self._tca = np.concatenate(ca).astype(np.int64) if th else np.zeros(0, np.int64)
        self._tcb = np.concatenate(cb).astype(np.int64) if th else np.zeros(0, np.int64)
        sides = []
        for i, t in enumerate(th):
            sides.extend(self._near.get(i, (t.cells_a, t.cells_b)))
        self._sptr = np.zeros(len(sides) + 1, np.int64)
        np.cumsum([c.size for c in sides], out=self._sptr[1:])
        self._sidx = np.concatenate(sides).astype(np.int64) if sides else np.zeros(0, np.int64)

    # ------------------------------------------------------------ geometry
    def _geom(self, up, zb):
        """Rounded up-vector, bath level and node heights for a pose.
        Heights are rounded to 1 nm (and the up-vector to 1e-12) so that
        cells at mathematically equal heights (e.g. exactly 45 deg) tie
        identically on every platform/BLAS; ties are then broken by index."""
        up, zb = self._pose(up, zb)
        return up, zb, stepk.geom_h(self.X, up)

    @staticmethod
    def _pose(up, zb):
        return np.round(np.asarray(up, float), 12), float(np.round(zb, 9))

    def _bath_atm(self, h, zb):
        """Bath / atmosphere: connected (without crossing throats) to the
        domain boundary below / above the bath level."""
        ptr, idx = self.nbr

        def connected(m):
            return par.connected_to(m, self.lab, ptr, idx, self.bnd)

        lo, hi = stepk.below_masks(self.ext, h, zb)
        if getattr(self, "_in_pool", False):
            B, A = connected(lo), connected(hi)
        else:
            B, A = self._both(connected, (lo,), connected, (hi,))
        if getattr(self, "suction", False) and B.any():
            # the submerged system: exterior connected to the bath without
            # passing through the atmosphere
            B = par.connected_to(self.ext & ~A, self.lab, ptr, idx, B)
        return B, A

    def _frame(self, up, zb, geom=None, BA=None):
        up, zb, h = geom if geom is not None else self._geom(up, zb)
        self.up = up
        self.zb = zb
        self.h = h
        self.e = 0.5 * self.grid.dx * np.abs(up).sum()     # half vertical extent
        # per node (fine sub-cell nodes are k times smaller)
        self.en = stepk.node_half_extent(self.nsize, np.abs(up).sum())
        self.eshape = box_shape(up) if self.portion == "box" else LINEAR
        self.B, self.A = BA if BA is not None else self._bath_atm(h, zb)

    # ------------------------------------------------------------- bodies
    def _bodies(self):
        """Separate liquid bodies: volume, level (the bath's is the bath
        level), free-surface area, and their nodes (grouped by body)."""
        ptr, idx = self.nbr
        mask = stepk.liquid_mask(self.fl, self.L)
        body, nb = par.label_bodies(mask, self.lab, ptr, idx)
        nb = int(nb)
        C = max(1, min(4 * numba.get_num_threads(), 20_000_000 // max(nb, 1)))
        grp, gst = stepk.group_by(body, nb, C)
        vol, level, area, isbath = stepk.body_stats(grp, gst, self.L, self.v, self.h,
                                                    self.en, self.B, self.zb,
                                                    float(self.grid.face_area), self.eshape)
        return dict(body=body, n=nb, vol=vol, level=level, area=area,
                    bath=isbath, grp=grp, gst=gst, top={})

    def _remove_top(self, bd, k, dV):
        cells = bd["top"].get(k)
        if cells is None:                    # nodes of body k, top first
            cells = bd["grp"][bd["gst"][k]:bd["gst"][k + 1]]
            cells = cells[np.argsort(-self.h[cells], kind="stable")]
            bd["top"][k] = cells
        rem = dV
        for c in cells:
            have = self.L[c] * self.v[c]
            take = min(have, rem)
            self.L[c] -= take / self.v[c]
            rem -= take
            if rem <= 0:
                break
        bd["vol"][k] -= dV - rem
        return dV - rem

    # ------------------------------------------------------------ throats
    def _throat_step(self, dt):
        th = self.comp.throats
        flows = np.zeros(len(th))
        if not th:
            return np.zeros(self.N), flows
        inj = np.zeros(self.N)
        bd = self._bodies()
        L = self.L
        fp = self.fluid
        nd = self.grid.ndim
        # venting. Air must enter a compartment that loses liquid and leave a
        # compartment that gains liquid. Besides the draining opening itself
        # (if it has air on both sides of some face), air can:
        #  - leave through any other opening where the compartment has air on
        #    its own side (it bubbles out even if that opening is submerged);
        #  - enter through any other opening with air on the far side.
        # plugged openings (open_at in the future) exchange nothing, not even air
        is_open = self._open_at <= self.t + 1e-12
        air_a, air_b, air_air = stepk.throat_air(self._tptr, self._tca, self._tcb, L)
        air_a &= is_open
        air_b &= is_open
        air_air &= is_open
        ta = np.array([t.a for t in th])
        tb = np.array([t.b for t in th])

        def can_gain(k, i):          # air can leave compartment k
            if k == 0 or air_air[i]:
                return True
            m = ((ta == k) & air_a) | ((tb == k) & air_b)
            m[i] = False
            return bool(m.any())

        def can_lose(k, i):          # air can enter compartment k
            if k == 0 or air_air[i]:
                return True
            m = ((ta == k) & air_b) | ((tb == k) & air_a)
            m[i] = False
            return bool(m.any())

        liq = stepk.comp_liquid(self.fl, self.lab, L, self.v, self.B, False, self.comp.n)
        air_room = self.comp.volume - liq
        slev, sbod = stepk.side_bodies(self._sptr, self._sidx, bd["body"], bd["level"])
        dcrit = self.tm.d_crit(fp, nd)
        for i, t in enumerate(th):
            if not is_open[i]:
                continue
            # highest body at each side (nodes near an explicit hole)
            lev = (slev[2 * i], slev[2 * i + 1])
            bods = (int(sbod[2 * i]), int(sbod[2 * i + 1]))
            if lev[0] == lev[1] or max(lev) == -np.inf:
                continue
            hf = 0.5 * (self.h[t.cells_a] + self.h[t.cells_b])
            s, d = (0, 1) if lev[0] > lev[1] else (1, 0)
            Hs, Hd = lev[s], lev[d]
            wet = hf < Hs
            if t.axis is not None:
                # explicit hole: wetted part of the true circle (slot in 2D)
                wfrac, zc, sill = self._hole_wet(t, Hs)
                if wfrac <= 0.0:
                    continue
                if not wet.any():
                    wet = hf <= hf.min() + 1e-12       # inject at the lowest faces
            else:
                if not wet.any():
                    continue
                if t.weights is None:
                    wfrac = wet.mean()
                    zc = hf[wet].mean()
                else:                              # whole and sub-cell faces
                    ww = t.weights[wet]
                    wfrac = ww.sum() / t.weights.sum()
                    zc = (hf[wet] * ww).sum() / ww.sum()
                sill = hf[wet].min()
            free = Hd < zc
            head = Hs - (zc if free else Hd)
            dcell = (t.cells_a, t.cells_b)[d]
            hold = self.tm.holdup_head(t.diameter, fp, nd) if free else 0.0
            head -= hold
            if head <= 0:
                continue
            area = t.area * wfrac
            comp_s = (t.a, t.b)[s]
            comp_d = (t.a, t.b)[d]
            if free and t.axis is not None and getattr(self.tm, "hole_profile", False):
                # orifice law over the wetted part of the hole (weir when
                # partly covered), with the level lowered by the hold-up
                Q = self.tm.Cd * self._hole_flux(t, Hs - hold, fp.g)
                if Q <= 0.0:
                    continue
            else:
                Q = self.tm.flow(area, head, fp)
            if not (can_lose(comp_s, i) and can_gain(comp_d, i)):
                if t.diameter < dcrit:
                    Q = 0.0
                else:
                    Q *= self.tm.counter_current_factor
            dV = Q * dt
            ks, kd = bods[s], bods[d]
            # limiters
            As = bd["area"][ks]
            if self.tm.instant:
                dV = np.inf
            if not bd["bath"][ks]:
                dV = min(dV, bd["vol"][ks], max(Hs - sill, 0.0) * As)
            # equalisation limiter (avoid overshooting the other level). Only
            # for closed receivers (internal compartments): in the exterior a
            # small receiving puddle just spills on, its area says nothing.
            if not free and kd >= 0 and comp_d != 0:
                Ad = bd["area"][kd]
                Aeq = As * Ad / (As + Ad) if np.isfinite(As + Ad) else min(As, Ad)
                frac = 1.0 if self.tm.instant else 0.5
                dV = min(dV, frac * (Hs - Hd) * Aeq)
            if comp_d != 0:
                dV = min(dV, max(air_room[comp_d], 0.0))
            if not np.isfinite(dV) or dV <= 0:
                continue
            if not bd["bath"][ks]:
                dV = self._remove_top(bd, ks, dV)
            dc = dcell[wet]
            if t.weights is None or t.axis is not None:
                np.add.at(inj, dc, dV / dc.size)
            else:
                ww = t.weights[wet]
                np.add.at(inj, dc, dV * ww / ww.sum())
            air_room[comp_d] -= dV
            air_room[comp_s] += dV
            flows[i] = (dV if s == 0 else -dV) / dt
        return inj, flows

    def _grow(self, seed, mask):
        """Cells of ``mask`` connected to ``seed`` (through self.nbr)."""
        got = np.zeros(mask.size, bool)
        front = np.unique(seed)
        got[front] = True
        ptr, idx = self.nbr
        while front.size:
            cnt = ptr[front + 1] - ptr[front]
            pos = np.repeat(ptr[front], cnt) + (np.arange(cnt.sum()) -
                                                np.repeat(np.cumsum(cnt) - cnt, cnt))
            nb = idx[pos]
            nb = nb[nb >= 0]
            nb = nb[mask[nb] & ~got[nb]]
            nb = np.unique(nb)
            got[nb] = True
            front = nb
        return np.flatnonzero(got)

    def _hole_wet(self, t, H):
        """Wetted fraction, centroid height and sill height of an explicit
        hole for a pool level H (circle of diameter d in 3D, slot in 2D)."""
        n = t.axis
        upp = self.up - (self.up @ n) * n
        sp = float(np.linalg.norm(upp))
        R = 0.5 * t.diameter
        c0 = float(t.centroid @ self.up)
        sill = c0 - R * sp
        if sp < 1e-9:                                 # hole in a horizontal sheet
            return (1.0, c0, c0) if H > c0 else (0.0, c0, c0)
        y = min(max((H - c0) / sp, -R), R)
        if y <= -R:
            return 0.0, sill, sill
        q = max(R * R - y * y, 0.0)
        if self.grid.ndim == 3:
            A = R * R * np.arccos(-y / R) + y * np.sqrt(q)
            frac = A / (np.pi * R * R)
            ybar = -(2.0 / 3.0) * q ** 1.5 / max(A, 1e-300)
        else:
            frac = (y + R) / (2 * R)
            ybar = 0.5 * (y - R)
        return float(frac), c0 + ybar * sp, sill

    def _hole_flux(self, t, H, g):
        """int over the wetted part of an explicit hole of sqrt(2 g (H - z))
        dA for a pool level H (m^3/s per unit Cd; per unit depth in 2D).

        The hole is a circle of diameter d (a slot in 2D) in the plane
        normal to its axis; z = c0 + y * sp along the steepest direction y
        in that plane (sp = the in-plane part of the up vector). The chord
        width is 2 sqrt(R^2 - y^2) (1 in 2D). Integrated with Gauss-Legendre
        after y = y0 + (Y - y0) sin^2(pi/2 s), which removes the square-root
        end points."""
        n = t.axis
        upp = self.up - (self.up @ n) * n
        sp = float(np.linalg.norm(upp))
        R = 0.5 * t.diameter
        c0 = float(t.centroid @ self.up)
        if sp < 1e-9:                                     # horizontal sheet
            return t.area * np.sqrt(2.0 * g * (H - c0)) if H > c0 else 0.0
        Y = min((H - c0) / sp, R)
        if Y <= -R:
            return 0.0
        x, w = _GL
        s = 0.5 * (x + 1.0)
        th = 0.5 * np.pi * s
        y = -R + (Y + R) * np.sin(th) ** 2
        dy = (Y + R) * np.pi * np.sin(th) * np.cos(th) * 0.5 * w   # dy/ds * ds weights
        head = np.maximum(H - (c0 + y * sp), 0.0)
        width = 2.0 * np.sqrt(np.maximum(R * R - y * y, 0.0)) if self.grid.ndim == 3 \
            else np.ones_like(y)
        return float(np.sum(width * np.sqrt(2.0 * g * head) * dy))

    # --------------------------------------------------------- equilibrate
    def _hydrostatic(self):
        """Ambient + hydrostatic pressure at each cell centre (Pa)."""
        fp = self.fluid
        dz = self.zb - self.h
        if getattr(self, "suction", False) and getattr(self, "B", None) is not None:
            # in the submerged system the pressure also falls below ambient
            # above the bath level
            return fp.p_atm + fp.rho * fp.g * np.where(self.B, dz, np.maximum(dz, 0.0))
        return fp.p_atm + fp.rho * fp.g * np.maximum(dz, 0.0)

    def _air_compressible(self, region_a, sink_a, md, iters=6, tol=1e-7):
        """Fill-spill of the trapped *gas* (Boyle, isothermal).

        The routed quantity is the gas amount G (m^3 at ambient pressure),
        and a cell of volume v at pressure p holds v p / p_atm of it. The
        pressure of a pocket is uniform and equal to the hydrostatic pressure
        at its liquid interface. That interface depends on how much the
        pocket is compressed, so a few fixed-point passes are made. Gas is
        conserved by the fill-spill routing, so pockets that merge add their
        gas and a pocket that splits divides it. Returns (gas per cell, air
        volume per cell).
        """
        fp = self.fluid
        v, e, h = self.v, self.en, self.h
        gas = np.where(self.ext & ~self.A, self.G, 0.0)
        # (below ambient only in the submerged system, see ``suction``;
        # floored at 5 % of ambient against columns of several metres)
        P = np.where(self.B, np.maximum(self.P, 0.05 * fp.p_atm), np.maximum(self.P, fp.p_atm)) \
            if self.suction else np.maximum(self.P, fp.p_atm)
        hyd = self._hydrostatic()
        RG = None
        for _ in range(iters):
            vg = v * P / fp.p_atm
            RG, _, _, _ = fill_spill(-h, region_a, self.nbr, sink_a, gas, vg,
                                     True, nregions=1, min_depth=md,
                                     ecell=self.en if self.spill_floor else None,
                                     eshape=self.eshape)
            f = np.where(vg > 0, RG / np.where(vg > 0, vg, 1.0), 0.0)
            air = self.ext & ~self.A & (f > 1e-9)
            body, nb = _label_bodies(air, self.lab, self.nbr)
            Pn = hyd.copy()
            if nb:
                idx = np.flatnonzero(body >= 0)
                bottom = h[idx] + e[idx] - 2.0 * e[idx] * f[idx]   # air fills from the top
                Hk = np.full(nb, np.inf)
                np.minimum.at(Hk, body[idx], bottom)
                dz = self.zb - Hk
                if self.suction:
                    inD = np.zeros(nb, bool)
                    inD[body[idx][self.B[idx]]] = True
                    dz = np.where(inD, dz, np.maximum(dz, 0.0))
                else:
                    dz = np.maximum(dz, 0.0)
                pk = np.maximum(fp.p_atm + fp.rho * fp.g * dz, 0.05 * fp.p_atm)
                Pn[idx] = pk[body[idx]]
            done = np.max(np.abs(Pn - P) / fp.p_atm) < tol
            P = Pn
            if done:
                break
        self.P = P
        return RG, RG * fp.p_atm / P

    def _equilibrate(self, inj):
        v = self.v
        L = self.L
        md = self.min_depth_cells * 2.0 * self.e
        ec = self.en if self.spill_floor else None
        # Height bands: liquid only moves down and air only up, so no pool
        # can rise above the highest liquid (or sink below the lowest air).
        # Nodes beyond that, with a margin of one cell, cannot change and are
        # left out of the sweep; the result is the same. While the pose is
        # unchanged, the whole domain is swept instead and its hierarchy is
        # reused in the next steps (see keep_l / keep_a below).
        static = self._static and self.reuse_static
        mrg = 2.0 * self.e
        if not static:
            self._fc_l.clear()
            self._fc_a.clear()
            self._last_keys = {}
        # --- liquid: all compartments at once (region = compartment label)
        region, sink_l, src, top = stepk.liquid_inputs(self.fl, self.lab, self.B, self.bnd,
                                                       self.ext, L, v, inj, self.h, self.en)
        # --- air: exterior only
        region_a, sink_a, src_a, bot = stepk.air_inputs(self.ext, self.A, self.bnd, L, v,
                                                        self.h, self.en)
        # air in the exterior can escape through a throat into a compartment
        # that has air at that opening (bubbling through a hole)
        ok = np.zeros(0, bool)
        if self._tx_ext.size:
            ok = (L[self._tx_int] < 0.5) & (self._open_at[self._tx_own] <= self.t + 1e-12)
            sink_a[self._tx_ext[ok]] = True
        # cache keys: with the pose unchanged, the sweep inputs other than the
        # sources only change with the air sinks at throats
        key_l = ("liquid", self._pose_id, self.spill_floor)
        key_a = ("air", self._pose_id, self.spill_floor, ok.tobytes())
        # A hierarchy is kept (whole domain) once a sweep's inputs were the
        # same in two steps in a row; otherwise the sweep uses the band.
        last = getattr(self, "_last_keys", {})
        keep_l = static and last.get("l") == key_l
        keep_a = static and last.get("a") == key_a
        self._last_keys = {"l": key_l, "a": key_a}

        def liquid():
            reg = region
            if self.height_bands and not keep_l:
                reg = stepk.band_region(region, self.h, self.en, sink_l, top + mrg, True)
            return fill_spill(self.h, reg, self.nbr, sink_l, src, v,
                              self.spill_routing, nregions=self.comp.n,
                              min_depth=md, ecell=ec, eshape=self.eshape,
                              cache=self._fc_l if keep_l else None, key=key_l)

        def air():
            if self.compressible_air:
                return self._air_compressible(region_a, sink_a, md)
            reg = region_a
            if self.height_bands and not keep_a:
                reg = stepk.band_region(region_a, self.h, self.en, sink_a, bot - mrg, False)
            return None, fill_spill(-self.h, reg, self.nbr, sink_a, src_a, v,
                                    True, nregions=1, min_depth=md, ecell=ec, eshape=self.eshape,
                                    cache=self._fc_a if keep_a else None, key=key_a)[0]

        (RL, drained, lost, _), (RG, RA) = self._both(liquid, (), air, ())
        self.drained_total += drained.sum()     # includes throat inflow into B
        self.L = stepk.combine(RL, RA, self._vsafe, self.B, self.fl)
        self.lost = lost
        if self.compressible_air:
            # gas carried to the next step: the routed gas for trapped air,
            # ambient-pressure air everywhere else (atmosphere, compartments)
            free = (1.0 - self.L) * v
            self.G = np.where(self.ext & ~self.A, RG, free)
            self.G[~self.fl] = 0.0
            self.P[~(self.ext & ~self.A)] = self.fluid.p_atm

    # ---------------------------------------------------------------- run
    def step(self, dt):
        up, zb = self.motion.frame(self.t + dt)
        # an unchanged pose (hanging still): same heights, bath and atmosphere
        upr, zbr = self._pose(up, zb)
        static = self.reuse_static and zbr == self.zb and np.array_equal(upr, self.up)
        if static:
            inj, flows = self._throat_step(dt)
        else:
            geom = self._geom(up, zb)
            # throat exchange is evaluated on the start-of-step state; the
            # bath and atmosphere of the new pose are labelled meanwhile
            fut = None
            if self._py_threads():
                if self._pool is None:
                    from concurrent.futures import ThreadPoolExecutor
                    self._pool = ThreadPoolExecutor(max_workers=1)

                def ba():
                    self._in_pool = True
                    try:
                        return self._bath_atm(geom[2], geom[1])
                    finally:
                        self._in_pool = False
                fut = self._pool.submit(ba)
            inj, flows = self._throat_step(dt)
            BA = fut.result() if fut is not None else self._bath_atm(geom[2], geom[1])
            self._frame(up, zb, geom=geom, BA=BA)
            self._pose_id += 1
        self._static = static
        inj = inj + self._pending              # film absorption / drips
        self._equilibrate(inj)
        self.t += dt
        if self.film is not None:
            self._pending = self.film.update(dt)
        return flows

    def _choose_dt(self, t_end):
        dt = self.dt_max
        if self.t < self.motion.t_end:
            probe = min(self.dt_max, self.motion.t_end - self.t)
            vmax = self.motion.speed_bound(self.t, max(probe, 1e-6), self.extent)
            if vmax > 0:
                dt = min(dt, self.cells_per_step * self.grid.dx / vmax)
            dt = min(dt, self.motion.t_end - self.t) if self.motion.t_end - self.t > 1e-9 else dt
        dt = min(dt, t_end - self.t)
        return max(dt, min(self.dt_min, t_end - self.t))

    def _snap(self, t):
        H = self.hist
        H.snapshots[t] = self.L.astype(np.float32)
        H.snap_bath[t] = self.B.copy()
        H.snap_atm[t] = self.A.copy()
        if self.film is not None:
            H.snap_film[t] = self.film.s.h.astype(np.float32)

    def run(self, t_end=None, snapshot_times=(), snapshot_every=None,
            progress=False):
        """Advance to ``t_end`` (default: end of motion).

        snapshot_times : times at which the liquid field is stored.
        snapshot_every : alternatively, a fixed interval (e.g. for animations).
        progress : True prints a line every 100 steps; a callable is called
                   with the simulation after every step.
        """
        t_end = self.motion.t_end if t_end is None else t_end
        if snapshot_every:
            n = int(np.floor((t_end - self.t) / snapshot_every + 1e-9))
            snapshot_times = list(snapshot_times) + list(
                self.t + snapshot_every * np.arange(n + 1))
        snaps = sorted(set(float(np.round(s, 9)) for s in snapshot_times))
        si = 0
        while si < len(snaps) and snaps[si] <= self.t + 1e-12:
            self._snap(self.t)
            si += 1
        nstep = 0
        while self.t < t_end - 1e-9:
            dt = self._choose_dt(t_end)
            if si < len(snaps) and self.t + dt > snaps[si]:
                dt = max(snaps[si] - self.t, 1e-9)
            flows = self.step(dt)
            self._record(flows)
            nstep += 1
            while si < len(snaps) and snaps[si] <= self.t + 1e-9:
                self._snap(snaps[si])
                si += 1
            if callable(progress):
                progress(self)                    # called after every step
            elif progress and nstep % 100 == 0:
                print(f"t={self.t:.3f}  retained={self.hist.liquid_retained[-1]:.4g}")
        self.nsteps = nstep
        return self.hist

    # ---------------------------------------------------------- diagnostics
    # What is *reported and drawn* as trapped uses the bath surface of the
    # current pose, the same for every compartment:
    #   trapped liquid = liquid (not bath) above the surface,
    #   trapped air    = air (not atmosphere) below the surface.
    # A node that straddles the surface counts with the part of its extent
    # on that side. The state itself is not touched: a compartment keeps the
    # liquid or air it holds while it crosses the surface (it is only
    # filled or emptied through its throats).
    def above_fraction(self, h=None, zb=None, en=None, eshape=None):
        """Part of each node's volume above the bath surface."""
        h = self.h if h is None else h
        zb = self.zb if zb is None else zb
        en = self.en if en is None else en
        w = getattr(self, "eshape", LINEAR) if eshape is None else eshape
        return 1.0 - cell_cdf_vec((zb - h + en) / (2.0 * en), np.asarray(w, float))

    def trapped_fields(self, L=None, B=None, A=None, h=None, zb=None, en=None):
        """Per node: trapped liquid and trapped air fractions (see above).
        Arguments default to the current state (copies may be passed)."""
        L = self.L if L is None else L
        B = self.B if B is None else B
        A = self.A if A is None else A
        a = self.above_fraction(h, zb, en)
        liq = np.where(self.fl & ~B, L, 0.0) * a
        air = np.where(self.fl & ~A, 1.0 - L, 0.0) * (1.0 - a)
        return liq, air

    def _record(self, flows):
        H = self.hist
        liq_comp, S = stepk.record_sums(self.fl, self.lab, self.L, self.v, self.B, self.A,
                                        self.h, self.en, float(self.zb), self.comp.n,
                                        self.eshape)
        H.t.append(self.t)
        H.bath_level.append(self.zb)
        H.liquid_retained.append(liq_comp.sum())
        H.liquid_above_bath.append(float(S[0]))
        H.liquid_exterior.append(liq_comp[0])
        H.liquid_comp.append(liq_comp)
        H.liquid_trapped.append(float(S[1]))
        H.air_trapped.append(float(S[2]))
        H.air_gas.append(float(self.G[self.B].sum()) if self.compressible_air
                         else float(S[3]))
        H.air_below_bath.append(float(S[4]))
        H.drained.append(self.drained_total)
        H.throat_flow.append(flows)
        f = getattr(self, "film", None)
        H.film_volume.append(f.volume if f is not None else 0.0)
        H.film_pending.append(float(self._pending.sum()) if f is not None else 0.0)
        H.drip_volume.append(f.drip_volume if f is not None else 0.0)
        H.n_drips.append(f.n_drops if f is not None else 0)     # not a sum over s.drips

    def pockets(self, min_volume=0.0):
        """Separate bodies of trapped liquid (above the bath surface) and
        trapped air (below it) at the current time: list of (volume,
        centroid). Same definitions as ``trapped_fields``."""
        out = {}
        v = self.v
        liq_t, air_t = self.trapped_fields()
        for name, w in (("liquid", liq_t), ("air", air_t)):
            mask = w > 1e-9
            body, nb = _label_bodies(mask, self.lab, self.nbr)
            idx = np.flatnonzero(body >= 0)
            vol = np.bincount(body[idx], weights=w[idx] * v[idx], minlength=nb)
            cen = np.stack([np.bincount(body[idx], weights=w[idx] * v[idx] * self.X[idx, d],
                                        minlength=nb) for d in range(self.grid.ndim)], 1)
            cen = cen / np.maximum(vol, 1e-300)[:, None]
            keep = vol > min_volume
            out[name] = sorted(zip(vol[keep], cen[keep]), key=lambda x: -x[0])
        return out
