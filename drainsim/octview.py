"""Uniform display grid for octree runs.

The movie and VTK tools draw iso-surfaces of cell fields on a uniform grid.
An octree run has no such grid, so node fields are resampled onto a uniform
*display grid* of spacing ``h * 2**j`` (aligned with the octree cells): every
display cell gets the volume-weighted mean of the node fractions inside it
(a coarse node spreads evenly over the display cells it covers). The solid
mask of the display grid marks cells whose level-0 cells are mostly closed.
The model state is not touched; this is for drawing only.
"""
from __future__ import annotations

import numpy as np
from numba import njit

from .octree import Octree, _coords, _key, _dims


@njit(cache=True, nogil=True)
def _spread(X, size, v, f, origin, dxd, shape, out, sel, mark):
    """out[cell] += f * v * overlap for the nodes in ``sel``: a node no
    larger than a display cell goes to the cell of its centre, a larger one
    evenly to the cells it covers. ``mark`` (uint8, ncells, or empty): set
    to 1 at every cell written."""
    mk = mark.shape[0] > 0
    nx, ny, nz = shape[0], shape[1], shape[2]
    for q in range(sel.shape[0]):
        i = sel[q]
        w = f[i] * v[i]
        if w == 0.0:
            continue
        s = size[i]
        if s <= dxd * 1.000001:
            a = int(np.floor((X[i, 0] - origin[0]) / dxd))
            b = int(np.floor((X[i, 1] - origin[1]) / dxd))
            c = int(np.floor((X[i, 2] - origin[2]) / dxd))
            if 0 <= a < nx and 0 <= b < ny and 0 <= c < nz:
                out[(a * ny + b) * nz + c] += w
                if mk:
                    mark[(a * ny + b) * nz + c] = 1
        else:
            n = int(np.rint(s / dxd))
            a0 = int(np.rint((X[i, 0] - 0.5 * s - origin[0]) / dxd))
            b0 = int(np.rint((X[i, 1] - 0.5 * s - origin[1]) / dxd))
            c0 = int(np.rint((X[i, 2] - 0.5 * s - origin[2]) / dxd))
            wc = w / (n * n * n)
            for a in range(a0, a0 + n):
                for b in range(b0, b0 + n):
                    for c in range(c0, c0 + n):
                        out[(a * ny + b) * nz + c] += wc
                        if mk:
                            mark[(a * ny + b) * nz + c] = 1
    return 0


@njit(cache=True, nogil=True)
def _fluid_nonzero(fl, f):
    """np.flatnonzero(np.where(fl, f, 0) != 0)."""
    m = 0
    for i in range(f.shape[0]):
        if fl[i] and f[i] != 0:
            m += 1
    sel = np.empty(m, np.int64)
    m = 0
    for i in range(f.shape[0]):
        if fl[i] and f[i] != 0:
            sel[m] = i
            m += 1
    return sel


@njit(cache=True)
def _argmax(X, size, w, origin, dxd, shape, best, arg, sel):
    """Per display cell the node of largest w among the nodes in ``sel``
    that cover it."""
    nx, ny, nz = shape[0], shape[1], shape[2]
    for q in range(sel.shape[0]):
        i = sel[q]
        s = size[i]
        if s <= dxd * 1.000001:
            a = int(np.floor((X[i, 0] - origin[0]) / dxd))
            b = int(np.floor((X[i, 1] - origin[1]) / dxd))
            c = int(np.floor((X[i, 2] - origin[2]) / dxd))
            if 0 <= a < nx and 0 <= b < ny and 0 <= c < nz:
                k = (a * ny + b) * nz + c
                if w[i] > best[k]:
                    best[k] = w[i]
                    arg[k] = i
        else:
            n = int(np.rint(s / dxd))
            a0 = int(np.rint((X[i, 0] - 0.5 * s - origin[0]) / dxd))
            b0 = int(np.rint((X[i, 1] - 0.5 * s - origin[1]) / dxd))
            c0 = int(np.rint((X[i, 2] - 0.5 * s - origin[2]) / dxd))
            for a in range(a0, a0 + n):
                for b in range(b0, b0 + n):
                    for c in range(c0, c0 + n):
                        k = (a * ny + b) * nz + c
                        if w[i] > best[k]:
                            best[k] = w[i]
                            arg[k] = i
    return 0


class DisplayGrid:
    """Uniform grid for drawing the fields of an octree run.

    ot      the ``Octree``;
    j       display spacing h * 2**j (0 <= j <= L);
    solid_fraction  a display cell is solid if at least this fraction of
            its level-0 cells is closed (and it holds no fluid node's centre
            for j = 0)."""

    def __init__(self, ot: Octree, j=1, solid_fraction=0.5):
        j = int(min(max(j, 0), ot.L))
        self.j = j
        self.dx = ot.h * 2 ** j
        self.origin = np.asarray(ot.origin, float)
        self.shape = tuple(int(v) for v in _dims(ot.dims0, j))
        self.ncells = int(np.prod(self.shape))
        self.ndim = 3
        self.triangles = None
        # solid: closed level-0 cells per display cell
        d = np.asarray(self.shape, np.int64)
        pk = _key(_coords(ot.cut, ot.dims(0)) >> j, d)
        cnt = np.bincount(pk, minlength=self.ncells)
        self.solid = (cnt >= max(1, solid_fraction * 8 ** j)).reshape(self.shape)
        self.fluid = ~self.solid
        self._den = None

    @classmethod
    def for_budget(cls, ot: Octree, max_cells=100e6, **kw):
        """Finest display spacing with at most ``max_cells`` cells."""
        j = 0
        while j < ot.L and np.prod(_dims(ot.dims0, j).astype(float)) > max_cells:
            j += 1
        return cls(ot, j, **kw)

    def _sum(self, sim, f, sel=None):
        out = np.zeros(self.ncells)
        if sel is None:
            sel = np.flatnonzero(f != 0)
        _spread(sim.X, sim.nsize, sim.v, np.asarray(f, float), self.origin, self.dx,
                np.asarray(self.shape, np.int64), out, sel.astype(np.int64),
                np.zeros(0, np.uint8))
        return out

    def fluid_volume(self, sim):
        if self._den is None:
            self._den = self._sum(sim, sim.fl.astype(float)).astype(np.float32)
        return self._den

    def field(self, sim, f):
        """Node fraction field -> display fraction (flat, ncells)."""
        num = self._sum(sim, np.where(sim.fl, f, 0.0))
        den = self.fluid_volume(sim)
        out = np.zeros(self.ncells, np.float32)
        i = np.flatnonzero(num > 0)
        out[i] = np.minimum(num[i] / np.maximum(den[i], 1e-30), 1.0)
        return out

    def field_nonzero(self, sim, f):
        """``field`` at its nonzero cells only: (cells ascending, float32
        values), the same numbers without scanning the whole display grid
        (the written cells are marked; the zeroed buffers are only touched
        where written)."""
        from .par import compact
        f = np.asarray(f, float)
        sel = _fluid_nonzero(np.asarray(sim.fl, np.bool_), f)
        num = np.zeros(self.ncells)
        mark = np.zeros(self.ncells, np.uint8)
        _spread(sim.X, sim.nsize, sim.v, f, self.origin, self.dx,
                np.asarray(self.shape, np.int64), num, sel, mark)
        i = compact(mark)
        nm = num[i]
        pos = nm > 0
        i, nm = i[pos], nm[pos]
        den = self.fluid_volume(sim)
        return i, np.minimum(nm / np.maximum(den[i], 1e-30), 1.0).astype(np.float32)

    def argmax_node(self, sim, w):
        """Per display cell: the node with the largest w > 0 covering it
        (-1 where none)."""
        sel = np.flatnonzero(np.asarray(w) > 0).astype(np.int64)
        best = np.zeros(self.ncells)
        arg = -np.ones(self.ncells, np.int64)
        _argmax(sim.X, sim.nsize, np.asarray(w, float), self.origin, self.dx,
                np.asarray(self.shape, np.int64), best, arg, sel)
        return arg


def display_grid(sim, max_cells=100e6):
    """The display grid of an octree run (made once, kept on ``sim``)."""
    dg = getattr(sim, "_display_grid", None)
    if dg is None:
        dg = DisplayGrid.for_budget(sim.grid, max_cells)
        sim._display_grid = dg
    return dg
