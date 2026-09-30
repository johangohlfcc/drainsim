"""Primitive shapes painted onto a Grid as solid (closed) cells.

Each primitive returns a boolean mask of cells whose centres lie inside the
shape. Use ``add(grid, mask)`` to make them solid and ``cut(grid, mask)`` to
open holes. Thin walls get a minimum half-thickness so that diagonal walls
stay leak-free for face-connected fluid.
"""
from __future__ import annotations

import numpy as np

from .grid import Grid


def coords(grid: Grid):
    axes = [grid.origin[d] + (np.arange(n) + 0.5) * grid.dx
            for d, n in enumerate(grid.shape)]
    return np.meshgrid(*axes, indexing="ij")


def _min_half(grid: Grid) -> float:
    # a centre-distance band of this half-width is face-connected-leak-free
    return 0.5 * np.sqrt(grid.ndim) * grid.dx * 1.01


def add(grid: Grid, mask: np.ndarray) -> Grid:
    grid.solid |= mask
    grid._nbr = None
    return grid


def cut(grid: Grid, mask: np.ndarray) -> Grid:
    grid.solid &= ~mask
    grid._nbr = None
    return grid


def box(grid: Grid, lo, hi) -> np.ndarray:
    """Axis-aligned filled box (rectangle in 2D)."""
    X = coords(grid)
    m = np.ones(grid.shape, bool)
    for d in range(grid.ndim):
        m &= (X[d] >= lo[d]) & (X[d] <= hi[d])
    return m


rect = box


def segment(grid: Grid, p0, p1, thickness: float) -> np.ndarray:
    """Thick line segment (2D) or capsule (3D)."""
    X = coords(grid)
    p0 = np.asarray(p0, float)
    p1 = np.asarray(p1, float)
    d = p1 - p0
    L2 = float(d @ d)
    rel = [X[k] - p0[k] for k in range(grid.ndim)]
    t = sum(rel[k] * d[k] for k in range(grid.ndim)) / max(L2, 1e-30)
    t = np.clip(t, 0.0, 1.0)
    dist2 = sum((rel[k] - t * d[k]) ** 2 for k in range(grid.ndim))
    r = max(0.5 * thickness, _min_half(grid))
    return dist2 <= r * r


def polyline(grid: Grid, points, thickness: float, closed=False) -> np.ndarray:
    pts = [np.asarray(p, float) for p in points]
    if closed:
        pts.append(pts[0])
    m = np.zeros(grid.shape, bool)
    for a, b in zip(pts[:-1], pts[1:]):
        m |= segment(grid, a, b, thickness)
    return m


def ball(grid: Grid, center, radius) -> np.ndarray:
    X = coords(grid)
    return sum((X[k] - center[k]) ** 2 for k in range(grid.ndim)) <= radius ** 2


disk = ball


def cylinder(grid: Grid, p0, p1, radius) -> np.ndarray:
    """Finite cylinder (3D) between p0 and p1 (flat ends)."""
    X = coords(grid)
    p0 = np.asarray(p0, float)
    p1 = np.asarray(p1, float)
    d = p1 - p0
    L2 = float(d @ d)
    rel = [X[k] - p0[k] for k in range(grid.ndim)]
    t = sum(rel[k] * d[k] for k in range(grid.ndim)) / L2
    dist2 = sum((rel[k] - t * d[k]) ** 2 for k in range(grid.ndim))
    return (t >= 0) & (t <= 1) & (dist2 <= radius ** 2)


def shell_box(grid: Grid, lo, hi, thickness: float, open_faces=()) -> np.ndarray:
    """Hollow box with walls of given thickness (outside dimensions lo..hi).

    ``open_faces`` is a list of (axis, side) with side in {-1, +1}, e.g.
    ``[(1, +1)]`` for an open top in 2D (y up) or ``[(2, +1)]`` in 3D.
    """
    lo = np.asarray(lo, float)
    hi = np.asarray(hi, float)
    t = max(thickness, grid.dx)          # axis-aligned walls: one cell suffices
    outer = box(grid, lo, hi)
    ilo = lo + t
    ihi = hi - t
    for ax, side in open_faces:
        if side > 0:
            ihi[ax] = hi[ax] + grid.dx
        else:
            ilo[ax] = lo[ax] - grid.dx
    inner = box(grid, ilo, ihi)
    return outer & ~inner
