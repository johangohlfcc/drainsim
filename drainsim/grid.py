"""Uniform Cartesian voxel grid in the object frame (2D or 3D).

Conventions
-----------
* Arrays are indexed ``[i, j]`` (2D) or ``[i, j, k]`` (3D); axis 0 = x, 1 = y, 2 = z.
* Cell ``idx`` has its centre at ``origin + (idx + 0.5) * dx``.
* Flat cell index = C-order ravel of the multi-index.
* The grid is fixed to the object. Gravity and the bath surface move instead
  (same idea as in the IBOFlow method).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Grid:
    shape: tuple
    dx: float
    origin: np.ndarray
    solid: np.ndarray                      # bool, shape == self.shape
    _nbr: np.ndarray | None = field(default=None, repr=False)
    # surface triangles (n, 3, 3) the grid was voxelised from, if any; used
    # for sub-cell volume fractions (volfrac.py)
    triangles: np.ndarray | None = field(default=None, repr=False)

    # ------------------------------------------------------------------ basics
    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def ncells(self) -> int:
        return int(np.prod(self.shape))

    @property
    def cell_volume(self) -> float:
        """Cell 'volume': m^3 in 3D, m^2 (per unit depth) in 2D."""
        return self.dx ** self.ndim

    @property
    def face_area(self) -> float:
        """Face 'area': m^2 in 3D, m (per unit depth) in 2D."""
        return self.dx ** (self.ndim - 1)

    @property
    def fluid(self) -> np.ndarray:
        return ~self.solid

    def centers(self) -> np.ndarray:
        """Cell centres, shape (ncells, ndim), flat C-order."""
        axes = [self.origin[d] + (np.arange(n) + 0.5) * self.dx
                for d, n in enumerate(self.shape)]
        mesh = np.meshgrid(*axes, indexing="ij")
        return np.stack([m.ravel() for m in mesh], axis=1)

    def boundary_mask(self) -> np.ndarray:
        """True for cells on the outer faces of the domain."""
        b = np.zeros(self.shape, bool)
        for d in range(self.ndim):
            sl = [slice(None)] * self.ndim
            sl[d] = 0
            b[tuple(sl)] = True
            sl[d] = -1
            b[tuple(sl)] = True
        return b

    # -------------------------------------------------------------- neighbours
    def neighbors(self) -> np.ndarray:
        """Face-neighbour table (ncells, 2*ndim) of flat indices.

        -1 where the neighbour is outside the domain or solid. Solid cells get
        an all -1 row. Column ``2*d`` is the -d neighbour, ``2*d+1`` the +d one.
        """
        if self._nbr is not None:
            return self._nbr
        n = self.ncells
        idx = np.arange(n, dtype=np.int64).reshape(self.shape)
        nbr = -np.ones((n, 2 * self.ndim), dtype=np.int64)
        fluid = self.fluid
        for d in range(self.ndim):
            # minus neighbour
            src = [slice(None)] * self.ndim
            dst = [slice(None)] * self.ndim
            src[d] = slice(1, None)
            dst[d] = slice(None, -1)
            a = idx[tuple(src)].ravel()
            b = idx[tuple(dst)].ravel()
            ok = fluid[tuple(src)].ravel() & fluid[tuple(dst)].ravel()
            nbr[a[ok], 2 * d] = b[ok]
            nbr[b[ok], 2 * d + 1] = a[ok]
        self._nbr = nbr
        return nbr

    # ------------------------------------------------------------ constructors
    @classmethod
    def empty(cls, lo, hi, dx) -> "Grid":
        lo = np.asarray(lo, float)
        hi = np.asarray(hi, float)
        shape = tuple(int(np.ceil((h - l) / dx - 1e-9)) for l, h in zip(lo, hi))
        return cls(shape=shape, dx=float(dx), origin=lo.copy(),
                   solid=np.zeros(shape, bool))

    @classmethod
    def from_mesh(cls, mesh, dx: float, pad: float | None = None,
                  fill_interior: bool = False, bounds=None, cut: str = "exact") -> "Grid":
        """Voxelise a triangle surface mesh (path or trimesh.Trimesh).

        Cells intersected by triangles become solid (as in IBOFlow, where cut
        cells are 'closed object cells'). Suitable for thin sheet-metal meshes
        that are not watertight. ``pad`` adds free space around the object so
        that the domain boundary does not touch it (default 3 cells).

        ``bounds = (lo, hi)`` voxelises only the part of the mesh inside that
        box (mesh coordinates) on a grid covering exactly that box - useful
        when only the lower part of a large object is ever submerged.
        ``cut``: "exact" marks every cell whose box touches a triangle
        (``voxel.surface_cells``); "sample" is the point sampling used before
        6.3 (it can miss cells a sheet only grazes).
        """
        import trimesh
        if not isinstance(mesh, trimesh.Trimesh):
            mesh = trimesh.load(mesh, force="mesh")
        if bounds is not None:
            lo = np.asarray(bounds[0], float)
            hi = np.asarray(bounds[1], float)
            # keep triangles whose bounding box overlaps the (padded) box
            tri = mesh.triangles
            tlo, thi = tri.min(1), tri.max(1)
            keep = np.all((thi >= lo - 2 * dx) & (tlo <= hi + 2 * dx), axis=1)
            m = mesh.submesh([np.flatnonzero(keep)], append=True)
            g = cls.empty(lo, hi, dx)
            _mark_surface(g, m.triangles, exact=(cut == "exact"))
            g.triangles = np.asarray(m.triangles, float)
            if fill_interior:
                from scipy import ndimage
                g.solid = ndimage.binary_fill_holes(g.solid)
            return g
        pad = 3 * dx if pad is None else pad
        vox = mesh.voxelized(pitch=dx)
        if fill_interior:
            vox = vox.fill()
        mat = vox.matrix.astype(bool)
        # voxel centre of index (0,0,0) in mesh coordinates
        c0 = np.asarray(vox.transform)[:3, 3]
        npad = int(np.ceil(pad / dx))
        solid = np.pad(mat, npad, constant_values=False)
        origin = c0 - 0.5 * dx - npad * dx
        g = cls(shape=solid.shape, dx=float(dx), origin=origin, solid=solid)
        g.triangles = np.asarray(mesh.triangles, float)
        return g

    def copy(self) -> "Grid":
        return Grid(self.shape, self.dx, self.origin.copy(), self.solid.copy(),
                    triangles=self.triangles)


def sample_triangles(tri: np.ndarray, h: float, spacing: float = 0.45,
                     max_points: int = 2_000_000):
    """Yield chunks of points on the triangles with spacing below
    ``spacing * h`` (barycentric lattice per triangle)."""
    tri = np.asarray(tri, float)
    emax = np.max(np.linalg.norm(tri[:, [1, 2, 0]] - tri, axis=2), axis=1)
    nsub = np.maximum(1, np.ceil(emax / (spacing * h))).astype(int)
    for n in np.unique(nsub):
        T = tri[nsub == n]
        i, j = np.meshgrid(np.arange(n + 1), np.arange(n + 1), indexing="ij")
        ok = i + j <= n
        w1 = (i[ok] / n)[:, None]
        w2 = (j[ok] / n)[:, None]
        w0 = 1.0 - w1 - w2
        chunk = max(1, int(max_points // len(w0)))
        for s0 in range(0, len(T), chunk):
            t = T[s0:s0 + chunk]
            yield (w0[None] * t[:, None, 0] + w1[None] * t[:, None, 1]
                   + w2[None] * t[:, None, 2]).reshape(-1, 3)


def _mark_surface(grid: "Grid", tri: np.ndarray, spacing: float = 0.45, exact: bool = True):
    """Mark every cell touched by the triangles as solid.

    exact=True (default, 3D): every cell whose closed box touches a triangle
    (``voxel.surface_cells``, separating-axis test). exact=False: each
    triangle is sampled on a barycentric lattice with point spacing
    below ``spacing * dx``, so consecutive samples fall in the same or a
    neighbouring (26-connected) cell: thin sheets become leak-free walls for
    face-connected fluid. Memory-light alternative to trimesh voxelisation.
    """
    dx = grid.dx
    tri = np.asarray(tri, float)
    if exact and grid.ndim == 3:
        from .voxel import cut_cells
        keys = cut_cells(tri, grid.origin, dx, grid.shape, spacing)
        grid.solid.reshape(-1)[keys] = True
        grid._nbr = None
        return
    shape = np.array(grid.shape)
    for P in sample_triangles(tri, dx, spacing=spacing):
        idx = np.floor((P - grid.origin) / dx).astype(np.int64)
        good = np.all((idx >= 0) & (idx < shape), axis=1)
        idx = idx[good]
        grid.solid[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    grid._nbr = None
