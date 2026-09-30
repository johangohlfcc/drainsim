"""Plotting (2D) and VTK export (2D/3D, open in ParaView)."""
from __future__ import annotations

import numpy as np

from .grid import Grid


def plot_state_2d(grid: Grid, L, ax=None, title=None, comp=None, up=None,
                  zb=None, show_throats=True, show_compartments=False):
    """Liquid fraction (blue) on a 2D grid, solids dark, optional bath line."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    ext = [grid.origin[0], grid.origin[0] + grid.shape[0] * grid.dx,
           grid.origin[1], grid.origin[1] + grid.shape[1] * grid.dx]
    Lg = np.asarray(L)[:grid.ncells].reshape(grid.shape)
    img = np.ma.masked_where(grid.solid, Lg)
    cmap = plt.get_cmap("Blues").copy()
    cmap.set_bad("#222222")
    ax.imshow(img.T, origin="lower", extent=ext, cmap=cmap, vmin=0, vmax=1.3,
              interpolation="nearest")
    if show_compartments and comp is not None:
        lab = np.ma.masked_where(comp.label[:grid.ncells].reshape(grid.shape) <= 0,
                                 comp.label[:grid.ncells].reshape(grid.shape))
        ax.imshow(lab.T, origin="lower", extent=ext, cmap=ListedColormap(
            ["#f4a261", "#2a9d8f", "#e76f51", "#8ab17d"]), alpha=0.25,
            interpolation="nearest")
    if show_throats and comp is not None:
        for t in comp.throats:
            ax.plot(*t.centroid, "o", ms=4, mfc="none", mec="#d62728")
    if up is not None and zb is not None:
        # bath line: points x with up.x = zb
        t = np.array([up[1], -up[0]])
        p0 = up * zb
        s = np.linspace(-3, 3, 2)
        ax.plot(p0[0] + s * t[0], p0[1] + s * t[1], "-", color="#1f77b4", lw=1)
        # arrow for gravity
    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=9)
    return ax


def write_vtk(path: str, grid: Grid, fields: dict):
    """Legacy VTK structured-points file with cell data (ParaView-readable)."""
    shape = list(grid.shape) + [1] * (3 - grid.ndim)
    origin = list(grid.origin) + [0.0] * (3 - grid.ndim)
    n = int(np.prod(shape))
    with open(path, "w") as f:
        f.write("# vtk DataFile Version 3.0\ndrainsim\nASCII\n")
        f.write("DATASET STRUCTURED_POINTS\n")
        f.write("DIMENSIONS {} {} {}\n".format(*[s + 1 for s in shape]))
        f.write("ORIGIN {} {} {}\n".format(*origin))
        f.write("SPACING {0} {0} {0}\n".format(grid.dx))
        f.write(f"CELL_DATA {n}\n")
        for name, arr in fields.items():
            a = np.asarray(arr, float)[:grid.ncells].reshape(grid.shape)
            # VTK wants x fastest -> Fortran order
            a = a.ravel(order="F")
            f.write(f"SCALARS {name} float 1\nLOOKUP_TABLE default\n")
            np.savetxt(f, a, fmt="%.5g")
