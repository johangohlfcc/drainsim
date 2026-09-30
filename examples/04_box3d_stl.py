"""3D pipeline from a triangle mesh (STL): voxelise, segment, dip, drain,
then export world-frame VTK (the box moves through a fixed bath) and render
an MP4 of it.

The mesh is generated here with trimesh (thin plates, like sheet metal):
a covered box with two chambers separated by a divider that has a gap at the
bottom, a drain hole in the floor of chamber 1, a small vent in the lid and an
open cup on the outside. Replace ``mesh`` with ``trimesh.load("part.stl")``
to run your own geometry (units: metres).
"""
import os
import time
import numpy as np
import trimesh
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from drainsim.grid import Grid
from drainsim.model import Simulation
from drainsim.motion import Keyframes
from drainsim.worldviz import export_world

out = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(out, exist_ok=True)
t_pl = 0.003          # plate thickness


def plate(lo, hi):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    return trimesh.creation.box(extents=hi - lo,
                                transform=trimesh.transformations.translation_matrix((lo + hi) / 2))


Lx, Ly, Lz = 0.40, 0.30, 0.20
hole, vent, gap = 0.03, 0.03, 0.04
parts = [
    # floor with a square drain hole centred at x = 0.1
    plate([0, 0, 0], [0.1 - hole / 2, Ly, t_pl]),
    plate([0.1 + hole / 2, 0, 0], [Lx, Ly, t_pl]),
    plate([0.1 - hole / 2, 0, 0], [0.1 + hole / 2, Ly / 2 - hole / 2, t_pl]),
    plate([0.1 - hole / 2, Ly / 2 + hole / 2, 0], [0.1 + hole / 2, Ly, t_pl]),
    # side walls
    plate([0, 0, 0], [t_pl, Ly, Lz]), plate([Lx - t_pl, 0, 0], [Lx, Ly, Lz]),
    plate([0, 0, 0], [Lx, t_pl, Lz]), plate([0, Ly - t_pl, 0], [Lx, Ly, Lz]),
    # lid with a vent above chamber 2
    plate([0, 0, Lz - t_pl], [0.3 - vent / 2, Ly, Lz]),
    plate([0.3 + vent / 2, 0, Lz - t_pl], [Lx, Ly, Lz]),
    plate([0.3 - vent / 2, 0, Lz - t_pl], [0.3 + vent / 2, Ly / 2 - vent / 2, Lz]),
    plate([0.3 - vent / 2, Ly / 2 + vent / 2, Lz - t_pl], [0.3 + vent / 2, Ly, Lz]),
    # divider with a gap at the bottom
    plate([0.2 - t_pl / 2, 0, gap], [0.2 + t_pl / 2, Ly, Lz]),
    # outside cup on the +x wall (open top)
    plate([Lx, 0.1, 0.08], [Lx + 0.06, 0.2, 0.08 + t_pl]),
    plate([Lx + 0.06 - t_pl, 0.1, 0.08], [Lx + 0.06, 0.2, 0.13]),
    plate([Lx, 0.1, 0.08], [Lx + 0.06, 0.1 + t_pl, 0.13]),
    plate([Lx, 0.2 - t_pl, 0.08], [Lx + 0.06, 0.2, 0.13]),
]
mesh = trimesh.util.concatenate(parts)
mesh.export(os.path.join(out, "04_box.stl"))

grid = Grid.from_mesh(os.path.join(out, "04_box.stl"), dx=0.008, pad=0.04)
print("grid", grid.shape, grid.ncells, "cells")

# dip 0.35 m in 4 s, hold 1 s, out in 4 s while tilting 20 deg about y, drain
pivot = np.array([Lx / 2, Ly / 2, Lz / 2])
T = [0, 4, 5, 9, 30]
ang = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 20, 0], [0, 20, 0]])
sh = np.array([[0, 0, 0.05], [0, 0, -0.30], [0, 0, -0.30], [0, 0, 0.05], [0, 0, 0.05]])
motion = Keyframes(T, ang, sh, ndim=3, pivot=pivot)

t0 = time.time()
sim = Simulation(grid, motion, dt_max=0.05, segment_kwargs=dict(d_free=0.06),
                 film=True)
print("compartments", sim.comp.n, "throats:")
for t in sim.comp.throats:
    print(f"   {t.a}-{t.b}: area {t.area*1e4:.1f} cm2, d {t.diameter*1e3:.0f} mm,"
          f" at {np.round(t.centroid, 3)}")
H = sim.run(snapshot_every=0.2)
print(f"{sim.nsteps} steps in {time.time()-t0:.0f} s")
a = H.arrays()
print(f"wall film {a['film_volume'][-1]*1e3:.3f} l, drops {a['n_drips'][-1]} "
      f"({a['drip_volume'][-1]*1e3:.3f} l)")
print("final pockets:", [(round(v * 1e3, 3), np.round(c, 3)) for v, c in sim.pockets(1e-6)["liquid"]])

# world-frame VTK series: open examples/out/vtk_box/*.pvd in ParaView and
# press Play, or run box_open_in_paraview.py from ParaView's Python shell
files = export_world(sim, H, os.path.join(out, "vtk_box"), prefix="box",
                     object_mesh=os.path.join(out, "04_box.stl"))
print("VTK:", files)
try:
    from drainsim.render3d import render_animation
    render_animation(os.path.join(out, "vtk_box"), "box",
                     os.path.join(out, "04_box3d.mp4"), fps=20)
    print("rendered", os.path.join(out, "04_box3d.mp4"))
except ImportError:
    print("install 'vtk' (pip) to render the MP4")

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(a["t"], a["liquid_above_bath"] * 1e3, label="liquid above bath")
for k in range(1, sim.comp.n):
    ax[0].plot(a["t"], a["liquid_comp"][:, k] * 1e3, "--", label=f"inside compartment {k}")
ax[0].plot(a["t"], a["liquid_exterior"] * 1e3, ":", label="exterior pools (outside cup)")
ax[1].plot(a["t"], a["air_below_bath"] * 1e3, label="air below bath")
ax[1].plot(a["t"], a["air_trapped"] * 1e3, "--", label="trapped exterior pockets")
for x in ax:
    x.set_xlabel("time [s]")
    x.set_ylabel("litres")
    x.legend(fontsize=8)
    x.axvspan(5, 9, color="#eeeeee", zorder=0)
fig.suptitle("3D box from STL: dip, tilted dip-out, drainage", fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(out, "04_box3d.png"), dpi=130)
