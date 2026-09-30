"""World-frame animation of the 2D door section: the door moves through a
fixed bath (the simulation itself moves the bath surface instead).

Writes examples/out/05_door.mp4, 05_door.gif and a VTK series in
examples/out/vtk_door/ (open door_*.pvd in ParaView, press Play).
"""
import os
import numpy as np

from drainsim import cases
from drainsim.model import Simulation
from drainsim.motion import Keyframes
from drainsim.worldviz import animate_2d, export_world

out = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(out, exist_ok=True)

# down 0.9 m in 6 s, hold 2 s, move up and 0.3 m sideways in 6 s while
# tilting 25 deg, then drain for 16 s
motion = Keyframes([0, 6, 8, 14, 30], [0, 0, 0, 25, 25],
                   np.array([[0, 0.15], [0, -0.75], [0, -0.75], [0.3, 0.15], [0.3, 0.15]]),
                   2, pivot=np.array([0, 0.35]))
grid = cases.door_section(dx=0.002, hole=0.008)
sim = Simulation(grid, motion, dt_max=0.05)
dt_snap = 0.1
H = sim.run(snapshot_every=dt_snap)
print(f"{sim.nsteps} steps, {len(H.snapshots)} frames")

animate_2d(sim, H, os.path.join(out, "05_door.mp4"), fps=int(1 / dt_snap),
           title="transient model, 8 mm drain holes")
# a lighter GIF (every 3rd frame, 3x real time)
times = sorted(H.snapshots)[::3]
animate_2d(sim, H, os.path.join(out, "05_door.gif"), fps=10, times=times,
           dpi=70, title="transient model, 8 mm drain holes")
print(export_world(sim, H, os.path.join(out, "vtk_door"), prefix="door"))
