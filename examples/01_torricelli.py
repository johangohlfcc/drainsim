"""Verification: a vented box draining through a hole in its floor (2D).

Compares the simulated level with the analytic orifice solution
sqrt(h - h_cap) = sqrt(h0 - h_cap) - Cd a/A sqrt(g/2) t.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from drainsim import cases
from drainsim.model import Simulation
from drainsim.physics import Fluid, ThroatModel, torricelli_level

out = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(out, exist_ok=True)

fig, ax = plt.subplots(figsize=(6, 4))
for hole, col in [(0.010, "#1f77b4"), (0.015, "#ff7f0e"), (0.025, "#2ca02c")]:
    g, interior = cases.torricelli_box(dx=0.005, hole=hole)
    sim = Simulation(g, cases.static(t_end=25), initial_L=interior.astype(float),
                     dt_max=0.02)
    H = sim.run().arrays()
    th = min(sim.comp.throats, key=lambda t: t.centroid[1])
    W = 0.38
    level = 0.31 + H["liquid_comp"][:, 1] / W - th.centroid[1]
    hc = ThroatModel().holdup_head(th.diameter, Fluid(), 2)
    ana = torricelli_level(H["t"], level[0], 0.6, th.area, W, h_cap=hc)
    ax.plot(H["t"], level, color=col, label=f"sim, opening {th.area*1e3:.0f} mm")
    ax.plot(H["t"], ana, "--", color=col, lw=1)
    print(f"opening {th.area*1e3:.1f} mm: max |sim-analytic| = "
          f"{np.abs(level-ana).max()*1e3:.2f} mm, steps {sim.nsteps}")
ax.set_xlabel("time [s]")
ax.set_ylabel("liquid level above opening [m]")
ax.set_title("2D box drainage: simulation (solid) vs analytic (dashed)", fontsize=10)
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(out, "01_torricelli.png"), dpi=130)
