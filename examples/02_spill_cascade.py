"""Spill routing: liquid leaving a cavity is routed to the next cavity.

A partly filled cup is tilted. With fill-spill-merge the overflow lands in the
cup below; with the original behaviour (spill_routing=False) it is lost.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from drainsim import cases
from drainsim.fsm import fill_spill
from drainsim.viz import plot_state_2d

out = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(out, exist_ok=True)

g = cases.two_cups(dx=0.004)
X = g.centers()
fl = g.fluid.ravel()
reg = np.where(fl, 0, -1)
nbr = g.neighbors()
sink = g.boundary_mask().ravel() & fl
src = np.where(fl & (X[:, 0] > 0.15) & (X[:, 0] < 0.35) & (X[:, 1] > 0.75)
               & (X[:, 1] < 0.95), g.cell_volume, 0.0)
upright, _, _, _ = fill_spill(X[:, 1], reg, nbr, sink, src, g.cell_volume)

fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
plot_state_2d(g, upright / g.cell_volume, axes[0], "upright")
th = np.radians(35)
up = np.array([-np.sin(th), np.cos(th)])
for ax, routing, name in [(axes[1], True, "tilted 35 deg: fill-spill-merge"),
                          (axes[2], False, "tilted 35 deg: escaped liquid lost")]:
    r, dr, _, _ = fill_spill(X @ up, reg, nbr, sink, upright, g.cell_volume,
                             spill_routing=routing)
    plot_state_2d(g, r / g.cell_volume, ax,
                  f"{name}\nretained {r.sum()*1e3:.1f} l/m, lost {dr.sum()*1e3:.1f} l/m")
fig.tight_layout()
fig.savefig(os.path.join(out, "02_spill_cascade.png"), dpi=130, bbox_inches="tight")
print("saved")
