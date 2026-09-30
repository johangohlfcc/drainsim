"""Door-like 2D section: dip-in, hold, dip-out while tilting, then drain.

Compares
  * transient     - compartments + throats (this work)
  * equilibrium   - every opening instantaneous, with spill routing
  * legacy        - every opening instantaneous, escaped liquid lost
and sweeps the drain-hole size for the transient model.
"""
import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from drainsim import cases
from drainsim.model import Simulation
from drainsim.motion import Keyframes
from drainsim.viz import plot_state_2d

out = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(out, exist_ok=True)

# down 0.9 m in 6 s, hold 2 s, up in 6 s while tilting 25 deg, drain 30 s
motion = Keyframes([0, 6, 8, 14, 44], [0, 0, 0, 25, 25],
                   np.array([[0, 0.15], [0, -0.75], [0, -0.75], [0, 0.15], [0, 0.15]]),
                   2, pivot=np.array([0, 0.35]))
snap_t = [3, 5, 7, 10, 12, 14, 18, 44]


def run(hole, **kw):
    g = cases.door_section(dx=0.002, hole=hole)
    sim = Simulation(g, motion, dt_max=0.05, **kw)
    t0 = time.time()
    H = sim.run(snapshot_times=snap_t)
    print(f"hole {hole*1e3:.0f} mm {kw}: {sim.nsteps} steps, "
          f"{time.time()-t0:.0f} s, final retained {H.liquid_retained[-1]*1e3:.2f} l/m")
    return sim, H.arrays(), H.snapshots


res = {"transient": run(0.008),
       "equilibrium + spill routing": run(0.008, split=False),
       "legacy (escaped liquid lost)": run(0.008, split=False, spill_routing=False)}

# ---- model comparison
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for (name, (sim, a, _)), col in zip(res.items(), ["#d62728", "#1f77b4", "#7f7f7f"]):
    ax[0].plot(a["t"], a["liquid_above_bath"] * 1e3, color=col, label=name)
    ax[1].plot(a["t"], a["air_below_bath"] * 1e3, color=col, label=name)
for x in ax:
    x.axvspan(8, 14, color="#eeeeee", zorder=0)
    x.set_xlabel("time [s]")
    x.legend(fontsize=8)
ax[0].set_ylabel("liquid above bath surface [l per m depth]")
ax[1].set_ylabel("air below bath surface [l per m depth]")
ax[0].set_title("carried liquid (grey band = dip-out + tilt)", fontsize=10)
ax[1].set_title("accessibility: air not yet displaced", fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(out, "03_door_models.png"), dpi=130)

# ---- snapshots of the transient run
sim, a, snaps = res["transient"]
fig, axes = plt.subplots(2, 4, figsize=(12, 7.5))
for ax, t in zip(axes.ravel(), snap_t):
    up, zb = motion.frame(t)
    i = np.argmin(abs(a["t"] - t))
    plot_state_2d(sim.grid, snaps[t], ax, f"t = {t} s   liquid above bath "
                  f"{a['liquid_above_bath'][i]*1e3:.1f} l/m", comp=sim.comp,
                  up=up, zb=zb)
fig.tight_layout()
fig.savefig(os.path.join(out, "03_door_snapshots.png"), dpi=110)

# ---- hole-size sweep (transient model only)
fig, ax = plt.subplots(figsize=(6, 4))
for hole, col in [(0.004, "#9467bd"), (0.008, "#d62728"), (0.016, "#ff7f0e")]:
    sim, a, _ = res["transient"] if hole == 0.008 else run(hole)
    ax.plot(a["t"], a["liquid_above_bath"] * 1e3, color=col,
            label=f"drain holes {hole*1e3:.0f} mm")
eq = res["equilibrium + spill routing"][1]
ax.plot(eq["t"], eq["liquid_above_bath"] * 1e3, ":", color="#1f77b4",
        label="equilibrium (instant)")
ax.axvspan(8, 14, color="#eeeeee", zorder=0)
ax.set_xlabel("time [s]")
ax.set_ylabel("liquid above bath [l per m depth]")
ax.set_title("drainage transient vs drain-hole size", fontsize=10)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(out, "03_door_holes.png"), dpi=130)
