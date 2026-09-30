"""Door section with the wall-film model (water).

Same dip as example 05: down, hold, up while tilting 25 deg, then 60 s of
dripping. Reports the carried liquid split into cavity liquid and wall film,
the drip rate, and the time of the last drop; writes an animation with the
film drawn along the walls, and a world-frame VTK series.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from drainsim import cases
from drainsim.model import Simulation
from drainsim.motion import Keyframes
from drainsim.worldviz import animate_2d, export_world

out = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(out, exist_ok=True)

motion = Keyframes([0, 6, 8, 14, 74], [0, 0, 0, 25, 25],
                   np.array([[0, 0.15], [0, -0.75], [0, -0.75], [0.3, 0.15], [0.3, 0.15]]),
                   2, pivot=np.array([0, 0.35]))
grid = cases.door_section(dx=0.002, hole=0.008)
sim = Simulation(grid, motion, dt_max=0.1, film=True)
print("film elements", sim.film.c.n)
H = sim.run(snapshot_every=0.2)
a = H.arrays()
f = sim.film
drips = np.array([(d[0], d[3]) for d in f.s.drips]) if f.s.drips else np.zeros((0, 2))
t_last = drips[-1, 0] if len(drips) else np.nan
print(f"final: cavity liquid {a['liquid_above_bath'][-1]*1e3:.2f} l/m, "
      f"wall film {a['film_volume'][-1]*1e3:.2f} l/m, drops {a['n_drips'][-1]}, "
      f"last drop at t = {t_last:.1f} s")
print(f"film taken from bath {f.from_bath_volume*1e3:.2f} l/m, dripped "
      f"{f.drip_volume*1e3:.2f} l/m, handed to cavities as puddles {f.bulk_volume*1e3:.2f} l/m")

fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
ax[0].plot(a["t"], a["liquid_above_bath"] * 1e3, color="#1f5fa8")
ax[0].set_title("cavity / pool liquid above bath", fontsize=10)
ax[0].set_ylabel("l per m depth")
ax[1].plot(a["t"], a["film_volume"] * 1e3, color="#d95f02", label="on the walls")
ax[1].plot(a["t"], a["drip_volume"] * 1e3, color="#7570b3", label="dripped (cumulative)")
ax[1].set_title("wall film", fontsize=10)
ax[1].legend(fontsize=8)
if len(drips):
    bins = np.arange(0, a["t"][-1] + 2, 2.0)
    cnt, _ = np.histogram(drips[:, 0], bins, weights=drips[:, 1])
    ax[2].bar(bins[:-1], cnt, width=2.0, align="edge", color="#7570b3")
    ax[2].axvline(t_last, color="k", lw=1, ls=":")
    ax[2].text(t_last, 0.9 * max(cnt.max(), 1), f" last drop {t_last:.1f} s", fontsize=8)
ax[2].set_title("drops per 2 s", fontsize=10)
for x in ax:
    x.set_xlabel("time [s]")
    x.axvspan(8, 14, color="#eeeeee", zorder=0)
fig.tight_layout()
fig.savefig(os.path.join(out, "07_door_film.png"), dpi=130)

animate_2d(sim, H, os.path.join(out, "07_door_film.mp4"), fps=10,
           times=sorted(H.snapshots)[::2], title="door section with wall film (water)")
print(export_world(sim, H, os.path.join(out, "vtk_door_film"), prefix="door",
                   volume=False))
