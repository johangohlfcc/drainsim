"""Film model verification.

(a) 2D vertical wall with an initial 100 um film vs Jeffreys' solution
    h = sqrt(mu x / (rho g t)) (x from the top edge), at 5, 20 and 60 s.
(b) 3D vertical plate, same check.
(c) Film deposited on a wall withdrawn from the bath at speed U vs the
    Landau-Levich-Derjaguin law used in the model.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from drainsim import cases, shapes as sh
from drainsim.grid import Grid
from drainsim.model import Simulation
from drainsim.motion import Keyframes

out = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(out, exist_ok=True)
mu, rho, g0 = 1e-3, 1000.0, 9.81
fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
cols = ["#1f77b4", "#ff7f0e", "#2ca02c"]

# (a) 2D
g = Grid.empty([0, 0], [0.3, 0.6], 0.002)
sh.add(g, sh.box(g, [0.1, 0.05], [0.15, 0.55]))
sim = Simulation(g, cases.static(t_end=60), dt_max=0.25, film=True)
c = sim.film.c
sim.film.s.h[c.normal[:, 0] > 0.7] = 1e-4
H = sim.run(snapshot_times=[5, 20, 60])
face = (c.normal[:, 0] > 0.95) & (c.x[:, 1] > 0.06) & (c.x[:, 1] < 0.54)
xs = 0.55 - c.x[face, 1]
o = np.argsort(xs)
for t, col in zip([5, 20, 60], cols):
    ax[0].plot(xs[o], H.snap_film[t][face][o] * 1e6, color=col, label=f"model t={t} s")
    xx = np.linspace(0, 0.5, 200)
    ax[0].plot(xx, np.minimum(np.sqrt(mu * xx / (rho * g0 * t)), 1e-4) * 1e6, "--",
               color=col, lw=1)
ax[0].set_title("(a) 2D wall: model (solid) vs Jeffreys (dashed)", fontsize=10)
ax[0].set_xlabel("distance from top edge [m]")
ax[0].set_ylabel("film thickness [µm]")
ax[0].legend(fontsize=8)

# (b) 3D
g = Grid.empty([-0.1, -0.1, 0], [0.1, 0.1, 0.4], 0.005)
sh.add(g, sh.box(g, [-0.03, -0.06, 0.05], [0.03, 0.06, 0.35]))
sim = Simulation(g, cases.static(ndim=3, t_end=60), dt_max=0.25, film=True)
c = sim.film.c
sim.film.s.h[c.normal[:, 0] > 0.7] = 1e-4
H = sim.run(snapshot_times=[5, 20, 60])
face = (c.normal[:, 0] > 0.95) & (np.abs(c.x[:, 1]) < 0.04)
xs = 0.35 - c.x[face, 2]
for t, col in zip([5, 20, 60], cols):
    ax[1].plot(xs, H.snap_film[t][face] * 1e6, ".", ms=2, color=col, label=f"model t={t} s")
    xx = np.linspace(0, 0.3, 200)
    ax[1].plot(xx, np.minimum(np.sqrt(mu * xx / (rho * g0 * t)), 1e-4) * 1e6, "--",
               color=col, lw=1)
ax[1].set_title("(b) 3D plate (triangles): model vs Jeffreys", fontsize=10)
ax[1].set_xlabel("distance from top edge [m]")
ax[1].legend(fontsize=8)

# (c) deposition vs withdrawal speed
Us = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
sim_h = []
for U in Us:
    g = Grid.empty([0, -0.05], [0.3, 0.45], 0.003)
    sh.add(g, sh.box(g, [0.1, 0.05], [0.16, 0.35]))
    T = 0.4 / U
    m = Keyframes([0, T], [0, 0], np.array([[0, -0.4], [0, 0.0]]), 2)
    sim = Simulation(g, m, dt_max=0.05, film=True)
    H = sim.run(snapshot_times=[T])
    c = sim.film.c
    low = (c.normal[:, 0] > 0.95) & (c.x[:, 1] > 0.08) & (c.x[:, 1] < 0.12)
    sim_h.append(np.mean(H.snap_film[T][low]))
UU = np.logspace(-3, 0, 100)
ax[2].loglog(UU, sim.film.deposit_thickness(UU) * 1e6, "-", color="#555",
             label="Landau-Levich / Derjaguin")
ax[2].loglog(Us, np.array(sim_h) * 1e6, "o", color="#d62728",
             label="model, film just after emergence")
ax[2].set_xlabel("withdrawal speed [m/s]")
ax[2].set_ylabel("deposited film [µm]")
ax[2].set_title("(c) deposition, water", fontsize=10)
ax[2].legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(out, "06_film_validation.png"), dpi=130)
print("saved")
