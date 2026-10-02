"""The car's hanging (drip-off) pose of the IPS XC90 case: side, front and
top views, and an oblique view, from points of the mesh."""
import os
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
OUT = os.path.join(REPO, "runs", "ips_xc90")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, os.path.join(REPO, "examples"))
sys.path.insert(0, REPO)
from car_article import ips_motion                               # noqa: E402
from drainsim.worldviz import to_world                           # noqa: E402

STL = os.path.join(REPO, "xc90_10mm.stl")
dt = np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
with open(STL, "rb") as f:
    f.seek(80)
    n = int(np.frombuffer(f.read(4), "<u4")[0])
V = np.memmap(STL, dtype=dt, mode="r", offset=84, shape=(n,))["v"]
lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
for s0 in range(0, n, 4_000_000):
    b = np.asarray(V[s0:s0 + 4_000_000], np.float64).reshape(-1, 3)
    lo, hi = np.minimum(lo, b.min(0)), np.maximum(hi, b.max(0))
centre = 0.5 * (lo + hi)
mo, info = ips_motion(os.path.join(REPO, "XC90_motion.xmo"), os.path.join(REPO, "bath.stl"),
                      centre, 1.0, 180.0)
rng = np.random.default_rng(1)
idx = np.sort(rng.choice(n, 1_500_000, replace=False))
T = np.asarray(V[idx], np.float64)
w = rng.random((len(idx), 2))
w = np.where(w.sum(1, keepdims=True) > 1, 1 - w, w)                 # uniform on each triangle
P = T[:, 0] + w[:, :1] * (T[:, 1] - T[:, 0]) + w[:, 1:] * (T[:, 2] - T[:, 0])
t = float(info["t_dipout"])
W = to_world(P - centre, mo, t)
ang = info["angles_deg"][-1]
print(f"hanging pose from t = {t:.2f} s to {mo.t_end:.2f} s: pitch {ang:.1f} deg "
      f"(= {ang % 360:.1f} deg, the pose of the STL), car z {W[:, 2].min():.3f} .. {W[:, 2].max():.3f} m "
      f"above the bath surface, x {W[:, 0].min():.2f} .. {W[:, 0].max():.2f} m")

import matplotlib                                                 # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402
fig = plt.figure(figsize=(16, 10))
x, y, z = W.T
kw = dict(s=0.02, c=z, cmap="viridis", lw=0, rasterized=True)
ax = fig.add_subplot(2, 2, 1)
o = np.argsort(-y)                                                # near side last
ax.scatter(x[o], z[o], **kw)
ax.axhline(0, color="b", lw=1)
ax.text(x.min(), 0.05, "bath surface", color="b", fontsize=8)
ax.set_aspect("equal"); ax.set_title(f"side (plant x-z), pitch {ang % 360:.1f} deg")
ax.set_xlabel("x (m)"); ax.set_ylabel("z (m, bath surface = 0)")
ax = fig.add_subplot(2, 2, 2)
o = np.argsort(x)
ax.scatter(y[o], z[o], **kw)
ax.axhline(0, color="b", lw=1)
ax.set_aspect("equal"); ax.set_title("end view (plant y-z)")
ax.set_xlabel("y (m)"); ax.set_ylabel("z (m)")
ax = fig.add_subplot(2, 2, 3)
o = np.argsort(z)
ax.scatter(x[o], y[o], **kw)
ax.set_aspect("equal"); ax.set_title("top (plant x-y)")
ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
ax = fig.add_subplot(2, 2, 4, projection="3d")
s = rng.choice(len(x), 400_000, replace=False)
ax.scatter(x[s], y[s], z[s], s=0.03, c=z[s], cmap="viridis", lw=0)
ax.view_init(elev=18, azim=-55)
ax.set_box_aspect((np.ptp(x), np.ptp(y), np.ptp(z)))
ax.set_title("oblique")
fig.suptitle(f"IPS XC90: hanging (drip-off) pose, t = {t:.1f}-{mo.t_end:.1f} s", fontsize=13)
fig.savefig(os.path.join(OUT, "final_pose.png"), dpi=100, bbox_inches="tight")
print("wrote", os.path.join(OUT, "final_pose.png"))
