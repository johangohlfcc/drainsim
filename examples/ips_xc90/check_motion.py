"""Setup check of the IPS XC90 case (no simulation): the motion built from
the .xmo (car_article.ips_motion) gives IPS's object pose at every keyframe
and the STL itself at t = 0; the car's depth below the bath over time; a
side view of the motion (runs/ips_xc90/motion_side.png)."""
import os
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
OUT = os.path.join(REPO, "runs", "ips_xc90")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, os.path.join(REPO, "examples"))
sys.path.insert(0, REPO)
from car_article import ips_motion, read_xmo                      # noqa: E402
from drainsim.motion import rot3                                  # noqa: E402
from drainsim.worldviz import to_world                            # noqa: E402

STL = os.path.join(REPO, "xc90_10mm.stl")
XMO = os.path.join(REPO, "XC90_motion.xmo")
BATH = os.path.join(REPO, "bath.stl")
HANG = 180.0

dt = np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
with open(STL, "rb") as f:
    f.seek(80)
    n = int(np.frombuffer(f.read(4), "<u4")[0])
V = np.memmap(STL, dtype=dt, mode="r", offset=84, shape=(n,))["v"]
lo = np.full(3, np.inf)
hi = np.full(3, -np.inf)
for s0 in range(0, n, 4_000_000):
    b = np.asarray(V[s0:s0 + 4_000_000], np.float64).reshape(-1, 3)
    lo, hi = np.minimum(lo, b.min(0)), np.maximum(hi, b.max(0))
centre = 0.5 * (lo + hi)                       # load_car's centre (metres: scale 1)
mo, info = ips_motion(XMO, BATH, centre, 1.0, HANG)
W0 = np.array(info["world_origin_plant"])
print(f"car {n} triangles, size {np.round(hi - lo, 3)} m, centre {np.round(centre, 3)}")
print(f"bath surface z = {info['bath_level_plant']:.3f} m (plant); world origin {np.round(W0, 3)}")
print(f"pivot in object coordinates {np.round(mo.pivot, 4)} m; offset {np.round(info['offset_m'], 4)} m")
print(f"keyframes {np.round(info['times'], 2).tolist()}")
print(f"angles (deg, unwrapped) {np.round(info['angles_deg'], 2).tolist()}")
print(f"motion to t = {info['t_dipout']:.2f} s, then {HANG:g} s hanging: t_end {mo.t_end:.2f} s")

# 1. t = 0: the STL itself
rng = np.random.default_rng(0)
idx = np.sort(rng.choice(n, 200_000, replace=False))
P = np.asarray(V[idx], np.float64).reshape(-1, 3)
err0 = np.abs(to_world(P - centre, mo, 0.0) - (P - W0)).max()
print(f"t = 0: the posed car equals the STL within {err0:.2e} m")

# 2. every keyframe: IPS's object pose (origin T or T + R off, rotation R)
fr = read_xmo(XMO)
q0 = fr[0][1]
R0 = rot3([0.0, np.degrees(2 * np.arctan2(q0[2], q0[0])), 0.0])
O0 = fr[0][2] if fr[0][3] is None else fr[0][2] + R0 @ fr[0][3]
B = np.array([[0, 0, 0], [1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])      # body points (object axes)
worst = 0.0
for (t, q, T, off), a in zip(fr, info["angles_deg"]):
    R = rot3([0.0, a, 0.0])
    O = T if off is None else T + R @ off
    ips = O + B @ (R @ R0.T).T                                   # IPS: body points in the plant
    ours = to_world(O0 + B @ np.eye(3) - centre, mo, t) + W0
    worst = max(worst, np.abs(ips - ours).max())
print(f"keyframes: our pose equals IPS's object pose within {worst:.2e} m")

# 3. depth of the car below the bath surface
print("   t (s)   angle   car lowest / highest point (m, bath surface = 0)")
for t in [0, 3, 5.45, 8, 10, 13.37, 20.58, 27.51, 34.71, 41.49, 45, 48, 50, 52, 54.53, 60]:
    W = to_world(P - centre, mo, t)
    a = np.interp(t, mo.times, mo.angles[:, 1]) + info["angles_deg"][0]
    print(f"  {t:6.2f}  {a:7.1f}   {W[:, 2].min():7.3f} / {W[:, 2].max():7.3f}"
          + ("   under" if W[:, 2].max() < 0 else "   out" if W[:, 2].min() > 0 else ""))

# 4. side view
import matplotlib                                                 # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402
fig, ax = plt.subplots(figsize=(14, 4))
sub = P[::20] - centre
for t in np.arange(0.0, info["t_dipout"] + 0.01, 2.5):
    W = to_world(sub, mo, t)
    ax.plot(W[:, 0], W[:, 2], ",", alpha=0.25)
    ax.text(W[:, 0].mean(), W[:, 2].max() + 0.1, f"{t:g}", fontsize=7, ha="center")
bx = np.array([995.61, 1035.61]) - W0[0]
ax.axhline(0.0, color="b", lw=0.8)
ax.plot([bx[0], bx[0], bx[1], bx[1]], [0, -4.0, -4.0, 0], "b-", lw=0.8)
ax.set_aspect("equal")
ax.set_xlabel("x (m, plant x - %.2f)" % W0[0])
ax.set_ylabel("z (m, bath surface = 0)")
ax.set_title("IPS XC90 dip: the car every 2.5 s (labels: t in s)")
fig.savefig(os.path.join(OUT, "motion_side.png"), dpi=110, bbox_inches="tight")
print("wrote", os.path.join(OUT, "motion_side.png"))
