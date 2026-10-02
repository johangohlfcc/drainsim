"""Time loop of a py-spy raw profile: the par.dual kernels (par.py call)
by the line that called them, and the fill-spill (_equilibrate) part by
line (innermost drainsim frame, par.call replaced by its caller)."""
import re
import sys
from collections import Counter
FR = re.compile(r"(.+?) \((.+?):(\d+)\)")
ours = ("model.py", "fsm.py", "film.py", "par.py", "stepk.py", "compartments.py")
kern, eq = Counter(), Counter()
L = E = 0
for line in open(sys.argv[1], encoding="utf-8", errors="replace"):
    stack, _, c = line.rstrip().rpartition(" ")
    try:
        c = int(c)
    except ValueError:
        continue
    fr = [FR.match(f.strip()) for f in stack.split(";")]
    fr = [(m.group(1), m.group(2).replace("\\", "/").split("/")[-1], int(m.group(3))) for m in fr if m]
    if not any(f[0] == "step" and f[1] == "model.py" for f in fr) or "writer (" in stack:
        continue
    fr = [f for f in fr if f[1] in ours]
    if not fr:
        continue
    L += c
    if fr[-1][1] == "par.py" and fr[-1][0] == "call":
        p = fr[-2] if len(fr) > 1 else ("?", "?", 0)
        kern[f"{p[1]}:{p[2]} {p[0]}"] += c
        key = f"[par] {p[1]}:{p[2]} {p[0]}"
    else:
        key = f"{fr[-1][1]}:{fr[-1][2]} {fr[-1][0]}"
    if any(f[0] == "_equilibrate" for f in fr):
        E += c
        eq[key] += c
print(f"time loop {L / 20 / 60:.1f} min; _equilibrate {E / 20 / 60:.1f} min ({100 * E / L:.0f} %)")
print("\n== dual kernels by caller (share of the time loop)")
for k, v in kern.most_common(25):
    print(f"  {100 * v / L:5.1f} %  {v / 20 / 60:5.2f} min  {k}")
print("\n== _equilibrate by line (share of _equilibrate)")
for k, v in eq.most_common(30):
    print(f"  {100 * v / E:5.1f} %  {k}")
