"""film.update in the time loop of a py-spy raw profile: by line (par.call
replaced by its caller), split by phase of the motion via the caller chain
not available -> overall only."""
import re, sys
from collections import Counter
FR = re.compile(r"(.+?) \((.+?):(\d+)\)")
c_line, c_func = Counter(), Counter()
F = L = 0
for line in open(sys.argv[1], encoding="utf-8", errors="replace"):
    stack, _, c = line.rstrip().rpartition(" ")
    try: c = int(c)
    except ValueError: continue
    fr = [FR.match(f.strip()) for f in stack.split(";")]
    fr = [(m.group(1), m.group(2).replace("\\", "/").split("/")[-1], int(m.group(3))) for m in fr if m]
    if not any(f[0] == "step" and f[1] == "model.py" for f in fr) or "writer (" in stack:
        continue
    L += c
    if not any(f[1] == "film.py" for f in fr):
        continue
    F += c
    ours = [f for f in fr if f[1] in ("film.py", "par.py", "fsm.py", "stepk.py")]
    leaf = ours[-1]
    if leaf[1] == "par.py" and len(ours) > 1:
        leaf = ours[-2]; key = f"[par] {leaf[1]}:{leaf[2]} {leaf[0]}"
    else:
        key = f"{leaf[1]}:{leaf[2]} {leaf[0]}"
    c_line[key] += c
    # innermost external (numpy/scipy) frame, if any
    ext = fr[-1]
    c_func[f"{ext[1]}:{ext[0]}"] += c
print(f"time loop {L/1200:.1f} min, film {F/1200:.1f} min ({100*F/L:.0f} %)")
for k, v in c_line.most_common(30): print(f"  {100*v/F:5.1f} %  {k}")
print("innermost frames:")
for k, v in c_func.most_common(12): print(f"  {100*v/F:5.1f} %  {k}")
