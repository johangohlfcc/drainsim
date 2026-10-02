"""Writer thread of a car_movie --record py-spy raw profile, by line in
worldviz/octview/car_movie/par/model (innermost such frame), and by function."""
import re, sys
from collections import Counter
FR = re.compile(r"(.+?) \((.+?):(\d+)\)")
keep = ("worldviz.py", "octview.py", "car_movie.py", "par.py", "model.py", "fsm.py", "npyio.py", "fromnumeric.py")
byline, top = Counter(), Counter()
W = 0
for line in open(sys.argv[1], encoding="utf-8", errors="replace"):
    stack, _, c = line.rstrip().rpartition(" ")
    try: c = int(c)
    except ValueError: continue
    if "_state_fields" not in stack and "car_movie.py)" not in stack: pass
    if not ("writer (" in stack and "car_movie" in stack): continue
    if "model.py" in stack and " step (" in stack: continue
    W += c
    fr = [FR.match(f.strip()) for f in stack.split(";")]
    fr = [(m.group(1), m.group(2).replace("\\", "/").split("/")[-1], int(m.group(3))) for m in fr if m]
    ours = [f for f in fr if f[1] in keep]
    if ours:
        byline[f"{ours[-1][1]}:{ours[-1][2]} {ours[-1][0]}"] += c
    leaf = fr[-1] if fr else ("?", "?", 0)
    top[f"{leaf[1]}:{leaf[0]}"] += c
print(f"writer {W} samples = {W/20/60:.1f} thread-min")
for k, v in byline.most_common(25): print(f"  {100*v/W:5.1f} %  {k}")
print("leaf:")
for k, v in top.most_common(12): print(f"  {100*v/W:5.1f} %  {k}")
