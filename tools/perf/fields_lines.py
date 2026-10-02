"""py-spy raw profile of bench_writer.py: the new _state_fields only (not
old_state_fields), by innermost drainsim/car_movie line and by leaf."""
import re
import sys
from collections import Counter
FR = re.compile(r"(.+?) \((.+?):(\d+)\)")
keep = ("worldviz.py", "octview.py", "car_movie.py", "par.py", "model.py", "fsm.py")
byline, leaf = Counter(), Counter()
W = 0
for line in open(sys.argv[1], encoding="utf-8", errors="replace"):
    stack, _, c = line.rstrip().rpartition(" ")
    try:
        c = int(c)
    except ValueError:
        continue
    if "_state_fields (" not in stack or "old_state_fields" in stack:
        continue
    W += c
    fr = [FR.match(f.strip()) for f in stack.split(";")]
    fr = [(m.group(1), m.group(2).replace("\\", "/").split("/")[-1], int(m.group(3))) for m in fr if m]
    ours = [f for f in fr if f[1] in keep]
    if ours:
        byline[f"{ours[-1][1]}:{ours[-1][2]} {ours[-1][0]}"] += c
    lf = fr[-1] if fr else ("?", "?", 0)
    leaf[f"{lf[1]}:{lf[0]}"] += c
print(f"new _state_fields: {W} samples = {W / 20:.1f} s")
for k, v in byline.most_common(22):
    print(f"  {100 * v / W:5.1f} %  {k}")
print("leaf:")
for k, v in leaf.most_common(10):
    print(f"  {100 * v / W:5.1f} %  {k}")
