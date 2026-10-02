"""Break a py-spy raw profile of car_movie --record into setup, time loop and
writer thread, and rank the time loop by drainsim line and by function."""
import re
import sys
from collections import Counter, defaultdict

path = sys.argv[1]
FR = re.compile(r"(.+?) \((.+?):(\d+)\)")


def frames(stack):
    out = []
    for f in stack.split(";"):
        m = FR.match(f.strip())
        if m:
            out.append((m.group(1), m.group(2).replace("\\", "/").split("/")[-1], int(m.group(3))))
        else:
            out.append((f.strip(), "", 0))
    return out


phase = Counter()
loop_line = Counter()
loop_func = Counter()
loop_part = Counter()
writer = Counter()
setup_func = Counter()
for line in open(path, encoding="utf-8", errors="replace"):
    stack, _, c = line.rstrip().rpartition(" ")
    try:
        c = int(c)
    except ValueError:
        continue
    fr = frames(stack)
    names = [f"{fn}:{func}" for func, fn, ln in fr]
    ours = [(func, fn, ln) for func, fn, ln in fr
            if fn in ("model.py", "fsm.py", "film.py", "par.py", "stepk.py", "gseg.py",
                      "octree.py", "compartments.py", "grid.py", "volfrac.py", "sidesplit.py",
                      "voxel.py", "car_movie.py", "car_article.py", "octview.py")]
    if any(n == "car_movie.py:writer" or "_state_fields" in n or "car_movie.py:_write" in n
           for n in names) and not any(n == "model.py:step" for n in names):
        phase["writer thread"] += c
        inner = [x for x in ours if x[1] != "car_movie.py" or True]
        writer[f"{ours[-1][1]}:{ours[-1][0]}:{ours[-1][2]}" if ours else "?"] += c
        continue
    if any(n == "model.py:__init__" for n in names) or any(n == "car_article.py:car_grid" for n in names) \
            or any(n == "car_article.py:load_car" for n in names):
        phase["setup"] += c
        # the function right below model.__init__/_init_* or car_grid
        key = ours[-1] if ours else ("?", "", 0)
        setup_func[f"{key[1]}:{key[0]}"] += c
        continue
    if any(n == "model.py:step" for n in names):
        phase["time loop (main thread)"] += c
        leaf = ours[-1] if ours else ("?", "", 0)
        loop_line[f"{leaf[1]}:{leaf[2]} {leaf[0]}"] += c
        loop_func[f"{leaf[1]}:{leaf[0]}"] += c
        # the part of the step: the frame right below model.step
        i = max(j for j, n in enumerate(names) if n == "model.py:step")
        part = names[i + 1] if i + 1 < len(names) else "(step itself)"
        loop_part[part] += c
        continue
    phase["other: " + (names[-1] if names else "?")] += c

tot = sum(phase.values())
print(f"samples {tot} ({tot / 20 / 60:.1f} thread-minutes at 20 Hz)")
for k, v in phase.most_common(8):
    print(f"  {100 * v / tot:5.1f} %  {k}")
L = phase["time loop (main thread)"]
print(f"\n== time loop, by part of the step (main thread, {L / 20 / 60:.1f} min)")
for k, v in loop_part.most_common(10):
    print(f"  {100 * v / L:5.1f} %  {k}")
print("\n== time loop, by innermost drainsim function")
for k, v in loop_func.most_common(18):
    print(f"  {100 * v / L:5.1f} %  {k}")
print("\n== time loop, by innermost drainsim line")
for k, v in loop_line.most_common(25):
    print(f"  {100 * v / L:5.1f} %  {k}")
S = phase["setup"]
print(f"\n== setup ({S / 20 / 60:.1f} min), by innermost drainsim function")
for k, v in setup_func.most_common(12):
    print(f"  {100 * v / S:5.1f} %  {k}")
W = phase["writer thread"]
if W:
    print(f"\n== writer thread ({W / 20 / 60:.1f} min)")
    for k, v in writer.most_common(8):
        print(f"  {100 * v / W:5.1f} %  {k}")
