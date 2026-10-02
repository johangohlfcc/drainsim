import re, sys
from collections import Counter
FR = re.compile(r"(.+?) \((.+?):(\d+)\)")
cnt = Counter(); U = 0
for line in open(sys.argv[1], encoding="utf-8", errors="replace"):
    stack, _, c = line.rstrip().rpartition(" ")
    try: c = int(c)
    except ValueError: continue
    fr = [FR.match(f.strip()) for f in stack.split(";")]
    fr = [(m.group(1), m.group(2).replace("\\", "/").split("/")[-1], int(m.group(3))) for m in fr if m]
    i = [k for k, f in enumerate(fr) if f[0] == "update" and f[1] == "film.py"]
    if not i: continue
    U += c
    ours = [f for f in fr[i[-1]:] if f[1] == "film.py"]
    leaf = ours[-1]
    cnt[f"film.py:{leaf[2]} {leaf[0]}"] += c
print(f"update samples {U}")
for k, v in cnt.most_common(int(sys.argv[2]) if len(sys.argv) > 2 else 30): print(f"  {100*v/U:5.1f} %  {k}")
