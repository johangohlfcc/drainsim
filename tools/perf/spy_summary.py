"""Summarise a py-spy raw (collapsed) profile: self and total time per
drainsim/examples function, and time per setup phase.

    python spy_summary.py spy.txt [--phase-depth N]
"""
import re
import sys
from collections import Counter

path = sys.argv[1]
only = sys.argv[2] if len(sys.argv) > 2 else None      # keep stacks containing this text
total = Counter()
selfc = Counter()
n = 0
paths = Counter()
for line in open(path, encoding="utf-8", errors="replace"):
    line = line.rstrip()
    if not line:
        continue
    stack, _, cnt = line.rpartition(" ")
    try:
        c = int(cnt)
    except ValueError:
        continue
    if only and only not in stack:
        continue
    frames = [f for f in stack.split(";") if f]
    # keep our own code: "func (drainsim\file.py:line)" or examples
    ours = []
    for f in frames:
        m = re.match(r"(.+?) \((.+?):(\d+)\)", f)
        if not m:
            continue
        fn, file = m.group(1), m.group(2).replace("\\", "/")
        if (file.startswith("drainsim/") or "/drainsim/" in file or "perf_setup" in file
                or "examples/" in file):
            ours.append(f"{file.split('/')[-1]}:{fn}")
    if not ours:
        ours = ["<other>"]
    n += c
    for f in set(ours):
        total[f] += c
    selfc[ours[-1]] += c
    paths[" > ".join(ours[-3:])] += c
print(f"samples: {n}")
print("\n-- self time (innermost drainsim frame) --")
for f, c in selfc.most_common(30):
    print(f"{100*c/n:6.1f} %  {f}")
print("\n-- total time (inclusive) --")
for f, c in total.most_common(40):
    print(f"{100*c/n:6.1f} %  {f}")
print("\n-- innermost 3 frames --")
for f, c in paths.most_common(25):
    print(f"{100*c/n:6.1f} %  {f}")
