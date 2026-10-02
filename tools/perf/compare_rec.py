"""Compare two car_movie recordings step by step (every array of every
step file, and static.npz)."""
import os
import sys

import numpy as np

a, b = sys.argv[1], sys.argv[2]
names = sorted(os.listdir(os.path.join(a, "steps")))
assert names == sorted(os.listdir(os.path.join(b, "steps"))), "different step files"
bad = 0
for f in ["../static.npz"] + names:
    x = np.load(os.path.join(a, "steps", f))
    y = np.load(os.path.join(b, "steps", f))
    if set(x.files) != set(y.files):
        print(f, "keys differ", x.files, y.files)
        bad += 1
        continue
    for k in x.files:
        if x[k].dtype != y[k].dtype or not np.array_equal(x[k], y[k], equal_nan=x[k].dtype.kind == "f"):
            print(f, k, "differs")
            bad += 1
print(f"{len(names)} steps, {bad} differences")
