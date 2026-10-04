"""Check the pockets of every saved state of a recording as the states
appear (car_movie.py --save-at): the biggest pockets of trapped air in each
state up to the last one, and of held liquid in the last one, against the
true geometry (pockets.py). One report per state, <rec>/pockets_t<T>.txt
(and pictures in <rec>/pockets_t<T>/).

    python tools/debug/validate_states.py REC [--top 10] [--end T_END]
"""
import argparse
import glob
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rec")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--end", type=float, default=None,
                    help="the time of the last state (liquid is checked there); default: "
                         "the state saved last once the recording is done")
    a = ap.parse_args()
    done = set()
    env = dict(os.environ)
    root = os.path.abspath(os.path.join(HERE, "..", ".."))
    env["PYTHONPATH"] = root + os.pathsep + os.path.join(root, "examples")
    while True:
        states = sorted(glob.glob(os.path.join(a.rec, "state_t*.pkl")))
        finished = os.path.exists(os.path.join(a.rec, "done"))
        todo = [s for s in states if s not in done]
        if not todo and finished:
            break
        for s in todo:
            t = float(os.path.basename(s)[7:-4])
            last = (a.end is not None and abs(t - a.end) < 1e-6) or \
                (finished and s == states[-1] and a.end is None)
            kind = ["--kind", "liquid", "--holdup"] if last else ["--kind", "air", "--width"]
            tag = os.path.basename(s)[6:-4]
            out = os.path.join(a.rec, f"pockets_{tag}.txt")
            cmd = [sys.executable, os.path.join(HERE, "pockets.py"), s, "--top", str(a.top),
                   "--geometry", "1.5", "--images", os.path.join(a.rec, f"pockets_{tag}")] + kind
            t0 = time.time()
            with open(out, "w") as f:
                subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
            print(f"{tag}: {' '.join(kind)} -> {out} ({time.time() - t0:.0f} s)", flush=True)
            done.add(s)
        if not finished:
            time.sleep(60)


if __name__ == "__main__":
    main()
