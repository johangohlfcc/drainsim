"""Render a car_movie.py recording with a resizable pool of workers.

Same result as `car_movie.py --render DIR --workers W`, but:
- every worker gets all the look options (car_movie.render() does not pass
  --orbit/--look/--sim-dt/--dt-hang on to its workers, so the parts lose the orbit);
- the frames are cut into --chunks parts that run --workers at a time; the
  number of workers can be changed while it runs by writing a number into
  <parts>/workers.txt;
- finished parts are kept, so a restart only renders what is missing;
- if free memory drops below --min-free GB, the running parts are stopped,
  requeued, and the pool halves its workers.

Usage (from the drainsim repo, PYTHONPATH as for car_movie.py):
  python tools/runs/render_pool.py --rec runs/car5mm/oct5n --out runs/car5mm/car_dip_oct5n.mp4
      --workers 4 --chunks 24  <car_movie.py look options>

--stagger is the wait between any two worker starts, not only the first ones:
keep it below the time of a part, or the parts run one at a time. (The film
shells are built once per recording by the first worker; the others wait for
them up to an hour.)
"""
import argparse
import json
import os
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import psutil

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
MOVIE = os.path.join(REPO, "examples", "car_movie.py")
sys.path.insert(0, os.path.join(REPO, "examples"))
sys.path.insert(0, REPO)


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def free_gb():
    return psutil.virtual_memory().available / 2**30


def tree_rss_gb(pid):
    try:
        p = psutil.Process(pid)
        return sum(q.memory_info().rss for q in [p] + p.children(recursive=True)) / 2**30
    except psutil.Error:
        return 0.0


def kill_tree(pid):
    subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rec", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--chunks", type=int, default=24)
    ap.add_argument("--min-free", type=float, default=8.0)
    ap.add_argument("--stagger", type=float, default=0.0,
                    help="seconds between part starts (spreads the load spikes)")
    a, look = ap.parse_known_args()

    # the frame times exactly as car_movie.render() computes them
    import car_movie
    lp = argparse.ArgumentParser()
    for name, typ, default in (("--fps", int, 24), ("--speedup", float, 2.0),
                               ("--slow-hang", float, 10.0), ("--speedup-late", float, 10.0),
                               ("--sim-dt", float, 0.1), ("--dt-hang", float, 0.5)):
        lp.add_argument(name, type=typ, default=default)
    ta, _ = lp.parse_known_args(look)
    with open(os.path.join(a.rec, "meta.json")) as f:
        meta = json.load(f)
    n = len(car_movie.timeline(ta, meta["t_dipout"], meta["t_end"]).frames())
    C = max(1, min(a.chunks, n))
    cuts = np.linspace(0, n, C + 1).round().astype(int)

    out = os.path.abspath(a.out)
    parts_dir = out.replace(".mp4", "") + "_parts"
    os.makedirs(parts_dir, exist_ok=True)
    wfile = os.path.join(parts_dir, "workers.txt")
    with open(wfile, "w") as f:
        f.write(f"{a.workers}\n")
    base = [sys.executable, "-X", "faulthandler", "-u", MOVIE, "--render", os.path.abspath(a.rec)]
    base += look + ["--sim-dt", str(ta.sim_dt), "--dt-hang", str(ta.dt_hang)]

    def part(c):
        return os.path.join(parts_dir, f"part{c:02d}.mp4")

    def plog(c):
        return os.path.join(parts_dir, f"part{c:02d}.log")

    def finished(c):
        if not os.path.exists(part(c)) or not os.path.exists(plog(c)):
            return False
        with open(plog(c), errors="replace") as f:
            return "] done:" in f.read()

    pending = [c for c in range(C) if not finished(c)]
    log(f"{n} frames in {C} parts of about {n // C} frames; {C - len(pending)} already done; "
        f"{a.workers} workers (change: write a number into {wfile})")
    log("worker command: " + " ".join(base[3:]))
    t0 = time.time()
    running = {}          # chunk -> (Popen, file, start time)
    failed = []
    peak_total = 0.0
    last_mem = 0.0
    last_start = -1e9
    while pending or running:
        try:
            with open(wfile) as f:
                W = max(1, int(f.read().strip()))
        except (OSError, ValueError):
            W = a.workers
        while (pending and len(running) < W and not failed
               and time.time() - last_start >= a.stagger):
            last_start = time.time()
            c = pending.pop(0)
            fh = open(plog(c), "w")
            p = subprocess.Popen(base + ["--frames", str(cuts[c]), str(cuts[c + 1]),
                                         "--part", part(c), "--part-id", str(c)],
                                 stdout=fh, stderr=subprocess.STDOUT, cwd=REPO)
            running[c] = (p, fh, time.time())
            log(f"start part {c:02d}: frames {cuts[c]}-{cuts[c + 1]} (PID {p.pid})")
        time.sleep(10.0)
        for c in list(running):
            p, fh, ts = running[c]
            if p.poll() is not None:
                fh.close()
                del running[c]
                if p.returncode == 0 and finished(c):
                    log(f"done  part {c:02d} in {(time.time() - ts) / 60:.1f} min "
                        f"({C - len(pending) - len(running) - len(failed)}/{C} parts finished)")
                else:
                    failed.append(c)
                    log(f"FAILED part {c:02d} (exit {p.returncode}); see {plog(c)}")
        fr = free_gb()
        per = {c: tree_rss_gb(running[c][0].pid) for c in running}
        total = sum(per.values())
        peak_total = max(peak_total, total)
        if fr < a.min_free and running:
            W2 = max(1, len(running) // 2)
            log(f"LOW MEMORY: free {fr:.1f} GB; stopping {len(running)} parts, "
                f"continuing with {W2} workers")
            for c in list(running):
                p, fh, _ = running.pop(c)
                kill_tree(p.pid)
                fh.close()
                if os.path.exists(part(c)):
                    os.remove(part(c))
                pending.insert(0, c)
            pending.sort()
            with open(wfile, "w") as f:
                f.write(f"{W2}\n")
        elif time.time() - last_mem > 60 and running:
            last_mem = time.time()
            log(f"memory: free {fr:.1f} GB, workers {total:.1f} GB in {len(running)} "
                f"(" + ", ".join(f"{v:.1f}" for v in per.values()) + f"), peak total {peak_total:.1f} GB")
        if failed and not running:
            log(f"stopping: part(s) {failed} failed; finished parts are kept, rerun to resume")
            sys.exit(1)

    lst = os.path.join(parts_dir, "parts.txt")
    with open(lst, "w") as f:
        for c in range(C):
            f.write(f"file '{part(c)}'\n")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
                    "-i", lst, "-c", "copy", "-movflags", "+faststart", out], check=True)
    log(f"wrote {out}: {n} frames, {(time.time() - t0) / 60:.1f} min, "
        f"peak worker memory {peak_total:.1f} GB")


if __name__ == "__main__":
    main()
