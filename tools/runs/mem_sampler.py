"""Sample the memory of a drainsim run and the system, and guard the C: drive.

Every --every seconds, writes one CSV row: time, the run's private and resident
GB, system free RAM, pagefile use, commit and C: free space. At each new peak of
the run's private memory (+--dump-step GB) it saves a py-spy stack dump, so the
setup phase that drives the peak can be found afterwards. If C: free space drops
below --min-c GB, it stops the run (whole process tree) and exits.

  python tools/runs/mem_sampler.py --pid <launcher PID> --out runs/car3mm/oct3n_mem
"""
import argparse
import csv
import os
import shutil
import subprocess
import sys
import time

import psutil


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--every", type=float, default=15.0)
    ap.add_argument("--dump-step", type=float, default=10.0)
    ap.add_argument("--min-c", type=float, default=15.0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    pyspy = os.path.join(os.path.dirname(sys.executable), "py-spy.exe")
    launcher = psutil.Process(a.pid)
    f = open(os.path.join(a.out, "mem.csv"), "a", newline="")
    w = csv.writer(f)
    if f.tell() == 0:
        w.writerow(["time", "run_private_gb", "run_rss_gb", "free_gb", "swap_used_gb",
                    "commit_gb", "c_free_gb", "pid"])
    peak, last_dump = 0.0, 0.0
    while launcher.is_running():
        procs = [launcher] + launcher.children(recursive=True)
        best = None
        for p in procs:
            try:
                mi = p.memory_info()
                if best is None or mi.private > best[1].private:
                    best = (p, mi)
            except psutil.Error:
                pass
        vm, sw = psutil.virtual_memory(), psutil.swap_memory()
        c_free = shutil.disk_usage("C:\\").free / 2**30
        priv = best[1].private / 2**30 if best else 0.0
        rss = best[1].rss / 2**30 if best else 0.0
        commit = (vm.total - vm.available) / 2**30 + sw.used / 2**30
        w.writerow([time.strftime("%H:%M:%S"), f"{priv:.2f}", f"{rss:.2f}",
                    f"{vm.available / 2**30:.1f}", f"{sw.used / 2**30:.1f}",
                    f"{commit:.1f}", f"{c_free:.1f}", best[0].pid if best else ""])
        f.flush()
        peak = max(peak, priv)
        if best and priv >= last_dump + a.dump_step and os.path.exists(pyspy):
            last_dump = priv
            name = os.path.join(a.out, f"stack_{time.strftime('%H%M%S')}_{priv:.0f}GB.txt")
            with open(name, "w") as g:
                subprocess.run([pyspy, "dump", "--pid", str(best[0].pid)], stdout=g,
                               stderr=subprocess.STDOUT, timeout=120)
        if c_free < a.min_c:
            with open(os.path.join(a.out, "STOPPED.txt"), "w") as g:
                g.write(f"{time.strftime('%H:%M:%S')}: C: free {c_free:.1f} GB < {a.min_c} GB; "
                        f"stopped run {a.pid} at private {priv:.1f} GB\n")
            subprocess.run(["taskkill", "/PID", str(a.pid), "/T", "/F"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            break
        time.sleep(a.every)
    f.close()
    with open(os.path.join(a.out, "summary.txt"), "w") as g:
        g.write(f"peak private {peak:.2f} GB\n")


if __name__ == "__main__":
    main()
