"""Wait for selected physical cards to be idle; never terminate other workloads."""

import argparse
import re
import subprocess
import time

p = argparse.ArgumentParser()
p.add_argument("--devices", default="0,1,2,3,4,5,6,7")
p.add_argument("--timeout", type=int, default=3600)
a = p.parse_args()
selected = {int(x) for x in a.devices.split(",")}
deadline = time.monotonic() + a.timeout
while True:
    result = subprocess.run(["mthreads-gmi"], capture_output=True, text=True, check=True)
    rows = re.findall(r"^\s*(\d+)\s+MTT S5000\s+\|[^|]+\|\s*(\d+)%", result.stdout, re.MULTILINE)
    usage = {int(i): int(u) for i, u in rows}
    if not selected <= usage.keys():
        raise RuntimeError("Could not identify all requested cards in mthreads-gmi")
    if all(usage[i] == 0 for i in selected):
        print("Selected cards have no active GPU compute:", sorted(selected), flush=True)
        break
    if time.monotonic() >= deadline:
        raise TimeoutError("Cards remain busy; acceptance was not started")
    time.sleep(10)
