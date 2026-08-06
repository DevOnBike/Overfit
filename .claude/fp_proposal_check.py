"""Wait until the guard has been up an hour, then print its floor proposals.

Separate file from run.py on purpose: run.py is scratch and is rewritten constantly, and this has to
survive an hour of other work. It reads only logs — nothing in the lab is touched.
"""
import datetime
import pathlib
import subprocess
import sys
import time

sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
ROOT = pathlib.Path(r"D:\Overfit")

# The guard came up at 11:08 UTC and proposes on an hourly interval measured from its first cycle, so the
# first one lands a little after 12:13. A few minutes of margin, then one look.
TARGET = datetime.datetime(2026, 8, 1, 12, 20, tzinfo=datetime.timezone.utc)

wait = (TARGET - datetime.datetime.now(datetime.timezone.utc)).total_seconds()

if wait > 0:
    print(f"waiting {wait / 60:.0f} min for the guard's first floor proposal")
    time.sleep(wait)

r = subprocess.run(
    ["kubectl", "logs", "-n", "lab", "deployment/anomaly-guard", "--tail=-1"],
    cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=180)

lines = (r.stdout + r.stderr).splitlines()
proposals = [l.strip() for l in lines if "Floor proposal for" in l]
cycles = [l.strip() for l in lines if "cycle: pods=" in l]

print(f"\ncycles completed: {len(cycles)}")
print(f"floor proposals emitted: {len(proposals)}\n")

for line in proposals:
    print("  " + line[:230])

if not proposals:
    print("  none yet — either fewer than 30 observations per metric, or every configured floor")
    print("  already sits above what the calibrator would propose (which is the quiet, correct case)")

print("\nlast 3 cycle summaries:")
for line in cycles[-3:]:
    print("  " + line[:190])
