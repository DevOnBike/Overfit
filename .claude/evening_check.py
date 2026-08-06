"""Did tonight's diurnal ramp open a RequestsPerSecond incident?

The two false incidents of the 24-hour run were the up and down slopes of the load driver's 1440-minute
curve: +10.6% of typical at 19:03Z and -15.2% at 23:43Z. `minTrendChange` was raised 0.194 -> 0.333 on
2026-08-05 16:40Z to stop them, and this is the only verification available — the fault panel can degrade a
replica but cannot move deployment-wide traffic, so unlike the CPU, heap and memory floors this one cannot
be checked by injection. It has to be waited for.

Reports the guard's health alongside the answer, because "no incident" is only good news if the guard was
running: a stalled guard also reports nothing, and that is the failure this subsystem exists to remove.
"""
import datetime
import pathlib
import re
import subprocess
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = pathlib.Path(r"D:\Overfit")

result = subprocess.run(
    ["kubectl", "logs", "deployment/anomaly-guard", "-n", "lab", "--tail=-1"],
    cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300)

lines = (result.stdout + result.stderr).splitlines()

if result.returncode != 0:
    print(f"CANNOT READ THE GUARD ({result.returncode}) — no verdict. {lines[-1][:120] if lines else ''}")
    sys.exit(1)

cycles = [l for l in lines if "cycle: pods=" in l]
stamps = []
current = None

for line in lines:
    found = re.match(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", line)

    if found:
        current = found.group(1)

    if "cycle: pods=" in line:
        stamps.append(current)

now = datetime.datetime.now(datetime.timezone.utc)

# Every incident this guard has opened since it restarted onto the new floors.
opened = []

for i, line in enumerate(lines):
    if "Anomaly incident in" not in line:
        continue

    signal = re.search(r": (\w+) on ", line)
    opened.append((stamps[-1] if stamps else "?", signal.group(1) if signal else "?", line.strip()))

rps = [row for row in opened if row[1] == "RequestsPerSecond"]

# The two moments under test, in UTC, on whichever day this runs.
ramps = [("up 19:03Z", 19, 3), ("down 23:43Z", 23, 43)]
passed = [name for name, h, m in ramps
          if (now.hour, now.minute) >= (h, m) or (h == 23 and now.hour < 12)]

print(f"now {now:%H:%M}Z   cycles since the floors changed: {len(cycles)}")
print(f"last cycle: {stamps[-1] if stamps else '(none)'}")
print(f"ramps covered so far: {', '.join(passed) if passed else 'none yet'}")
print()

if not cycles:
    print("NO VERDICT — the guard has run no cycle since the restart, so silence proves nothing.")
    sys.exit(0)

print(f"incidents opened since the restart: {len(opened)}")

for stamp, signal, line in opened[-6:]:
    print(f"   {stamp}  {signal}")
    print(f"      {line[line.find('Anomaly incident'):][:190]}")

print()

if rps:
    print(f"RequestsPerSecond incidents: {len(rps)} — THE RAISED FLOOR DID NOT STOP THE RAMP")
else:
    print("RequestsPerSecond incidents: 0")

    if len(passed) == 2:
        print("both ramps have passed and neither opened one — the floor did what it was raised to do")
    else:
        print("not both ramps have passed yet; this is not the final answer")
