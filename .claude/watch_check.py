"""Runs the hourly check and prints it ONLY when something changed.

Wraps `hourly_check.py` rather than modifying it, on purpose: that script has produced every report of this
run so far, and editing a reporting instrument while it is reporting makes the series it produced
non-uniform. This one adds a decision on top and touches nothing underneath.

When nothing has changed it prints a single heartbeat line rather than nothing at all. Silence would be
indistinguishable from the watcher having died — which is the exact failure this whole subsystem exists to
remove, and it would be absurd to reintroduce it in the thing watching the thing that removes it.

"Changed" is mechanical, not a judgement call:
  * a new incident opened, or a new signal appeared;
  * a cycle failed, or blind metrics rose above the one expected;
  * the guard went stale (no cycle for over 12 minutes at a 5-minute cadence);
  * resident memory set a new peak above 200 MB, or crossed 400 MB;
  * the verdict stopped being OK;
  * the run finished.
"""
import json
import pathlib
import re
import subprocess
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = pathlib.Path(r"D:\Overfit")
STATE = ROOT / "Tests" / "bin" / "watch-state.json"

result = subprocess.run(
    [sys.executable, str(ROOT / ".claude" / "hourly_check.py")],
    cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600)

report = result.stdout + result.stderr


def number(pattern, default=-1.0):
    found = re.search(pattern, report)

    return float(found.group(1)) if found else default


now = {
    "elapsed": number(r"elapsed ([\d.]+) h"),
    "cycles": number(r"cycles (\d+) of"),
    "failures": number(r"cycle failures (\d+)"),
    "blind": number(r"blind (\d+) "),
    "opened": number(r"RATE\s+(\d+) opened"),
    "findings": number(r"findings (\d+)"),
    "peak": number(r"peak (\d+) MB"),
    "verdict": 1.0 if "VERDICT OK" in report else 0.0,
    "signals": sorted(set(re.findall(r"^   (\w+)\s+\d+\s+[\d.]+%", report, re.MULTILINE))),
}

previous = {}

if STATE.exists():
    try:
        previous = json.loads(STATE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        previous = {}

reasons = []

if not previous:
    reasons.append("first run of the watcher")

for key, label in (("opened", "an incident opened"), ("failures", "a cycle failed"),
                   ("findings", "a finding appeared")):
    if previous and now[key] > previous.get(key, 0):
        reasons.append(f"{label} ({previous.get(key)} -> {now[key]:.0f})")

if previous and now["signals"] != previous.get("signals", []):
    reasons.append(f"signals changed: {previous.get('signals')} -> {now['signals']}")

if now["blind"] > 1:
    reasons.append(f"blind metrics above the expected one ({now['blind']:.0f})")

if now["verdict"] == 0.0:
    reasons.append("verdict is NOT OK")

if "last cycle" in report:
    age = number(r"last cycle ([\d.]+) min ago")

    if age > 12:
        reasons.append(f"the guard has gone stale ({age:.0f} min since the last cycle)")

if now["peak"] > 400:
    reasons.append(f"memory peak {now['peak']:.0f} MB is approaching the 512 MiB limit")
elif previous and now["peak"] > previous.get("peak", 0) + 30:
    reasons.append(f"memory peak rose {previous.get('peak'):.0f} -> {now['peak']:.0f} MB")

if now["elapsed"] >= 24.0:
    reasons.append("the 24-hour window has closed")

STATE.write_text(json.dumps(now), encoding="utf-8")

if reasons:
    print("CHANGED: " + "; ".join(reasons))
    print()
    print(report)
else:
    print(f"no change — {now['elapsed']:.1f} h elapsed, {now['cycles']:.0f} cycles, "
          f"{now['opened']:.0f} incident(s), {now['findings']:.0f} findings, "
          f"peak {now['peak']:.0f} MB, verdict OK")
