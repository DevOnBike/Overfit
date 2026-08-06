"""The end-of-run analysis for a 24-hour false-positive measurement.

Written and dry-run BEFORE the run ends, deliberately. Three separate harnesses in this project produced
confident wrong answers on 2026-08-02 — one grepped for text that never reaches the log, one counted noise
from before a fault was injected, one matched on the wrong field — and in every case the harness was written
in a hurry at the moment its answer was wanted. This one gets to be wrong now, when there is time to notice.

Answers four questions the running tally cannot:

  1. the rate, with a Poisson interval, over the whole day rather than the last poll;
  2. how the rate moved across the load driver's 1440-minute curve — a rate that tracks traffic is a
     different problem from one that does not;
  3. why the quiet fraction rose from 31% to 35% — opened versus ongoing, which the per-poll line conflates;
  4. whether any cycle overlapped a recorded build window, which is the contamination check that was done by
     hand yesterday and should not be.
"""
import datetime
import pathlib
import re
import subprocess
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = pathlib.Path(r"D:\Overfit")

CYCLE = re.compile(
    r"cycle: pods=(\d+) findings=(\d+) incidents=(\d+) opened=(\d+) ongoing=(\d+) "
    r"resolved=(\d+) blind=(\d+) unevaluable=(\d+)")
STAMP = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
INCIDENT = re.compile(r"Anomaly incident in \S+: (\w+) on ")


def marker_start():
    text = (ROOT / "Tests" / "bin" / "fp-run-clean-start.txt").read_text(encoding="utf-8")
    stamp = text.split()[-1]

    return datetime.datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=datetime.timezone.utc)


def guard_log(since):
    r = subprocess.run(
        ["kubectl", "logs", "deployment/anomaly-guard", "-n", "lab",
         f"--since-time={since:%Y-%m-%dT%H:%M:%SZ}", "--tail=-1"],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300)

    return (r.stdout + r.stderr).splitlines()


def build_windows():
    """Recorded machine-load windows, so contaminated cycles can be named rather than assumed absent."""
    path = ROOT / "Tests" / "bin" / "build-windows.txt"

    if not path.exists():
        return []

    windows = []

    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = re.match(r"(\S+) \.\. (\S+)", line.strip())

        if match is None:
            continue

        try:
            start = datetime.datetime.strptime(match.group(1), "%Y-%m-%dT%H:%M:%SZ").replace(
                tzinfo=datetime.timezone.utc)
            end_text = match.group(2)
            end = datetime.datetime.strptime(
                f"{start:%Y-%m-%d}T{end_text}", "%Y-%m-%dT%H:%M:%SZ").replace(
                tzinfo=datetime.timezone.utc)
        except ValueError:
            continue

        windows.append((start, end))

    return windows


def main():
    start = marker_start()
    now = datetime.datetime.now(datetime.timezone.utc)
    elapsed = (now - start).total_seconds() / 3600.0

    print(f"run started {start:%Y-%m-%d %H:%M}Z, {elapsed:.1f} h elapsed\n")

    lines = guard_log(start)
    cycles = []
    signals = {}
    last_stamp = None

    for line in lines:
        stamp = STAMP.match(line.strip())

        if stamp:
            last_stamp = datetime.datetime.strptime(
                stamp.group(1), "%Y-%m-%d %H:%M:%S").replace(tzinfo=datetime.timezone.utc)

        found = CYCLE.search(line)

        if found:
            pods, findings, incidents, opened, ongoing, resolved, blind, unevaluable = (
                int(x) for x in found.groups())
            cycles.append({
                "at": last_stamp,
                "pods": pods, "findings": findings, "incidents": incidents,
                "opened": opened, "ongoing": ongoing, "resolved": resolved,
                "blind": blind, "unevaluable": unevaluable,
            })

            continue

        signal = INCIDENT.search(line)

        if signal:
            signals[signal.group(1)] = signals.get(signal.group(1), 0) + 1

    if not cycles:
        print("no cycles parsed — the log window or the pattern is wrong, and this is the failure the "
              "dry run exists to catch")

        return

    opened = sum(c["opened"] for c in cycles)
    findings = sum(c["findings"] for c in cycles)
    quiet = sum(1 for c in cycles if c["findings"] == 0)
    per_cycle = opened / len(cycles)

    print(f"cycles {len(cycles)}   findings {findings}   opened {opened}   "
          f"quiet {quiet} ({quiet / len(cycles):.0%})")
    print(f"rate   {per_cycle:.3f}/cycle   ->  {per_cycle * 288:.0f} per day at a 5-minute cadence")

    lo = max(0.0, opened - 1.96 * (opened ** 0.5))
    hi = opened + 1.96 * (opened ** 0.5)
    print(f"       95% Poisson on {opened} events: {lo:.0f}-{hi:.0f} "
          f"({lo / len(cycles) * 288:.0f}-{hi / len(cycles) * 288:.0f} per day)\n")

    print("by signal:")

    for name in sorted(signals, key=lambda k: -signals[k]):
        share = signals[name] / max(1, sum(signals.values()))
        print(f"   {name:<26} {signals[name]:4d}  {share:5.1%}")

    # ---- question 2: does the rate follow the load curve? ----
    print("\nrate by hour of the run (opened / cycles):")
    buckets = {}

    for c in cycles:
        if c["at"] is None:
            continue

        hour = int((c["at"] - start).total_seconds() // 3600)
        entry = buckets.setdefault(hour, [0, 0])
        entry[0] += c["opened"]
        entry[1] += 1

    for hour in sorted(buckets):
        got, total = buckets[hour]
        bar = "#" * int(round(got / max(1, total) * 20))
        print(f"   +{hour:2d}h  {got:3d}/{total:3d}  {got / max(1, total):.2f}  {bar}")

    # ---- question 3: is the quiet fraction rising because incidents persist? ----
    half = len(cycles) // 2
    first, second = cycles[:half], cycles[half:]

    def summarise(part, label):
        if not part:
            return

        o = sum(c["opened"] for c in part)
        g = sum(c["ongoing"] for c in part)
        r = sum(c["resolved"] for c in part)
        q = sum(1 for c in part if c["findings"] == 0)
        print(f"   {label:<12} opened {o:4d}  ongoing {g:4d}  resolved {r:4d}  "
              f"quiet {q / len(part):.0%}")

    print("\nfirst half against second — if ongoing grows while opened does not, incidents are persisting")
    summarise(first, "first half")
    summarise(second, "second half")

    # ---- question 4: contamination ----
    windows = build_windows()
    touched = [c for c in cycles
               if c["at"] is not None and any(s <= c["at"] <= e for s, e in windows)]

    print(f"\nrecorded machine-load windows: {len(windows)}")
    print(f"cycles overlapping one: {len(touched)}")

    if touched:
        clean = [c for c in cycles if c not in touched]
        clean_rate = sum(c["opened"] for c in clean) / max(1, len(clean))
        print(f"rate excluding them: {clean_rate:.3f}/cycle ({clean_rate * 288:.0f} per day)")
        print("   if that differs from the headline rate, the contamination mattered")


main()
