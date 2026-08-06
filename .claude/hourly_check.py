"""Hourly status of the running 24-hour false-positive measurement.

Answers three questions in this order, because the third is worthless without the first two:

  1. Is the guard alive and seeing the cluster? A guard that has stopped, or whose queries fail, produces
     no incidents — indistinguishable from a healthy cluster, which is the failure this whole subsystem
     exists to remove. Silence is only good news once it is verified silence.
  2. Is anything contaminating the count? Cycle failures, blind metrics beyond the expected one, an
     unevaluable count that is climbing.
  3. Only then: the rate, its Poisson interval, and where it comes from — and beside it, **how long anything
     was open at all**. The rate counts openings, which under-reports the one shape an operator feels most:
     an incident that opens once and never closes is a single unit in the rate and a permanently red screen
     in the room. Both numbers, or neither is honest.

Also tracks the one known non-comparability with the 2026-08-02 baseline: the workload pods restarted
minutes before this run began, so their heaps started cold. If GcGen2HeapBytes is inflated by that rather
than by the floor, its share falls as the hours pass — which is visible here and nowhere else.
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

# Baseline to compare against: the 2026-08-02 run on the pre-sweep build, same configuration.
BASELINE_PER_DAY = 112
BASELINE_NON_HEAP_PER_DAY = 4


def marker_start():
    text = (ROOT / "Tests" / "bin" / "fp-run-clean-start.txt").read_text(encoding="utf-8")
    stamp = text.split()[-1]

    return datetime.datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=datetime.timezone.utc)


def main():
    start = marker_start()
    now = datetime.datetime.now(datetime.timezone.utc)
    elapsed = (now - start).total_seconds() / 3600.0

    r = subprocess.run(
        ["kubectl", "logs", "deployment/anomaly-guard", "-n", "lab",
         f"--since-time={start:%Y-%m-%dT%H:%M:%SZ}", "--tail=-1"],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300)

    lines = (r.stdout + r.stderr).splitlines()

    if r.returncode != 0:
        print(f"!! kubectl failed — the guard may be gone. {(r.stdout + r.stderr)[:300]}")

        return

    cycles = []
    signals = {}
    failures = 0
    last_stamp = None

    for line in lines:
        stamp = STAMP.match(line.strip())

        if stamp:
            last_stamp = datetime.datetime.strptime(
                stamp.group(1), "%Y-%m-%d %H:%M:%S").replace(tzinfo=datetime.timezone.utc)

        if "cycle failed" in line or "cycle threw" in line:
            failures += 1

        found = CYCLE.search(line)

        if found:
            values = [int(x) for x in found.groups()]
            cycles.append(dict(zip(
                ["pods", "findings", "incidents", "opened", "ongoing", "resolved", "blind",
                 "unevaluable"], values)) | {"at": last_stamp})

            continue

        signal = INCIDENT.search(line)

        if signal:
            signals[signal.group(1)] = signals.get(signal.group(1), 0) + 1

    print(f"RUN  started {start:%m-%d %H:%M}Z   elapsed {elapsed:.1f} h of 24   "
          f"ends {start + datetime.timedelta(hours=24):%m-%d %H:%M}Z")

    if not cycles:
        print("!! NOT OK — no cycle lines parsed. Either the guard is not running its loop, or the log "
              "shape changed. Nothing below can be trusted.")

        return

    last = cycles[-1]
    age = (now - last["at"]).total_seconds() / 60.0 if last["at"] else 999

    # ---- 1. is it alive ----
    alive = age < 12 and failures == 0
    expected_cycles = int(elapsed * 12)
    coverage = len(cycles) / max(1, expected_cycles)

    print(f"\nALIVE  last cycle {age:.1f} min ago (cadence 5)   cycles {len(cycles)} of "
          f"~{expected_cycles} expected ({coverage:.0%})   cycle failures {failures}")

    if not alive:
        print("!! NOT OK — the guard has stopped or is failing cycles. Everything below is stale.")

    # ---- 2. is the count clean ----
    blind = last["blind"]
    unevaluable = sum(c["unevaluable"] for c in cycles) / len(cycles)

    print(f"COVER  pods {last['pods']}   blind {blind} (1 expected: CpuThrottleRatio has no binding)   "
          f"unevaluable {unevaluable:.1f}/cycle avg")

    if blind > 1:
        print("!! more metrics blind than expected — the guard is partly not looking")

    # ---- 2b. the guard's own footprint ----
    # Read from the container's /proc rather than from an instrument, because the guard does not export
    # anything about itself: its fifteen series are all about detection. cAdvisor has the container's
    # working set, but nothing here distinguishes managed heap from native or shows GC pressure, so a
    # guard heading for trouble looks identical to one that is fine until it hits the limit.
    pod = subprocess.run(
        ["kubectl", "get", "pods", "-n", "lab", "-l", "app.kubernetes.io/name=anomaly-guard",
         "-o", "jsonpath={.items[0].metadata.name}"],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=60).stdout.strip()

    if pod:
        status = subprocess.run(
            ["kubectl", "exec", f"pod/{pod}", "-n", "lab", "--", "cat", "/proc/1/status"],
            cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=90).stdout

        rss = next((l.split()[1] for l in status.splitlines() if l.startswith("VmRSS")), "?")
        peak = next((l.split()[1] for l in status.splitlines() if l.startswith("VmHWM")), "?")
        threads = next((l.split()[1] for l in status.splitlines() if l.startswith("Threads")), "?")

        try:
            mb = int(rss) / 1024.0
            peak_mb = int(peak) / 1024.0
            note = "" if peak_mb < 400 else "  !! approaching the 512Mi limit"
            print(f"MEM    {mb:.0f} MB resident, peak {peak_mb:.0f} MB, {threads} threads "
                  f"(request 128Mi, limit 512Mi){note}")
        except ValueError:
            print(f"MEM    unreadable: VmRSS={rss}")

    # ---- 3. the number ----
    opened = sum(c["opened"] for c in cycles)
    findings = sum(c["findings"] for c in cycles)
    quiet = sum(1 for c in cycles if c["findings"] == 0)
    per_cycle = opened / len(cycles)
    per_day = per_cycle * 288

    lo = max(0.0, opened - 1.96 * (opened ** 0.5))
    hi = opened + 1.96 * (opened ** 0.5)

    print(f"\nRATE   {opened} opened over {len(cycles)} cycles = {per_cycle:.3f}/cycle "
          f"-> {per_day:.0f}/day   95% Poisson {lo / len(cycles) * 288:.0f}-{hi / len(cycles) * 288:.0f}")
    print(f"       findings {findings}   quiet cycles {quiet} ({quiet / len(cycles):.0%})   "
          f"baseline 2026-08-02 was {BASELINE_PER_DAY}/day on the SAME config, older build")

    # ---- 3b. how long was anything open at all ----
    # The rate above counts OPENINGS, and that under-reports the case an operator feels most. Measured
    # 2026-08-06: one MemoryWorkingSetBytes incident on a single pod (peer gap 9.99 MB, Cliff's delta 1.0)
    # opened once and stayed open for hours while `opened` sat at zero every cycle. In the rate that is one
    # unit; on the operator's screen it is a permanently red entry. Those are different costs and only the
    # first was being reported, so the rate alone would have called that run quiet.
    # Latched on the EVENTS (opened/resolved), not on the `incidents` snapshot — and the difference is not
    # academic. Measured 2026-08-06 at 13:44:18: one cycle read `findings=0 incidents=0 opened=0 ongoing=0
    # resolved=0`, with the incident back as `ongoing=1` five minutes later and NO resolve in between. The
    # snapshot field evidently means "incidents that produced a finding this cycle", not "incidents that are
    # open". Counting on it split a single 5-hour incident into "longest 255 min, now 25 min" — under-reporting
    # exactly the number this line exists to stop anyone under-reporting.
    live = 0
    open_cycles = 0
    snapshot_cycles = 0
    disagreements = 0
    longest = 0
    current = 0

    for c in cycles:
        live = max(0, live + c["opened"] - c["resolved"])
        snapshot_cycles += 1 if c["incidents"] > 0 else 0

        if (live > 0) != (c["incidents"] > 0):
            disagreements += 1

        if live > 0:
            open_cycles += 1
            current += 1
            longest = max(longest, current)
            continue

        current = 0

    open_hours = open_cycles * 5 / 60.0
    tail = (f"   {live} open NOW, for {current * 5} min" if live
            else "   nothing open right now")

    print(f"OPEN   something was open in {open_cycles}/{len(cycles)} cycles "
          f"({open_cycles / len(cycles):.0%}) = {open_hours:.1f} h of {elapsed:.1f} h elapsed   "
          f"longest streak {longest * 5} min{tail}")
    print("       the rate counts OPENINGS, this counts TIME — an incident that never closes is 1 there "
          "and a permanent alarm here")

    # Surfaced every run rather than left for somebody to notice again. Either `incidents` means something
    # narrower than its name suggests, or the guard's own incident bookkeeping has a hole; only the code
    # settles which, and a silent divergence between an event log and a snapshot is worth knowing about
    # either way.
    if disagreements:
        print(f"       NOTE: the guard's `incidents=` snapshot disagreed with the opened/resolved events in "
              f"{disagreements} of {len(cycles)} cycles (snapshot said open in {snapshot_cycles}). "
              f"This line trusts the events.")

    # NOT max(1, ...). The first version divided by a floored denominator, so zero incidents reported
    # "1 of 1 non-heap -> 288/day" — an invented incident and a rate three hundred times the baseline,
    # from an empty log. Caught by the dry run, which is what dry runs are for.
    total = sum(signals.values())
    heap = signals.get("GcGen2HeapBytes", 0)
    non_heap = total - heap

    if total == 0:
        print("\nBY SIGNAL  nothing reported yet")
    else:
        print("\nBY SIGNAL")

        for name in sorted(signals, key=lambda k: -signals[k]):
            print(f"   {name:<26}{signals[name]:5d}  {signals[name] / total:5.1%}")

    print(f"\nNON-HEAP  {non_heap} of {total} incident row(s)   "
          f"-> {non_heap / len(cycles) * 288:.1f}/day against a baseline of {BASELINE_NON_HEAP_PER_DAY}/day")
    print("          this is the number that says whether the floor repairs bought detection with noise")

    # ---- the known non-comparability: cold pod heaps at the start ----
    if len(cycles) >= 24:
        half = len(cycles) // 2
        early = sum(c["opened"] for c in cycles[:half]) / half
        late = sum(c["opened"] for c in cycles[half:]) / (len(cycles) - half)
        print(f"\nDECAY  first half {early:.3f}/cycle vs second half {late:.3f}/cycle — the workload pods "
              f"restarted just before this run, so a falling rate is cold heaps settling, not the guard "
              f"improving")

    verdict = "OK" if alive and blind <= 1 else "NOT OK"
    print(f"\nVERDICT {verdict}")


main()
