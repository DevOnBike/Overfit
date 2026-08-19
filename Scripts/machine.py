"""Was the machine quiet while that measurement ran?

Import it, wrap the measured window, read the verdict:

    import sys
    sys.path.insert(0, r"D:\\Overfit\\Scripts")
    from machine import quiet_guard

    with quiet_guard("VGG-16 ABAB") as window:
        run_the_thing()

    if not window.quiet:
        print("discard this reading")

**Why this exists.** A benchmark on this box holds cores, L3 and memory bandwidth still on purpose. Windows
does not cooperate: Defender runs scheduled scans, Windows Update downloads and installs, the search indexer
walks the disk, and none of them announces itself in the benchmark's output. The result is a number that is
wrong and looks exactly like a number that is right — the same shape as every other failure this repository
keeps guards for.

**The unit is core-seconds, and that was chosen rather than assumed.** A foreign process hurts a measurement
by taking cores, cache and bandwidth away from it, so the quantity is foreign CPU time over the window as a
share of the core-seconds the window had available. "Percent of one core" and "number of running processes"
are both the wrong unit and would put the threshold in the wrong place.

**The threshold comes from a measurement on this machine, 2026-08-19**, not from a guess:

  - idle, 30 s:                          7.59 core-seconds of 481.9 = **1.58%**
  - during a 16-core VGG-16 run, 32 s:  16.81 core-seconds of 518.5 = **3.24%**

Background load roughly **doubles while a benchmark runs** — the compositor, the remote-desktop agent and an
open Task Manager all react to the activity — so a ceiling set from an idle sample would fire on every run.
`QUIET_CEILING` sits above the measured busy figure with headroom.

**Two probes, not one, because they fail differently.** The share catches anything, including software
nobody put on a list. The named-process probe catches Defender and Windows Update at loads the share would
forgive, because those two also thrash L3 and the disk far beyond what their CPU seconds suggest.

**What was checked when this was written**, so a later reader knows what the probe can and cannot see:

  - **All 296 processes reported CPU time, zero denied.** Protected processes including `MsMpEng` are
    readable without administrator rights, so the detector is not blind to the thing it exists to detect.
  - **`Get-Counter` with English counter names FAILS on this machine** — Windows is localised and the
    object name does not resolve. Do not reach for a machine-wide performance counter here; per-process
    times are the reliable source.
  - Defender real-time protection was **off** on this box at the time and scheduled scans were still
    enabled, so the risk is reduced but not removed.
"""

import subprocess
import time

#: Foreign CPU share above which a window is reported as contaminated. Measured busy baseline is 3.24%.
QUIET_CEILING = 0.05

#: Core-seconds from any one named process that condemn a window on its own.
LOUD_SECONDS = 1.0

#: Physical cores. Logical count is deliberately not used: SMT siblings do not add FMA throughput on this
#: machine (16 workers measured at 31.22 ms against 31.55 for 32), so core-seconds are counted per core.
PHYSICAL_CORES = 16

#: Processes whose presence condemns a window regardless of how little CPU they show.
LOUD_PROCESSES = {
    "MsMpEng",               # Defender scan engine
    "MpCmdRun",              # Defender command-line scan
    "MpDefenderCoreService",
    "TiWorker",              # Windows Update installer worker
    "TrustedInstaller",
    "MoUsoCoreWorker",       # Update Session Orchestrator
    "UsoClient",
    "wuauclt",
    "SearchIndexer",
    "SearchProtocolHost",
    "CompatTelRunner",       # compatibility telemetry, runs after updates
    "OneDrive",
    "backgroundTaskHost",
}

#: Names belonging to the measurement itself, excluded from the foreign total.
MEASUREMENT_PROCESSES = {
    "dotnet",
    "ProfHarness",
    "testhost",
    "MSBuild",
    "VBCSCompiler",
    "vstest.console",
}

_SAMPLE = r"""
$out = @()
foreach ($p in Get-Process) {
  try { $out += ('{0}|{1}|{2}' -f $p.Id, $p.ProcessName, $p.TotalProcessorTime.TotalSeconds) } catch { }
}
$out -join "`n"
"""


def snapshot():
    """Every readable process as ``pid -> (name, cpu seconds)``."""
    proc = subprocess.run(
        ["powershell", "-NoProfile", "-Command", _SAMPLE],
        capture_output=True, text=True, timeout=180)

    table = {}

    for line in proc.stdout.splitlines():
        parts = line.strip().split("|")

        # A process that exits between the enumeration and the property read leaves an empty field rather
        # than an error, and float('') would take the whole sample down with it.
        if len(parts) == 3 and parts[2].strip():
            table[int(parts[0])] = (parts[1], float(parts[2].replace(",", ".")))

    if not table:
        raise RuntimeError(
            "process sample came back empty — the window cannot be judged, so do not treat it as quiet")

    return table


class Verdict:
    """What the machine was doing while the window was open."""

    def __init__(self, quiet, share, foreign_seconds, available, offenders, named):
        self.quiet = quiet
        self.share = share
        self.foreign_seconds = foreign_seconds
        self.available = available
        self.offenders = offenders
        self.named = named

    def report(self, label):
        state = "QUIET" if self.quiet else "*** CONTAMINATED ***"

        print(f"  [machine] {label}: {state} — foreign {self.foreign_seconds:.2f} core-s of "
              f"{self.available:.0f} = {100 * self.share:.2f}% (ceiling {100 * QUIET_CEILING:.0f}%)")

        for name, seconds in self.named:
            print(f"  [machine]   scanner/updater active: {name} used {seconds:.2f} core-s")

        if not self.quiet:
            for seconds, name, pid in self.offenders[:5]:
                print(f"  [machine]   {name:<24} pid {pid:<7} {seconds:6.2f} core-s")

        return self.quiet


def judge(before, after, wall_seconds, cores=PHYSICAL_CORES):
    """Compares two snapshots and decides whether the window between them is usable."""
    deltas = []

    for pid, (name, cpu) in after.items():
        previous = before.get(pid)
        moved = cpu - previous[1] if previous else cpu

        if moved > 0.01:
            deltas.append((moved, name, pid))

    deltas.sort(reverse=True)

    foreign = [(seconds, name, pid) for seconds, name, pid in deltas
               if name not in MEASUREMENT_PROCESSES]
    total = sum(seconds for seconds, _, _ in foreign)
    available = max(wall_seconds, 1e-9) * cores
    share = total / available

    named = [(name, seconds) for seconds, name, _ in foreign
             if name in LOUD_PROCESSES and seconds >= LOUD_SECONDS]

    return Verdict(share <= QUIET_CEILING and not named, share, total, available, foreign, named)


class quiet_guard:
    """Samples the machine around a block and reports whether the window is usable.

    It reports rather than raises. A contaminated window is a fact about the reading, and the caller is the
    one that knows whether to discard it, repeat it or record it with the caveat attached.
    """

    def __init__(self, label, cores=PHYSICAL_CORES, quiet_output=False):
        self.label = label
        self.cores = cores
        self.quiet_output = quiet_output
        self.verdict = None
        self.quiet = True

    def __enter__(self):
        self._before = snapshot()
        self._start = time.monotonic()

        return self

    def __exit__(self, exc_type, exc, tb):
        elapsed = time.monotonic() - self._start
        self.verdict = judge(self._before, snapshot(), elapsed, self.cores)
        self.quiet = self.verdict.quiet

        if not self.quiet_output:
            self.verdict.report(self.label)

        return False
