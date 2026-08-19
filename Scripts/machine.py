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

**The threshold comes from measurements on this machine, 2026-08-19**, not from a guess — and it was
**re-derived once, from more samples, after the first value fired on ordinary conditions**:

  - idle, 30 s:                                  7.59 core-seconds of 481.9 = **1.58%**
  - during a 16-core VGG-16 run, 32 s:          16.81 of 518.5 = **3.24%**
  - three ONNX Runtime profiling runs:          **6.44%**, **5.99%**, **6.15%**

Background load roughly doubles while a benchmark runs — the compositor, the remote-desktop agent, an open
Task Manager and the agent driving the run all react to the activity — so a ceiling set from an idle sample
fires on every run. **The first ceiling was 5%, derived from two samples, and it condemned three consecutive
runs whose readings agreed to within 1% of each other.** That is a threshold with too little evidence behind
it, so it was re-derived from all five: 8% clears the highest observed background with headroom, and the
capability check still fires at 53%.

**Headroom on this desktop is genuinely poor and that is worth knowing before trusting the share probe.**
Parsec, the desktop compositor, Task Manager and the agent together account for most of the 6%; a machine
without them would sit far lower. The share probe is therefore the weaker of the two here. **The named
scanner probe is the one that catches the failure this module was written for**, and it fires at any load.

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

#: Foreign CPU share above which a window is reported as contaminated. Highest measured background while a
#: benchmark runs on this desktop is 6.44%; see the module docstring for all five samples and for why the
#: first value of 5% was wrong.
QUIET_CEILING = 0.08

#: Core-seconds from any one named process that condemn a window on its own.
LOUD_SECONDS = 1.0

#: Physical cores. Logical count is deliberately not used: SMT siblings do not add FMA throughput on this
#: machine (16 workers measured at 31.22 ms against 31.55 for 32), so core-seconds are counted per core.
PHYSICAL_CORES = 16

#: Processes whose presence condemns a window regardless of how little CPU they show, because their cost is
#: not proportional to their CPU time: a scan or an update also thrashes L3 and the disk.
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

#: Ordinary desktop software that competes for cores without being a scanner.
#:
#: These are **named and reported** whenever they cost more than :data:`DESKTOP_SECONDS`, and they condemn a
#: window only above :data:`DESKTOP_LOUD_SECONDS`. The two-tier treatment is deliberate and measured: on this
#: desktop, Parsec, the compositor and Task Manager together account for most of a 6% background, so putting
#: them in :data:`LOUD_PROCESSES` would condemn every window ever measured and the guard would be useless.
#: Naming them costs nothing and tells a reader which of their own windows to close.
#:
#: **Task Manager is on this list because it is easy to leave open and it is not free** — measured at
#: 0.77 to 5.89 core-seconds across the windows recorded on 2026-08-19, i.e. up to a quarter of the whole
#: foreign background. It samples every process on a timer, which is the same work this module does.
DESKTOP_PROCESSES = {
    "Taskmgr",               # samples every process on a timer; measured up to 5.89 core-s in one window
    "parsecd",               # remote desktop: encodes a video stream of whatever the run is drawing
    "dwm",                   # the compositor, which wakes for that stream — REPORT ONLY, see below
    "chrome",
    "msedge",
    "msedgewebview2",
    "firefox",
    "brave",
    "opera",
    "Discord",
    "Slack",
    "Teams",
    "ms-teams",
    "Spotify",
    "steam",
    "steamwebhelper",
    "EpicGamesLauncher",
    "obs64",
    "Docker Desktop",
    "com.docker.backend",
    "vmmem",                 # WSL / Hyper-V guest memory process
    "vmmemWSL",
    "Code",                  # VS Code and its language servers
    "devenv",                # Visual Studio
    "rider64",
    "jetbrains-toolbox",
    "Everything",
    "Dropbox",
    "GoogleDriveFS",
    "RadeonSoftware",
    "NVIDIA Share",
    "nvcontainer",
    "explorer",
}

#: Core-seconds from a desktop application above which it is named in the report.
DESKTOP_SECONDS = 0.5

#: Share of the window's core-seconds, taken by a single desktop application, that condemns it on its own.
#: Set well above the measured background of the always-present ones so it fires on something that started,
#: not on the desktop.
#:
#: **This was an absolute 8.0 core-seconds until 2026-08-19, and that was a defect that only shows on long
#: windows.** Every sample it was calibrated on ran 30-32 s, so 8.0 core-s meant "1.67% of the window"
#: without anybody writing the denominator down. `XC-88` then measured for 296 s: Parsec's ordinary,
#: unchanged background reached 21.84 core-s of 4745 - **0.46%, a third of the calibrated share** - and the
#: window was condemned while the share probe beside it reported 2.31% against an 8% ceiling. A guard whose
#: verdict contradicts its own printed number teaches the reader to ignore both.
#:
#: The failure direction is the bad one: it gets **stricter** the longer you measure, so the runs most
#: expensive to repeat are the ones most likely to be thrown away. Expressed as a share it is invariant, and
#: it keeps the calibration it was given rather than replacing it with a fresh guess.
DESKTOP_LOUD_SHARE = 8.0 / (30.0 * PHYSICAL_CORES)

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

    def __init__(self, quiet, share, foreign_seconds, available, offenders, named, desktop):
        self.quiet = quiet
        self.share = share
        self.foreign_seconds = foreign_seconds
        self.available = available
        self.offenders = offenders
        self.named = named
        self.desktop = desktop

    def report(self, label):
        state = "QUIET" if self.quiet else "*** CONTAMINATED ***"

        print(f"  [machine] {label}: {state} — foreign {self.foreign_seconds:.2f} core-s of "
              f"{self.available:.0f} = {100 * self.share:.2f}% (ceiling {100 * QUIET_CEILING:.0f}%)")

        for name, seconds in self.named:
            print(f"  [machine]   scanner/updater active: {name} used {seconds:.2f} core-s")

        loud = DESKTOP_LOUD_SHARE * self.available

        for name, seconds in self.desktop:
            closable = name not in ("dwm", "explorer")
            verdict = " — CLOSE IT" if seconds >= loud and closable else ""
            print(f"  [machine]   desktop app: {name} used {seconds:.2f} core-s{verdict}")

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

    # Merged by name: a browser is a dozen processes and a dozen rows would bury the one that matters.
    totals = {}

    for seconds, name, _ in foreign:
        if name in DESKTOP_PROCESSES:
            totals[name] = totals.get(name, 0.0) + seconds

    desktop = sorted(((name, seconds) for name, seconds in totals.items() if seconds >= DESKTOP_SECONDS),
                     key=lambda pair: -pair[1])

    # The compositor and the shell cannot be closed by anyone, so condemning a window on them is advice
    # nobody can act on. They are still named, because knowing they were busy explains a slow reading.
    UNCLOSABLE = {"dwm", "explorer"}

    shouting = [name for name, seconds in desktop
                if seconds >= DESKTOP_LOUD_SHARE * available and name not in UNCLOSABLE]
    quiet = share <= QUIET_CEILING and not named and not shouting

    return Verdict(quiet, share, total, available, foreign, named, desktop)


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
