// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.GpuProbe
{
    /// <summary>
    /// Proves, on every run and before anything is measured, that a repaint requested from INSIDE a
    /// timed region is dropped and counted.
    /// <para>
    /// <b>Why this exists.</b> The guard in <see cref="LiveView.Refresh"/> is the only thing standing
    /// between a misplaced repaint and a whole report of numbers that carry a console frame inside them.
    /// Until this file, nothing in the project ever called <see cref="LiveView.Refresh"/> from inside a
    /// timed region: <c>--live-perturbation</c> refreshes BETWEEN rounds, where
    /// <see cref="TimedRegion.IsInside"/> is already false, and the real run suspends the view around
    /// every timed phase. So the guard could have been deleted and every run would have looked exactly
    /// the same - the counter reads zero either way, and zero is the value a healthy run prints.
    /// </para>
    /// <para>
    /// <b>Why a self-check and not a test project.</b> <c>Demo/GpuProbe</c> is deliberately outside
    /// <c>Overfit.sln</c>, so a test project added beside it would be run by nothing: not by
    /// <c>dotnet test ./Tests/Tests.csproj</c>, not by CI, and not by the stranger this probe is handed
    /// to. That is the silent half of the failure this repository already records twice - a guard that
    /// quietly stops looks identical to a guard that has nothing to say. A check wired into the entry
    /// point runs on every invocation of the probe, including the one run on the machine whose numbers
    /// matter, and it says so in the report the stranger pastes back.
    /// </para>
    /// <para>
    /// <b>It costs the report nothing.</b> No display is attached, so <see cref="LiveView.Refresh"/>
    /// returns before it reads a sensor or lays out a frame in every arm below; the check asserts that
    /// no frame was painted, then resets the counters it moved, so the numbers the report prints
    /// describe the measured run and not this. It needs no GPU, no driver and no terminal.
    /// </para>
    /// <para>
    /// <b>What it does NOT cover.</b> It says nothing about the panels rendering, nothing about NVML,
    /// and nothing about what a repaint COSTS - that is <c>--live-perturbation</c>, and it is a separate
    /// measurement. It also cannot see a timed region that nobody declared: a new stopwatch written
    /// without <see cref="TimedRegion.Enter"/> around it is invisible to the flag and therefore to this.
    /// </para>
    /// </summary>
    internal static class GuardSelfCheck
    {
        /// <summary>What the check found, in the words the report prints. Never null after <see cref="Run"/>.</summary>
        public static string Note { get; private set; } = "did not run";

        /// <summary>
        /// True when the guard is BROKEN. A check that could not run is not a failure and does not set
        /// this - it sets <see cref="Note"/> to say so, which is the distinction between a negative
        /// result and no result.
        /// </summary>
        public static bool Failed { get; private set; }

        /// <summary>
        /// Runs the five arms. Call once, from the entry point, before anything is timed.
        /// </summary>
        /// <param name="seed">Seeds the stub sensor. Nothing below reads a sample, so it only has to exist.</param>
        public static void Run(int seed)
        {
            if (IncrementalReport.Snapshots != 0 || IncrementalReport.DroppedInsideTimedRegion != 0)
            {
                Fail(
                    $"it ran after the snapshot writer had already been used ({IncrementalReport.Snapshots} " +
                    $"snapshots, {IncrementalReport.DroppedInsideTimedRegion} drops). The check resets those " +
                    "counters, so running it here would erase evidence from the run itself.");
                return;
            }

            if (LiveView.FramesPainted != 0 || LiveView.DroppedInsideTimedRegion != 0)
            {
                Fail(
                    $"it ran after the live view had already been used ({LiveView.FramesPainted} frames, " +
                    $"{LiveView.DroppedInsideTimedRegion} drops). The check resets those counters, so running " +
                    "it here would erase evidence from the run itself. This is a defect in the probe's " +
                    "start-up order, not in the guard.");
                return;
            }

            if (TimedRegion.IsInside)
            {
                Fail(
                    "a timed region was already open when the check started, which means some earlier " +
                    "TimedRegion.Enter was never matched by a Leave. Every repaint for the rest of the run " +
                    "would be dropped as inside-a-clock and the live view would silently stop moving.");
                return;
            }

            // ARM 5 first, because it is the one arm that needs no console - and the give-up path below,
            // for a machine where Spectre cannot be prepared, returns before any of the repaint arms run.
            // Ordered second it would have been skipped on exactly the machines least like this one.
            //
            // It covers the other expensive thing that happens between cells: writing the report to disk.
            // IncrementalReport.Write renders tens of kilobytes and touches the file system, so a
            // snapshot requested from inside a clock would move that number the same way a repaint would.
            // This arm is what makes its drop counter mean something; without it the counter reads zero
            // whether the guard works or has been deleted, which is the shape of the defect this whole
            // file exists to prevent.
            if (!CheckSnapshotGuard())
            {
                return;
            }

            IncrementalReport.ResetCounters();

            var telemetry = new StubTelemetry(seed);
            var view = LiveView.TryCreate(telemetry, new LiveProbeState(), force: true);

            if (view is null)
            {
                telemetry.Dispose();
                Note =
                    "PARTLY RUN. The snapshot guard passed: a report snapshot requested from inside a timed " +
                    "region was refused and wrote no file. The four REPAINT arms could not run, because the " +
                    "console could not be prepared here (" + (LiveView.Unavailable ?? "no reason recorded") +
                    "), so the repaint drop guard is UNVERIFIED for this run. That is not the same as " +
                    "passing. Nothing else about this run changed.";
                return;
            }

            using (view)
            {
                // ARM 1 - outside a timed region, not suspended: must NOT be counted as a drop. This is
                // the arm that proves the guard can tell the difference. Without it, a guard hard-wired
                // to increment on every call would pass every other arm here.
                view.Suspended = false;
                view.Refresh();

                if (LiveView.DroppedInsideTimedRegion != 0)
                {
                    Fail(
                        "a repaint requested OUTSIDE every timed region was counted as a drop. The guard " +
                        "fires when it should not, so the report's drop count says nothing about where " +
                        "repaints actually happen.");
                    return;
                }

                // ARM 2 - inside a timed region: must be dropped and counted. This is the arm the whole
                // guard exists for, and it is the one no other run in this project reaches.
                TimedRegion.Enter();
                view.Refresh();
                TimedRegion.Leave();

                if (LiveView.DroppedInsideTimedRegion != 1)
                {
                    Fail(
                        "a repaint requested from INSIDE a timed region was not dropped. A sensor read is a " +
                        "driver round trip and a frame is tens of kilobytes of allocation; either one between " +
                        "a timestamp and its synchronise moves that number and leaves the report looking " +
                        "entirely normal. Every timing this probe produces is suspect until this is fixed.");
                    return;
                }

                // ARM 3 - inside a timed region AND suspended: must still be dropped and counted, so the
                // guard cannot be made conditional on the suspension. That is the property measured on
                // 2026-08-22: the timed phases are exactly the phases the view is suspended for, so a
                // guard that skips the count while suspended reads zero for precisely the call sites it
                // exists to catch.
                //
                // Measured, so that this comment does not claim more than it covers: moving the guard
                // BELOW the suspension test is caught by arm 2, not by this one, because no display is
                // attached here and the null-context test in the same `if` returns first. What only this
                // arm catches is the guard being weakened to `IsInside && !Suspended` - run as a mutation
                // and it reddens here and nowhere else.
                view.Suspended = true;
                TimedRegion.Enter();
                view.Refresh();
                TimedRegion.Leave();
                view.Suspended = false;

                if (LiveView.DroppedInsideTimedRegion != 2)
                {
                    Fail(
                        "a repaint requested from inside a timed region was swallowed by the suspension check " +
                        "instead of being counted as a drop. The counter would then read zero for precisely " +
                        "the call sites it exists to catch, and the report would carry a reassuring zero it " +
                        "had not earned.");
                    return;
                }

                // ARM 4 - after Leave, a repaint must be accepted again. This pins TimedRegion.Leave
                // itself: if it stops clearing the flag, every repaint for the rest of the run is dropped,
                // the live view freezes, and the report accuses the probe of a defect it does not have.
                if (TimedRegion.IsInside)
                {
                    Fail("TimedRegion.Leave did not clear the flag.");
                    return;
                }

                view.Refresh();

                if (LiveView.DroppedInsideTimedRegion != 2)
                {
                    Fail(
                        "a repaint requested after a timed region had closed was still dropped, so the flag " +
                        "is leaking out of the region. Every later repaint is dropped too: the view freezes " +
                        "and the report reports a defect that is in the flag rather than at any call site.");
                    return;
                }

                if (LiveView.FramesPainted != 0)
                {
                    Fail(
                        $"the check itself painted {LiveView.FramesPainted} frames. It attaches no display " +
                        "precisely so that it cannot, and a frame here means it has started costing the run " +
                        "it is supposed to be free of.");
                    return;
                }
            }

            LiveView.ResetCounters();

            Note =
                "PASSED. Five arms, before anything was timed: a repaint outside a timed region was " +
                "accepted, a repaint inside one was dropped and counted, a repaint inside one was dropped " +
                "and counted even with the view suspended, a repaint after the region closed was " +
                "accepted again, and a REPORT SNAPSHOT requested from inside a timed region was refused " +
                "with no file written. So the drop counts printed below are measurements and not defaults. " +
                "This says nothing about what a repaint or a snapshot COSTS - that is --live-perturbation - " +
                "and nothing about a stopwatch written without TimedRegion.Enter around it, which no flag " +
                "can see. It also does not prove the report DIRECTORY is writable: that is established by " +
                "the first real snapshot, which is taken before any cell for exactly that reason.";
        }

        /// <summary>
        /// Arm 5. Drives <see cref="IncrementalReport.Write"/> from inside a timed region and requires
        /// that it refuses and writes nothing. Returns false when the arm failed.
        /// </summary>
        private static bool CheckSnapshotGuard()
        {
            var stem = Path.Combine(
                Path.GetTempPath(), "gpu-probe-guard-" + Guid.NewGuid().ToString("N"));
            var textPath = stem + ".txt";
            var jsonPath = stem + ".json";

            TimedRegion.Enter();
            var wrote = IncrementalReport.Write(new Report(), textPath, jsonPath);
            TimedRegion.Leave();

            var landed = File.Exists(textPath) || File.Exists(jsonPath);
            if (landed)
            {
                Delete(textPath);
                Delete(jsonPath);
            }

            if (wrote || landed)
            {
                Fail(
                    "a report snapshot requested from INSIDE a timed region was written instead of being " +
                    "refused. Rendering the report allocates tens of kilobytes and writing it is a file " +
                    "system round trip; between a timestamp and its synchronise, either one moves that " +
                    "number and leaves the report looking entirely normal.");
                return false;
            }

            if (IncrementalReport.DroppedInsideTimedRegion != 1)
            {
                Fail(
                    $"a report snapshot requested from inside a timed region wrote nothing, but the drop " +
                    $"counter reads {IncrementalReport.DroppedInsideTimedRegion} instead of 1. The write was " +
                    "stopped by something other than the guard - most likely it failed - so the guard itself " +
                    "is unverified and the count the report prints is not a measurement of it.");
                return false;
            }

            if (IncrementalReport.Snapshots != 0)
            {
                Fail(
                    $"a refused snapshot still counted as one of {IncrementalReport.Snapshots} written. The " +
                    "report would then claim a snapshot that never reached the disk.");
                return false;
            }

            return true;
        }

        private static void Delete(string path)
        {
            try
            {
                File.Delete(path);
            }
            catch (IOException)
            {
                // The arm has already failed and the run is about to abort; a leftover file in the temp
                // directory is not worth taking the process down for.
            }
            catch (UnauthorizedAccessException)
            {
            }
        }

        private static void Fail(string what)
        {
            Failed = true;
            Note = "FAILED - " + what;
        }
    }
}
