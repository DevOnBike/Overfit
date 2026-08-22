// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.GpuProbe
{
    /// <summary>
    /// Writes the report to disk BETWEEN cells, so that a run which never reaches its end still leaves
    /// the results it had.
    /// <para>
    /// <b>Why this exists.</b> The report used to be emitted once, at the end. The README budgets an
    /// hour for a full sweep, and the full sweep has never once completed - not even here, where it was
    /// stopped after three of fifteen combinations. The failure modes are ordinary rather than exotic:
    /// the largest shape holds a 1.2 GiB weight and the README already warns about running out of
    /// memory, a driver can reset, and a stranger can close a window on a run that looks stuck. A
    /// failure at minute fifty used to return NOTHING. Twelve completed combinations answer the question
    /// this probe was written to ask; zero do not.
    /// </para>
    /// <para>
    /// <b>It must not perturb the measurement, so it refuses to run inside a clock.</b> Rendering the
    /// report allocates tens of kilobytes and writing it is a file-system round trip - either one
    /// between a timestamp and its synchronise moves that number while the report looks entirely
    /// normal. That is the same hazard <see cref="LiveView.Refresh"/> faces, so this takes the same
    /// remedy and the same evidence: the write is DROPPED and COUNTED when
    /// <see cref="TimedRegion.IsInside"/> is true, and <see cref="GuardSelfCheck"/> drives exactly that
    /// case on every invocation of the probe so the count printed in the report is a measurement rather
    /// than a default. Dropping a write costs nothing permanent - the next cell writes the same content
    /// plus one more cell - which is why dropping is the right response and writing anyway is not.
    /// </para>
    /// <para>
    /// <b>A failed write must never cost the measurement in memory.</b> A full disk, a file held open by
    /// an editor or a virus scanner, a read-only directory: none of those is a reason to lose an hour of
    /// timings. Every failure is caught, counted, described in the next report, and the run carries on.
    /// The final emit in <see cref="Program"/> is unchanged and still happens.
    /// </para>
    /// </summary>
    internal static class IncrementalReport
    {
        /// <summary>
        /// Suffix of the file each snapshot is written to before it replaces the real one. A process
        /// killed halfway through a <c>File.WriteAllText</c> leaves a TRUNCATED file, which is worse
        /// than a stale one because nothing about it says it is truncated; writing beside the target and
        /// renaming turns that window into "the previous snapshot survives".
        /// </summary>
        private const string TempSuffix = ".writing";

        /// <summary>How many snapshots this process has attempted after passing the timed-region guard.</summary>
        public static int Snapshots { get; private set; }

        /// <summary>
        /// How many writes were refused because a clock was running. Any value above zero is a defect in
        /// the probe: the write is called from between cells, where no region is open.
        /// </summary>
        public static int DroppedInsideTimedRegion { get; private set; }

        /// <summary>How many snapshots could not reach the disk.</summary>
        public static int FailedWrites { get; private set; }

        /// <summary>The most recent write failure, with its exception type. Null when none has happened.</summary>
        public static string? LastFailure { get; private set; }

        /// <summary>
        /// Renders the report as it stands and replaces both files with it. Returns false when the write
        /// was refused or failed; the caller carries on either way.
        /// </summary>
        public static bool Write(Report report, string textPath, string jsonPath)
        {
            if (TimedRegion.IsInside)
            {
                DroppedInsideTimedRegion++;
                return false;
            }

            Snapshots++;
            report.SnapshotsWritten = Snapshots;
            report.WriteNote = Describe();

            try
            {
                // Both rendered before either is written, so the two files can never describe different
                // sets of cells. Rendering is also the step most likely to throw something unexpected,
                // and it must not do so with one file already replaced.
                var text = report.RenderText();
                var json = report.RenderJson();

                ReplaceFile(textPath, text);
                ReplaceFile(jsonPath, json);
                return true;
            }
            // Deliberately every exception, not the two that Emit catches. An incremental snapshot is
            // a convenience, and no failure of it - including one nobody predicted - may cost the timings
            // held in memory or stop the run reaching its final emit. The failure is counted and printed
            // in the next snapshot rather than swallowed, which is the part that makes this honest.
            catch (Exception ex)
            {
                FailedWrites++;
                LastFailure = ex.GetType().Name + ": " + ex.Message;
                return false;
            }
        }

        /// <summary>One sentence about the snapshots, for the report to print beside the numbers.</summary>
        public static string Describe()
        {
            var note = Snapshots == 0
                ? "none written yet"
                : $"{Snapshots} written, one after each shape/batch combination";

            if (DroppedInsideTimedRegion > 0)
            {
                note +=
                    $". {DroppedInsideTimedRegion} were REFUSED because a clock was running - that is a defect " +
                    "in the probe and it means a snapshot was requested from the wrong place";
            }

            if (FailedWrites > 0)
            {
                note +=
                    $". {FailedWrites} FAILED to reach the disk, most recently: {LastFailure}. The timings " +
                    "in memory were not affected and the run carried on";
            }

            return note;
        }

        /// <summary>
        /// Clears the counters. Called by <see cref="GuardSelfCheck"/> after it has driven the refusal
        /// path, so the counts the report prints describe the measured run and not the check.
        /// </summary>
        public static void ResetCounters()
        {
            Snapshots = 0;
            DroppedInsideTimedRegion = 0;
            FailedWrites = 0;
            LastFailure = null;
        }

        private static void ReplaceFile(string path, string content)
        {
            var temp = path + TempSuffix;
            File.WriteAllText(temp, content);
            File.Move(temp, path, overwrite: true);
        }
    }
}
