// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Diagnostics;

namespace DevOnBike.Overfit.Tests
{
    /// <summary>
    /// <see cref="MeasurementRefusalMarker"/> — the file the measurement guard writes when it refuses, and the
    /// only channel it has, since <c>XC-17</c> measured that neither an exit code nor a message survives the
    /// VSTest bridge.
    ///
    /// <para><b>What these pin, and it is content rather than mechanism.</b> The marker exists to answer one
    /// question in one read: <i>something alive is holding the lock, here is how to find it</i>. A marker that
    /// is written but omits that is the 2026-08-13 hour spent again, so the sentences carrying it are asserted
    /// individually rather than trusted to a smoke test that only checks the file appeared.</para>
    ///
    /// <para><b>The end-to-end case is the one that matters</b>, because everything else here tests a helper
    /// nobody would call by accident. <see cref="TheRefusalPathActuallyWritesTheMarker"/> makes a child process
    /// hit the real refusal — the suite running these tests already holds the mutex, so the child cannot
    /// acquire it — and asserts the file lands. It is safe to run because <c>-list classes</c> makes the child
    /// discover rather than execute: if the guard ever stopped refusing, the child prints class names and
    /// exits instead of running the suite recursively.</para>
    /// </summary>
    public sealed class MeasurementRefusalMarkerTests
    {
        [Fact]
        public void TheMarkerSaysTheHolderIsAliveRatherThanDead()
        {
            // The decisive sentence. Exit 2 is reachable ONLY when WaitOne returns false; a holder that died
            // raises AbandonedMutexException, which MeasurementExclusion catches and treats as acquired. A
            // reader who does not know that goes looking for stale state to clean up, which is what happened.
            var text = Compose();

            Assert.Contains("ALIVE", text, StringComparison.Ordinal);
            Assert.Contains("AbandonedMutexException", text, StringComparison.Ordinal);
            Assert.Contains("never a process that died holding it", text, StringComparison.Ordinal);
        }

        [Fact]
        public void TheMarkerNamesTheLockAndWhenItWasWritten()
        {
            var text = Compose();

            Assert.Contains(MeasurementExclusion.MutexName, text, StringComparison.Ordinal);
            Assert.Contains("2026-03-04 05:06:07", text, StringComparison.Ordinal);
            Assert.Contains("UTC", text, StringComparison.Ordinal);
        }

        [Fact]
        public void TheMarkerSaysToSearchByNameAndCarriesTheOrphanSignature()
        {
            // Both halves of the one-glance diagnostic: the search that works (name pattern) and the shape to
            // look for (test executable with no testhost beside it). Three filters built on process class
            // found nothing on 2026-08-13 while the orphan was in the list under its assembly name.
            var text = Compose();

            Assert.Contains("Get-CimInstance Win32_Process", text, StringComparison.Ordinal);
            Assert.Contains("Overfit|Tests", text, StringComparison.Ordinal);
            Assert.Contains("NAME PATTERN", text, StringComparison.Ordinal);
            Assert.Contains("MATCHED PAIR", text, StringComparison.Ordinal);
            Assert.Contains("testhost", text, StringComparison.Ordinal);
            Assert.Contains("process TREE", text, StringComparison.Ordinal);
        }

        [Fact]
        public void TheMarkerIdentifiesTheRunThatWasRefused()
        {
            // Only the latest refusal is kept, so a reader coming back an hour later needs to tell their own
            // blocked run from a later one. The pid and the command line are what let them.
            var text = Compose();

            Assert.Contains("pid 4242", text, StringComparison.Ordinal);
            Assert.Contains("DevOnBike.Overfit.Tests", text, StringComparison.Ordinal);
            Assert.Contains("--filter Something", text, StringComparison.Ordinal);
        }

        [Fact]
        public void CandidateProcessesAreListedWhenThereAreAnyAndSaidToBeAbsentWhenThereAreNot()
        {
            var withCandidates = Compose();
            var withNone = MeasurementRefusalMarker.Compose(
                Timestamp,
                MeasurementExclusion.MutexName,
                4242,
                "DevOnBike.Overfit.Tests",
                "DevOnBike.Overfit.Tests.exe --filter Something",
                []);

            Assert.Contains("pid 26056  DevOnBike.Overfit.Tests", withCandidates, StringComparison.Ordinal);

            // An empty section reads as "nothing was checked". Saying so explicitly is the difference between
            // "no holder found" and "the list could not be read", and the reader needs to know which.
            Assert.Contains("(none matched", withNone, StringComparison.Ordinal);
            Assert.DoesNotContain("pid 26056", withNone, StringComparison.Ordinal);
        }

        [Fact]
        public void LiveCandidatesFindThisProcess()
        {
            // Whatever launched this run — `testhost` under dotnet test, `DevOnBike.Overfit.Tests` when the
            // executable is run directly — normally matches the fragment list, so the enumeration missing it
            // means the enumeration is broken rather than that the box is quiet.
            //
            // The one case that legitimately does not match is a host launched as bare `dotnet` — no apphost,
            // which is possible on Linux CI — because `dotnet` is deliberately excluded from the fragment
            // list as too noisy. That is a fact about the runner, not a defect, so it skips.
            //
            // Named literally rather than by asking the fragment list, and that is not fussiness: the first
            // version of this guard skipped when the list matched nothing, which meant emptying the list — the
            // exact corruption this test exists to catch — made the test SKIP ITSELF. Measured: the mutation
            // went green.
            using var self = Process.GetCurrentProcess();

            if (self.ProcessName.Equals("dotnet", StringComparison.OrdinalIgnoreCase))
            {
                Assert.Skip("This run is hosted by a bare `dotnet` process, which the fragment list excludes.");
            }

            var candidates = MeasurementRefusalMarker.LiveCandidates();

            Assert.NotEmpty(candidates);
            Assert.Contains(candidates, line => line.Contains($"pid {Environment.ProcessId} ", StringComparison.Ordinal));
        }

        [Fact]
        public void TheMarkerIsWrittenWhereTheCallerAsked()
        {
            var directory = Path.Combine(Path.GetTempPath(), "overfit-marker-" + Guid.NewGuid().ToString("N"));

            try
            {
                var path = MeasurementRefusalMarker.TryWrite(directory, "content");

                Assert.NotNull(path);
                Assert.Equal(Path.Combine(directory, MeasurementRefusalMarker.FileName), path);
                Assert.Equal("content", File.ReadAllText(path));
            }
            finally
            {
                if (Directory.Exists(directory))
                {
                    Directory.Delete(directory, recursive: true);
                }
            }
        }

        [Fact]
        public void AnUnwritableTargetReturnsNullInsteadOfThrowing()
        {
            // The write happens on top of a refusal that has already ended the run. An exception here would
            // replace a legible failure with an illegible one, which is the exact defect being fixed.
            var occupied = Path.Combine(Path.GetTempPath(), "overfit-marker-" + Guid.NewGuid().ToString("N"));

            try
            {
                File.WriteAllText(occupied, "not a directory");

                Assert.Null(MeasurementRefusalMarker.TryWrite(occupied, "content"));
            }
            finally
            {
                File.Delete(occupied);
            }
        }

        [Fact]
        public void TheDefaultDirectoryIsTestsBinBesideTheSolution()
        {
            // Tests/bin because a test run already writes there and git already ignores it. Resolved by
            // walking up to Overfit.sln, so it survives a change in output-directory depth.
            var directory = MeasurementRefusalMarker.ResolveDirectory();

            if (directory == AppContext.BaseDirectory)
            {
                Assert.Skip($"No Overfit.sln above {AppContext.BaseDirectory}, so the fallback path is in use.");
            }

            Assert.EndsWith(Path.Combine("Tests", "bin"), directory, StringComparison.Ordinal);
        }

        [Fact]
        public void TheRefusalPathActuallyWritesTheMarker()
        {
            // The refusal ends in Environment.Exit(2) and cannot be exercised in-process without ending this
            // test host, so it is exercised in a CHILD. The suite running this test already holds the mutex,
            // so the child meets a genuinely live holder — the real code path, not a simulation of it.
            //
            // `-list classes` is what makes this safe: the framework is constructed for discovery, so the
            // guard runs, but no test executes. If the refusal ever stopped happening, the child prints class
            // names and exits rather than running the suite inside itself.
            if (Environment.GetEnvironmentVariable("OVERFIT_ALLOW_CONCURRENT_MEASUREMENT") == "1")
            {
                Assert.Skip("The exclusion is switched off for this run, so no mutex is held to refuse against.");
            }

            var executable = Path.Combine(
                AppContext.BaseDirectory,
                "DevOnBike.Overfit.Tests" + (OperatingSystem.IsWindows() ? ".exe" : string.Empty));

            if (!File.Exists(executable))
            {
                Assert.Skip($"No test executable at {executable} to run as a child.");
            }

            var marker = Path.Combine(MeasurementRefusalMarker.ResolveDirectory(), MeasurementRefusalMarker.FileName);
            var before = DateTime.UtcNow.AddSeconds(-2);

            var start = new ProcessStartInfo(executable)
            {
                RedirectStandardError = true,
                RedirectStandardOutput = true,
                UseShellExecute = false,
                WorkingDirectory = AppContext.BaseDirectory
            };

            start.ArgumentList.Add("-list");
            start.ArgumentList.Add("classes");

            using var child = Process.Start(start);

            Assert.NotNull(child);

            var error = child.StandardError.ReadToEnd();

            _ = child.StandardOutput.ReadToEnd();

            if (!child.WaitForExit(60_000))
            {
                // Kill the TREE, not the process: the orphan in XC-42 survived precisely because a kill
                // stopped at the runner and left the test executable running with the mutex.
                child.Kill(entireProcessTree: true);

                Assert.Fail("The child did not exit within 60s; it may now be holding the measurement mutex.");
            }

            Assert.True(File.Exists(marker), $"No marker at {marker}. Child stderr: {error}");
            Assert.True(
                File.GetLastWriteTimeUtc(marker) >= before,
                $"Marker at {marker} is stale ({File.GetLastWriteTimeUtc(marker):O}), so this refusal did not write it.");

            var text = File.ReadAllText(marker);

            Assert.Contains(MeasurementExclusion.MutexName, text, StringComparison.Ordinal);
            Assert.Contains("-list", text, StringComparison.Ordinal);
            Assert.Contains("ALIVE", text, StringComparison.Ordinal);

            // The console line matters too: the path is the one thing the operator can act on, and it is the
            // only part of the refusal that reaches a terminal running the executable directly.
            Assert.Contains("Details written to", error, StringComparison.Ordinal);
        }

        private static string Compose()
        {
            return MeasurementRefusalMarker.Compose(
                Timestamp,
                MeasurementExclusion.MutexName,
                4242,
                "DevOnBike.Overfit.Tests",
                "DevOnBike.Overfit.Tests.exe --filter Something",
                ["pid 26056  DevOnBike.Overfit.Tests  (started 2026-08-13 09:14:02)"]);
        }

        private static readonly DateTimeOffset Timestamp =
            new(2026, 3, 4, 5, 6, 7, TimeSpan.FromHours(1));
    }
}
