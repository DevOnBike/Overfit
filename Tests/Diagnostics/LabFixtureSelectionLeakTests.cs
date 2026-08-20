// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Text;
using DevOnBike.Overfit.Tests.TestSupport;

namespace DevOnBike.Overfit.Tests.Diagnostics
{
    /// <summary>
    /// No test may write <c>OVERFIT_LAB_FIXTURE_NAME</c>.
    ///
    /// <para><b>The incident this exists for is <c>TG-T14</c>.</b> One diagnostic set that variable to choose
    /// its own recording and never put it back. The variable is process-wide and it selects the recording
    /// that <see cref="LabWindowFixture"/> hands to <i>every</i> test, so from that point on five plain
    /// <c>[Fact]</c> tests in three unrelated classes loaded the twelve-replica HEALTHY window where they
    /// expected the four-replica one with an injected throttle. They failed with an empty collection and an
    /// out-of-range index — neither of which points anywhere near the test that caused it.</para>
    ///
    /// <para><b>It survived at least six days, and the reason is why a fast guard is worth having.</b> The
    /// culprit is a <c>[LongFact]</c>, so the ordinary suite never ran it; the area gate that would have
    /// shown it hit its own timeout before reaching a verdict; and the number of victims VARIED between 7, 8
    /// and 9 with the order xunit happened to pick, which reads as flakiness rather than as a leak. Finding
    /// it took a 64-minute run. This test does the same job in milliseconds.</para>
    ///
    /// <para><b>Reading the variable is fine; writing it from a test is not.</b>
    /// <see cref="LabWindowFixture.Load(string)"/> takes the path as an argument, so no test needs the
    /// global. The knob itself stays — it is how somebody points the whole suite at another recording on
    /// purpose, from outside the process.</para>
    ///
    /// <para><b>Why this rule and not the general one.</b> "Any test that writes a process-wide setting must
    /// restore it" is the real rule, but detecting a restore in source text is guesswork: a restore can be a
    /// <c>finally</c>, a <c>Dispose</c>, or a captured previous value written back somewhere else entirely.
    /// A first pass of that heuristic flagged <c>ModelFactTests</c>, which is correct and restores in
    /// <c>Dispose</c>. A guard that cries wolf teaches people to ignore it, and this repository already
    /// carries two analyzers that failed in exactly those two opposite directions. So the check here is the
    /// narrow one that cannot produce a false positive.</para>
    /// </summary>
    public sealed class LabFixtureSelectionLeakTests
    {
        /// <summary>The variable that selects the recording, named once so the test and the message agree.</summary>
        private const string Variable = "OVERFIT_LAB_FIXTURE_NAME";

        [Fact]
        public void NoTestWritesTheLabFixtureSelector()
        {
            var root = RepositoryPaths.TryFindRoot()
                       ?? throw new InvalidOperationException(
                           $"could not find Overfit.sln above {AppContext.BaseDirectory}; this test reads "
                           + "the source tree, so it cannot run against a binary-only layout");

            var tests = Path.Combine(root, "Tests");

            Assert.True(Directory.Exists(tests), $"expected a test source tree at {tests}");

            var offenders = new StringBuilder();
            var scanned = 0;

            foreach (var file in Directory.EnumerateFiles(tests, "*.cs", SearchOption.AllDirectories))
            {
                if (file.Contains($"{Path.DirectorySeparatorChar}obj{Path.DirectorySeparatorChar}",
                        StringComparison.Ordinal)
                    || file.Contains($"{Path.DirectorySeparatorChar}bin{Path.DirectorySeparatorChar}",
                        StringComparison.Ordinal))
                {
                    continue;
                }

                scanned++;

                // This file names the variable in its own documentation and in the constant above, so it
                // would report itself. Skipping by path rather than by a cleverer pattern keeps the match
                // simple enough to be obviously correct.
                if (Path.GetFileName(file) == "LabFixtureSelectionLeakTests.cs")
                {
                    continue;
                }

                var text = File.ReadAllText(file);
                var lines = text.Split('\n');

                for (var i = 0; i < lines.Length; i++)
                {
                    if (!lines[i].Contains("SetEnvironmentVariable", StringComparison.Ordinal))
                    {
                        continue;
                    }

                    // The name can sit on the same line as the call or on the next one, so both are read.
                    var window = lines[i] + (i + 1 < lines.Length ? lines[i + 1] : string.Empty);

                    if (!window.Contains(Variable, StringComparison.Ordinal))
                    {
                        continue;
                    }

                    offenders.Append($"\n    {Path.GetRelativePath(root, file)}:{i + 1}  {lines[i].Trim()}");
                }
            }

            // The scan must have SEEN something. An empty walk passes this assertion for the wrong reason,
            // and "no offenders" would then mean "no files read" — the exact shape of failure this
            // repository keeps finding: a guard that quietly stops checking.
            Assert.True(scanned > 100, $"only {scanned} test source files were scanned under {tests}; the "
                                       + "walk found almost nothing, so a green result here would be "
                                       + "meaningless");

            Assert.True(offenders.Length == 0,
                $"{Variable} is written by a test. It selects the recording that LabWindowFixture hands to "
                + "EVERY test in the process, so a write here silently changes what unrelated tests read — "
                + $"see TG-T14. Pass the path to LabWindowFixture.Load(path) instead.{offenders}");
        }
    }
}
