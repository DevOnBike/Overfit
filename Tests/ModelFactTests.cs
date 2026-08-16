// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Tests
{
    /// <summary>
    /// <see cref="ModelFact"/> — the guard that turns 25 silent passes into visible skips, guarded in turn.
    ///
    /// <para><b>Why this cannot be demonstrated by running one of those 25.</b> Every fixture they name is
    /// present on the development box, so all of them genuinely execute here; the defect they carried only
    /// ever showed up somewhere the models are missing, which is CI — the one place nobody was reading a
    /// per-test outcome. Proving the mechanism therefore has to be done against a path chosen to be absent,
    /// deterministically, in the fast suite. That is what these are.</para>
    ///
    /// <para>The distinction under test is not cosmetic. <c>Skip</c> makes the runner report
    /// <c>NotExecuted</c>; the early <c>return</c> these replaced made it report <c>Passed</c>, which is
    /// indistinguishable from "checked and correct" in every report anyone reads.</para>
    /// </summary>
    public sealed class ModelFactTests : IDisposable
    {
        private const string Missing = @"X:\definitely\not\here\model.gguf";

        private readonly string _original =
            Environment.GetEnvironmentVariable(LongFact.RunVariable);

        /// <summary>
        /// The attribute reads <c>OVERFIT_RUN_LONG</c> in its constructor and returns early when the test
        /// would be skipped as long-running anyway, so the file check is only reachable with the variable
        /// set. Setting it process-wide is safe here because it is restored in <see cref="Dispose"/> and
        /// because the variable only ever *lifts* a skip — a leak would make long tests run, which is loud,
        /// not silent.
        /// </summary>
        public ModelFactTests()
        {
            Environment.SetEnvironmentVariable(LongFact.RunVariable, "1");
        }

        public void Dispose()
        {
            Environment.SetEnvironmentVariable(LongFact.RunVariable, _original);
        }

        [Fact]
        public void AMissingFixtureSkips_ItDoesNotPass()
        {
            var attribute = new ModelFact(Missing);

            Assert.NotNull(attribute.Skip);
            Assert.Contains(Missing, attribute.Skip);

            // The message has to say the test checked nothing. "Fixture not present" alone reads as an
            // explanation for a pass, which is exactly the reading that let this go unnoticed.
            Assert.Contains("SKIPPED, not passed", attribute.Skip);
        }

        [Fact]
        public void APresentFixtureDoesNotSkip()
        {
            // The test assembly itself: a file that certainly exists wherever this runs, including CI.
            var present = typeof(ModelFactTests).Assembly.Location;

            Assert.True(File.Exists(present), "the running assembly should exist on disk");
            Assert.Null(new ModelFact(present).Skip);
        }

        [Fact]
        public void TheFirstMissingFileIsNamed_NotJustTheCount()
        {
            var present = typeof(ModelFactTests).Assembly.Location;
            var attribute = new ModelFact([present, Missing]);

            Assert.NotNull(attribute.Skip);
            Assert.Contains(Missing, attribute.Skip);
        }

        [Fact]
        public void TheRuntimeStillTravels()
        {
            Assert.Equal("45s", new ModelFact(typeof(ModelFactTests).Assembly.Location, "45s").Runtime);
        }

        [Fact]
        public void WithoutTheEnvironmentVariableItSkipsAsLongRunning_NotAsAMissingFixture()
        {
            // Polarity: the long-running skip is the general reason and must win, so that turning the
            // variable off cannot accidentally turn a fixture problem into a passing test.
            Environment.SetEnvironmentVariable(LongFact.RunVariable, null);

            var attribute = new ModelFact(Missing);

            Assert.NotNull(attribute.Skip);
            Assert.Contains("Long-running", attribute.Skip);
            Assert.DoesNotContain(Missing, attribute.Skip);
        }
    }
}
