// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Tests.TestSupport.Assemblies
{
    /// <summary>
    /// The XC-34 gate: the candidate <c>DevOnBike.Overfit.dll</c> against the last package on nuget.org.
    ///
    /// <para><b>Two halves, and they run in different places.</b> The first test needs a downloaded baseline
    /// and therefore skips unless <c>OVERFIT_API_BASELINE</c> names one —
    /// <c>Scripts/api_compat_check.py</c> is what sets it, and that script is where a missing baseline
    /// becomes an error rather than a skip. The rest need nothing at all: they compile two tiny assemblies in
    /// memory and pin <see cref="ApiCompatibilityGate.HighestAllowedLevel"/>, so relaxing the threshold turns
    /// the ordinary <c>dotnet test</c> run red instead of only the release gate nobody runs locally.</para>
    ///
    /// <para><b>Why the second half is not redundant.</b> A gate whose only test needs the network is a gate
    /// that is green on CI for the wrong reason. These four cases are the ones that establish the threshold is
    /// load-bearing; the network test is what establishes it is pointed at the right two files.</para>
    /// </summary>
    public sealed class PublishedApiCompatibilityTests
    {
        /// <summary>
        /// Compares the built library against the published baseline and fails on anything above
        /// <see cref="ApiCompatibilityGate.HighestAllowedLevel"/>, naming every finding.
        ///
        /// <para><b>It SKIPS when the baseline is absent, and that is not a formality.</b> A gate that passes
        /// with no input is indistinguishable from a gate that checked something, and this repository has
        /// spent a week converting exactly that shape before (TG-T1). Run
        /// <c>python Scripts/api_compat_check.py</c> to fetch the baseline and set the variable.</para>
        ///
        /// <para><b>Measured 2026-08-13</b>, against the real <c>10.0.31</c> package: red, with seven
        /// <c>AC-TYPE-REMOVED (CP0001)</c> findings naming the <c>FeatureImportance*</c> types deleted from
        /// the working tree. Relaxing <see cref="ApiCompatibilityGate.HighestAllowedLevel"/> to
        /// <see cref="ChangeLevel.BinaryBreaking"/> turns it green — so the constant, and nothing else, is what
        /// makes it fail.</para>
        /// </summary>
        [Fact]
        public void TheCandidateDoesNotBreakTheLastPublishedPackage()
        {
            var baseline = Environment.GetEnvironmentVariable(ApiCompatibilityGate.BaselineVariable);

            if (string.IsNullOrWhiteSpace(baseline))
            {
                Assert.Skip(
                    ApiCompatibilityGate.BaselineVariable + " is not set, so there is no published baseline to "
                    + "compare against. This test is SKIPPED, not passed — it has checked nothing. Run "
                    + "`python Scripts/api_compat_check.py`, which downloads the latest DevOnBike.Overfit "
                    + "package from nuget.org, sets the variable and runs this test.");
            }

            baseline = baseline.Trim();

            Assert.True(File.Exists(baseline),
                ApiCompatibilityGate.BaselineVariable + " points at '" + baseline + "', which does not exist. "
                + "A baseline that is named and missing is an error, not a skip: somebody asked for this gate.");

            var (candidate, overridden) = ApiCompatibilityGate.ResolveCandidate(
                Environment.GetEnvironmentVariable(ApiCompatibilityGate.CandidateVariable));

            Assert.True(File.Exists(candidate),
                "the candidate assembly '" + candidate + "' does not exist. Build it with "
                + "`dotnet build -c Release`, or point " + ApiCompatibilityGate.CandidateVariable
                + " at the assembly to check.");

            AssertCandidateIsThisBuild(candidate, overridden);

            var comparison = AssemblyComparer.CompareFiles(baseline, candidate);
            var description = ApiCompatibilityGate.Describe(comparison, baseline, candidate);
            var verdictPath = ApiCompatibilityGate.VerdictPath();

            // Written BEFORE the assertion, so a breaking verdict reaches the file too. A passing test prints
            // nothing a runner shows, which is why `unchanged` and `additive` are otherwise the same green.
            Directory.CreateDirectory(Path.GetDirectoryName(verdictPath));
            File.WriteAllText(verdictPath, description);

            Assert.True(
                ApiCompatibilityGate.Judge(comparison) != ApiCompatibilityVerdict.Breaking, description);
        }

        /// <summary>
        /// Removing a public type fails the gate.
        ///
        /// <para>This is the XC-34 incident in miniature and it needs no network: seven public types were
        /// removed from the package on 2026-08-12/13 and nothing noticed. Compiled here rather than
        /// downloaded, so the expected answer is true by construction.</para>
        /// </summary>
        [Fact]
        public void RemovingAPublicTypeIsBreaking()
        {
            var before = TinyAssemblyCompiler.Compile(
                "public class Kept { public int Value; } public class Dropped { public int Value; }");
            var after = TinyAssemblyCompiler.Compile("public class Kept { public int Value; }");

            var comparison = AssemblyComparer.CompareImages(before, after);

            Assert.Equal(ApiCompatibilityVerdict.Breaking, ApiCompatibilityGate.Judge(comparison));

            var blocking = ApiCompatibilityGate.Blocking(comparison);

            Assert.NotEmpty(blocking);
            Assert.Contains(blocking, change => change.Target.Contains("Dropped", StringComparison.Ordinal));
        }

        /// <summary>
        /// A purely additive change passes.
        ///
        /// <para><b>The half that stops the gate being a rubber stamp in the other direction.</b> A threshold
        /// tightened to <see cref="ChangeLevel.InternalOnly"/> would fail every release that adds anything,
        /// which is how a gate gets routed around; this is what would go red first.</para>
        /// </summary>
        [Fact]
        public void AddingAPublicTypeAndAMemberIsAdditiveAndPasses()
        {
            var before = TinyAssemblyCompiler.Compile("public class Kept { public int Value; }");
            var after = TinyAssemblyCompiler.Compile(
                "public class Kept { public int Value; public int Added; } public class Fresh { }");

            var comparison = AssemblyComparer.CompareImages(before, after);

            Assert.Equal(ApiCompatibilityVerdict.Additive, ApiCompatibilityGate.Judge(comparison));
            Assert.Empty(ApiCompatibilityGate.Blocking(comparison));
        }

        /// <summary>
        /// A changed <c>const</c> fails, even though nothing was removed and nothing fails to compile.
        ///
        /// <para>The case the whole taxonomy exists for: a literal is copied into the consumer's own IL at
        /// their compile time, so they do not fail, do not rebuild, and are simply already wrong. A gate that
        /// only watched for removals would ship this.</para>
        /// </summary>
        [Fact]
        public void AChangedPublicConstantIsBreaking()
        {
            var before = TinyAssemblyCompiler.Compile("public class Limits { public const int Max = 16; }");
            var after = TinyAssemblyCompiler.Compile("public class Limits { public const int Max = 32; }");

            var comparison = AssemblyComparer.CompareImages(before, after);

            Assert.Equal(ApiCompatibilityVerdict.Breaking, ApiCompatibilityGate.Judge(comparison));
            Assert.Contains(ApiCompatibilityGate.Blocking(comparison),
                change => change.Level == ChangeLevel.SilentBehaviourChange);
        }

        /// <summary>
        /// Two builds of an identical surface are <see cref="ApiCompatibilityVerdict.Unchanged"/>, despite
        /// differing as files — <c>TinyAssemblyCompiler</c> emits non-deterministically on purpose, so the MVID
        /// and timestamp move on every build exactly as they do for a real release.
        /// </summary>
        [Fact]
        public void AnIdenticalSurfaceRebuiltIsUnchanged()
        {
            const string source = "public class Kept { public int Value; }";

            var comparison = AssemblyComparer.CompareImages(
                TinyAssemblyCompiler.Compile(source), TinyAssemblyCompiler.Compile(source));

            Assert.Equal(ApiCompatibilityVerdict.Unchanged, ApiCompatibilityGate.Judge(comparison));
        }

        /// <summary>
        /// With nothing configured the candidate is the packable project's own Release output, and the
        /// staleness check below applies to it.
        /// </summary>
        [Theory]
        [InlineData(null)]
        [InlineData("")]
        [InlineData("   ")]
        // `string?`, not `string`: the first case IS null, and that is the point of the theory —
        // "unset" is the case this method exists to pin. Reported by `xunit.analyzers` 2.0.0 as xUnit1012
        // when the analyzer arrived transitively with xunit.v3 4.0.0 (XC-70); the rule is right and the
        // signature was wrong, so it is fixed rather than suppressed.
        public void AnUnsetOverrideUsesTheProjectsOwnReleaseOutput(string? configured)
        {
            var (path, overridden) = ApiCompatibilityGate.ResolveCandidate(configured);

            Assert.False(overridden);
            Assert.Equal(ApiCompatibilityGate.DefaultCandidatePath(), path);
        }

        /// <summary>
        /// A configured override is used as given, and is reported as an override — which is what turns the
        /// staleness check off. Both halves matter: silently keeping the default would compare the wrong
        /// assembly, and silently keeping the check would make the override unusable.
        /// </summary>
        [Fact]
        public void AConfiguredOverrideIsUsedAndIsReportedAsOne()
        {
            var (path, overridden) = ApiCompatibilityGate.ResolveCandidate("  /artifacts/lib.dll  ");

            Assert.True(overridden);
            Assert.Equal("/artifacts/lib.dll", path);
        }

        /// <summary>
        /// The candidate on disk must be the build this test process loaded.
        ///
        /// <para><b>The failure this prevents.</b> <c>Sources/Main/bin/Release/net10.0/</c> is not cleaned
        /// between builds, so a stale assembly from an earlier branch survives there indefinitely. Comparing a
        /// baseline against last week's build produces a confident, well-formatted, wrong verdict — and the
        /// MVID is the cheapest thing that separates the two, because a file copy preserves it and a rebuild
        /// never does.</para>
        /// </summary>
        private static void AssertCandidateIsThisBuild(string candidate, bool overridden)
        {
            if (overridden)
            {
                // Pointing the gate somewhere else is a deliberate act — at a CI artefact, or at an older
                // package to compare two releases. Asserting it equals the loaded build would make the
                // override useless.
                return;
            }

            var loaded = ApiCompatibilityGate.LoadedCandidatePath();

            Assert.True(File.Exists(loaded),
                "no DevOnBike.Overfit.dll beside the test assembly at '" + loaded
                + "', so there is nothing to check the candidate against.");

            using var candidateFacts = AssemblyFacts.FromFile(candidate);
            using var loadedFacts = AssemblyFacts.FromFile(loaded);

            Assert.True(candidateFacts.Mvid == loadedFacts.Mvid,
                "the candidate assembly is NOT the build under test. '" + candidate + "' has MVID "
                + candidateFacts.Mvid + " (version " + candidateFacts.AssemblyVersion + "), while the copy "
                + "beside the tests at '" + loaded + "' has MVID " + loadedFacts.Mvid + " (version "
                + loadedFacts.AssemblyVersion + "). Rebuild with `dotnet build -c Release` before running the "
                + "gate — a stale candidate produces a confident wrong verdict.");
        }
    }
}
