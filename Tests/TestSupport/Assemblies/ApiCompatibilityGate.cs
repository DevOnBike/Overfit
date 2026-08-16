// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Text;

namespace DevOnBike.Overfit.Tests.TestSupport.Assemblies
{
    /// <summary>
    /// The release gate built on <see cref="AssemblyComparer"/>: does the candidate <c>DevOnBike.Overfit.dll</c>
    /// break the last package already on nuget.org?
    ///
    /// <para><b>Why this exists (XC-34).</b> The comparator and its six-level taxonomy landed on 2026-08-12
    /// with 48 tests and no caller. On 2026-08-12/13 seven public types were removed from the package and the
    /// version was raised from <c>10.0.31</c> to <c>10.1.0</c> <b>by hand, because nothing was watching</b> —
    /// and <c>10.0.31</c> was the version already published, so a build-and-push without that manual step
    /// would have republished a breaking change under an existing number.</para>
    ///
    /// <para><b>The baseline is the published package, not a checked-in binary and not a text snapshot.</b> A
    /// committed baseline DLL is ~1.2 MB per release and grows for ever; a text snapshot would be a second
    /// implementation of what the comparator already does, and the two would drift. <c>Scripts/api_compat_check.py</c>
    /// resolves the latest version from nuget.org, downloads it, and points
    /// <see cref="BaselineVariable"/> at the extracted assembly.</para>
    ///
    /// <para><b>Absence is loud in the script, not here.</b> With <see cref="BaselineVariable"/> unset the test
    /// that uses this SKIPS — it cannot pass, because a gate that passes when its input is missing is the
    /// defect this repository names most often. Making absence an error instead would put a network dependency
    /// on every <c>dotnet test</c> run, so the script owns that half: it exits non-zero when the baseline
    /// cannot be fetched or extracted.</para>
    ///
    /// <para><b>What this does NOT answer.</b> Behaviour behind an unchanged signature. Two assemblies with an
    /// identical public surface can do entirely different things, and <see cref="ApiCompatibilityVerdict.Unchanged"/>
    /// says only that no consumer's call sites move. <see cref="AssemblyComparison.Warnings"/> is printed in
    /// every report for the same reason, and deliberately does not change the verdict: a warning is a limit on
    /// what was examined, not a finding about the surface.</para>
    /// </summary>
    internal static class ApiCompatibilityGate
    {
        /// <summary>
        /// Path to the baseline assembly — the <c>lib/net10.0/DevOnBike.Overfit.dll</c> of the last published
        /// package. Unset means SKIP; it never means pass.
        /// </summary>
        internal const string BaselineVariable = "OVERFIT_API_BASELINE";

        /// <summary>
        /// Optional override for the candidate assembly. Unset uses the packable project's own Release output,
        /// which is the artefact that would actually be pushed.
        ///
        /// <para>It exists so the gate can be pointed at a build produced somewhere else — a CI artefact
        /// directory, or a previously published package when checking one release against another. Overriding
        /// it turns off the staleness check below, because the point of that check is that the default path
        /// and the assembly this test process loaded must be the same build.</para>
        /// </summary>
        internal const string CandidateVariable = "OVERFIT_API_CANDIDATE";

        /// <summary>
        /// The worst finding that ships without a decision.
        ///
        /// <para><b>This constant is the gate.</b> Everything above it — <see cref="ChangeLevel.SourceBreaking"/>,
        /// <see cref="ChangeLevel.BinaryBreaking"/>, <see cref="ChangeLevel.SilentBehaviourChange"/> — fails.
        /// <c>PublishedApiCompatibilityTests</c> pins it with in-memory assemblies that need no network, so
        /// relaxing it turns the ordinary suite red rather than only the release gate nobody runs locally.</para>
        /// </summary>
        internal const ChangeLevel HighestAllowedLevel = ChangeLevel.Additive;

        /// <summary>The findings that fail the gate: everything above <see cref="HighestAllowedLevel"/>.</summary>
        internal static IReadOnlyList<ApiChange> Blocking(AssemblyComparison comparison)
        {
            ArgumentNullException.ThrowIfNull(comparison);

            return comparison.AtOrAbove(HighestAllowedLevel + 1);
        }

        /// <summary>
        /// The one-word answer. See <see cref="ApiCompatibilityVerdict"/> for what each word obliges.
        /// </summary>
        internal static ApiCompatibilityVerdict Judge(AssemblyComparison comparison)
        {
            ArgumentNullException.ThrowIfNull(comparison);

            if (Blocking(comparison).Count > 0)
            {
                return ApiCompatibilityVerdict.Breaking;
            }

            // Below Additive the surface did not move at all: what is left is IL, build stamps, or nothing.
            // Those are real differences and they are not API-compatibility findings.
            if (comparison.HighestLevel >= ChangeLevel.Additive)
            {
                return ApiCompatibilityVerdict.Additive;
            }

            return ApiCompatibilityVerdict.Unchanged;
        }

        /// <summary>
        /// The assembly that would be packed: <c>Sources/Main/bin/Release/net10.0/DevOnBike.Overfit.dll</c>.
        ///
        /// <para>Only <c>DevOnBike.Overfit</c> is in scope — <c>Anomalies</c>, <c>Cli</c>, <c>Server*</c> and
        /// <c>Mcp</c> are not packable, so no consumer can be broken by a change to them.</para>
        /// </summary>
        internal static string DefaultCandidatePath()
        {
            return Path.Combine(
                RepositoryPaths.Root, "Sources", "Main", "bin", "Release", "net10.0", "DevOnBike.Overfit.dll");
        }

        /// <summary>
        /// The copy of <c>DevOnBike.Overfit.dll</c> sitting beside the running test assembly — the build this
        /// test process is actually exercising, and the thing the candidate must be identical to.
        /// </summary>
        internal static string LoadedCandidatePath()
        {
            return Path.Combine(AppContext.BaseDirectory, "DevOnBike.Overfit.dll");
        }

        /// <summary>
        /// Where the gate leaves its verdict for <c>Scripts/api_compat_check.py</c> to read.
        ///
        /// <para><b>Why a file and not the test output.</b> A passing test prints nothing a runner shows, so
        /// from outside, <see cref="ApiCompatibilityVerdict.Unchanged"/> and
        /// <see cref="ApiCompatibilityVerdict.Additive"/> are the same green — and the release gate is asked
        /// for three answers, not two. The script deletes this file before every run, so a stale verdict from
        /// an earlier comparison cannot be read as this one's.</para>
        /// </summary>
        internal static string VerdictPath()
        {
            return RepositoryPaths.TestsBin("api-compat-verdict.txt");
        }

        /// <summary>
        /// The candidate to compare, and whether it came from <see cref="CandidateVariable"/>.
        ///
        /// <para><b>Takes the raw environment value as a parameter</b>, following the same reasoning as
        /// <c>AnomalyGuardReplayDiagnostics.ResolveReplayStart</c>: the rule is then checkable in
        /// milliseconds, with no process-wide environment mutation and no race against a test class running
        /// in parallel.</para>
        /// </summary>
        internal static (string Path, bool Overridden) ResolveCandidate(string configured)
        {
            if (!string.IsNullOrWhiteSpace(configured))
            {
                return (configured.Trim(), true);
            }

            return (DefaultCandidatePath(), false);
        }

        /// <summary>
        /// A report a person can act on: the verdict, then every blocking finding, then the comparator's own
        /// evidence. This is the failure message, so it has to say what broke and not merely that something did.
        /// </summary>
        internal static string Describe(AssemblyComparison comparison, string baselinePath, string candidatePath)
        {
            ArgumentNullException.ThrowIfNull(comparison);

            var verdict = Judge(comparison);
            var blocking = Blocking(comparison);
            var report = new StringBuilder();

            report.Append("API compatibility verdict: ").Append(verdict.ToString().ToUpperInvariant())
                .Append('\n');
            report.Append("  baseline : ").Append(baselinePath).Append('\n');
            report.Append("  candidate: ").Append(candidatePath).Append('\n');
            report.Append("  highest allowed level: ").Append((int)HighestAllowedLevel).Append(' ')
                .Append(HighestAllowedLevel).Append('\n');
            report.Append("  blocking findings: ").Append(blocking.Count).Append('\n');

            // BOUND: one iteration per blocking finding.
            foreach (var change in blocking)
            {
                report.Append("  !! ").Append(change).Append('\n');
            }

            report.Append('\n').Append(comparison.Report());

            return report.ToString();
        }
    }
}
