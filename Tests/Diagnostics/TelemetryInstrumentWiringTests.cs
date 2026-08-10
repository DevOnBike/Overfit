// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Text.RegularExpressions;
using DevOnBike.Overfit.Tests.TestSupport;

namespace DevOnBike.Overfit.Tests.Diagnostics
{
    /// <summary>
    /// Every declared telemetry instrument must be fed by something.
    ///
    /// <para><b>An instrument that is never written exports a flat zero, and a flat zero is
    /// indistinguishable from "this never happens".</b> That is the same pathology the anomaly guard exists
    /// to remove, living inside the tooling: a dashboard built on it looks healthy precisely because nothing
    /// is measuring. It is not caught by any other test, because every other test agrees with the code — it
    /// is the <i>description</i> that is false, and descriptions compile.</para>
    ///
    /// <para><b>Measured, which is why this exists.</b> On 2026-08-02 two independent reviews found the same
    /// shape twice in one afternoon: <c>overfit_guard_state_failures_total</c> in the anomaly guard, exported
    /// with a documented meaning and no caller anywhere; and eleven of forty-two instruments in
    /// <c>OverfitTelemetry</c>, each with a real description and zero call sites. Seven of the nine defects
    /// found by review that day were of this family — prose or instrumentation claiming something the code
    /// does not do.</para>
    ///
    /// <para><b>A ratchet, not a wall.</b> The eleven known-dead instruments are listed below rather than
    /// failing the build, because a red test that everyone learns to ignore protects nothing. What the test
    /// enforces is that the list only ever shrinks: a new dead instrument fails, and an instrument on the
    /// list that has since been wired up <b>also</b> fails, so the list cannot quietly rot into a permanent
    /// exemption.</para>
    /// </summary>
    public sealed class TelemetryInstrumentWiringTests
    {
        /// <summary>
        /// Instruments declared and documented but fed by nothing, as of 2026-08-02.
        ///
        /// <para><b>This list may only shrink.</b> Each entry is a metric somebody can put on a dashboard and
        /// read as evidence. Wire it or delete it; leaving it here for ever is choosing to export a lie
        /// slowly.</para>
        /// </summary>
        private static readonly string[] KnownUnwired =
        [
            "AllocationBytes",
            "GraphAllocatedBytes",
            "GraphBackwardDurationMs",
            "GraphCount",
            "KernelCount",
            "KernelDurationMs",
            "ModuleAllocatedBytes",
            "ModuleCount",
            "ModuleDurationMs",
            "NativeMemoryBytes",
            "TapeOpCount",
        ];

        /// <summary>
        /// Matches an instrument field: <c>Counter&lt;long&gt; Name</c>, <c>Histogram&lt;double&gt; Name</c>
        /// and the two gauge shapes. Deliberately narrow — a pattern that also matched local variables would
        /// produce phantom "dead instruments" and the test would be abandoned within a week.
        /// </summary>
        private static readonly Regex Declaration = new(
            @"\b(?:Counter|UpDownCounter|Histogram|ObservableGauge|ObservableCounter)<[^>]+>\s+(\w+)\s*(?:=|;)",
            RegexOptions.Compiled | RegexOptions.CultureInvariant);

        [Fact]
        public void EveryDeclaredInstrumentIsFedBySomething()
        {
            var root = RepositoryRoot();
            var telemetry = Path.Combine(root, "Sources", "Main", "Diagnostics", "OverfitTelemetry.cs");

            Assert.True(File.Exists(telemetry),
                $"expected the telemetry source at {telemetry}; this test scans source rather than metadata "
                + "because a never-called instrument is perfectly well-formed at runtime");

            var declared = new List<string>();

            foreach (Match match in Declaration.Matches(File.ReadAllText(telemetry)))
            {
                var name = match.Groups[1].Value;

                if (!declared.Contains(name, StringComparer.Ordinal))
                {
                    declared.Add(name);
                }
            }

            Assert.True(declared.Count > 10,
                $"only {declared.Count} instrument(s) matched — the declaration pattern has drifted from the "
                + "source, and a test that matches nothing passes for the wrong reason");

            var body = ReadSources(root);
            var unwired = new List<string>();

            foreach (var name in declared)
            {
                // `Name.Add(`, `Name.Record(`, and the null-conditional forms. Whitespace between the name
                // and the dot is allowed because a call can be wrapped across lines.
                var used = Regex.IsMatch(
                    body,
                    @"\b" + Regex.Escape(name) + @"\s*[.?]\s*\.?\s*(?:Add|Record)\s*\(",
                    RegexOptions.CultureInvariant);

                if (!used)
                {
                    unwired.Add(name);
                }
            }

            var appeared = unwired.Except(KnownUnwired, StringComparer.Ordinal).Order(StringComparer.Ordinal);
            var revived = KnownUnwired.Except(unwired, StringComparer.Ordinal).Order(StringComparer.Ordinal);

            Assert.True(!appeared.Any(),
                "these instruments are declared, documented, and fed by nothing — they will export a flat "
                + "zero that reads as 'this never happens': " + string.Join(", ", appeared));

            Assert.True(!revived.Any(),
                "these instruments are on the known-unwired list but now have a call site. Remove them from "
                + "KnownUnwired — a list that keeps entries it no longer needs stops being read: "
                + string.Join(", ", revived));
        }

        /// <summary>
        /// Every C# file under <c>Sources/Main</c>, concatenated.
        ///
        /// <para>The whole tree rather than a guess at which module ought to feed a given instrument: the
        /// question is whether <i>anything</i> writes it, and a narrower scan would answer a different one.</para>
        /// </summary>
        private static string ReadSources(string root)
        {
            var text = new System.Text.StringBuilder(1 << 20);
            var main = Path.Combine(root, "Sources", "Main");

            foreach (var file in Directory.EnumerateFiles(main, "*.cs", SearchOption.AllDirectories))
            {
                // obj/ and bin/ hold generated copies; counting them would let a stale build satisfy the test.
                if (file.Contains($"{Path.DirectorySeparatorChar}obj{Path.DirectorySeparatorChar}",
                        StringComparison.Ordinal)
                    || file.Contains($"{Path.DirectorySeparatorChar}bin{Path.DirectorySeparatorChar}",
                        StringComparison.Ordinal))
                {
                    continue;
                }

                text.Append(File.ReadAllText(file)).Append('\n');
            }

            return text.ToString();
        }

        /// <summary>
        /// Walks up from the test binary until it finds the solution.
        ///
        /// <para>Fails loudly rather than skipping when it cannot: a skipped test is indistinguishable from a
        /// passing one, and this project has already lost a real failure that way.</para>
        /// </summary>
        private static string RepositoryRoot()
        {
            // The walk is shared (XC-4); the POLICY is not. This one fails loudly on purpose — see the
            // remarks above — so it keeps its own message rather than inheriting RepositoryPaths.Root's.
            return RepositoryPaths.TryFindRoot()
                   ?? throw new InvalidOperationException(
                       $"could not find Overfit.sln above {AppContext.BaseDirectory}; this test reads the "
                       + "source tree, so it cannot run against a binary-only layout");
        }
    }
}
