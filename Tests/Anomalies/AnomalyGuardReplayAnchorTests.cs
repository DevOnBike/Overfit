// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Tests.Anomalies.Diagnostics;

namespace DevOnBike.Overfit.Tests.Anomalies
{
    /// <summary>
    /// The replay driver's start anchor, which is a gate rather than a comment.
    ///
    /// <para><b>Why this is worth a test at all.</b> The requirement was documented in
    /// <see cref="AnomalyGuardReplayDiagnostics"/> and enforced by nothing, and the run it protects is a
    /// <c>[LabFact]</c> that cannot be part of the fast suite — so the gate itself would never be exercised
    /// unless somebody happened to run the replay wrong. Resolving the anchor takes the raw environment value
    /// as a parameter for exactly this reason: the rule is checkable in milliseconds, with no cluster, no
    /// process-wide environment mutation and no race against a parallel test class.</para>
    /// </summary>
    public sealed class AnomalyGuardReplayAnchorTests
    {
        private static readonly TimeSpan Cadence = TimeSpan.FromMinutes(5);

        private static readonly DateTimeOffset Now = new(2026, 8, 8, 14, 07, 33, TimeSpan.Zero);

        private const int Cycles = 288;

        /// <summary>
        /// An unset anchor stops the run, and says what to set it to.
        ///
        /// <para><b>Measured:</b> restoring the old "fall back to a day ago" behaviour leaves this red.</para>
        /// </summary>
        [Theory]
        [InlineData(null)]
        [InlineData("")]
        [InlineData("   ")]
        public void AnUnpinnedReplayIsRefused(string? configured)
        {
            var refused = Assert.Throws<InvalidOperationException>(
                () => AnomalyGuardReplayDiagnostics.ResolveReplayStart(configured, Cadence, Cycles, Now));

            Assert.Contains("OVERFIT_REPLAY_START_UTC", refused.Message, StringComparison.Ordinal);

            // The suggestion is a day back from the supplied clock, snapped down to a whole cadence, so it can
            // be pasted straight back in. Without this the message names a rule and no way to satisfy it.
            Assert.Contains("2026-08-07 14:05:00Z", refused.Message, StringComparison.Ordinal);
        }

        /// <summary>
        /// An anchor that is set but is not a timestamp is refused too.
        ///
        /// <para><b>The silent case is the dangerous one</b>: this used to parse as nothing and fall through
        /// to the wall clock, so a run the operator believed was pinned was not, and the incident counts moved
        /// for a reason nothing in the output named.</para>
        ///
        /// <para><b>Measured:</b> replacing the throw with the old <c>TryParse</c>-then-fall-back leaves this
        /// red.</para>
        /// </summary>
        [Theory]
        [InlineData("yesterday")]
        [InlineData("\"2026-08-07T00:00:00Z\"")]
        [InlineData("1754524800")]
        public void AnAnchorThatIsNotATimestampIsRefusedRatherThanReplaced(string configured)
        {
            var refused = Assert.Throws<InvalidOperationException>(
                () => AnomalyGuardReplayDiagnostics.ResolveReplayStart(configured, Cadence, Cycles, Now));

            Assert.Contains(configured, refused.Message, StringComparison.Ordinal);
        }

        /// <summary>
        /// A pinned anchor is used exactly as given — the same instant, whatever offset it was written in, and
        /// with no snapping.
        ///
        /// <para>The value below deliberately sits off the cadence grid: an implementation that "tidied" an
        /// explicit anchor would silently replay a different window from the one the operator named, which is
        /// the failure this gate exists to prevent, one step further along.</para>
        ///
        /// <para><b>Measured:</b> snapping the parsed value down to a whole cadence leaves this red.</para>
        /// </summary>
        [Theory]
        [InlineData("2026-08-07T12:03:17Z")]
        [InlineData("2026-08-07T14:03:17+02:00")]
        [InlineData("2026-08-07 12:03:17")]
        public void APinnedAnchorIsUsedExactlyAsGiven(string configured)
        {
            var anchor = AnomalyGuardReplayDiagnostics.ResolveReplayStart(configured, Cadence, Cycles, Now);

            Assert.Equal(new DateTimeOffset(2026, 8, 7, 12, 03, 17, TimeSpan.Zero), anchor.ToUniversalTime());
        }
    }
}
