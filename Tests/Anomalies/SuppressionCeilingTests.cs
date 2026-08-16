// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Anomalies.Contracts;
using DevOnBike.Overfit.Anomalies.Monitoring;

namespace DevOnBike.Overfit.Tests.Anomalies
{
    /// <summary>
    /// What a dismissal covers, and what it must not.
    ///
    /// <para><b>Written because the mechanism had an argument and no number, and this module's README
    /// promises numbers.</b> The measurement previously cited for it was withdrawn: it came from a replay
    /// that matched incidents on the subject without checking the signal, so an unrelated finding on the
    /// same pod counted as a detection. With the check added, a week of dismissals cost no detection at
    /// all and the ceiling did nothing observable — in <i>that</i> scenario.</para>
    ///
    /// <para><b>That was the wrong scenario, which is why it measured nothing.</b> A replay over a healthy
    /// shadow week and then a fault only exercises the ceiling if a dismissal happens to land on the same
    /// pod and signal as the fault, inside the mute window. The claim being made is not statistical — it
    /// is behavioural, and it is exactly this: <i>a mute must not hide a fault materially larger than the
    /// one somebody dismissed</i>. So the scenario is built directly rather than waited for.</para>
    /// </summary>
    public sealed class SuppressionCeilingTests
    {
        private static readonly DateTimeOffset T0 = new(2026, 8, 3, 9, 0, 0, TimeSpan.Zero);

        private static SignalSuppression Mute(double magnitude)
        {
            return new SignalSuppression(
                Pod: "svc-a",
                Workload: "svc",
                Signal: "MemoryWorkingSetBytes",
                Until: T0.AddDays(7),
                IncidentId: 1,
                Reason: "known noisy",
                Magnitude: magnitude);
        }

        private static IncidentSubject Subject()
        {
            return new IncidentSubject("lab", "svc", string.Empty, "svc-a", string.Empty);
        }

        /// <summary>The same event again stays muted — otherwise the acknowledgement buys nothing.</summary>
        [Fact]
        public void AMuteCoversTheEventThatWasDismissed()
        {
            var mute = Mute(4_000_000);

            Assert.True(mute.Covers(Subject(), "MemoryWorkingSetBytes", 4_000_000));
            Assert.True(mute.Covers(Subject(), "MemoryWorkingSetBytes", 4_400_000));
        }

        /// <summary>
        /// The number the mechanism exists for. A ten-times-larger fault on the muted signal is a different
        /// event, and a mute scoped to pod and signal alone would hide it — which is literally what the
        /// operator asked for and nothing anyone means by it.
        /// </summary>
        [Fact]
        public void AMuteDoesNotCoverAMateriallyLargerEvent()
        {
            var mute = Mute(4_000_000);

            Assert.False(mute.Covers(Subject(), "MemoryWorkingSetBytes", 40_000_000));
        }

        /// <summary>
        /// The margin is 1.25, the same headroom the floor proposals use, and the boundary is where a
        /// behaviour claim is worth pinning: 25% more is still the same event, 26% more is not.
        /// </summary>
        [Fact]
        public void TheCeilingSitsAtTwentyFivePercentAbove()
        {
            var mute = Mute(1_000_000);

            Assert.True(mute.Covers(Subject(), "MemoryWorkingSetBytes", 1_250_000));
            Assert.False(mute.Covers(Subject(), "MemoryWorkingSetBytes", 1_260_000));
        }

        /// <summary>
        /// A dismissed finding with no measurable size mutes without a ceiling. Refusing to mute would
        /// make the acknowledgement do nothing, which is worse than muting too much: an operator who
        /// presses the button and sees no effect stops using the feature.
        /// </summary>
        [Fact]
        public void AnUnmeasurableDismissalMutesWithoutACeiling()
        {
            var mute = Mute(double.NaN);

            Assert.Equal(double.PositiveInfinity, mute.Ceiling);
            Assert.True(mute.Covers(Subject(), "MemoryWorkingSetBytes", 40_000_000));
        }

        /// <summary>
        /// A finding whose own size is unknown is covered. The alternative — reporting everything
        /// unmeasurable through an active mute — routes exactly the findings that carry no evidence to a
        /// human who has already said they do not want them.
        /// </summary>
        [Fact]
        public void AnUnmeasurableFindingIsCovered()
        {
            var mute = Mute(4_000_000);

            Assert.True(mute.Covers(Subject(), "MemoryWorkingSetBytes", double.NaN));
        }

        /// <summary>Another signal on the same pod is not covered, ceiling or no ceiling.</summary>
        [Fact]
        public void AMuteIsScopedToItsSignal()
        {
            var mute = Mute(4_000_000);

            Assert.False(mute.Covers(Subject(), "CpuUsageRatio", 1_000));
        }

        /// <summary>
        /// The guard honours the ceiling, not just the record. A mechanism correct in a value type and
        /// unwired in the pipeline is the shape of two other defects found this week.
        /// </summary>
        [Fact]
        public void TheGuardStopsMutingOnceTheEventGrowsPastTheCeiling()
        {
            var store = new SuppressionStore();

            store.Add(Mute(4_000_000), T0);

            var subject = Subject();

            Assert.True(
                store.IsSuppressed(subject, "MemoryWorkingSetBytes", T0.AddHours(1), 4_000_000),
                "the dismissed event must stay muted through the store, not only through the record");

            Assert.False(
                store.IsSuppressed(subject, "MemoryWorkingSetBytes", T0.AddHours(1), 40_000_000),
                "a ten-times-larger event reaches the operator, which is the whole point of the ceiling");
        }
    }
}
