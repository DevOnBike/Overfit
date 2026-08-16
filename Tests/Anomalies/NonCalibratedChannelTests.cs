// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Anomalies.Contracts;
using DevOnBike.Overfit.Anomalies.Monitoring;

namespace DevOnBike.Overfit.Tests.Anomalies
{
    /// <summary>
    /// A custom channel whose healthy state <b>is</b> a constant, and the exemption that stops the calibrator
    /// treating that as evidence of a dead binding or as a floor to fit.
    ///
    /// <para><b>The built-in enum has had this since <c>PeerSignalCatalog.IsCountedEvent</c>; a channel the
    /// customer named had no way to say the same thing.</b> The rule it encodes is about units, not about
    /// which detector reads it: a quantity whose SCALE is arbitrary can be calibrated from data, and a
    /// quantity whose UNIT is already the thing you care about cannot. A restart counter is the first case
    /// the project met; scrape coverage, pinned at 1.0 while healthy, is the second.</para>
    ///
    /// <para>The first test here is the defect rather than the fix, and it is deliberately kept: an exemption
    /// whose absence changes nothing is an exemption nobody can justify keeping.</para>
    /// </summary>
    public sealed class NonCalibratedChannelTests
    {
        private const string Coverage = "ScrapeCoverage";
        private const string Heap = "GcCommittedBytes";
        private const int Pods = 4;
        private const int Samples = 40;
        private const int Cycles = 30;

        /// <summary>Observations behind the fixture — one per pod per window, which is the unit that matters.</summary>
        private const int Observations = Pods * Cycles;

        /// <summary>
        /// <b>Without the exemption, a correctly-working coverage channel is accused of being dead.</b> Every
        /// step is satisfied by health: the folded range has <c>Min == Max == 1.0</c>, so
        /// <c>ObservedRange.IsConstant</c> holds, and <c>InertChannel.IsConclusive</c> tests
        /// <c>Value != 0.0</c> unconditionally — so the verdict is not "ambiguous, ask an operator" but
        /// "confirmed defect".
        /// </summary>
        [Fact]
        public void WithoutTheExemptionAConstantHealthyChannelIsCalledAConfirmedDefect()
        {
            var calibrator = Observe(constant: true);

            var found = calibrator.InertChannels(minimumObservations: Observations);
            var coverage = Assert.Single(found, c => string.Equals(c.Name, Coverage, StringComparison.Ordinal));

            Assert.Equal(1.0, coverage.Value);
            Assert.True(coverage.IsConclusive, "constant non-zero is what that type calls conclusive");
        }

        /// <summary>The exemption, doing the one thing it exists for.</summary>
        [Fact]
        public void AnExemptChannelIsNotJudgedInert()
        {
            var calibrator = Observe(constant: true);

            calibrator.ExemptFromCalibration([Coverage]);

            Assert.DoesNotContain(
                calibrator.InertChannels(minimumObservations: Observations),
                c => string.Equals(c.Name, Coverage, StringComparison.Ordinal));
        }

        /// <summary>
        /// It is a list of names, not a switch. Exempting one channel must not quietly stop the check working
        /// on every other one — that would trade a false accusation for total silence, which is worse.
        /// </summary>
        [Fact]
        public void ExemptingOneChannelLeavesTheOthersJudged()
        {
            var calibrator = Observe(constant: true, secondChannelConstant: true);

            calibrator.ExemptFromCalibration([Coverage]);

            var found = calibrator.InertChannels(minimumObservations: Observations);

            Assert.DoesNotContain(found, c => string.Equals(c.Name, Coverage, StringComparison.Ordinal));
            Assert.Contains(found, c => string.Equals(c.Name, Heap, StringComparison.Ordinal));
        }

        /// <summary>
        /// <b>Why the exemption is binary rather than "fit the floor but suppress the report".</b> A fitted
        /// floor over a constant channel is not merely low-value, it is exactly <c>0.0</c> — the value
        /// <c>MinAbsoluteGap</c> documents as "gate off" — because every folded peer gap is
        /// <c>|1.0 − 1.0|</c>. There is nothing on the fitting side to preserve.
        /// </summary>
        [Fact]
        public void FittingAFloorOverAConstantChannelProposesZeroAnyway()
        {
            var proposal = Observe(constant: true).Propose(Coverage);

            Assert.True(proposal.IsUsable, "the fixture must be past the sample and window bars to mean anything");
            Assert.Equal(0.0, proposal.ProposedMinAbsoluteGap);
        }

        /// <summary>
        /// The control that makes the test above mean something: on a channel that <i>does</i> vary, the same
        /// calibrator proposes a non-zero floor. Without this, "proposes zero" would be consistent with a
        /// calibrator that proposes zero for everything.
        /// </summary>
        [Fact]
        public void FittingAFloorOverAChannelThatVariesProposesSomething()
        {
            var proposal = Observe(constant: false).Propose(Coverage);

            Assert.True(proposal.IsUsable);
            Assert.True(
                proposal.ProposedMinAbsoluteGap > 0.0,
                $"a varying channel must produce a floor; got {proposal.ProposedMinAbsoluteGap}");
        }

        /// <summary>
        /// An exempt channel is still observed, and the observations are still readable. The built-in path
        /// makes the same split for a counted event: how far peers differ on a channel is a real fact about
        /// the cluster even when it must not become a threshold.
        /// </summary>
        [Fact]
        public void AnExemptChannelIsStillObservedAndItsFloorIsStillZero()
        {
            var calibrator = Observe(constant: false);

            calibrator.ExemptFromCalibration([Coverage]);

            var proposal = calibrator.Propose(Coverage);

            Assert.Equal(Observations, proposal.Samples);
            Assert.Equal(0.0, proposal.ProposedMinAbsoluteGap);
            Assert.Equal(0.0, proposal.ProposedMinAbsoluteTrendChange);
            Assert.Equal(0.0, proposal.ProposedMinAbsoluteLevelShift);
        }

        /// <summary>Passing nothing restores the default, so the exemption cannot be a one-way door.</summary>
        [Fact]
        public void ClearingTheExemptionRestoresFitting()
        {
            var calibrator = Observe(constant: false);

            calibrator.ExemptFromCalibration([Coverage]);
            calibrator.ExemptFromCalibration(null);

            Assert.True(calibrator.Propose(Coverage).ProposedMinAbsoluteGap > 0.0);
        }

        /// <summary>
        /// Windows carrying one or two custom channels, over enough cycles to clear both bars a proposal has
        /// to pass — <c>Samples &gt;= 30</c> and <c>Windows &gt;= 24</c>, which are different questions.
        /// </summary>
        private static FloorCalibrator Observe(bool constant, bool secondChannelConstant = false)
        {
            var calibrator = new FloorCalibrator();
            var pods = new List<string>(Pods);

            for (var p = 0; p < Pods; p++)
            {
                pods.Add($"pod-{p}");
            }

            string[] channels = secondChannelConstant ? [Coverage, Heap] : [Coverage];

            for (var cycle = 0; cycle < Cycles; cycle++)
            {
                var window = new MetricWindow(
                    pods, Samples, DateTime.UnixEpoch.AddMinutes(cycle * 5), TimeSpan.FromSeconds(30),
                    channels);

                for (var p = 0; p < Pods; p++)
                {
                    var coverage = window.Series(p, Coverage);

                    for (var i = 0; i < Samples; i++)
                    {
                        // Varying case: each pod sits at a different level, so the peer gap the calibrator
                        // folds is genuinely non-zero rather than merely noisy.
                        coverage[i] = constant ? 1.0 : 1.0 + (0.05 * p) + (0.001 * i);
                    }

                    if (!secondChannelConstant)
                    {
                        continue;
                    }

                    var heap = window.Series(p, Heap);

                    for (var i = 0; i < Samples; i++)
                    {
                        heap[i] = 1.0;
                    }
                }

                calibrator.Observe(window);
            }

            return calibrator;
        }
    }
}
