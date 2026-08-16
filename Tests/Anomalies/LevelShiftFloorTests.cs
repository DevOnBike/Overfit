// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Anomalies.Contracts;
using DevOnBike.Overfit.Anomalies.Monitoring;
using DevOnBike.Overfit.Statistics;
using DevOnBike.Overfit.Tests.TestSupport;

namespace DevOnBike.Overfit.Tests.Anomalies
{
    /// <summary>
    /// The step gate's floor must be calibrated on steps.
    ///
    /// <para><b>This family has now been given the wrong floor twice.</b> First the peer-gap floor, which is
    /// about six times off on the lab; then the trend-change floor, which is fitted to a single pod's slope
    /// while the gate judges a median across every replica. Both times the units matched and the
    /// distributions did not, and both times nothing failed — a floor that is too high looks exactly like a
    /// healthy cluster, which is the one symptom this whole subsystem exists to make impossible.</para>
    ///
    /// <para>So the defect is pinned as a <i>ratio between two measured quantities</i> rather than as a
    /// threshold. A future change that re-borrows a floor from another family fails here regardless of what
    /// the numbers happen to be that day.</para>
    /// </summary>
    public sealed class LevelShiftFloorTests
    {
        private static readonly DateTimeOffset T0 = new(2026, 8, 3, 0, 0, 0, TimeSpan.Zero);

        /// <summary>
        /// <see cref="LevelShiftDetector.StepSize"/> is what the calibrator measures and
        /// <c>MinAbsoluteChange</c> is what the gate compares. They are computed by two pieces of code, so
        /// they are pinned to each other here rather than assumed to agree.
        /// </summary>
        [Fact]
        public void StepSizeMatchesWhatTheGateCompares()
        {
            var detector = new LevelShiftDetector();

            // A flat series, a stepped one, and one with gaps — the third because the compaction is where
            // the two implementations could most easily disagree.
            double[][] cases =
            [
                [1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 5, 5, 5, 5],
                [2, double.NaN, 2, 2, 2, 9, 9, double.NaN, 9, 9],
            ];

            for (var i = 0; i < cases.Length; i++)
            {
                var result = detector.Detect(
                    cases[i], LevelShiftOptions.Balanced with
                    {
                        MinimumSamples = 4
                    });

                Assert.Equal(
                    Math.Abs(result.AbsoluteChange),
                    LevelShiftDetector.StepSize(cases[i]),
                    10);
            }
        }

        /// <summary>
        /// The defect itself, as a ratio. On a healthy population the trend floor sits far above the step
        /// floor for the same signal, because one is fitted per pod and the other is measured on the median
        /// across twelve of them.
        ///
        /// <para>The bar is deliberately loose — a factor of two, against a measured factor near four on
        /// CPU. What is being pinned is that the two are <b>different distributions</b>, not any particular
        /// separation, so the test does not become a tuning parameter for the generator.</para>
        /// </summary>
        [Fact]
        public void TheStepFloorIsFarBelowTheTrendFloorOnAHealthyPopulation()
        {
            var proposal = Calibrate()[(int)MetricIndex.CpuUsageRatio];

            Assert.True(proposal.IsUsable, "not enough windows observed for a proposal");

            Assert.True(
                proposal.ProposedMinAbsoluteLevelShift > 0.0,
                "a zero step floor means the accumulator is not being fed, which reads as 'no defect' and "
                + "is the opposite");

            Assert.True(
                proposal.ProposedMinAbsoluteLevelShift * 2.0 < proposal.ProposedMinAbsoluteTrendChange,
                $"step floor {proposal.ProposedMinAbsoluteLevelShift:G4} against trend floor "
                + $"{proposal.ProposedMinAbsoluteTrendChange:G4}: if these are close, the step gate has "
                + "been handed the trend distribution again");
        }

        /// <summary>
        /// The floor a healthy cluster produces must sit below a fault worth catching. Stated against the
        /// fault that exposed the defect: a 2.5× CPU rise on every replica moves the common level by about
        /// 0.4 of a core, and the borrowed floor stood at 0.81.
        /// </summary>
        [Fact]
        public void TheStepFloorLeavesRoomForTheFaultItGates()
        {
            var proposal = Calibrate()[(int)MetricIndex.CpuUsageRatio];

            Assert.True(
                proposal.ProposedMinAbsoluteLevelShift < 0.39,
                $"step floor {proposal.ProposedMinAbsoluteLevelShift:G4} is above the 0.39-core step a 2.5x "
                + "cluster-wide CPU rise produces, so the only fault this family can see stays invisible");
        }

        /// <summary>
        /// A state file written before the step accumulator existed still loads. Refusing it would drop a
        /// week of peer and trend calibration on upgrade, and a guard with no floors was measured at 209
        /// false incidents a day — so a strict reader turns an upgrade into that.
        /// </summary>
        [Fact]
        public void ReadsAStateFileWrittenBeforeTheStepAccumulator()
        {
            var current = new FloorCalibrator();

            foreach (var window in Windows(12))
            {
                current.Observe(window);
            }

            var written = current.Write();
            var lines = written.Split('\n', StringSplitOptions.RemoveEmptyEntries);
            var legacy = new System.Text.StringBuilder();

            for (var i = 0; i < lines.Length; i++)
            {
                var parts = lines[i].Split('\t');

                // The window count is a two-column record and a pre-change writer emitted none, so dropping
                // it is part of producing a genuinely legacy file rather than an accommodation. Keeping it
                // would make this test assert about a format that never existed — and it did, until the
                // count was added: the file still carried the marker and the "legacy" reader saw a window
                // count no old file could have had.
                if (parts.Length == 2)
                {
                    continue;
                }

                Assert.Equal(5, parts.Length);

                legacy.Append(string.Join('\t', parts[..4])).Append('\n');
            }

            var restored = FloorCalibrator.Read(legacy.ToString());
            var proposal = restored.Propose()[(int)MetricIndex.CpuUsageRatio];

            Assert.True(proposal.IsUsable, "the four-column form must still restore the other accumulators");
            Assert.Equal(0.0, proposal.ProposedMinAbsoluteLevelShift);
        }

        private static FloorProposal[] Calibrate()
        {
            var calibrator = new FloorCalibrator();

            foreach (var window in Windows(12))
            {
                calibrator.Observe(window);
            }

            return calibrator.Propose();
        }

        /// <summary>
        /// Successive windows over one healthy synthetic cluster, at the cadence the guard runs — the same
        /// shape the calibrator sees in production, because a floor learned from one window is a different
        /// quantity from a floor learned from a hundred.
        /// </summary>
        private static IEnumerable<MetricWindow> Windows(int pods)
        {
            const double scrape = 15.0;
            var cluster = new SyntheticCluster(pods, 6, scrape, 20260801, restartsPerPodPerDay: 0.0);
            var samples = (int)(20 * 60 / scrape);
            var step = (int)(5 * 60 / scrape);
            var names = new List<string>(pods);

            for (var p = 0; p < pods; p++)
            {
                names.Add(SyntheticCluster.PodName(p));
            }

            for (var start = 0; start + samples <= cluster.Samples; start += step)
            {
                var window = new MetricWindow(
                    names, samples, T0.AddSeconds(start * scrape), TimeSpan.FromSeconds(scrape));

                for (var p = 0; p < pods; p++)
                {
                    for (var m = 0; m < (int)MetricIndex.Count; m++)
                    {
                        cluster.Window(p, (MetricIndex)m, start, samples)
                            .CopyTo(window.Series(p, (MetricIndex)m));
                    }
                }

                yield return window;
            }
        }
    }
}
