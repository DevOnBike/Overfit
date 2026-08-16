// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Statistics;

namespace DevOnBike.Overfit.Tests.Statistics
{
    /// <summary>
    /// <see cref="CrossPeerBaseline"/>, and the decomposition it enables.
    ///
    /// <para>The load-bearing pair is the last two tests: a deployment-wide drift must stop producing a
    /// finding on every pod, and a single pod drifting against a steady group must still produce one. Getting
    /// only the first is easy and useless — it is what plain suppression would do, and it would throw away
    /// the one thing the trend family can see that a peer comparison cannot.</para>
    /// </summary>
    public sealed class CrossPeerBaselineTests
    {
        private const int Samples = 120;
        private const double StepSeconds = 15.0;

        [Fact]
        public void TakesTheMedianAcrossPeersAtEachInstant()
        {
            var peers = Group(
                [1.0, 2.0, 3.0],
                [5.0, 6.0, 7.0],
                [9.0, 100.0, 11.0]);

            var expectation = new double[3];

            Assert.True(CrossPeerBaseline.TryBuild(peers, expectation, new double[peers.Count]));

            Assert.Equal(5.0, expectation[0]);
            Assert.Equal(6.0, expectation[1]);
            Assert.Equal(7.0, expectation[2]);
        }

        /// <summary>Three members is the smallest group with a majority to take a median of.</summary>
        [Fact]
        public void RefusesAGroupTooSmallToHaveACommonComponent()
        {
            Assert.False(CrossPeerBaseline.TryBuild(
                Group([1.0], [2.0]), new double[1], new double[2]));

            Assert.True(CrossPeerBaseline.TryBuild(
                Group([1.0], [2.0], [3.0]), new double[1], new double[3]));
        }

        /// <summary>
        /// Too few reporters is silence, not a baseline. Subtracting a "median" computed from one pod would
        /// zero that pod's own residual and silence precisely the member whose data survived.
        /// </summary>
        [Fact]
        public void AnInstantWithTooFewReporters_HasNoExpectation()
        {
            var peers = Group(
                [1.0, double.NaN, 3.0],
                [2.0, double.NaN, 4.0],
                [3.0, 7.0, double.NaN]);

            var expectation = new double[3];
            CrossPeerBaseline.TryBuild(peers, expectation, new double[peers.Count]);

            Assert.Equal(2.0, expectation[0]);
            Assert.True(double.IsNaN(expectation[1]), "one reporter is not a group");
            Assert.True(double.IsNaN(expectation[2]), "two reporters are not a group");
        }

        [Fact]
        public void AMemberShorterThanTheWindow_ContributesWhatItHas()
        {
            var peers = Group(
                [1.0, 2.0, 3.0],
                [2.0, 3.0, 4.0],
                [3.0, 4.0, 5.0],
                [4.0]);

            var expectation = new double[3];

            Assert.True(CrossPeerBaseline.TryBuild(peers, expectation, new double[peers.Count]));
            Assert.All(expectation, v => Assert.True(double.IsFinite(v)));
        }

        /// <summary>
        /// The reported case: the whole deployment warming up. Every pod's latency falls together, and after
        /// decomposition no pod is drifting <i>differently</i>.
        /// </summary>
        [Fact]
        public void ADeploymentWideDrift_LeavesNoPerPodTrend()
        {
            var peers = Group(
                Drift(1000.0, -4.0, 1.0, seed: 1),
                Drift(1030.0, -4.0, 1.0, seed: 2),
                Drift(970.0, -4.0, 1.0, seed: 3));

            var expectation = new double[Samples];
            Assert.True(CrossPeerBaseline.TryBuild(peers, expectation, new double[peers.Count]));

            // The common component itself carries the drift — the deployment-level question still has its
            // answer, reported once instead of once per pod.
            Assert.Equal(DetectionStatus.Anomalous, DetectOn(expectation).Status);

            for (var p = 0; p < peers.Count; p++)
            {
                var result = DetectOn(peers[p].Values.Span, expectation);

                Assert.Equal(DetectionStatus.Healthy, result.Status);
            }
        }

        /// <summary>
        /// The half that suppression would break: one pod drifting while the group holds steady is exactly
        /// what this is supposed to surface, and it must survive the decomposition.
        /// </summary>
        [Fact]
        public void OnePodDriftingAgainstASteadyGroup_IsStillCaught()
        {
            var peers = Group(
                Drift(1000.0, 0.0, 1.0, seed: 4),
                Drift(1010.0, 0.0, 1.0, seed: 5),
                Drift(990.0, 0.0, 1.0, seed: 6),
                Drift(1000.0, 6.0, 1.0, seed: 7));

            var expectation = new double[Samples];
            Assert.True(CrossPeerBaseline.TryBuild(peers, expectation, new double[peers.Count]));

            // Nothing common to report.
            Assert.Equal(DetectionStatus.Healthy, DetectOn(expectation).Status);

            for (var p = 0; p < 3; p++)
            {
                Assert.Equal(DetectionStatus.Healthy, DetectOn(peers[p].Values.Span, expectation).Status);
            }

            var sick = DetectOn(peers[3].Values.Span, expectation);

            Assert.Equal(DetectionStatus.Anomalous, sick.Status);
            Assert.Equal(TrendDirection.Rising, sick.Direction);
        }

        private static TrendResult DetectOn(
            ReadOnlySpan<double> values,
            ReadOnlySpan<double> expectation = default)
        {
            var times = new double[values.Length];

            for (var i = 0; i < values.Length; i++)
            {
                times[i] = i * StepSeconds;
            }

            return new TrendDetector().Detect(values, times, TrendOptions.Balanced, double.NaN, expectation);
        }

        /// <summary>A level plus a per-sample drift plus noise — the shape a warming or leaking pod has.</summary>
        private static double[] Drift(double start, double perSample, double noise, int seed)
        {
            var rng = new Random(seed);
            var values = new double[Samples];

            for (var i = 0; i < Samples; i++)
            {
                values[i] = start + (i * perSample) + ((rng.NextDouble() - 0.5) * noise);
            }

            return values;
        }

        private static List<PeerSeries> Group(params double[][] series)
        {
            var peers = new List<PeerSeries>(series.Length);

            for (var i = 0; i < series.Length; i++)
            {
                peers.Add(new PeerSeries($"pod-{i}", series[i]));
            }

            return peers;
        }
    }
}
