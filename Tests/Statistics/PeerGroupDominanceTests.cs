// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Statistics;

namespace DevOnBike.Overfit.Tests.Statistics
{
    /// <summary>
    /// When members pull in both directions, whether that means "no norm" or "an outlier plus some scatter".
    ///
    /// <para><b>The behaviour these pin was a real detection hole, found by the detection matrix.</b> A replica
    /// running at 2.5× the CPU of its eleven peers — Cliff's delta 1.00, a 190% gap — was never named in any
    /// cycle, because two ordinary replicas happened to sit about 10% <i>under</i> the group and the
    /// bidirectional gate read "somebody above and somebody below" as an ambiguous group. A 190% deviation was
    /// vetoed by two 10% ones.</para>
    ///
    /// <para>Both directions of the fix have to hold, and the second is the one that keeps it honest: a group
    /// genuinely split at comparable size must still come back <see cref="DetectionStatus.Inconclusive"/>,
    /// because naming a culprit there is a confident answer to a question the data cannot settle.</para>
    /// </summary>
    public sealed class PeerGroupDominanceTests
    {
        private static readonly PeerGroupOutlierDetector Detector = new();

        [Fact]
        public void AnOutlierIsNotVetoedByMildScatterTheOtherWay()
        {
            var peers = new List<PeerSeries>();

            for (var p = 0; p < 9; p++)
            {
                peers.Add(new PeerSeries($"pod-{p}", Series(seed: p, level: 100.0)));
            }

            // Two replicas a tenth under the group: real, consistent, and nobody's problem.
            peers.Add(new PeerSeries("pod-cool-a", Series(seed: 51, level: 90.0)));
            peers.Add(new PeerSeries("pod-cool-b", Series(seed: 52, level: 90.0)));

            // One replica at three times the group.
            peers.Add(new PeerSeries("pod-hot", Series(seed: 99, level: 300.0)));

            var findings = new PeerOutlierFinding[peers.Count];
            var result = Detector.Detect(peers, PeerSignalKind.LoadIndependent, PeerOutlierOptions.Balanced, findings);

            Assert.Equal(DetectionStatus.Anomalous, result.Status);

            var hot = Find(findings, "pod-hot");

            Assert.Equal(PeerDeviation.High, hot.Deviation);

            // The mild ones are demoted to noise rather than reported as findings of their own...
            Assert.Equal(PeerDeviation.None, Find(findings, "pod-cool-a").Deviation);
            Assert.Equal(PeerDeviation.None, Find(findings, "pod-cool-b").Deviation);
            Assert.Equal(0, result.LowCount);
            Assert.Equal(1, result.HighCount);

            // ...and the verdict says so, because silently dropping a member that did deviate would be the
            // detector hiding its own reasoning.
            Assert.Contains("noise", result.Reason, StringComparison.OrdinalIgnoreCase);
        }

        /// <summary>
        /// The case that must NOT be resolved. Two members high and two low at the same magnitude is a group
        /// without a centre, and picking a side would be arbitrary.
        /// </summary>
        [Fact]
        public void AGenuinelySplitGroupStaysInconclusive()
        {
            var peers = new List<PeerSeries>();

            for (var p = 0; p < 8; p++)
            {
                peers.Add(new PeerSeries($"pod-{p}", Series(seed: p, level: 100.0)));
            }

            peers.Add(new PeerSeries("pod-high-a", Series(seed: 61, level: 140.0)));
            peers.Add(new PeerSeries("pod-high-b", Series(seed: 62, level: 140.0)));
            peers.Add(new PeerSeries("pod-low-a", Series(seed: 63, level: 60.0)));
            peers.Add(new PeerSeries("pod-low-b", Series(seed: 64, level: 60.0)));

            var findings = new PeerOutlierFinding[peers.Count];
            var result = Detector.Detect(peers, PeerSignalKind.LoadIndependent, PeerOutlierOptions.Balanced, findings);

            Assert.Equal(DetectionStatus.Inconclusive, result.Status);
        }

        /// <summary>
        /// The demotion is by <b>size</b>, not by how many members are on each side. Nine mild members must
        /// not out-vote one large one, because the gate is about magnitude and the count is what the majority
        /// gate ahead of it already handles.
        /// </summary>
        [Fact]
        public void ManyMildDeviationsDoNotOutvoteOneLargeOne()
        {
            var peers = new List<PeerSeries>();

            for (var p = 0; p < 4; p++)
            {
                peers.Add(new PeerSeries($"pod-{p}", Series(seed: p, level: 100.0)));
            }

            for (var p = 0; p < 3; p++)
            {
                peers.Add(new PeerSeries($"pod-cool-{p}", Series(seed: 40 + p, level: 88.0)));
            }

            peers.Add(new PeerSeries("pod-hot", Series(seed: 99, level: 400.0)));

            var findings = new PeerOutlierFinding[peers.Count];
            var result = Detector.Detect(peers, PeerSignalKind.LoadIndependent, PeerOutlierOptions.Balanced, findings);

            Assert.Equal(DetectionStatus.Anomalous, result.Status);
            Assert.Equal(PeerDeviation.High, Find(findings, "pod-hot").Deviation);
        }

        private static PeerOutlierFinding Find(PeerOutlierFinding[] findings, string pod)
        {
            for (var i = 0; i < findings.Length; i++)
            {
                if (string.Equals(findings[i].Name, pod, StringComparison.Ordinal))
                {
                    return findings[i];
                }
            }

            throw new InvalidOperationException($"no finding for {pod}");
        }

        /// <summary>Tight noise around a level, so the rank test is decisive and the gaps are the variable.</summary>
        private static double[] Series(int seed, double level, int samples = 60)
        {
            var rng = new Random(seed);
            var values = new double[samples];

            for (var i = 0; i < samples; i++)
            {
                values[i] = level * (0.98 + (0.04 * rng.NextDouble()));
            }

            return values;
        }
    }
}
