// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Statistics;

namespace DevOnBike.Overfit.Tests.Statistics
{
    public sealed class PeerGroupOutlierDetectorTests
    {
        private static readonly PeerGroupOutlierDetector Detector = new();

        [Fact]
        public void OneLeakingReplicaAmongEleven_IsNamed()
        {
            // The canonical case: eleven pods around 400 MB, one at 1.7 GB.
            var peers = new List<PeerSeries>();
            for (var p = 0; p < 11; p++)
            {
                peers.Add(new PeerSeries($"pod-{p}", Working(seed: p, megabytes: 400)));
            }

            peers.Add(new PeerSeries("pod-leaky", Working(seed: 99, megabytes: 1700)));

            var findings = new PeerOutlierFinding[peers.Count];
            var result = Detector.Detect(peers, PeerSignalKind.LoadIndependent, PeerOutlierOptions.Balanced, findings);

            Assert.Equal(DetectionStatus.Anomalous, result.Status);
            Assert.Equal(1, result.OutlierCount);
            Assert.True(result.IsDecided);
            Assert.False(result.IsHealthy);

            var flagged = findings.Single(f => f.IsOutlier);
            Assert.Equal("pod-leaky", flagged.Name);
            Assert.Equal(PeerDeviation.High, flagged.Deviation);
            Assert.True(flagged.Comparison.EffectSize > 0.9, $"effect = {flagged.Comparison.EffectSize}");

            // Bonferroni over two one-sided tests for each of twelve members.
            Assert.Equal(0.05 / 24.0, result.CorrectedAlpha, 12);
        }

        [Fact]
        public void AUniformGroup_IsHealthy()
        {
            var peers = new List<PeerSeries>();
            for (var p = 0; p < 8; p++)
            {
                peers.Add(new PeerSeries($"pod-{p}", Working(seed: p, megabytes: 400)));
            }

            var findings = new PeerOutlierFinding[peers.Count];
            var result = Detector.Detect(peers, PeerSignalKind.LoadIndependent, PeerOutlierOptions.Balanced, findings);

            Assert.Equal(DetectionStatus.Healthy, result.Status);
            Assert.Equal(0, result.OutlierCount);
            Assert.All(findings, f => Assert.Equal(PeerDeviation.None, f.Deviation));

            // Findings are produced for every member, not only the flagged ones — the explanation has to be
            // checkable against the whole group.
            Assert.All(findings, f => Assert.Equal(60, f.UsableSamples));
        }

        [Fact]
        public void ASplitGroup_IsInconclusive_NotAListOfOutliers()
        {
            // A rollout moved six of ten pods. Members now deviate in both directions at once, so the group has
            // no single norm — and saying so is the correct answer. Naming a list here would be a confident
            // reply to a question the data cannot settle.
            var peers = new List<PeerSeries>();
            for (var p = 0; p < 4; p++)
            {
                peers.Add(new PeerSeries($"pod-old-{p}", Working(seed: p, megabytes: 400)));
            }

            for (var p = 0; p < 6; p++)
            {
                peers.Add(new PeerSeries($"pod-new-{p}", Working(seed: 50 + p, megabytes: 1500)));
            }

            var findings = new PeerOutlierFinding[peers.Count];
            var result = Detector.Detect(peers, PeerSignalKind.LoadIndependent, PeerOutlierOptions.Balanced, findings);

            Assert.Equal(DetectionStatus.Inconclusive, result.Status);
            Assert.False(result.IsDecided);
            Assert.False(result.IsHealthy);
            Assert.True(result.HighCount > 0 && result.LowCount > 0,
                $"expected deviation on both sides, got {result.HighCount} high / {result.LowCount} low");
            Assert.Contains("no coherent norm", result.Reason);
        }

        [Fact]
        public void AMajorityRegression_IsNotReportedAsHealthy()
        {
            // The failure a one-sided detector hides. Eight of ten pods regressed: leave-one-out caps each
            // regressed pod's effect at roughly (clean peers)/(n-1) = 1/9, far under any usable threshold, so
            // testing only the high side would return Healthy on a workload that is mostly broken. The two
            // untouched pods stand out below their peers instead, which keeps the group visible.
            var peers = new List<PeerSeries>();
            for (var p = 0; p < 2; p++)
            {
                peers.Add(new PeerSeries($"pod-old-{p}", Working(seed: p, megabytes: 400)));
            }

            for (var p = 0; p < 8; p++)
            {
                peers.Add(new PeerSeries($"pod-new-{p}", Working(seed: 50 + p, megabytes: 1500)));
            }

            var findings = new PeerOutlierFinding[peers.Count];
            var result = Detector.Detect(peers, PeerSignalKind.LoadIndependent, PeerOutlierOptions.Balanced, findings);

            Assert.NotEqual(DetectionStatus.Healthy, result.Status);
            Assert.Equal(2, result.LowCount);
            Assert.Equal(0, result.HighCount);
            Assert.All(
                findings.Where(f => f.IsOutlier),
                f => Assert.Equal(PeerDeviation.Low, f.Deviation));

            // Attribution is genuinely undecidable from peer data alone; the detector must not pretend
            // otherwise, and the low-side effect must be reported as a positive magnitude.
            Assert.True(findings.First(f => f.IsOutlier).Comparison.EffectSize > 0.5);
        }

        [Fact]
        public void UnevenLoad_OnARawSignal_ProducesAFalsePositive_WhichNormalisationRemoves()
        {
            // This is the whole reason PeerSignalKind exists. One pod handles three times the traffic and so
            // uses three times the memory: correct behaviour that a raw comparison reports as a fault.
            var peers = new List<PeerSeries>();
            for (var p = 0; p < 6; p++)
            {
                peers.Add(new PeerSeries(
                    $"pod-{p}",
                    Working(seed: p, megabytes: 400),
                    Constant(100.0, 60)));
            }

            peers.Add(new PeerSeries("pod-hot", Working(seed: 77, megabytes: 1200), Constant(300.0, 60)));

            var findings = new PeerOutlierFinding[peers.Count];

            var raw = Detector.Detect(peers, PeerSignalKind.LoadIndependent, PeerOutlierOptions.Balanced, findings);
            Assert.Equal(DetectionStatus.Anomalous, raw.Status);
            Assert.Equal("pod-hot", findings.Single(f => f.IsOutlier).Name);

            // Same data, same thresholds — but compared as memory per request, the busy pod is ordinary.
            var normalised = Detector.Detect(peers, PeerSignalKind.LoadSensitive, PeerOutlierOptions.Balanced, findings);
            Assert.Equal(DetectionStatus.Healthy, normalised.Status);
        }

        [Fact]
        public void NormalisedComparison_StillCatchesARealPerRequestRegression()
        {
            // Guard against the previous test's fix over-correcting: a pod that costs more *per request* must
            // still be caught, even when its absolute usage looks like everyone else's.
            var peers = new List<PeerSeries>();
            for (var p = 0; p < 6; p++)
            {
                peers.Add(new PeerSeries($"pod-{p}", Working(seed: p, megabytes: 400), Constant(100.0, 60)));
            }

            peers.Add(new PeerSeries("pod-inefficient", Working(seed: 78, megabytes: 400), Constant(25.0, 60)));

            var findings = new PeerOutlierFinding[peers.Count];
            var result = Detector.Detect(peers, PeerSignalKind.LoadSensitive, PeerOutlierOptions.Balanced, findings);

            Assert.Equal(DetectionStatus.Anomalous, result.Status);
            Assert.Equal("pod-inefficient", findings.Single(f => f.IsOutlier).Name);
        }

        [Fact]
        public void LoadSensitiveSignalWithoutWork_IsInsufficientData_NotAGuess()
        {
            var peers = new List<PeerSeries>
            {
                new("pod-0", Working(seed: 0, megabytes: 400), Constant(100.0, 60)),
                new("pod-1", Working(seed: 1, megabytes: 400), Constant(100.0, 60)),
                new("pod-2", Working(seed: 2, megabytes: 1700)),
            };

            var findings = new PeerOutlierFinding[peers.Count];
            var result = Detector.Detect(peers, PeerSignalKind.LoadSensitive, PeerOutlierOptions.Balanced, findings);

            Assert.Equal(DetectionStatus.InsufficientData, result.Status);
            Assert.False(result.IsHealthy);
            Assert.Contains("pod-2", result.Reason);
            Assert.Contains("load-sensitive", result.Reason);
        }

        [Fact]
        public void TwoPeers_CannotYieldAnOutlier()
        {
            var peers = new List<PeerSeries>
            {
                new("pod-0", Working(seed: 0, megabytes: 400)),
                new("pod-1", Working(seed: 1, megabytes: 1700)),
            };

            var findings = new PeerOutlierFinding[peers.Count];
            var result = Detector.Detect(peers, PeerSignalKind.LoadIndependent, PeerOutlierOptions.Balanced, findings);

            Assert.Equal(DetectionStatus.InsufficientData, result.Status);
            Assert.Contains("at least 3", result.Reason);
        }

        [Fact]
        public void TooFewSamplesOnOnePeer_BlocksTheWholeGroup()
        {
            var peers = new List<PeerSeries>
            {
                new("pod-0", Working(seed: 0, megabytes: 400)),
                new("pod-1", Working(seed: 1, megabytes: 400)),
                new("pod-short", Working(seed: 2, megabytes: 1700, samples: 5)),
            };

            var findings = new PeerOutlierFinding[peers.Count];
            var result = Detector.Detect(peers, PeerSignalKind.LoadIndependent, PeerOutlierOptions.Balanced, findings);

            Assert.Equal(DetectionStatus.InsufficientData, result.Status);
            Assert.Contains("pod-short", result.Reason);
        }

        [Fact]
        public void NonFiniteSamplesAreDropped_NotPropagated()
        {
            var values = Working(seed: 3, megabytes: 400).ToArray();
            values[10] = double.NaN;
            values[11] = double.PositiveInfinity;

            var peers = new List<PeerSeries>
            {
                new("pod-0", Working(seed: 0, megabytes: 400)),
                new("pod-1", Working(seed: 1, megabytes: 400)),
                new("pod-gappy", values),
            };

            var findings = new PeerOutlierFinding[peers.Count];
            var result = Detector.Detect(peers, PeerSignalKind.LoadIndependent, PeerOutlierOptions.Balanced, findings);

            Assert.Equal(DetectionStatus.Healthy, result.Status);
            Assert.Equal(58, findings[2].UsableSamples);
        }

        [Fact]
        public void StricterProfile_DemandsALargerDeviation()
        {
            // Deterministic, overlapping distributions so the *effect size* is what separates the profiles —
            // not the sample floor. Peers ramp over [100, 200]; the odd one over [120, 220], which puts
            // Cliff's delta at about 0.36: past Balanced's 0.33, short of Strict's 0.474.
            var peers = new List<PeerSeries>();
            for (var p = 0; p < 6; p++)
            {
                peers.Add(new PeerSeries($"pod-{p}", Ramp(100.0, 200.0, 120)));
            }

            peers.Add(new PeerSeries("pod-warm", Ramp(120.0, 220.0, 120)));

            var findings = new PeerOutlierFinding[peers.Count];

            var balanced = Detector.Detect(peers, PeerSignalKind.LoadIndependent, PeerOutlierOptions.Balanced, findings);
            Assert.Equal(DetectionStatus.Anomalous, balanced.Status);
            Assert.Equal("pod-warm", findings.Single(f => f.IsOutlier).Name);
            Assert.InRange(findings.Single(f => f.IsOutlier).Comparison.EffectSize, 0.33, 0.474);

            var strict = Detector.Detect(peers, PeerSignalKind.LoadIndependent, PeerOutlierOptions.Strict, findings);
            Assert.Equal(DetectionStatus.Healthy, strict.Status);
        }

        [Fact]
        public void DefaultOptions_AreRejectedRatherThanSilentlyAcceptingAnything()
        {
            var peers = new List<PeerSeries>
            {
                new("pod-0", Working(seed: 0, megabytes: 400)),
                new("pod-1", Working(seed: 1, megabytes: 400)),
                new("pod-2", Working(seed: 2, megabytes: 400)),
            };

            Assert.False(default(PeerOutlierOptions).IsValid);
            Assert.Throws<ArgumentException>(() =>
                Detector.Detect(peers, PeerSignalKind.LoadIndependent, default, new PeerOutlierFinding[3]));
        }

        [Fact]
        public void FindingsBufferTooShort_Throws()
        {
            var peers = new List<PeerSeries>
            {
                new("pod-0", Working(seed: 0, megabytes: 400)),
                new("pod-1", Working(seed: 1, megabytes: 400)),
                new("pod-2", Working(seed: 2, megabytes: 400)),
            };

            Assert.Throws<ArgumentException>(() =>
                Detector.Detect(peers, PeerSignalKind.LoadIndependent, PeerOutlierOptions.Balanced,
                    new PeerOutlierFinding[2]));
        }

        [Fact]
        public void DetectorRunsOnTheInjectedComparer()
        {
            var peers = new List<PeerSeries>
            {
                new("pod-0", Working(seed: 0, megabytes: 400)),
                new("pod-1", Working(seed: 1, megabytes: 400)),
                new("pod-2", Working(seed: 2, megabytes: 1700)),
            };

            var stub = new CountingComparer();
            var detector = new PeerGroupOutlierDetector(stub);
            var result = detector.Detect(peers, PeerSignalKind.LoadIndependent, PeerOutlierOptions.Balanced,
                new PeerOutlierFinding[3]);

            // Two one-sided comparisons per peer, and the stub's verdict — not Mann-Whitney's — decided it,
            // even though pod-2 is four times the others.
            Assert.Equal(6, stub.Calls);
            Assert.Equal(DetectionStatus.Healthy, result.Status);
        }

        /// <summary>A stable, mildly noisy resource series around <paramref name="megabytes"/>.</summary>
        private static double[] Working(int seed, double megabytes, int samples = 60)
        {
            var rng = new Random(seed);
            var values = new double[samples];
            for (var i = 0; i < samples; i++)
            {
                values[i] = megabytes * (0.95 + (0.1 * rng.NextDouble()));
            }

            return values;
        }

        /// <summary>Evenly spaced values over [from, to] — deterministic, so effect sizes are exact.</summary>
        private static double[] Ramp(double from, double to, int samples)
        {
            var values = new double[samples];
            for (var i = 0; i < samples; i++)
            {
                values[i] = from + ((to - from) * i / (samples - 1));
            }

            return values;
        }

        private static double[] Constant(double value, int samples)
        {
            var values = new double[samples];
            Array.Fill(values, value);

            return values;
        }

        private sealed class CountingComparer : ITwoSampleComparer
        {
            public int Calls
            {
                get; private set;
            }

            public string Name => "stub";

            public TwoSampleComparison Compare(ReadOnlySpan<double> baseline, ReadOnlySpan<double> candidate)
            {
                Calls++;

                return new TwoSampleComparison(1.0, 0.0, 0.5, baseline.Length, candidate.Length);
            }

            public TwoSampleComparison CompareHistograms(
                ReadOnlySpan<long> baselineCounts,
                ReadOnlySpan<long> candidateCounts) => throw new NotSupportedException();
        }
    }
}
