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
        public void AnIdenticallyZeroSignal_IsHealthy_NotInconclusive()
        {
            // Most of the signals in a healthy cluster are counters sitting at zero: OOM kills, 5xx responses,
            // GC pause, thread-pool queue depth. A group on which nobody disagrees is Healthy.
            //
            // Regression for a defect the unit suite missed and the live lab caught. The size gate reports an
            // infinite relative gap when the centre has no usable scale, which is the right reading for
            // "should this finding be blocked" and the wrong one for "how many members depart" — every member
            // of an all-zero group counted as a departure, so four identically-zero metrics were reported as
            // fully split groups with no coherent norm.
            var peers = new List<PeerSeries>();
            for (var p = 0; p < 6; p++)
            {
                peers.Add(new PeerSeries($"pod-{p}", Constant(0.0, 60)));
            }

            var findings = new PeerOutlierFinding[peers.Count];
            var result = Detector.Detect(peers, PeerSignalKind.LoadIndependent, PeerOutlierOptions.Balanced, findings);

            Assert.Equal(DetectionStatus.Healthy, result.Status);
            Assert.Equal(0, result.OutlierCount);
            Assert.All(findings, f => Assert.Equal(PeerDeviation.None, f.Deviation));
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

            // pod-2 carries no work series, so none of its samples can be normalised and it is dropped. Two
            // comparable members are below the minimum, so the group still refuses — but for the honest
            // reason, and the verdict says which member went missing rather than pretending nothing did.
            Assert.Equal(DetectionStatus.InsufficientData, result.Status);
            Assert.False(result.IsHealthy);
            Assert.Equal(1, result.ExcludedCount);
            Assert.Equal(2, result.PeerCount);
            Assert.Contains("load-sensitive", result.Reason);
        }

        /// <summary>
        /// <b>The lab regression.</b> Scaling one deployment from four replicas to eight made nine of eleven
        /// metrics return <c>InsufficientData</c> and the degraded replica vanish — the same traffic over
        /// twice as many pods left some of them with sparse quantiles, and the first member under the sample
        /// floor ended the evaluation for everyone. One starving pod must cost its own coverage and nothing
        /// else.
        /// </summary>
        [Fact]
        public void AStarvedPeer_IsDropped_NotAllowedToBlindTheGroup()
        {
            var peers = new List<PeerSeries>();
            for (var p = 0; p < 6; p++)
            {
                peers.Add(new PeerSeries($"pod-{p}", Working(seed: p, megabytes: 400)));
            }

            peers.Add(new PeerSeries("pod-leaky", Working(seed: 99, megabytes: 1700)));

            // Freshly created: it is in the group, it reports, and it has almost no history in the window.
            peers.Add(new PeerSeries("pod-new", Working(seed: 7, megabytes: 400, samples: 4)));

            var findings = new PeerOutlierFinding[peers.Count];
            var result = Detector.Detect(peers, PeerSignalKind.LoadIndependent, PeerOutlierOptions.Balanced, findings);

            Assert.Equal(DetectionStatus.Anomalous, result.Status);
            Assert.Equal("pod-leaky", findings.Single(f => f.IsOutlier).Name);

            // The correction is over the members that were actually tested: counting the dropped one would
            // quietly raise the bar for everybody else.
            Assert.Equal(7, result.PeerCount);
            Assert.Equal(0.05 / 14.0, result.CorrectedAlpha, 12);
        }

        [Fact]
        public void ADroppedPeer_IsCounted_AndGetsAFindingThatExplainsIt()
        {
            var peers = new List<PeerSeries>();
            for (var p = 0; p < 5; p++)
            {
                peers.Add(new PeerSeries($"pod-{p}", Working(seed: p, megabytes: 400)));
            }

            peers.Add(new PeerSeries("pod-new", Working(seed: 7, megabytes: 400, samples: 4)));

            var findings = new PeerOutlierFinding[peers.Count];
            var result = Detector.Detect(peers, PeerSignalKind.LoadIndependent, PeerOutlierOptions.Balanced, findings);

            Assert.Equal(DetectionStatus.Healthy, result.Status);

            // Healthy over five of six is a weaker statement than healthy over six, and both the count and the
            // wording have to carry that — this is the shape that reads as health and is not.
            Assert.False(result.HasFullCoverage);
            Assert.Equal(1, result.ExcludedCount);
            Assert.Equal(5, result.PeerCount);
            Assert.Contains("dropped", result.Reason);

            var dropped = findings.Single(f => f.Name == "pod-new");
            Assert.Equal(4, dropped.UsableSamples);
            Assert.Equal(PeerDeviation.None, dropped.Deviation);

            // NaN, not zero: it was never measured against anything, and zero would read as "sits exactly on
            // its peers' median".
            Assert.True(double.IsNaN(dropped.RelativeGap));
            Assert.True(double.IsNaN(dropped.AbsoluteGap));
        }

        /// <summary>
        /// The dropped member must leave no trace in the pooled baseline either. If compaction were wrong the
        /// group would still be evaluated, quietly, against a baseline containing the excluded observations —
        /// a silent wrong answer, which is worse than the abort it replaced.
        /// </summary>
        [Fact]
        public void DroppingAPeer_LeavesTheSurvivorsComparedOnlyAgainstEachOther()
        {
            var peers = new List<PeerSeries>();
            for (var p = 0; p < 5; p++)
            {
                peers.Add(new PeerSeries($"pod-{p}", Working(seed: p, megabytes: 400)));
            }

            peers.Add(new PeerSeries("pod-leaky", Working(seed: 99, megabytes: 1700)));

            var findings = new PeerOutlierFinding[peers.Count];
            var reference = Detector.Detect(
                peers, PeerSignalKind.LoadIndependent, PeerOutlierOptions.Balanced, findings);

            // The same six, plus a starved member sitting at a wildly different level in the middle of the
            // list — so a compaction bug would both shift indices and poison the baseline.
            var withStarved = new List<PeerSeries>(peers);
            withStarved.Insert(3, new PeerSeries("pod-new", Constant(50_000.0, 4)));

            var afterFindings = new PeerOutlierFinding[withStarved.Count];
            var after = Detector.Detect(
                withStarved, PeerSignalKind.LoadIndependent, PeerOutlierOptions.Balanced, afterFindings);

            Assert.Equal(reference.Status, after.Status);
            Assert.Equal(reference.PeerCount, after.PeerCount);
            Assert.Equal("pod-leaky", afterFindings.Single(f => f.IsOutlier).Name);

            // Findings stay aligned with the caller's list, not with the compacted one.
            for (var i = 0; i < withStarved.Count; i++)
            {
                Assert.Equal(withStarved[i].Name, afterFindings[i].Name);
            }

            var flagged = findings.Single(f => f.IsOutlier);
            var flaggedAfter = afterFindings.Single(f => f.IsOutlier);
            Assert.Equal(flagged.RelativeGap, flaggedAfter.RelativeGap, 9);
            Assert.Equal(flagged.Comparison.EffectSize, flaggedAfter.Comparison.EffectSize, 9);
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

        /// <summary>
        /// Dropping starved members must not become a way of comparing two pods. Once the survivors fall
        /// under <see cref="PeerOutlierOptions.MinimumPeers"/> there is no "rest of the group" left and the
        /// answer is a refusal — the same refusal as before, reached for a reason that is now stated.
        /// </summary>
        [Fact]
        public void DroppingStarvedPeersBelowTheMinimum_StillRefuses()
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
            Assert.Equal(1, result.ExcludedCount);
            Assert.Equal(2, result.PeerCount);
            Assert.Contains("30", result.Reason);
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
