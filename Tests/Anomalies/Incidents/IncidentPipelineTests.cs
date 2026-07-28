// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System;
using System.Collections.Generic;
using DevOnBike.Overfit.Anomalies.Incidents;
using DevOnBike.Overfit.Anomalies.Incidents.Contracts;
using DevOnBike.Overfit.Statistics;

namespace DevOnBike.Overfit.Tests.Anomalies.Incidents
{
    /// <summary>
    /// End-to-end through the real detectors: synthetic metric series go into
    /// <see cref="TrendDetector"/> and <see cref="PeerGroupOutlierDetector"/>, their verdicts go into
    /// <see cref="IncidentPipeline"/>, and what comes out is checked. Nothing here hand-constructs a
    /// <see cref="SignalFinding"/> — that is what <c>IncidentGrouperTests</c> is for, and it would defeat the
    /// purpose of this suite.
    /// </summary>
    public sealed class IncidentPipelineTests
    {
        private const double ScrapeSeconds = 30.0;

        private static readonly DateTimeOffset WindowStart = new(2026, 7, 28, 12, 0, 0, TimeSpan.Zero);
        private static readonly DateTimeOffset WindowEnd = WindowStart.AddMinutes(30);

        private static readonly TrendDetector Trend = new();
        private static readonly PeerGroupOutlierDetector Peers = new();

        [Fact]
        public void AHealthyTrend_ProducesNothing()
        {
            var pipeline = new IncidentPipeline();
            var (values, times) = FlatSeries(120);

            var result = Trend.Detect(values, times, TrendOptions.Balanced);

            Assert.False(pipeline.Observe(Subject("pod-a"), "process_resident_memory_bytes", result, WindowStart, WindowEnd));
            Assert.Equal(0, pipeline.Count);
            Assert.Empty(pipeline.Group(IncidentGroupingOptions.Balanced));
        }

        [Fact]
        public void AnUndecidableVerdict_IsNotTreatedAsAFinding()
        {
            // WarmingUp is not Healthy and it is not an anomaly either. Reporting "we could not tell" as
            // "something is wrong" is the fastest way to make an alerting product untrustworthy.
            var pipeline = new IncidentPipeline();
            var (values, times) = LeakySeries(TrendOptions.Balanced.MinimumSamples - 5);

            var result = Trend.Detect(values, times, TrendOptions.Balanced);

            Assert.Equal(DetectionStatus.WarmingUp, result.Status);
            Assert.False(pipeline.Observe(Subject("pod-a"), "process_resident_memory_bytes", result, WindowStart, WindowEnd));
            Assert.Equal(0, pipeline.Count);
        }

        [Fact]
        public void ALeakingSeries_BecomesAnIncidentCarryingTheDetectorsOwnReason()
        {
            var pipeline = new IncidentPipeline();
            var (values, times) = LeakySeries(120);

            var result = Trend.Detect(values, times, TrendOptions.Balanced);

            Assert.Equal(DetectionStatus.Anomalous, result.Status);
            Assert.True(pipeline.Observe(Subject("pod-a"), "process_resident_memory_bytes", result, WindowStart, WindowEnd));

            var incident = Assert.Single(pipeline.Group(IncidentGroupingOptions.Balanced));

            Assert.Equal("process_resident_memory_bytes", incident.Primary.Signal);
            Assert.Equal(SignalClass.Resource, incident.Primary.Class);
            Assert.Equal(result.Reason, incident.Primary.Reason);
            Assert.Equal(Math.Abs(result.KendallTau), incident.Primary.Severity, 12);
        }

        [Fact]
        public void ADeviatingPeer_BecomesAFindingAgainstItsOwnSubject()
        {
            var pipeline = new IncidentPipeline();
            var (peers, subjects) = PeerGroup(deviatingIndex: 2, factor: 1.6);

            var findings = new PeerOutlierFinding[peers.Count];
            var result = Peers.Detect(peers, PeerSignalKind.LoadIndependent, PeerOutlierOptions.Balanced, findings);

            Assert.Equal(DetectionStatus.Anomalous, result.Status);

            var added = pipeline.ObservePeerGroup(
                "kube_pod_container_status_restarts_total",
                result,
                findings.AsSpan(0, peers.Count),
                subjects.AsSpan(0, peers.Count),
                WindowStart,
                WindowEnd);

            Assert.Equal(result.OutlierCount, added);

            var incident = Assert.Single(pipeline.Group(IncidentGroupingOptions.Balanced));

            Assert.Equal(SignalClass.Infrastructure, incident.Primary.Class);
            Assert.Equal("pod-2", incident.Primary.Subject.Pod);
            Assert.Contains("pod-2", incident.Primary.Reason, StringComparison.Ordinal);
            Assert.Contains("Cliff's delta", incident.Primary.Reason, StringComparison.Ordinal);
        }

        [Fact]
        public void TheLabShape_TwoDetectorsOnOnePod_CollapseIntoOneIncidentLedByTheInfrastructureSignal()
        {
            // The measured shape from the cluster lab: a CPU-throttled replica whose p95 response time rose
            // 2.74x while it kept serving every request. Two detectors, two signals, one pod — and the thing
            // an engineer can act on is the throttling, not the latency it explains.
            var pipeline = new IncidentPipeline();
            var subject = Subject("pod-2");

            var (latency, times) = LeakySeries(120);
            var latencyResult = Trend.Detect(latency, times, TrendOptions.Balanced);
            Assert.Equal(DetectionStatus.Anomalous, latencyResult.Status);
            Assert.True(pipeline.Observe(
                subject, "overfit_chat_response_time_seconds", latencyResult, WindowStart, WindowEnd));

            var (peers, subjects) = PeerGroup(deviatingIndex: 2, factor: 2.2);
            var findings = new PeerOutlierFinding[peers.Count];
            var peerResult = Peers.Detect(peers, PeerSignalKind.LoadIndependent, PeerOutlierOptions.Balanced, findings);
            Assert.Equal(DetectionStatus.Anomalous, peerResult.Status);
            Assert.True(pipeline.ObservePeerGroup(
                "container_cpu_cfs_throttled_periods_total",
                peerResult,
                findings.AsSpan(0, peers.Count),
                subjects.AsSpan(0, peers.Count),
                WindowStart,
                WindowEnd) > 0);

            var incident = Assert.Single(pipeline.Group(IncidentGroupingOptions.Balanced));

            Assert.Equal("container_cpu_cfs_throttled_periods_total", incident.Primary.Signal);
            Assert.Equal(SignalClass.Infrastructure, incident.Primary.Class);
            Assert.Equal(2, incident.DistinctSignals);
            Assert.Contains("+1 related finding", incident.Summary, StringComparison.Ordinal);
        }

        [Fact]
        public void SeverityComesFromTheEffectSize_SoTheTwoDetectorsAreOrderedAgainstEachOther()
        {
            // A weak-but-significant trend must not outrank a strong peer deviation just because it was
            // measured over more samples. That is precisely what would happen if severity carried a p-value.
            var pipeline = new IncidentPipeline();

            var (weak, times) = LeakySeries(200, driftPerSample: 0.35, noise: 0.02);
            var weakResult = Trend.Detect(weak, times, TrendOptions.Balanced);
            Assert.Equal(DetectionStatus.Anomalous, weakResult.Status);
            pipeline.Observe(Subject("pod-far", workload: "other", node: "node-9"),
                             "process_resident_memory_bytes", weakResult, WindowStart, WindowEnd);

            var (peers, subjects) = PeerGroup(deviatingIndex: 1, factor: 3.0);
            var findings = new PeerOutlierFinding[peers.Count];
            var peerResult = Peers.Detect(peers, PeerSignalKind.LoadIndependent, PeerOutlierOptions.Balanced, findings);
            pipeline.ObservePeerGroup("container_oom_events_total", peerResult,
                                      findings.AsSpan(0, peers.Count), subjects.AsSpan(0, peers.Count),
                                      WindowStart, WindowEnd);

            var incidents = pipeline.Group(IncidentGroupingOptions.Balanced);

            Assert.Equal(2, incidents.Count);
            Assert.Equal("container_oom_events_total", incidents[0].Primary.Signal);
            Assert.True(
                incidents[0].PeakSeverity > incidents[1].PeakSeverity,
                $"peer deviation {incidents[0].PeakSeverity} did not outrank trend {incidents[1].PeakSeverity}");
        }

        [Fact]
        public void ASmallGroupWithOneStrongOutlier_IsInconclusive()
        {
            // Masking, and the pipeline must not launder it into a finding. Leave-one-out puts the outlier
            // inside every other member's baseline: at four peers it is 1/3 of that baseline, which drags the
            // three healthy members far enough to clear the materiality gate in the opposite direction. The
            // detector then sees one member High and three Low — no coherent norm — and says so rather than
            // picking a side.
            var pipeline = new IncidentPipeline();
            var (peers, subjects) = PeerGroup(deviatingIndex: 2, factor: 1.6, members: 4);

            var findings = new PeerOutlierFinding[peers.Count];
            var result = Peers.Detect(peers, PeerSignalKind.LoadIndependent, PeerOutlierOptions.Balanced, findings);

            Assert.Equal(DetectionStatus.Inconclusive, result.Status);
            Assert.True(result.HighCount > 0 && result.LowCount > 0, "expected contradictory directions");

            var added = pipeline.ObservePeerGroup(
                "kube_pod_container_status_restarts_total",
                result,
                findings.AsSpan(0, peers.Count),
                subjects.AsSpan(0, peers.Count),
                WindowStart,
                WindowEnd);

            Assert.Equal(0, added);
            Assert.Empty(pipeline.Group(IncidentGroupingOptions.Balanced));
        }

        [Fact]
        public void MisalignedSubjects_AreRejectedRatherThanSilentlyMisattributed()
        {
            // Attributing a finding to the wrong pod is worse than not reporting it, so the alignment
            // contract is enforced rather than trusted.
            var pipeline = new IncidentPipeline();
            var (peers, subjects) = PeerGroup(deviatingIndex: 2, factor: 1.6);
            var findings = new PeerOutlierFinding[peers.Count];
            var result = Peers.Detect(peers, PeerSignalKind.LoadIndependent, PeerOutlierOptions.Balanced, findings);

            Assert.Throws<ArgumentException>(() => pipeline.ObservePeerGroup(
                "kube_pod_container_status_restarts_total",
                result,
                findings.AsSpan(0, peers.Count),
                subjects.AsSpan(0, peers.Count - 1),
                WindowStart,
                WindowEnd));
        }

        [Fact]
        public void ClearResetsTheCycle()
        {
            var pipeline = new IncidentPipeline();
            var (values, times) = LeakySeries(120);

            pipeline.Observe(Subject("pod-a"), "process_resident_memory_bytes",
                             Trend.Detect(values, times, TrendOptions.Balanced), WindowStart, WindowEnd);
            Assert.Equal(1, pipeline.Count);

            pipeline.Clear();

            Assert.Equal(0, pipeline.Count);
            Assert.Empty(pipeline.Group(IncidentGroupingOptions.Balanced));
        }

        [Theory]
        [InlineData("kube_pod_container_status_restarts_total", SignalClass.Infrastructure)]
        [InlineData("container_cpu_cfs_throttled_periods_total", SignalClass.Infrastructure)]
        [InlineData("container_oom_events_total", SignalClass.Infrastructure)]
        [InlineData("process_resident_memory_bytes", SignalClass.Resource)]
        [InlineData("process_cpu_seconds_total", SignalClass.Resource)]
        [InlineData("dotnet_threadpool_queue_length", SignalClass.Resource)]
        [InlineData("overfit_chat_response_time_seconds", SignalClass.Symptom)]
        [InlineData("overfit_chat_ttft_seconds", SignalClass.Symptom)]
        [InlineData("overfit_pool_rejected_total", SignalClass.Symptom)]
        [InlineData("something_nobody_has_seen_before", SignalClass.Symptom)]
        [InlineData("", SignalClass.Symptom)]
        public void TheCatalogClassifiesTheMetricsTheLabActuallyExports(string signal, SignalClass expected)
        {
            Assert.Equal(expected, SignalCatalog.Classify(signal));
        }

        [Fact]
        public void AnExplicitClassOverridesTheCatalog()
        {
            var pipeline = new IncidentPipeline();
            var (values, times) = LeakySeries(120);

            pipeline.Observe(
                Subject("pod-a"),
                "process_resident_memory_bytes",
                Trend.Detect(values, times, TrendOptions.Balanced),
                WindowStart,
                WindowEnd,
                signalClass: SignalClass.Infrastructure);

            var incident = Assert.Single(pipeline.Group(IncidentGroupingOptions.Balanced));

            Assert.Equal(SignalClass.Infrastructure, incident.Primary.Class);
        }

        private static IncidentSubject Subject(string pod, string workload = "overfit-server", string node = "node-1")
            => new("overfit", workload, pod, node);

        private static (double[] Values, double[] Times) FlatSeries(int samples)
        {
            var rng = new Random(11);
            var values = new double[samples];
            var times = new double[samples];

            for (var i = 0; i < samples; i++)
            {
                times[i] = i * ScrapeSeconds;
                values[i] = 400.0 * (1.0 + ((rng.NextDouble() - 0.5) * 0.005));
            }

            return (values, times);
        }

        private static (double[] Values, double[] Times) LeakySeries(
            int samples,
            double driftPerSample = 3.0,
            double noise = 0.01)
        {
            var rng = new Random(20260728);
            var values = new double[samples];
            var times = new double[samples];
            var level = 400.0;

            for (var i = 0; i < samples; i++)
            {
                times[i] = i * ScrapeSeconds;
                level += driftPerSample;
                values[i] = level * (1.0 + ((rng.NextDouble() - 0.5) * noise));
            }

            return (values, times);
        }

        /// <summary>
        /// Peers on a common baseline, one of them scaled away from the rest.
        ///
        /// <para>Eight members by default, and the number matters. The detector compares each member against
        /// the pooled others, so a single outlier sits inside everybody else's baseline — at eight members it
        /// contributes 1/7 of that baseline, which caps the resulting Cliff's delta near 0.14 and leaves the
        /// healthy peers below the 0.33 materiality gate. See
        /// <see cref="ASmallGroupWithOneStrongOutlier_IsInconclusive"/> for what happens when the group is
        /// small enough that it does not.</para>
        /// </summary>
        private static (List<PeerSeries> Peers, IncidentSubject[] Subjects) PeerGroup(
            int deviatingIndex,
            double factor,
            int members = 8)
        {
            const int Samples = 60;

            var rng = new Random(4242);
            var peers = new List<PeerSeries>(members);
            var subjects = new IncidentSubject[members];

            for (var p = 0; p < members; p++)
            {
                var values = new double[Samples];
                var scale = p == deviatingIndex ? factor : 1.0;

                for (var i = 0; i < Samples; i++)
                {
                    values[i] = 100.0 * scale * (1.0 + ((rng.NextDouble() - 0.5) * 0.05));
                }

                peers.Add(new PeerSeries($"pod-{p}", values));
                subjects[p] = Subject($"pod-{p}");
            }

            return (peers, subjects);
        }
    }
}
