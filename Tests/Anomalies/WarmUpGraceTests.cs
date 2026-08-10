// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Anomalies.Contracts;
using DevOnBike.Overfit.Anomalies.Incidents;
using DevOnBike.Overfit.Anomalies.Incidents.Abstractions;
using DevOnBike.Overfit.Anomalies.Monitoring.Abstractions;
using DevOnBike.Overfit.Statistics;

namespace DevOnBike.Overfit.Tests.Anomalies
{
    /// <summary>
    /// A pod that has just started is warming up, not leaking.
    ///
    /// <para><b>Measured, not supposed.</b> On the lab, across a rollout, a manual scale-up and an HPA
    /// scale-up, a freshly created replica's working set climbed <b>13–17% of typical over its first 10–20
    /// minutes</b> at a Kendall tau of 0.70–0.94. The scale-up phases ran <b>zero quiet cycles out of
    /// seven</b>; the opposite transition ran four of six. The whole asymmetry was new pods being judged for
    /// getting warm. See <c>docs/aiops/aiops-day-one-events.md</c>.</para>
    ///
    /// <para>The last two tests are the ones that matter, because a grace is a licence to stay silent and
    /// every silence has to be justified: an unknown age must not buy an exemption, and the peer family must
    /// keep judging young pods.</para>
    /// </summary>
    public sealed class WarmUpGraceTests
    {
        private static readonly DateTimeOffset T0 = new(2026, 8, 2, 12, 0, 0, TimeSpan.Zero);

        [Fact]
        public void AFreshPodsClimbIsNotReported()
        {
            var sink = new CapturingSink();

            // Created two minutes before the window ends, against a fifteen-minute grace.
            var guard = Guard(sink, created: T0.AddMinutes(18));

            guard.RunCycle(Climbing(), T0.AddMinutes(20));

            Assert.DoesNotContain(sink.Messages, m => m.Contains("rose by", StringComparison.Ordinal));
        }

        [Fact]
        public void TheSameClimbOnAnEstablishedPodIsReported()
        {
            var sink = new CapturingSink();
            var guard = Guard(sink, created: T0.AddHours(-6));

            guard.RunCycle(Climbing(), T0.AddMinutes(20));

            Assert.Contains(sink.Messages, m => m.Contains("rose by", StringComparison.Ordinal));
        }

        /// <summary>
        /// The gate fails closed. A pod whose creation time the topology cannot supply — kube-state-metrics
        /// lagging, or a topology that does not report it at all — is judged exactly as before, because
        /// reading "unknown" as "young" would hand a silent exemption to a pod that may have been running
        /// for a week.
        /// </summary>
        [Fact]
        public void AnUnknownAgeBuysNoExemption()
        {
            var sink = new CapturingSink();
            var guard = Guard(sink, created: default);

            guard.RunCycle(Climbing(), T0.AddMinutes(20));

            Assert.Contains(sink.Messages, m => m.Contains("rose by", StringComparison.Ordinal));
        }

        [Fact]
        public void AZeroGraceDisablesIt()
        {
            var sink = new CapturingSink();
            var guard = Guard(sink, created: T0.AddMinutes(18), grace: TimeSpan.Zero);

            guard.RunCycle(Climbing(), T0.AddMinutes(20));

            Assert.Contains(sink.Messages, m => m.Contains("rose by", StringComparison.Ordinal));
        }

        /// <summary>
        /// The grace covers the trend family only. A young replica that differs from its peers <b>right
        /// now</b> is worth reporting whatever its age — and during a rollout every pod is young, so a
        /// peer-wide grace would blind the guard precisely while a bad version is going out.
        /// </summary>
        [Fact]
        public void PeerComparisonStillJudgesAYoungPod()
        {
            var sink = new CapturingSink();
            var guard = Guard(sink, created: T0.AddMinutes(18));

            guard.RunCycle(OneReplicaHigh(), T0.AddMinutes(20));

            Assert.Contains(sink.Messages, m => m.Contains("peers", StringComparison.Ordinal));
        }

        /// <summary>
        /// The same grace on a CUSTOM channel, which it did not cover until 2026-08-10 (`XC-9`).
        ///
        /// <para><b>Only `RunTrend` and `RunSilentPods` called `IsWarmingUp`</b> — mapped across every
        /// detector method. So the trend family refused to judge a young pod on the thirteen built-in
        /// channels and judged it on every custom one, with no reason recorded for the difference. On the
        /// deployed configuration that is five channels, and a rollout creates twelve young pods at once.</para>
        ///
        /// <para>The measurement behind the grace does not care which enum a channel is keyed by: a fresh
        /// replica climbs 13-17% of typical over its first 10-20 minutes at a Kendall tau of 0.70-0.94.</para>
        /// </summary>
        [Fact]
        public void AFreshPodsClimbOnACustomChannelIsNotReportedEither()
        {
            var sink = new CapturingSink();
            var guard = Guard(sink, created: T0.AddMinutes(18), custom: CustomBinding());

            guard.RunCycle(ClimbingCustom(), T0.AddMinutes(20));

            Assert.DoesNotContain(sink.Messages, m => m.Contains("rose by", StringComparison.Ordinal));
        }

        /// <summary>
        /// The control, and it is what stops the test above passing because nothing was ever reported: the
        /// identical climb on an established pod must still be caught on the same custom channel.
        /// </summary>
        [Fact]
        public void TheSameCustomClimbOnAnEstablishedPodIsReported()
        {
            var sink = new CapturingSink();
            var guard = Guard(sink, created: T0.AddHours(-6), custom: CustomBinding());

            guard.RunCycle(ClimbingCustom(), T0.AddMinutes(20));

            Assert.Contains(sink.Messages, m => m.Contains("rose by", StringComparison.Ordinal));
        }

        /// <summary>A custom channel shaped like the deployed ones: load-independent, its own trend floor.</summary>
        private static CustomMetricBinding CustomBinding()
        {
            return new CustomMetricBinding(
                Name: "GcCommittedBytes",
                Source: "dotnet_gc_committed_bytes",
                Kind: MetricSourceKind.Gauge,
                SignalKind: PeerSignalKind.LoadIndependent,
                Class: SignalClass.Resource,
                MinAbsoluteGap: 1.089e6,
                MinAbsoluteTrendChange: 1.089e6);
        }

        /// <summary>`Climbing()`, on a custom channel instead of a built-in one. Same shape, same noise.</summary>
        private static MetricWindow ClimbingCustom()
        {
            var names = new List<string> { "pod-0", "pod-1", "pod-2", "pod-3" };
            var window = new MetricWindow(
                names, 80, T0, TimeSpan.FromSeconds(15), ["GcCommittedBytes"]);
            var rng = new Random(20260810);

            for (var pod = 0; pod < names.Count; pod++)
            {
                var series = window.Series(pod, "GcCommittedBytes");
                var start = 40e6 * (1.0 + ((rng.NextDouble() - 0.5) * 0.02));

                for (var i = 0; i < window.Length; i++)
                {
                    var phase = i / (double)(window.Length - 1);

                    // Noise for the same reason as Climbing(): a noiseless ramp is rejected by the trend
                    // detector's variance inflation, and the test would pass without the gate doing anything.
                    series[i] = start * (1.0 + (0.15 * phase)) * (1.0 + ((rng.NextDouble() - 0.5) * 0.05));
                }
            }

            return window;
        }

        private static AnomalyGuard Guard(
            CapturingSink sink,
            DateTimeOffset created,
            TimeSpan? grace = null,
            CustomMetricBinding? custom = null)
        {
            return new AnomalyGuard(
                new AnomalyGuardOptions
                {
                    CustomMetrics = custom is null ? [] : [custom.Value],
                    Namespace = "lab",
                    Workload = "svc",
                    PodTopology = new FakeTopology(created),
                    MinimumHistoryDays = 0,
                    WarmUpGrace = grace ?? TimeSpan.FromMinutes(15),

                    // The floor the LAB runs, not the shipped default. AnomalyGuardOptions defaults the
                    // working-set trend floor to 256 MiB over a window — a figure measured on pods carrying
                    // 1.23 GB — and this fixture's pods carry 40 MB, so the default would reject a 6 MB
                    // climb before the warm-up rule ever saw it and every assertion here would pass for the
                    // wrong reason. The deployed lab ConfigMap overrides it to 1.089 MB; this matches.
                    MinAbsoluteTrendChange = TrendFloors(),

                    // Off, so the only thing that can produce a "rose by" line is the per-pod trend family —
                    // the workload-level one would report the same climb from a subject the grace does not
                    // cover, and the test would pass for the wrong reason.
                    DecomposeCommonMode = false,
                    Grouping = IncidentGroupingOptions.Balanced with { Topology = TopologyWeights.SingleNode },
                },
                sink,
                IncidentTrackingOptions.Balanced);
        }

        /// <summary>A floor table sized to this fixture's 40 MB pods rather than to the shipped default.</summary>
        private static double[] TrendFloors()
        {
            var floors = new double[(int)MetricIndex.Count];
            floors[(int)MetricIndex.MemoryWorkingSetBytes] = 1.089e6;

            return floors;
        }

        /// <summary>Four replicas whose memory climbs together, the way freshly started pods do.</summary>
        private static MetricWindow Climbing()
        {
            var names = new List<string> { "pod-0", "pod-1", "pod-2", "pod-3" };
            var window = new MetricWindow(names, 80, T0, TimeSpan.FromSeconds(15));
            var rng = new Random(20260802);

            for (var pod = 0; pod < names.Count; pod++)
            {
                var memory = window.Series(pod, MetricIndex.MemoryWorkingSetBytes);
                var cpu = window.Series(pod, MetricIndex.CpuUsageRatio);
                var start = 40e6 * (1.0 + ((rng.NextDouble() - 0.5) * 0.02));

                for (var i = 0; i < window.Length; i++)
                {
                    var phase = i / (double)(window.Length - 1);

                    // +15% across the window, on 5% noise. The noise is not decoration: a perfectly smooth
                    // ramp has a lag-1 autocorrelation near 1.0, which the trend detector deliberately
                    // penalises through variance inflation, so a noiseless fixture is REJECTED and the test
                    // would pass for the wrong reason. The lab's real warm-ups sat at 0.70-0.90.
                    memory[i] = start * (1.0 + (0.15 * phase)) * (1.0 + ((rng.NextDouble() - 0.5) * 0.05));
                    cpu[i] = 0.2;
                }
            }

            return window;
        }

        /// <summary>Flat memory, but one replica sitting far above the others — a peer question, not a trend one.</summary>
        private static MetricWindow OneReplicaHigh()
        {
            var names = new List<string> { "pod-0", "pod-1", "pod-2", "pod-3" };
            var window = new MetricWindow(names, 80, T0, TimeSpan.FromSeconds(15));
            var rng = new Random(20260802);

            for (var pod = 0; pod < names.Count; pod++)
            {
                var memory = window.Series(pod, MetricIndex.MemoryWorkingSetBytes);
                var cpu = window.Series(pod, MetricIndex.CpuUsageRatio);
                var level = pod == 0 ? 90e6 : 40e6;

                for (var i = 0; i < window.Length; i++)
                {
                    memory[i] = level * (1.0 + ((rng.NextDouble() - 0.5) * 0.01));
                    cpu[i] = 0.2;
                }
            }

            return window;
        }

        private sealed class FakeTopology : IPodTopology
        {
            private readonly DateTimeOffset _created;

            public FakeTopology(DateTimeOffset created)
            {
                _created = created;
            }

            public bool TryResolve(string pod, out PodPlacement placement)
            {
                placement = new PodPlacement("svc", "rs-1", "node-0", string.Empty, _created);

                return true;
            }
        }

        private sealed class CapturingSink : IIncidentSink
        {
            public List<string> Messages { get; } = [];

            public void Report(ReadOnlySpan<IncidentLogRecord> rows)
            {
                for (var i = 0; i < rows.Length; i++)
                {
                    Messages.Add(rows[i].Message);
                }
            }
        }
    }
}
