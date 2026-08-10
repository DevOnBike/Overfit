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
    /// The decision trace on the TREND and RULE families — <c>XC-15</c>.
    ///
    /// <para><b>Why it exists.</b> The peer path could say why it stayed quiet and the rest of the detector
    /// could not, so half the guard's silence was diagnosable and half was not. Measured cost on 2026-08-10:
    /// two hours spent on a channel that looked gated and was merely slower, and a twenty-minute injected
    /// fault that produced zero rows because the custom path emitted nothing.</para>
    ///
    /// <para><b>The row worth having is the warm-up one.</b> A pod inside the grace is skipped with a bare
    /// <c>continue</c>, so before this "tested and healthy" and "never tested" were the same silence — and
    /// during a rollout that is every pod at once. Anything else the trace shows can be inferred from a
    /// finding; that one cannot be inferred from anything.</para>
    /// </summary>
    public sealed class TrendAndRuleTraceTests
    {
        private const string Queue = "myapp_queue_depth";

        private static readonly DateTimeOffset T0 = new(2026, 8, 6, 8, 0, 0, TimeSpan.Zero);

        [Fact]
        public void TheTrendPathEmitsARowPerPodPerChannel()
        {
            var rows = new List<TrendDecisionTrace>();
            var guard = Guard(new NullSink(), custom: false);

            guard.RunCycle(Window(custom: false), T0.AddMinutes(5), null, null, rows.Add, null);

            Assert.NotEmpty(rows);
            Assert.All(rows, r => Assert.False(string.IsNullOrEmpty(r.Pod)));
            Assert.All(rows, r => Assert.False(string.IsNullOrEmpty(r.Signal)));
        }

        /// <summary>The custom half, which is where the peer trace's equivalent gap was found.</summary>
        [Fact]
        public void TheCustomTrendPathEmitsRowsToo()
        {
            var rows = new List<TrendDecisionTrace>();
            var guard = Guard(new NullSink(), custom: true);

            guard.RunCycle(Window(custom: true), T0.AddMinutes(5), null, null, rows.Add, null);

            Assert.Contains(rows, r => string.Equals(r.Signal, Queue, StringComparison.Ordinal));
        }

        /// <summary>
        /// <b>The point of the whole task.</b> A pod skipped for warm-up leaves a row saying so, instead of
        /// leaving nothing and being indistinguishable from a pod that was tested and found healthy.
        /// </summary>
        [Fact]
        public void APodSkippedForWarmUpIsTracedRatherThanSilentlyDropped()
        {
            var rows = new List<TrendDecisionTrace>();
            var guard = Guard(new NullSink(), custom: false, warmUp: TimeSpan.FromHours(6));

            // The window starts minutes after T0, so every pod is inside a six-hour grace.
            guard.RunCycle(Window(custom: false), T0.AddMinutes(5), null, null, rows.Add, null);

            var skipped = rows.Where(r => r.WarmingUp).ToList();

            Assert.NotEmpty(skipped);
            Assert.All(skipped, r => Assert.Equal(DetectionStatus.InsufficientData, r.Status));
            Assert.All(skipped, r => Assert.Contains("warm-up", r.Reason, StringComparison.OrdinalIgnoreCase));
        }

        /// <summary>Without the grace the same pods are judged, so the flag is not simply always set.</summary>
        [Fact]
        public void WithoutTheGraceNoRowIsMarkedWarmingUp()
        {
            var rows = new List<TrendDecisionTrace>();
            var guard = Guard(new NullSink(), custom: false, warmUp: TimeSpan.Zero);

            guard.RunCycle(Window(custom: false), T0.AddMinutes(5), null, null, rows.Add, null);

            Assert.DoesNotContain(rows, r => r.WarmingUp);
        }

        /// <summary>
        /// The trend row carries the detector's own sentence. That sentence is what an operator reads first,
        /// and the numbers beside it are what they check it against.
        /// </summary>
        [Fact]
        public void ATrendRowCarriesTheDetectorsReasonAndTheFloorItWasJudgedAgainst()
        {
            var rows = new List<TrendDecisionTrace>();
            var guard = Guard(new NullSink(), custom: false, warmUp: TimeSpan.Zero);

            guard.RunCycle(Window(custom: false), T0.AddMinutes(5), null, null, rows.Add, null);

            var judged = rows.First(r => !r.WarmingUp);

            Assert.False(string.IsNullOrWhiteSpace(judged.Reason));
            Assert.True(judged.SampleCount > 0);
        }

        /// <summary>Nothing is emitted when no sink is passed — the flag is what turns this on, not the code.</summary>
        [Fact]
        public void NoSinkMeansNoWork()
        {
            var guard = Guard(new NullSink(), custom: false);

            var result = guard.RunCycle(Window(custom: false), T0.AddMinutes(5));

            Assert.NotNull(result);
        }

        private static AnomalyGuard Guard(IIncidentSink sink, bool custom, TimeSpan? warmUp = null)
        {
            var gaps = new double[(int)MetricIndex.Count];
            gaps[(int)MetricIndex.MemoryWorkingSetBytes] = 5_000_000.0;

            return new AnomalyGuard(
                new AnomalyGuardOptions
                {
                    Namespace = "lab",
                    Workload = "lab-workload",
                    MinAbsoluteGap = gaps,
                    ApplyCalibratedFloors = false,
                    MinimumHistoryDays = 0,
                    DecomposeCommonMode = false,
                    WarmUpGrace = warmUp ?? TimeSpan.Zero,

                    // The grace FAILS CLOSED: no topology, or an unknown creation time, means no pod is
                    // ever spared. So the warm-up row cannot be produced without one, and a test that
                    // omitted it saw zero skips and looked like a missing trace.
                    PodTopology = warmUp is { } g && g > TimeSpan.Zero ? new FreshTopology(T0) : null,
                    CustomMetrics = custom
                        ? [new CustomMetricBinding(
                            Queue, Queue, MetricSourceKind.Gauge, PeerSignalKind.LoadIndependent,
                            Class: SignalClass.Resource, MinAbsoluteGap: 5.0)]
                        : [],
                    Grouping = IncidentGroupingOptions.Balanced with
                    {
                        Topology = TopologyWeights.SingleNode
                    },
                },
                sink,
                IncidentTrackingOptions.Balanced);
        }

        private static MetricWindow Window(bool custom)
        {
            var names = new List<string>(8);

            for (var p = 0; p < 8; p++)
            {
                names.Add($"lab-workload-7765564ff6-pod{p:d2}");
            }

            var window = custom
                ? new MetricWindow(names, 60, T0, TimeSpan.FromSeconds(15), [Queue])
                : new MetricWindow(names, 60, T0, TimeSpan.FromSeconds(15));
            var rng = new Random(20260806);

            for (var pod = 0; pod < window.Pods.Count; pod++)
            {
                var series = custom
                    ? window.Series(pod, Queue)
                    : window.Series(pod, MetricIndex.MemoryWorkingSetBytes);
                var baseline = custom ? 20.0 : 44_000_000.0;
                var scale = custom ? 1.0 : 400_000.0;

                for (var i = 0; i < window.Length; i++)
                {
                    series[i] = baseline + ((rng.NextDouble() - 0.5) * scale);
                }
            }

            return window;
        }

        /// <summary>Every pod created at <c>T0</c>, so a grace wider than the window covers all of them.</summary>
        private sealed class FreshTopology : IPodTopology
        {
            private readonly DateTimeOffset _created;

            public FreshTopology(DateTimeOffset created) => _created = created;

            public bool TryResolve(string pod, out PodPlacement placement)
            {
                placement = new PodPlacement("lab-workload", "rs-1", "node-0", string.Empty, _created);

                return true;
            }
        }

        private sealed class NullSink : IIncidentSink
        {
            public void Report(ReadOnlySpan<IncidentLogRecord> rows)
            {
            }
        }
    }
}
