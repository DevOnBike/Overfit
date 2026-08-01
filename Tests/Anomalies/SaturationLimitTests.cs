// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Anomalies.Contracts;
using DevOnBike.Overfit.Anomalies.Incidents;
using DevOnBike.Overfit.Anomalies.Incidents.Abstractions;

namespace DevOnBike.Overfit.Tests.Anomalies
{
    /// <summary>
    /// Time-to-limit, which the detector has always computed and the deployed path never had a limit for.
    ///
    /// <para>The guard passed <c>double.NaN</c> at both trend call sites, so <c>ProjectTimeToLimit</c>
    /// returned null every time and the capability was dead in exactly the place it mattered. The difference
    /// it makes is not cosmetic: "working set rose by 11% of typical" and "working set reaches its limit in
    /// 40 minutes" are the same measurement, and only the second tells an operator whether to get up.</para>
    /// </summary>
    public sealed class SaturationLimitTests
    {
        private static readonly DateTimeOffset T0 = new(2026, 8, 1, 12, 0, 0, TimeSpan.Zero);

        [Fact]
        public void WithoutALimit_NoProjectionIsMade()
        {
            var sink = new CapturingSink();

            Guard(sink, limits: null).RunCycle(LeakingWindow(), T0);

            Assert.NotEmpty(sink.Messages);
            Assert.DoesNotContain(sink.Messages, m => m.Contains("the limit is reached in", StringComparison.Ordinal));
        }

        /// <summary>
        /// The same leak against a stated ceiling. The number itself is the detector's; what is asserted here
        /// is that the ceiling reaches it at all, which is what was broken.
        /// </summary>
        [Fact]
        public void WithALimit_TheFindingSaysWhenItWillBeReached()
        {
            var sink = new CapturingSink();
            var limits = new double[(int)MetricIndex.Count];

            // 1 GiB, against a series climbing from 400 MB — comfortably ahead, so a projection exists and is
            // inside the detector's own horizon for one worth reporting.
            limits[(int)MetricIndex.MemoryWorkingSetBytes] = 1024.0 * 1024 * 1024;

            Guard(sink, limits).RunCycle(LeakingWindow(), T0);

            Assert.Contains(sink.Messages, m => m.Contains("the limit is reached in", StringComparison.Ordinal));
        }

        [Fact]
        public void AnAbsentEntryIsNotZero()
        {
            var table = new double[(int)MetricIndex.Count];
            table[(int)MetricIndex.MemoryWorkingSetBytes] = 1000.0;

            // A missing ceiling must read as "do not project", not as a ceiling of zero — which every series
            // is already above, and which would make the projection nonsense rather than absent.
            Assert.Equal(1000.0, AnomalyGuardOptions.LimitFor(table, MetricIndex.MemoryWorkingSetBytes));
            Assert.True(double.IsNaN(AnomalyGuardOptions.LimitFor(table, MetricIndex.CpuUsageRatio)));
            Assert.True(double.IsNaN(AnomalyGuardOptions.LimitFor(null, MetricIndex.MemoryWorkingSetBytes)));
        }

        private static AnomalyGuard Guard(CapturingSink sink, IReadOnlyList<double>? limits)
            => new(
                new AnomalyGuardOptions
                {
                    Namespace = "lab",
                    Workload = "svc",
                    SaturationLimit = limits,

                    // Off: the decomposition tests each pod against the group's common component, and here
                    // every pod leaks together, so the residual would be flat and there would be no trend to
                    // project at all.
                    DecomposeCommonMode = false,
                    Grouping = IncidentGroupingOptions.Balanced with { Topology = TopologyWeights.SingleNode },
                },
                sink,
                IncidentTrackingOptions.Balanced);

        /// <summary>Four replicas whose working set climbs steadily from 400 MB.</summary>
        private static MetricWindow LeakingWindow()
        {
            var names = new List<string> { "pod-0", "pod-1", "pod-2", "pod-3" };
            var window = new MetricWindow(names, 80, T0, TimeSpan.FromSeconds(15));
            var rng = new Random(20260801);

            for (var pod = 0; pod < names.Count; pod++)
            {
                var memory = window.Series(pod, MetricIndex.MemoryWorkingSetBytes);
                var rps = window.Series(pod, MetricIndex.RequestsPerSecond);

                for (var i = 0; i < window.Length; i++)
                {
                    memory[i] = (400e6 + (i * 4e6)) * (1.0 + ((rng.NextDouble() - 0.5) * 0.01));
                    rps[i] = 5.0;
                }
            }

            return window;
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
