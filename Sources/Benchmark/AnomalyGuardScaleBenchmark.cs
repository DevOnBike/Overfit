// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using BenchmarkDotNet.Attributes;
using DevOnBike.Overfit.Anomalies.Incidents;
using DevOnBike.Overfit.Anomalies.Incidents.Abstractions;
using DevOnBike.Overfit.Anomalies.Contracts;
using DevOnBike.Overfit.Statistics;

namespace Benchmarks
{
    /// <summary>
    /// How many pods one guard instance can evaluate inside its cadence.
    ///
    /// <para><b>This number is the client's second question and it has been unanswered for a day.</b> One
    /// instance per cluster is the recommended shape, so "how big a cluster" is not a curiosity — it decides
    /// whether the recommendation holds at all. Estimating it from the algorithm has been the fallback and is
    /// not good enough: Theil-Sen is O(n²) in samples, which says the shape of the curve but not where it
    /// crosses five minutes.</para>
    ///
    /// <para>The whole cycle is measured, not a kernel: rules, peer comparison, the common-mode-decomposed
    /// trend, grouping, tracking and reporting. A kernel benchmark would answer a question nobody asked —
    /// what is wanted is wall-clock against the cadence.</para>
    ///
    /// <para>The window is 80 samples, which is the guard's default twenty minutes at a fifteen-second
    /// scrape. Sample count matters more than it looks because of the quadratic, so a benchmark at a
    /// different window length would not transfer.</para>
    ///
    /// <para><see cref="SimpleJobAttribute"/> rather than the shared config: this is milliseconds-to-seconds
    /// work, and the shared <c>InvocationCount=1</c> exists for multi-millisecond model runs.</para>
    /// </summary>
    [SimpleJob(warmupCount: 1, iterationCount: 5)]
    [MemoryDiagnoser]
    public class AnomalyGuardScaleBenchmark
    {
        private MetricWindow _window = null!;
        private AnomalyGuard _guard = null!;
        private DateTimeOffset _now;

        /// <summary>
        /// Replicas in the evaluated group. 4 is the lab, 20 the synthetic population, and the rest are the
        /// sizes a real namespace reaches.
        /// </summary>
        [Params(4, 20, 50, 100, 200)]
        public int Pods
        {
            get; set;
        }

        [GlobalSetup]
        public void Setup()
        {
            _now = new DateTimeOffset(2026, 7, 31, 12, 0, 0, TimeSpan.Zero);

            var names = new List<string>(Pods);

            for (var p = 0; p < Pods; p++)
            {
                names.Add($"srv-111-pod{p:d5}");
            }

            // 80 samples: twenty minutes at a fifteen-second scrape, the guard's default window.
            _window = new MetricWindow(names, 80, _now, TimeSpan.FromSeconds(15));

            var rng = new Random(20260731);

            // Every channel filled, because an unreported one is skipped and would flatter the measurement.
            for (var p = 0; p < Pods; p++)
            {
                for (var m = 0; m < (int)MetricIndex.Count; m++)
                {
                    var series = _window.Series(p, (MetricIndex)m);
                    var level = 100.0 * (m + 1) * (1.0 + ((rng.NextDouble() - 0.5) * 0.1));

                    for (var i = 0; i < series.Length; i++)
                    {
                        series[i] = level * (1.0 + ((rng.NextDouble() - 0.5) * 0.2));
                    }
                }
            }

            _guard = new AnomalyGuard(
                new AnomalyGuardOptions
                {
                    Namespace = "bench",
                    Workload = "srv",
                    Grouping = IncidentGroupingOptions.Balanced with
                    {
                        Topology = TopologyWeights.SingleNode
                    },
                },
                new NullSink(),
                IncidentTrackingOptions.Balanced);
        }

        [Benchmark]
        public int OneCycle()
        {
            return _guard.RunCycle(_window, _now).Findings;
        }

        private sealed class NullSink : IIncidentSink
        {
            public void Report(ReadOnlySpan<IncidentLogRecord> rows)
            {
            }
        }
    }
}
