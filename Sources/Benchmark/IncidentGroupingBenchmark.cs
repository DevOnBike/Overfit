// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using BenchmarkDotNet.Attributes;
using DevOnBike.Overfit.Anomalies.Incidents;
using DevOnBike.Overfit.Anomalies.Contracts;

namespace Benchmarks
{
    /// <summary>
    /// What one grouping cycle costs, and what the correlation step adds to it.
    ///
    /// <para>This is the product-level number: the guard runs this once per evaluation cycle over whatever
    /// the detectors produced. Two arms, differing in exactly one lever — <c>Strict</c> sets
    /// <c>MinCorrelation = 1.0</c>, which disables correlation and leaves time and topology, while
    /// <c>Balanced</c> consults <see cref="DevOnBike.Overfit.Statistics.SpearmanCorrelation"/> for every pair
    /// topology cannot already justify.</para>
    ///
    /// <para><b>The fixtures are a deliberate worst case for the correlated arm.</b> Every finding sits in one
    /// namespace with its own workload and its own node, so topology scores 0.25 for every pair and the
    /// correlation step is reached by all of them. Real batches are nowhere near this adversarial — replicas
    /// share a workload, which short-circuits the expensive term — so treat these as the ceiling rather than
    /// the expectation. A benchmark built from friendly data would price the cheap path and say nothing about
    /// the one that can hurt.</para>
    ///
    /// <para><see cref="ScatteredNoSeries"/> is the same adversarial topology with no series attached, which
    /// isolates how much of the correlated arm is scoring and how much is correlation.</para>
    /// </summary>
    [SimpleJob]
    [MemoryDiagnoser]
    public class IncidentGroupingBenchmark
    {
        private const int SeriesLength = 120;

        private static readonly DateTimeOffset Origin = new(2026, 7, 28, 12, 0, 0, TimeSpan.Zero);

        private readonly IncidentGrouper _grouper = new();

        private SignalFinding[] _scattered = [];
        private SignalFinding[] _scatteredNoSeries = [];

        /// <summary>Findings in the batch. 256 is a busy cycle; the per-call bound is 1024.</summary>
        [Params(16, 64, 256)]
        public int Findings
        {
            get; set;
        }

        [GlobalSetup]
        public void Setup()
        {
            var rng = new Random(20260728);

            _scattered = new SignalFinding[Findings];
            _scatteredNoSeries = new SignalFinding[Findings];

            for (var i = 0; i < Findings; i++)
            {
                var series = new double[SeriesLength];
                for (var s = 0; s < SeriesLength; s++)
                {
                    series[s] = Math.Sin((s + i) * 0.15) * 50.0 + (rng.NextDouble() * 10.0);
                }

                // Overlapping windows, so the temporal term never short-circuits the pair away.
                var finding = new SignalFinding(
                    new IncidentSubject("overfit", $"workload-{i}", string.Empty, $"pod-{i}", $"node-{i}"),
                    $"signal_{i % 13}",
                    (SignalClass)(i % 3),
                    Origin.AddSeconds(i),
                    Origin.AddMinutes(10),
                    0.5,
                    "synthetic");

                _scatteredNoSeries[i] = finding;
                _scattered[i] = finding with
                {
                    Series = series
                };
            }
        }

        /// <summary>Time and topology only — the floor.</summary>
        [Benchmark(Baseline = true)]
        public int Strict_NoCorrelation()
        {
            return _grouper.Group(_scattered, IncidentGroupingOptions.Strict).Count;
        }

        /// <summary>Correlation enabled but no series to correlate: isolates the scoring loop itself.</summary>
        [Benchmark]
        public int ScatteredNoSeries()
        {
            return _grouper.Group(_scatteredNoSeries, IncidentGroupingOptions.Balanced).Count;
        }

        /// <summary>The default, on the topology that forces every pair through the lag scan.</summary>
        [Benchmark]
        public int Balanced_WithCorrelation()
        {
            return _grouper.Group(_scattered, IncidentGroupingOptions.Balanced).Count;
        }
    }
}
