// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using BenchmarkDotNet.Attributes;
using DevOnBike.Overfit.Statistics;

namespace Benchmarks
{
    /// <summary>
    /// What one rank correlation costs, and what the lag scan multiplies it by.
    ///
    /// <para>The reference measurement for <see cref="SpearmanCorrelation"/>, written alongside its
    /// correctness tests rather than after someone wonders whether it is slow. It sits under
    /// <c>IncidentGrouper</c>, which consults it once per finding pair — so this number gets multiplied by
    /// <c>O(N²)</c> before it reaches the product, and a cost that looks negligible here does not stay
    /// negligible there.</para>
    ///
    /// <para>The lag scan cannot share work between offsets: shifting the window changes which samples
    /// overlap, which changes the ranks, so every offset is a fresh ranking of both series. The arms below
    /// exist to price that honestly — the expectation is a factor of 2·L+1, and the point of measuring is to
    /// find out where it is not.</para>
    ///
    /// <para><see cref="SimpleJobAttribute"/> for the reason given in <see cref="MannWhitneyBenchmark"/>: the
    /// shared config's <c>InvocationCount=1</c> would leave a microsecond routine measuring the timer.</para>
    /// </summary>
    [SimpleJob]
    [MemoryDiagnoser]
    public class SpearmanCorrelationBenchmark
    {
        private double[] _leader = [];
        private double[] _follower = [];

        /// <summary>Window length. 120 is an hour at a 30 s scrape; 300 is two and a half.</summary>
        [Params(60, 120, 300)]
        public int Samples
        {
            get; set;
        }

        [GlobalSetup]
        public void Setup()
        {
            var rng = new Random(20260728);

            _leader = new double[Samples];
            _follower = new double[Samples];

            for (var i = 0; i < Samples; i++)
            {
                _leader[i] = Math.Sin(i * 0.17) * 100.0 + (rng.NextDouble() * 8.0);
            }

            // A delayed, rescaled echo with its own noise — the shape the grouper is looking for.
            for (var i = 0; i < Samples; i++)
            {
                var source = Math.Max(0, i - 4);
                _follower[i] = (_leader[source] * 0.4) + 30.0 + (rng.NextDouble() * 5.0);
            }
        }

        /// <summary>No scan: one ranking of each series, one Pearson over the ranks.</summary>
        [Benchmark(Baseline = true)]
        public double Correlate()
        {
            return SpearmanCorrelation.Correlate(_leader, _follower).Rho;
        }

        /// <summary>Nine offsets.</summary>
        [Benchmark]
        public double CorrelateWithLag_4()
        {
            return SpearmanCorrelation.CorrelateWithLag(_leader, _follower, 4).Rho;
        }

        /// <summary>Twenty-one offsets — the <c>IncidentGroupingOptions.Balanced</c> default.</summary>
        [Benchmark]
        public double CorrelateWithLag_10()
        {
            return SpearmanCorrelation.CorrelateWithLag(_leader, _follower, 10).Rho;
        }

        /// <summary>
        /// The shape this class shipped with for one afternoon: exact Spearman inside every offset's overlap,
        /// which means re-sorting both series 2L+1 times. Kept as the arm that justifies the replacement —
        /// the ratio against <see cref="CorrelateWithLag_10"/> is the whole argument.
        /// </summary>
        [Benchmark]
        public double CorrelateWithLag_10_PerWindowRanks()
        {
            return SpearmanCorrelation.CorrelateWithLagPerWindowRanks(_leader, _follower, 10).Rho;
        }
    }
}
