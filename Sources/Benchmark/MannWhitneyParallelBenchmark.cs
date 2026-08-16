// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using BenchmarkDotNet.Attributes;
using DevOnBike.Overfit.Statistics;

namespace Benchmarks
{
    /// <summary>
    /// A whole canary evaluation rather than a single comparison: one Mann-Whitney per metric, sequential
    /// versus spread across cores. This is the shape the analyser actually runs, and the only place in the
    /// statistic where parallelism has real headroom — the metrics are completely independent.
    /// </summary>
    [SimpleJob]
    [MemoryDiagnoser]
    public class MannWhitneyParallelBenchmark
    {
        private double[][] _baselines = [];
        private double[][] _candidates = [];
        private MannWhitneyResult[] _results = [];

        /// <summary>Metrics compared per evaluation: CPU, latency, errors, working set, tokens/s, and so on.</summary>
        [Params(4, 16)]
        public int Metrics
        {
            get; set;
        }

        [Params(2_000, 10_000)]
        public int SamplesPerArm
        {
            get; set;
        }

        [GlobalSetup]
        public void Setup()
        {
            var rng = new Random(20260724);

            _baselines = new double[Metrics][];
            _candidates = new double[Metrics][];
            _results = new MannWhitneyResult[Metrics];

            for (var m = 0; m < Metrics; m++)
            {
                _baselines[m] = SampleLatencies(rng, SamplesPerArm, 1.0);
                _candidates[m] = SampleLatencies(rng, SamplesPerArm, 1.1);
            }
        }

        [Benchmark(Baseline = true)]
        public void Sequential()
        {
            for (var m = 0; m < _baselines.Length; m++)
            {
                _results[m] = MannWhitneyU.Compare(_baselines[m], _candidates[m]);
            }
        }

        [Benchmark]
        public void Parallel()
        {
            MannWhitneyU.CompareMany(_baselines, _candidates, _results);
        }

        private static double[] SampleLatencies(Random rng, int count, double scale)
        {
            var values = new double[count];

            for (var i = 0; i < count; i++)
            {
                var u1 = 1.0 - rng.NextDouble();
                var u2 = 1.0 - rng.NextDouble();
                var normal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
                values[i] = scale * 40.0 * Math.Exp(0.35 * normal);
            }

            return values;
        }
    }
}
