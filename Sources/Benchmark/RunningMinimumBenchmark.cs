// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using BenchmarkDotNet.Attributes;
using DevOnBike.Overfit.Statistics;

namespace Benchmarks
{
    /// <summary>
    /// What the sawtooth floor costs, and whether the deque was worth writing.
    ///
    /// <para>The reference measurement for <see cref="RunningMinimum"/>, written alongside its correctness
    /// tests rather than after someone wonders. It runs per pod, per sawtooth-shaped signal, per evaluation
    /// cycle — so on a 200-pod namespace with two such signals it is 400 sweeps a cycle, and a cost that looks
    /// negligible for one series does not stay negligible there.</para>
    ///
    /// <para><b>The baseline is the version anyone would write first.</b> "Minimum of the last k" as a nested
    /// loop is obviously correct and O(n·k); the deque is O(n) whatever k is. The ratio should therefore grow
    /// with <see cref="Lookback"/> and stay flat without it, and the reason to measure rather than assert that
    /// is the constant factor: the naive loop is a tight, branch-predictable, vectorisable scan over
    /// contiguous doubles, while the deque chases indices. At a short lookback the simple loop can win, and
    /// the operational lookback here — one GC cycle, tens of samples — is not obviously past that crossover.</para>
    ///
    /// <para><see cref="SimpleJobAttribute"/> because the shared config's <c>InvocationCount=1</c> fits
    /// multi-millisecond model runs and would leave a microsecond routine measuring the timer.</para>
    /// </summary>
    [SimpleJob]
    [MemoryDiagnoser]
    public class RunningMinimumBenchmark
    {
        private double[] _series = [];
        private double[] _destination = [];
        private int[] _scratch = [];

        /// <summary>Window length. 80 is the detector's 20 minutes at a 15 s scrape; 240 is an hour.</summary>
        [Params(80, 240, 960)]
        public int Samples
        {
            get; set;
        }

        /// <summary>
        /// Collection cycle in samples. 8 is a short gen-2 interval, 57 is the one measured on the synthetic
        /// population, 240 is an hour — the range over which the deque's advantage is supposed to appear.
        /// </summary>
        [Params(8, 57, 240)]
        public int Lookback
        {
            get; set;
        }

        [GlobalSetup]
        public void Setup()
        {
            var rng = new Random(20260730);

            _series = new double[Samples];
            _destination = new double[Samples];
            _scratch = new int[RunningMinimum.RequiredScratchLength(Samples)];

            // A sawtooth with noise and the occasional gap, which is the shape this exists for — a monotone
            // series would let the deque stay empty and flatter it.
            var level = 1.15e9;

            for (var i = 0; i < Samples; i++)
            {
                level += 1.2e6 * (1.0 + ((rng.NextDouble() - 0.5) * 0.4));

                if (level > 1.15e9 * 1.06)
                {
                    level = 1.15e9;
                }

                _series[i] = rng.NextDouble() < 0.005 ? double.NaN : level;
            }
        }

        [Benchmark(Baseline = true)]
        public double Naive()
        {
            for (var i = 0; i < _series.Length; i++)
            {
                var best = double.NaN;
                var from = Math.Max(0, i - Lookback + 1);

                for (var j = from; j <= i; j++)
                {
                    if (double.IsFinite(_series[j]) && (double.IsNaN(best) || _series[j] < best))
                    {
                        best = _series[j];
                    }
                }

                _destination[i] = best;
            }

            return _destination[^1];
        }

        [Benchmark]
        public double Deque()
        {
            RunningMinimum.Compute(_series, Lookback, _destination, _scratch);

            return _destination[^1];
        }
    }
}
