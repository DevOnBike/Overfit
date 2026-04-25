// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;
using BenchmarkDotNet.Order;
using DevOnBike.Overfit.Randomization;

namespace Benchmarks
{
    [MemoryDiagnoser]
    [SimpleJob(RuntimeMoniker.Net10_0)]
    [Orderer(SummaryOrderPolicy.FastestToSlowest)]
    public class RandomizationBenchmarks
    {
        [Params(256, 1024, 8192, 65536)]
        public int Count { get; set; }

        private float[] _buffer = null!;
        private Random _random = null!;
        private VectorizedRandom _vectorized;

        [GlobalSetup]
        public void Setup()
        {
            _buffer = new float[Count];
            _random = new Random(12345);
            _vectorized = new VectorizedRandom(12345);
        }

        [Benchmark(Baseline = true)]
        public float Random_NextSingle()
        {
            var buffer = _buffer;
            var rnd = _random;

            for (var i = 0; i < buffer.Length; i++)
            {
                buffer[i] = rnd.NextSingle();
            }

            return buffer[^1];
        }

        [Benchmark]
        public float RandomShared_NextSingle()
        {
            var buffer = _buffer;

            for (var i = 0; i < buffer.Length; i++)
            {
                buffer[i] = Random.Shared.NextSingle();
            }

            return buffer[^1];
        }

        [Benchmark]
        public float VectorizedPRNG_Fill()
        {
            _vectorized.Fill(_buffer);
            return _buffer[^1];
        }
    }
}