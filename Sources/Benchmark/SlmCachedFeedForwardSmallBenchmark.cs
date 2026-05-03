// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
//
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using BenchmarkDotNet.Attributes;
using Benchmarks.Helpers;
using DevOnBike.Overfit.LanguageModels.Runtime;

namespace Benchmarks
{
    /// <summary>
    /// Small SLM single-token feed-forward decode benchmark.
    ///
    /// Shape:
    ///
    /// dModel = 64
    /// dFF = 256
    ///
    /// This isolates:
    ///
    /// hidden -> Linear(64, 256) -> activation -> Linear(256, 64)
    ///
    /// No attention, no LayerNorm, no residual, no LM head.
    /// </summary>
    [Config(typeof(BenchmarkConfig))]
    [MemoryDiagnoser]
    public class SlmCachedFeedForwardSmallBenchmark
    {
        private const int OperationsPerInvoke = 8_192;

        private const int DModel = 64;
        private const int DFF = 256;

        private CachedFeedForwardBlock _block = null!;

        private float[] _hidden = null!;
        private float[] _w1 = null!;
        private float[] _b1 = null!;
        private float[] _w2 = null!;
        private float[] _b2 = null!;
        private float[] _output = null!;

        private float _checksum;

        [Params(FeedForwardActivation.ReLU, FeedForwardActivation.GeLU)]
        public FeedForwardActivation Activation { get; set; }

        [GlobalSetup]
        public void Setup()
        {
            _block = new CachedFeedForwardBlock(
                DModel,
                DFF,
                Activation);

            _hidden = new float[DModel];
            _w1 = new float[DModel * DFF];
            _b1 = new float[DFF];
            _w2 = new float[DFF * DModel];
            _b2 = new float[DModel];
            _output = new float[DModel];

            FillDeterministic(_hidden, seed: 101);
            FillDeterministic(_w1, seed: 201);
            FillDeterministic(_b1, seed: 301);
            FillDeterministic(_w2, seed: 401);
            FillDeterministic(_b2, seed: 501);

            _block.Decode(
                _hidden,
                _w1,
                _b1,
                _w2,
                _b2,
                _output);
        }

        [Benchmark(OperationsPerInvoke = OperationsPerInvoke)]
        public float Decode_FeedForward_Small()
        {
            var checksum = 0f;

            for (var i = 0; i < OperationsPerInvoke; i++)
            {
                _block.Decode(
                    _hidden,
                    _w1,
                    _b1,
                    _w2,
                    _b2,
                    _output);

                checksum += _output[i & (DModel - 1)];
            }

            _checksum = checksum;
            return checksum;
        }

        [GlobalCleanup]
        public void Cleanup()
        {
            GC.KeepAlive(_checksum);
        }

        private static void FillDeterministic(float[] data, int seed)
        {
            var rng = new Random(seed);

            for (var i = 0; i < data.Length; i++)
            {
                data[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
            }
        }
    }
}
