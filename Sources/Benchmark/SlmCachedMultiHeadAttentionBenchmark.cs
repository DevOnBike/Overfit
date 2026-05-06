// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using BenchmarkDotNet.Attributes;
using Benchmarks.Helpers;
using DevOnBike.Overfit.LanguageModels.Runtime;

namespace Benchmarks
{
    [Config(typeof(BenchmarkConfig))]
    public class SlmCachedMultiHeadAttentionBenchmark : IDisposable
    {
        private const int OperationsPerInvoke = 128;

        private CachedMultiHeadAttention _decoder = null!;
        private KeyValueCache _cache = null!;

        private float[] _hidden = null!;
        private BlockWeights _blockWeights;
        private float[] _output = null!;

        private int _headDimension;
        private int _position;
        private float _checksum;
        private bool _disposed;

        [Params(64, 768)]
        public int DModel { get; set; }

        [Params(4, 12)]
        public int HeadCount { get; set; }

        [Params(16, 64, 256, 512)]
        public int SequenceLength { get; set; }

        [GlobalSetup]
        public void Setup()
        {
            if (DModel % HeadCount != 0)
            {
                throw new InvalidOperationException("DModel must be divisible by HeadCount.");
            }

            _headDimension = DModel / HeadCount;

            _decoder = new CachedMultiHeadAttention(
            DModel,
            HeadCount,
            SequenceLength);

            _cache = KeyValueCache.Create(
            layerCount: 1,
            kvHeadCount: HeadCount,
            maxSequenceLength: SequenceLength,
            headDimension: _headDimension);

            _hidden = new float[DModel];
            _output = new float[DModel];

            var heads = new SingleHeadWeights[HeadCount];
            for (var h = 0; h < HeadCount; h++)
            {
                var wq = new float[DModel * _headDimension]; FillDeterministic(wq, 1000 + h);
                var wk = new float[DModel * _headDimension]; FillDeterministic(wk, 2000 + h);
                var wv = new float[DModel * _headDimension]; FillDeterministic(wv, 3000 + h);
                var wo = new float[_headDimension * DModel]; FillDeterministic(wo, 4000 + h);
                heads[h] = new SingleHeadWeights(wq: wq, wk: wk, wv: wv, wo: wo);
            }
            _blockWeights = new BlockWeights(heads: heads);

            FillDeterministic(_hidden, seed: 101);
            PrefillCache(_cache, HeadCount, SequenceLength, _headDimension);

            _position = SequenceLength - 1;

            _decoder.Decode(_hidden, in _blockWeights, _cache, 0, _position, _output);
        }

        [Benchmark(OperationsPerInvoke = OperationsPerInvoke)]
        public float Decode_MultiHead()
        {
            var checksum = 0f;

            for (var i = 0; i < OperationsPerInvoke; i++)
            {
                _decoder.Decode(_hidden, in _blockWeights, _cache, 0, _position, _output);

                checksum += _output[i % _output.Length];
            }

            _checksum = checksum;
            return checksum;
        }

        [GlobalCleanup]
        public void Cleanup()
        {
            if (_disposed)
            {
                return;
            }

            _disposed = true;
            _cache.Dispose();

            GC.KeepAlive(_checksum);
        }

        public void Dispose()
        {
            Cleanup();
        }

        private static void PrefillCache(
            KeyValueCache cache,
            int headCount,
            int sequenceLength,
            int headDimension)
        {
            for (var position = 0; position < sequenceLength; position++)
            {
                for (var head = 0; head < headCount; head++)
                {
                    var key = cache.GetKeyWriteSpan(0, head, position);
                    var value = cache.GetValueWriteSpan(0, head, position);

                    for (var i = 0; i < headDimension; i++)
                    {
                        key[i] = ((position + 1) * 0.001f) + (head * 0.01f) + (i * 0.0001f);
                        value[i] = ((position + 1) * 0.002f) - (head * 0.01f) - (i * 0.0001f);
                    }
                }

                cache.Advance();
            }
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