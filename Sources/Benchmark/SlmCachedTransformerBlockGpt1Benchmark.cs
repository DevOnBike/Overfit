// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using BenchmarkDotNet.Attributes;
using Benchmarks.Helpers;
using DevOnBike.Overfit.LanguageModels.Runtime;

namespace Benchmarks
{
    /// <summary>
    /// GPT-1 style cached transformer block benchmark.
    ///
    /// Shape:
    ///
    /// dModel = 768
    /// heads = 12
    /// headDim = 64
    /// dFF = 3072
    ///
    /// Measures one cached transformer layer for one token:
    ///
    /// LN1 -> cached MHA -> residual -> LN2 -> FFN -> residual
    ///
    /// No embedding, no LM head, no full GPT stack.
    /// </summary>
    [Config(typeof(BenchmarkConfig))]
    [MemoryDiagnoser]
    public class SlmCachedTransformerBlockGpt1Benchmark : IDisposable
    {
        private const int OperationsPerInvoke = 128;

        private const int DModel = 768;
        private const int HeadCount = 12;
        private const int DFF = 3072;

        private CachedTransformerBlock _block = null!;
        private KeyValueCache _cache = null!;

        private float[] _input = null!;
        private BlockWeights _blockWeights;
        private float[] _output = null!;

        private int _headDimension;
        private int _position;
        private float _checksum;
        private bool _disposed;

        [Params(16, 64, 256, 512)]
        public int SequenceLength
        {
            get; set;
        }

        [GlobalSetup]
        public void Setup()
        {
            _headDimension = DModel / HeadCount;

            _block = new CachedTransformerBlock(
                DModel,
                HeadCount,
                DFF,
                SequenceLength,
                layerNormEpsilon: 1e-5f,
                feedForwardActivation: FeedForwardActivation.GeLU);

            _cache = KeyValueCache.Create(
                layerCount: 1,
                kvHeadCount: HeadCount,
                maxSequenceLength: SequenceLength,
                headDimension: _headDimension);

            _input = new float[DModel];
            _output = new float[DModel];

            var heads = new SingleHeadWeights[HeadCount];
            for (var h = 0; h < HeadCount; h++)
            {
                var wq = new float[DModel * _headDimension];
                FillDeterministic(wq, 1000 + h);
                var wk = new float[DModel * _headDimension];
                FillDeterministic(wk, 2000 + h);
                var wv = new float[DModel * _headDimension];
                FillDeterministic(wv, 3000 + h);
                var wo = new float[_headDimension * DModel];
                FillDeterministic(wo, 4000 + h);
                heads[h] = new SingleHeadWeights(wq: wq, wk: wk, wv: wv, wo: wo);
            }

            var ln1Gamma = new float[DModel];
            FillAffineGamma(ln1Gamma);
            var ln2Gamma = new float[DModel];
            FillAffineGamma(ln2Gamma);
            var ffnW1 = new float[DModel * DFF];
            FillDeterministic(ffnW1, 501);
            var ffnW2 = new float[DFF * DModel];
            FillDeterministic(ffnW2, 701);

            _blockWeights = new BlockWeights(
                heads: heads,
                ln1Gamma: ln1Gamma,
                ln2Gamma: ln2Gamma,
                ffnW1: ffnW1,
                ffnW2: ffnW2);

            FillDeterministic(_input, seed: 101);

            PrefillCache(_cache, HeadCount, SequenceLength, _headDimension);

            _position = SequenceLength - 1;

            _block.Decode(_input, in _blockWeights, _cache, 0, _position, _output);
        }

        [Benchmark(OperationsPerInvoke = OperationsPerInvoke)]
        public float Decode_TransformerBlock_Gpt1()
        {
            var checksum = 0f;

            for (var i = 0; i < OperationsPerInvoke; i++)
            {
                _block.Decode(_input, in _blockWeights, _cache, 0, _position, _output);

                checksum += _output[i % DModel];
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

        private static void FillAffineGamma(float[] data)
        {
            for (var i = 0; i < data.Length; i++)
            {
                data[i] = 1f;
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
