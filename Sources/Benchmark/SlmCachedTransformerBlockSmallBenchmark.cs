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
    /// Small SLM cached transformer block benchmark.
    ///
    /// Shape:
    ///
    /// dModel = 64
    /// heads = 4
    /// headDim = 16
    /// dFF = 256
    ///
    /// Measures one cached transformer layer for one token:
    ///
    /// LN1 -> cached MHA -> residual -> LN2 -> FFN -> residual
    ///
    /// No embedding, no LM head, no full GPT stack.
    /// </summary>
    [Config(typeof(BenchmarkConfig))]
    [MemoryDiagnoser]
    public class SlmCachedTransformerBlockSmallBenchmark : IDisposable
    {
        private const int OperationsPerInvoke = 4_096;

        private const int DModel = 64;
        private const int HeadCount = 4;
        private const int DFF = 256;

        private CachedTransformerBlock _block = null!;
        private KeyValueCache _cache = null!;

        private float[] _input = null!;
        private float[] _ln1Gamma = null!;
        private float[] _ln1Beta = null!;
        private float[][] _wqHeads = null!;
        private float[][] _wkHeads = null!;
        private float[][] _wvHeads = null!;
        private float[][] _woHeads = null!;
        private float[] _attentionOutputBias = null!;
        private float[] _ln2Gamma = null!;
        private float[] _ln2Beta = null!;
        private float[] _ffnW1 = null!;
        private float[] _ffnB1 = null!;
        private float[] _ffnW2 = null!;
        private float[] _ffnB2 = null!;
        private float[] _output = null!;

        private int _headDimension;
        private int _position;
        private float _checksum;
        private bool _disposed;

        [Params(16, 64, 256, 512)]
        public int SequenceLength { get; set; }

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
                headCount: HeadCount,
                maxSequenceLength: SequenceLength,
                headDimension: _headDimension);

            _input = new float[DModel];
            _ln1Gamma = new float[DModel];
            _ln1Beta = new float[DModel];
            _attentionOutputBias = new float[DModel];
            _ln2Gamma = new float[DModel];
            _ln2Beta = new float[DModel];
            _ffnW1 = new float[DModel * DFF];
            _ffnB1 = new float[DFF];
            _ffnW2 = new float[DFF * DModel];
            _ffnB2 = new float[DModel];
            _output = new float[DModel];

            _wqHeads = CreateHeadWeights(
                HeadCount,
                DModel * _headDimension,
                seedBase: 1000);

            _wkHeads = CreateHeadWeights(
                HeadCount,
                DModel * _headDimension,
                seedBase: 2000);

            _wvHeads = CreateHeadWeights(
                HeadCount,
                DModel * _headDimension,
                seedBase: 3000);

            _woHeads = CreateHeadWeights(
                HeadCount,
                _headDimension * DModel,
                seedBase: 4000);

            FillDeterministic(_input, seed: 101);
            FillAffineGamma(_ln1Gamma);
            FillDeterministic(_ln1Beta, seed: 201);
            FillDeterministic(_attentionOutputBias, seed: 301);
            FillAffineGamma(_ln2Gamma);
            FillDeterministic(_ln2Beta, seed: 401);
            FillDeterministic(_ffnW1, seed: 501);
            FillDeterministic(_ffnB1, seed: 601);
            FillDeterministic(_ffnW2, seed: 701);
            FillDeterministic(_ffnB2, seed: 801);

            PrefillCache(
                _cache,
                HeadCount,
                SequenceLength,
                _headDimension);

            _position = SequenceLength - 1;

            _block.Decode(
                _input,
                _ln1Gamma,
                _ln1Beta,
                _wqHeads,
                _wkHeads,
                _wvHeads,
                Array.Empty<float[]>(),  // bqHeads
                Array.Empty<float[]>(),  // bkHeads
                Array.Empty<float[]>(),  // bvHeads
                _woHeads,
                _attentionOutputBias,
                _ln2Gamma,
                _ln2Beta,
                _ffnW1,
                _ffnB1,
                _ffnW2,
                _ffnB2,
                _cache,
                0,  // layerIndex
                _position,  // position
                _output);
        }

        [Benchmark(OperationsPerInvoke = OperationsPerInvoke)]
        public float Decode_TransformerBlock_Small()
        {
            var checksum = 0f;

            for (var i = 0; i < OperationsPerInvoke; i++)
            {
                _block.Decode(
                    _input,
                    _ln1Gamma,
                    _ln1Beta,
                    _wqHeads,
                    _wkHeads,
                    _wvHeads,
                    Array.Empty<float[]>(),  // bqHeads
                    Array.Empty<float[]>(),  // bkHeads
                    Array.Empty<float[]>(),  // bvHeads
                    _woHeads,
                    _attentionOutputBias,
                    _ln2Gamma,
                    _ln2Beta,
                    _ffnW1,
                    _ffnB1,
                    _ffnW2,
                    _ffnB2,
                    _cache,
                    0,  // layerIndex
                    _position,  // position
                    _output);

                checksum += _output[i & (DModel - 1)];
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

        private static float[][] CreateHeadWeights(
            int headCount,
            int length,
            int seedBase)
        {
            var weights = new float[headCount][];

            for (var h = 0; h < headCount; h++)
            {
                weights[h] = new float[length];
                FillDeterministic(weights[h], seedBase + h);
            }

            return weights;
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
