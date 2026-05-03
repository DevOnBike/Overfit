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
    /// GPT-1 style cached multi-head decode benchmark.
    ///
    /// Shape:
    ///
    /// dModel = 768
    /// heads = 12
    /// headDim = 64
    ///
    /// This is the relevant shape for the current GPT-1 style branch.
    /// </summary>
    [Config(typeof(BenchmarkConfig))]
    [MemoryDiagnoser]
    public class SlmCachedMultiHeadAttentionGpt1Benchmark : IDisposable
    {
        private const int OperationsPerInvoke = 512;

        private const int DModel = 768;
        private const int HeadCount = 12;

        private CachedMultiHeadAttention _decoder = null!;
        private KeyValueCache _cache = null!;

        private float[] _hidden = null!;
        private float[][] _wqHeads = null!;
        private float[][] _wkHeads = null!;
        private float[][] _wvHeads = null!;
        private float[][] _woHeads = null!;
        private float[] _outputBias = null!;
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

            _decoder = new CachedMultiHeadAttention(
                DModel,
                HeadCount,
                SequenceLength);

            _cache = KeyValueCache.Create(
                layerCount: 1,
                headCount: HeadCount,
                maxSequenceLength: SequenceLength,
                headDimension: _headDimension);

            _hidden = new float[DModel];
            _outputBias = new float[DModel];
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

            FillDeterministic(_hidden, seed: 101);
            FillDeterministic(_outputBias, seed: 202);

            PrefillCache(
                _cache,
                HeadCount,
                SequenceLength,
                _headDimension);

            _position = SequenceLength - 1;

            _decoder.Decode(
                _hidden,
                _wqHeads,
                _wkHeads,
                _wvHeads,
                _woHeads,
                _outputBias,
                _cache,
                layerIndex: 0,
                position: _position,
                _output);
        }

        [Benchmark(OperationsPerInvoke = OperationsPerInvoke)]
        public float Decode_MultiHead_Gpt1()
        {
            var checksum = 0f;

            for (var i = 0; i < OperationsPerInvoke; i++)
            {
                _decoder.Decode(
                    _hidden,
                    _wqHeads,
                    _wkHeads,
                    _wvHeads,
                    _woHeads,
                    _outputBias,
                    _cache,
                    layerIndex: 0,
                    position: _position,
                    _output);

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
