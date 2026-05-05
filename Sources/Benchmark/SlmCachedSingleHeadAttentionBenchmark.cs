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
    /// Benchmarks cached single-head decode:
    ///
    /// hidden -> Q/K/V -> write K/V cache -> attention -> O projection
    ///
    /// This does not call GPT1Model and does not include LayerNorm, residuals,
    /// FFN or LM head. It isolates one cached attention head.
    /// </summary>
    [Config(typeof(BenchmarkConfig))]
    [MemoryDiagnoser]
    public class SlmCachedSingleHeadAttentionBenchmark : IDisposable
    {
        private const int OperationsPerInvoke = 8_192;

        private CachedSingleHeadAttention _decoder = null!;
        private KeyValueCache _cache = null!;

        private float[] _hidden = null!;
        private float[] _wq = null!;
        private float[] _wk = null!;
        private float[] _wv = null!;
        private float[] _wo = null!;
        private float[] _output = null!;

        private int _position;
        private float _checksum;
        private bool _disposed;

        [Params(64, 768)]
        public int DModel { get; set; }

        [Params(64)]
        public int HeadDimension { get; set; }

        [Params(16, 64, 256, 512)]
        public int SequenceLength { get; set; }

        [GlobalSetup]
        public void Setup()
        {
            if (HeadDimension > DModel)
            {
                throw new InvalidOperationException("HeadDimension cannot exceed DModel.");
            }

            _decoder = new CachedSingleHeadAttention(
                DModel,
                HeadDimension,
                SequenceLength);

            _cache = KeyValueCache.Create(
                layerCount: 1,
                headCount: 1,
                maxSequenceLength: SequenceLength,
                headDimension: HeadDimension);

            _hidden = new float[DModel];
            _wq = new float[DModel * HeadDimension];
            _wk = new float[DModel * HeadDimension];
            _wv = new float[DModel * HeadDimension];
            _wo = new float[HeadDimension * DModel];
            _output = new float[DModel];

            FillDeterministic(_hidden, seed: 101);
            FillDeterministic(_wq, seed: 201);
            FillDeterministic(_wk, seed: 301);
            FillDeterministic(_wv, seed: 401);
            FillDeterministic(_wo, seed: 501);

            PrefillCache(
                _cache,
                SequenceLength,
                HeadDimension);

            _position = SequenceLength - 1;

            _decoder.DecodeWithoutOutputBias(
                _hidden,
                _wq,
                _wk,
                _wv,
                ReadOnlySpan<float>.Empty,  // bq
                ReadOnlySpan<float>.Empty,  // bk
                ReadOnlySpan<float>.Empty,  // bv
                _wo,
                _cache,
                0,  // layerIndex
                0,  // headIndex
                _position,  // position
                _output);
        }

        [Benchmark(OperationsPerInvoke = OperationsPerInvoke)]
        public float Decode_SingleHead()
        {
            var checksum = 0f;

            for (var i = 0; i < OperationsPerInvoke; i++)
            {
                _decoder.DecodeWithoutOutputBias(
                    _hidden,
                    _wq,
                    _wk,
                    _wv,
                    ReadOnlySpan<float>.Empty,  // bq
                    ReadOnlySpan<float>.Empty,  // bk
                    ReadOnlySpan<float>.Empty,  // bv
                    _wo,
                    _cache,
                    0,  // layerIndex
                    0,  // headIndex
                    _position,  // position
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
            int sequenceLength,
            int headDimension)
        {
            for (var position = 0; position < sequenceLength; position++)
            {
                var key = cache.GetKeyWriteSpan(0, 0, position);
                var value = cache.GetValueWriteSpan(0, 0, position);

                for (var i = 0; i < headDimension; i++)
                {
                    key[i] = ((position + 1) * 0.001f) + (i * 0.0001f);
                    value[i] = ((position + 1) * 0.002f) - (i * 0.0001f);
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
