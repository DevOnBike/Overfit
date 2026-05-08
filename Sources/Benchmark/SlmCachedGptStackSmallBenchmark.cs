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
    /// Small SLM cached GPT stack benchmark.
    ///
    /// Shape:
    ///
    /// layers = 2
    /// dModel = 64
    /// heads = 4
    /// dFF = 256
    /// vocab = 256
    ///
    /// Measures one cached decode step:
    ///
    /// inputHidden
    /// -> N x CachedTransformerBlock
    /// -> final LayerNorm
    /// -> LM head
    /// -> logits
    ///
    /// Not included:
    ///
    /// - token embedding,
    /// - positional embedding,
    /// - token sampling,
    /// - SlmSession.
    /// </summary>
    [Config(typeof(BenchmarkConfig))]
    [MemoryDiagnoser]
    public class SlmCachedGptStackSmallBenchmark : IDisposable
    {
        private const int OperationsPerInvoke = 2_048;

        private const int LayerCount = 2;
        private const int DModel = 64;
        private const int HeadCount = 4;
        private const int DFF = 256;
        private const int VocabSize = 256;

        private CachedGptStack _stack = null!;
        private KeyValueCache _cache = null!;

        private float[] _inputHidden = null!;
        private StackWeights _weights = null!;
        private float[] _logits = null!;

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

            _stack = new CachedGptStack(
                LayerCount,
                DModel,
                HeadCount,
                DFF,
                VocabSize,
                SequenceLength,
                layerNormEpsilon: 1e-5f,
                feedForwardActivation: FeedForwardActivation.GeLU);

            _cache = KeyValueCache.Create(
                layerCount: LayerCount,
                kvHeadCount: HeadCount,
                maxSequenceLength: SequenceLength,
                headDimension: _headDimension);

            _inputHidden = new float[DModel];
            _logits      = new float[VocabSize];
            FillDeterministic(_inputHidden, seed: 101);

            var finalNormGamma = CreateVector(DModel, fillOnes: true);
            var finalNormBeta  = CreateVector(DModel, seed: 201);
            var lmHead         = CreateVector(DModel * VocabSize, seed: 301);

            _weights = StackWeights.ForTest(
                LayerCount, HeadCount,
                l => new BlockWeights(
                    heads:         CreateHeadWeights(HeadCount, DModel, _headDimension, l),
                    ln1Gamma:      CreateVector(DModel, fillOnes: true,  seedBase: 1000 + l),
                    ln1Beta:       CreateVector(DModel,                  seedBase: 2000 + l),
                    attentionBias: CreateVector(DModel,                  seedBase: 3000 + l),
                    ln2Gamma:      CreateVector(DModel, fillOnes: true,  seedBase: 4000 + l),
                    ln2Beta:       CreateVector(DModel,                  seedBase: 5000 + l),
                    ffnW1:         CreateVector(DModel * DFF,            seedBase: 6000 + l),
                    ffnB1:         CreateVector(DFF,                     seedBase: 7000 + l),
                    ffnW2:         CreateVector(DFF * DModel,            seedBase: 8000 + l),
                    ffnB2:         CreateVector(DModel,                  seedBase: 9000 + l)),
                finalNormGamma: finalNormGamma,
                finalNormBeta:  finalNormBeta,
                lmHead:         lmHead);

            PrefillCache(
                _cache,
                LayerCount,
                HeadCount,
                SequenceLength,
                _headDimension);

            _position = SequenceLength - 1;

            _stack.Decode(
                    _inputHidden,
                    _weights,
                    _cache,
                    _position,
                    _logits);
        }

        [Benchmark(OperationsPerInvoke = OperationsPerInvoke)]
        public float Decode_GptStack_Small()
        {
            var checksum = 0f;

            for (var i = 0; i < OperationsPerInvoke; i++)
            {
                _stack.Decode(
                    _inputHidden,
                    _weights,
                    _cache,
                    _position,
                    _logits);

                checksum += _logits[i & (VocabSize - 1)];
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
            int layerCount,
            int headCount,
            int sequenceLength,
            int headDimension)
        {
            for (var position = 0; position < sequenceLength; position++)
            {
                for (var layer = 0; layer < layerCount; layer++)
                {
                    for (var head = 0; head < headCount; head++)
                    {
                        var key = cache.GetKeyWriteSpan(layer, head, position);
                        var value = cache.GetValueWriteSpan(layer, head, position);

                        for (var i = 0; i < headDimension; i++)
                        {
                            key[i] = ((position + 1) * 0.001f) + (layer * 0.02f) + (head * 0.01f) + (i * 0.0001f);
                            value[i] = ((position + 1) * 0.002f) - (layer * 0.02f) - (head * 0.01f) - (i * 0.0001f);
                        }
                    }
                }

                cache.Advance();
            }
        }

        private static SingleHeadWeights[] CreateHeadWeights(int headCount, int dModel, int headDim, int layer)
        {
            var heads = new SingleHeadWeights[headCount];
            for (var h = 0; h < headCount; h++)
            {
                var wq = new float[dModel * headDim]; FillDeterministic(wq, 10_000 + layer * 100 + h);
                var wk = new float[dModel * headDim]; FillDeterministic(wk, 20_000 + layer * 100 + h);
                var wv = new float[dModel * headDim]; FillDeterministic(wv, 30_000 + layer * 100 + h);
                var wo = new float[headDim * dModel]; FillDeterministic(wo, 40_000 + layer * 100 + h);
                heads[h] = new SingleHeadWeights(wq: wq, wk: wk, wv: wv, wo: wo);
            }
            return heads;
        }

        private static float[] CreateVector(int length, bool fillOnes = false, int seed = 0, int seedBase = 0)
        {
            var v = new float[length];
            if (fillOnes)
            {
                FillOnes(v);
            }
            else
            {
                FillDeterministic(v, seed + seedBase);
            }
            return v;
        }

        private static void FillOnes(float[] data)
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
