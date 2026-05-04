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
    /// GPT-1 style cached GPT stack benchmark.
    ///
    /// Shape:
    ///
    /// layers = 12
    /// dModel = 768
    /// heads = 12
    /// dFF = 3072
    /// vocab = 40478
    ///
    /// Measures one cached decode step:
    ///
    /// inputHidden
    /// -> 12 x CachedTransformerBlock
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
    public class SlmCachedGptStackGpt1Benchmark : IDisposable
    {
        private const int OperationsPerInvoke = 16;

        private const int LayerCount = 12;
        private const int DModel = 768;
        private const int HeadCount = 12;
        private const int DFF = 3072;
        private const int VocabSize = 40_478;

        private CachedGptStack _stack = null!;
        private KeyValueCache _cache = null!;

        private float[] _inputHidden = null!;
        private float[][] _ln1Gammas = null!;
        private float[][] _ln1Betas = null!;
        private IReadOnlyList<float[]>[] _wqHeadsByLayer = null!;
        private IReadOnlyList<float[]>[] _wkHeadsByLayer = null!;
        private IReadOnlyList<float[]>[] _wvHeadsByLayer = null!;
        private IReadOnlyList<float[]>[] _woHeadsByLayer = null!;
        private float[][] _attentionOutputBiases = null!;
        private float[][] _ln2Gammas = null!;
        private float[][] _ln2Betas = null!;
        private float[][] _ffnW1ByLayer = null!;
        private float[][] _ffnB1ByLayer = null!;
        private float[][] _ffnW2ByLayer = null!;
        private float[][] _ffnB2ByLayer = null!;
        private float[] _finalLayerNormGamma = null!;
        private float[] _finalLayerNormBeta = null!;
        private float[] _lmHeadWeights = null!;
        private float[] _lmHeadBias = null!;
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
                headCount: HeadCount,
                maxSequenceLength: SequenceLength,
                headDimension: _headDimension);

            _inputHidden = new float[DModel];
            _ln1Gammas = CreateLayerVectors(LayerCount, DModel, fillOnes: true, seedBase: 1000);
            _ln1Betas = CreateLayerVectors(LayerCount, DModel, fillOnes: false, seedBase: 2000);
            _attentionOutputBiases = CreateLayerVectors(LayerCount, DModel, fillOnes: false, seedBase: 3000);
            _ln2Gammas = CreateLayerVectors(LayerCount, DModel, fillOnes: true, seedBase: 4000);
            _ln2Betas = CreateLayerVectors(LayerCount, DModel, fillOnes: false, seedBase: 5000);
            _ffnW1ByLayer = CreateLayerVectors(LayerCount, DModel * DFF, fillOnes: false, seedBase: 6000);
            _ffnB1ByLayer = CreateLayerVectors(LayerCount, DFF, fillOnes: false, seedBase: 7000);
            _ffnW2ByLayer = CreateLayerVectors(LayerCount, DFF * DModel, fillOnes: false, seedBase: 8000);
            _ffnB2ByLayer = CreateLayerVectors(LayerCount, DModel, fillOnes: false, seedBase: 9000);

            _wqHeadsByLayer = CreateHeadsByLayer(LayerCount, HeadCount, DModel * _headDimension, seedBase: 10_000);
            _wkHeadsByLayer = CreateHeadsByLayer(LayerCount, HeadCount, DModel * _headDimension, seedBase: 20_000);
            _wvHeadsByLayer = CreateHeadsByLayer(LayerCount, HeadCount, DModel * _headDimension, seedBase: 30_000);
            _woHeadsByLayer = CreateHeadsByLayer(LayerCount, HeadCount, _headDimension * DModel, seedBase: 40_000);

            _finalLayerNormGamma = new float[DModel];
            _finalLayerNormBeta = new float[DModel];
            _lmHeadWeights = new float[DModel * VocabSize];
            _lmHeadBias = new float[VocabSize];
            _logits = new float[VocabSize];

            FillDeterministic(_inputHidden, seed: 101);
            FillOnes(_finalLayerNormGamma);
            FillDeterministic(_finalLayerNormBeta, seed: 201);
            FillDeterministic(_lmHeadWeights, seed: 301);
            FillDeterministic(_lmHeadBias, seed: 401);

            PrefillCache(
                _cache,
                LayerCount,
                HeadCount,
                SequenceLength,
                _headDimension);

            _position = SequenceLength - 1;

            _stack.Decode(
                _inputHidden,
                _ln1Gammas,
                _ln1Betas,
                _wqHeadsByLayer,
                _wkHeadsByLayer,
                _wvHeadsByLayer,
                _woHeadsByLayer,
                _attentionOutputBiases,
                _ln2Gammas,
                _ln2Betas,
                _ffnW1ByLayer,
                _ffnB1ByLayer,
                _ffnW2ByLayer,
                _ffnB2ByLayer,
                _finalLayerNormGamma,
                _finalLayerNormBeta,
                _lmHeadWeights,
                _lmHeadBias,
                _cache,
                _position,
                _logits);
        }

        [Benchmark(OperationsPerInvoke = OperationsPerInvoke)]
        public float Decode_GptStack_Gpt1()
        {
            var checksum = 0f;

            for (var i = 0; i < OperationsPerInvoke; i++)
            {
                _stack.Decode(
                    _inputHidden,
                    _ln1Gammas,
                    _ln1Betas,
                    _wqHeadsByLayer,
                    _wkHeadsByLayer,
                    _wvHeadsByLayer,
                    _woHeadsByLayer,
                    _attentionOutputBiases,
                    _ln2Gammas,
                    _ln2Betas,
                    _ffnW1ByLayer,
                    _ffnB1ByLayer,
                    _ffnW2ByLayer,
                    _ffnB2ByLayer,
                    _finalLayerNormGamma,
                    _finalLayerNormBeta,
                    _lmHeadWeights,
                    _lmHeadBias,
                    _cache,
                    _position,
                    _logits);

                checksum += _logits[i % VocabSize];
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

        private static IReadOnlyList<float[]>[] CreateHeadsByLayer(
            int layerCount,
            int headCount,
            int length,
            int seedBase)
        {
            var layers = new IReadOnlyList<float[]>[layerCount];

            for (var layer = 0; layer < layerCount; layer++)
            {
                var heads = new float[headCount][];

                for (var head = 0; head < headCount; head++)
                {
                    heads[head] = new float[length];
                    FillDeterministic(heads[head], seedBase + layer * 100 + head);
                }

                layers[layer] = heads;
            }

            return layers;
        }

        private static float[][] CreateLayerVectors(
            int layerCount,
            int length,
            bool fillOnes,
            int seedBase)
        {
            var values = new float[layerCount][];

            for (var layer = 0; layer < layerCount; layer++)
            {
                values[layer] = new float[length];

                if (fillOnes)
                {
                    FillOnes(values[layer]);
                }
                else
                {
                    FillDeterministic(values[layer], seedBase + layer);
                }
            }

            return values;
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
