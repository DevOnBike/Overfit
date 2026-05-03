// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
//
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using BenchmarkDotNet.Attributes;
using Benchmarks.Helpers;
using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.LanguageModels.Runtime;

namespace Benchmarks
{
    /// <summary>
    /// GPT-1 style cached adapter benchmark.
    ///
    /// Shape:
    ///
    /// vocab = 40478
    /// context = 512
    /// dModel = 768
    /// heads = 12
    /// layers = 12
    /// dFF = 3072
    ///
    /// Measures:
    ///
    /// token id
    /// -> token embedding
    /// -> position embedding
    /// -> CachedGptStack
    /// -> logits
    ///
    /// Not included:
    ///
    /// - token sampling,
    /// - tokenizer,
    /// - SlmSession.
    /// </summary>
    [Config(typeof(BenchmarkConfig))]
    [MemoryDiagnoser]
    public class SlmCachedGpt1AdapterGpt1Benchmark : IDisposable
    {
        private GPT1Model _model = null!;
        private CachedGpt1ModelAdapter _adapter = null!;
        private int[] _tokens = null!;
        private float[] _logits = null!;
        private float _checksum;
        private bool _disposed;

        [Params(1, 10)]
        public int TokenCount { get; set; }

        [GlobalSetup]
        public void Setup()
        {
            var config = new GPT1Config
            {
                VocabSize = 40_478,
                ContextLength = 512,
                DModel = 768,
                NHeads = 12,
                NLayers = 12,
                DFF = 3072,
                TieWeights = true,
                PreLayerNorm = true
            };

            _model = new GPT1Model(config);
            _model.Eval();

            _adapter = new CachedGpt1ModelAdapter(_model);
            _tokens = CreateTokens(TokenCount, config.VocabSize, seed: 456);
            _logits = new float[config.VocabSize];

            RunDecode();
        }

        [Benchmark]
        public float Decode_Tokens_Gpt1()
        {
            _adapter.Reset();

            var checksum = RunDecode();

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

            _adapter.Dispose();
            _model.Dispose();

            GC.KeepAlive(_checksum);
        }

        public void Dispose()
        {
            Cleanup();
        }

        private float RunDecode()
        {
            var checksum = 0f;

            for (var i = 0; i < _tokens.Length; i++)
            {
                _adapter.DecodeNextToken(
                    _tokens[i],
                    _logits);

                checksum += _logits[i % _logits.Length];
            }

            return checksum;
        }

        private static int[] CreateTokens(
            int count,
            int vocabSize,
            int seed)
        {
            var tokens = new int[count];
            var rng = new Random(seed);

            for (var i = 0; i < tokens.Length; i++)
            {
                tokens[i] = rng.Next(0, vocabSize);
            }

            return tokens;
        }
    }
}
