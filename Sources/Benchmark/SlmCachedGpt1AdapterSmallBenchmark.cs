// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using BenchmarkDotNet.Attributes;
using Benchmarks.Helpers;
using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.LanguageModels.Runtime;

namespace Benchmarks
{
    /// <summary>
    /// Small cached GPT-1 adapter benchmark.
    ///
    /// Shape:
    ///
    /// vocab = 256
    /// context = 128
    /// dModel = 64
    /// heads = 4
    /// layers = 2
    /// dFF = 256
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
    public class SlmCachedGpt1AdapterSmallBenchmark : IDisposable
    {
        private GPT1Model _model = null!;
        private CachedGpt1ModelAdapter _adapter = null!;
        private int[] _tokens = null!;
        private float[] _logits = null!;
        private float _checksum;
        private bool _disposed;

        [Params(1, 10, 100)]
        public int TokenCount
        {
            get; set;
        }

        [GlobalSetup]
        public void Setup()
        {
            var config = new GPT1Config
            {
                VocabSize = 256,
                ContextLength = 128,
                DModel = 64,
                NHeads = 4,
                NLayers = 2,
                DFF = 256,
                TieWeights = true,
                PreLayerNorm = true
            };

            _model = new GPT1Model(config);
            _model.Eval();

            _adapter = new CachedGpt1ModelAdapter(_model);
            _tokens = CreateTokens(TokenCount, config.VocabSize, seed: 123);
            _logits = new float[config.VocabSize];

            RunDecode();
        }

        [Benchmark]
        public float Decode_Tokens_Small()
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
