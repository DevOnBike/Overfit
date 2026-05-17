// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using BenchmarkDotNet.Attributes;
using Benchmarks.Helpers;
using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.LanguageModels.Contracts;
using DevOnBike.Overfit.LanguageModels.Runtime;

namespace Benchmarks
{
    /// <summary>
    /// GPT-1 style session benchmark:
    ///
    /// legacy SlmSession
    /// vs
    /// CachedSlmSession
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
    /// prompt prefill + Generate(maxNewTokens)
    ///
    /// This benchmark is intentionally limited to 1 and 10 generated tokens.
    /// The legacy path is slow and allocation-heavy.
    /// </summary>
    [Config(typeof(BenchmarkConfig))]
    [MemoryDiagnoser]
    public class SlmCachedVsLegacySessionGpt1Benchmark : IDisposable
    {
        private const int PromptLength = 16;

        private GPT1Model _model = null!;
        private SlmInferenceEngine _legacyEngine = null!;
        private ISlmSession _legacySession = null!;
        private CachedSlmSession _cachedSession = null!;

        private int[] _prompt = null!;
        private int[] _legacyOutput = null!;
        private int[] _cachedOutput = null!;
        private GenerationOptions _options;
        private int _checksum;
        private bool _disposed;

        [Params(1, 10)]
        public int MaxNewTokens { get; set; }

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

            _legacyEngine = SlmInferenceEngine.FromGpt1(_model);
            _legacySession = _legacyEngine.CreateSession(config.ContextLength);
            _cachedSession = new CachedSlmSession(_model);

            _prompt = CreateTokens(
                PromptLength,
                config.VocabSize,
                seed: 456);

            _legacyOutput = new int[MaxNewTokens];
            _cachedOutput = new int[MaxNewTokens];

            _options = new GenerationOptions(
                maxNewTokens: MaxNewTokens,
                maxContextLength: config.ContextLength,
                sampling: SamplingOptions.Greedy,
                stopOnEndOfTextToken: false);

            _legacySession.Generate(
                _prompt,
                _legacyOutput,
                in _options);

            _cachedSession.Generate(
                _prompt,
                _cachedOutput,
                in _options);
        }

        [Benchmark(Baseline = true)]
        public int LegacySession_Generate()
        {
            Array.Clear(_legacyOutput);

            var generated = _legacySession.Generate(
                _prompt,
                _legacyOutput,
                in _options);

            _checksum ^= generated;
            _checksum ^= _legacyOutput[0];

            return _checksum;
        }

        [Benchmark]
        public int CachedSession_Generate()
        {
            Array.Clear(_cachedOutput);

            var generated = _cachedSession.Generate(
                _prompt,
                _cachedOutput,
                in _options);

            _checksum ^= generated;
            _checksum ^= _cachedOutput[0];

            return _checksum;
        }

        [GlobalCleanup]
        public void Cleanup()
        {
            if (_disposed)
            {
                return;
            }

            _disposed = true;

            _cachedSession.Dispose();
            _legacySession.Dispose();
            _legacyEngine.Dispose();
            _model.Dispose();

            GC.KeepAlive(_checksum);
        }

        public void Dispose()
        {
            Cleanup();
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
