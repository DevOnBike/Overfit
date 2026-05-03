// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
//
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
    /// Small SLM continuation-only benchmark.
    ///
    /// This benchmark excludes prompt prefill from the measured region.
    ///
    /// Iteration setup:
    ///   session.Reset(prompt)
    ///
    /// Measured method:
    ///   GenerateNextToken(...) repeated MaxNewTokens times
    ///
    /// This separates steady-state continuation decode from first-request prompt
    /// prefill cost.
    /// </summary>
    [Config(typeof(BenchmarkConfig))]
    [MemoryDiagnoser]
    public class SlmCachedVsLegacyContinuationSmallBenchmark : IDisposable
    {
        private const int PromptLength = 8;

        private GPT1Model _model = null!;
        private SlmInferenceEngine _legacyEngine = null!;
        private ISlmSession _legacySession = null!;
        private CachedSlmSession _cachedSession = null!;

        private int[] _prompt = null!;
        private SamplingOptions _sampling;
        private int _checksum;
        private bool _disposed;

        [Params(1, 10, 100)]
        public int MaxNewTokens { get; set; }

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

            _legacyEngine = SlmInferenceEngine.FromGpt1(_model);
            _legacySession = _legacyEngine.CreateSession(config.ContextLength);
            _cachedSession = new CachedSlmSession(_model);

            _prompt = CreateTokens(
                PromptLength,
                config.VocabSize,
                seed: 123);

            _sampling = SamplingOptions.Greedy;
        }

        [IterationSetup(Target = nameof(LegacySession_GenerateNextToken_ContinuationOnly))]
        public void SetupLegacyContinuation()
        {
            _legacySession.Reset(_prompt);
        }

        [IterationSetup(Target = nameof(CachedSession_GenerateNextToken_ContinuationOnly))]
        public void SetupCachedContinuation()
        {
            _cachedSession.Reset(_prompt);
        }

        [Benchmark(Baseline = true)]
        public int LegacySession_GenerateNextToken_ContinuationOnly()
        {
            var checksum = 0;

            for (var i = 0; i < MaxNewTokens; i++)
            {
                checksum ^= _legacySession.GenerateNextToken(in _sampling);
            }

            _checksum ^= checksum;
            return _checksum;
        }

        [Benchmark]
        public int CachedSession_GenerateNextToken_ContinuationOnly()
        {
            var checksum = 0;

            for (var i = 0; i < MaxNewTokens; i++)
            {
                checksum ^= _cachedSession.GenerateNextToken(in _sampling);
            }

            _checksum ^= checksum;
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
