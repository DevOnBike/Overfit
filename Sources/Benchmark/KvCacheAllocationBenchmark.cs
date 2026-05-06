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
    /// Compares legacy SlmSession (full forward pass per token) vs CachedSlmSession (KV-cache).
    ///
    /// Key questions:
    ///   1. Does CachedSlmSession allocate 0 bytes per GenerateNextToken call?
    ///   2. Is CachedSlmSession faster for longer sequences?
    ///
    /// Expected results for KV-cache (CachedSlmSession):
    ///   - Allocated = 0 B per token (after warmup/setup)
    ///   - Gen0/Gen1/Gen2 = 0
    ///   - Speed scales as O(N) not O(N²) with MaxNewTokens
    ///
    /// Expected results for legacy (SlmSession):
    ///   - Allocated grows with MaxNewTokens (new context array + ComputationGraph per step)
    ///   - Speed scales as O(N²) — context grows each step
    ///
    /// Run:
    ///   dotnet run -c Release --project Sources/Benchmark --filter "*KvCacheAllocation*"
    /// </summary>
    [Config(typeof(BenchmarkConfig))]
    public class KvCacheAllocationBenchmark : IDisposable
    {
        private const int PromptLength = 8;

        private static readonly GPT1Config SmallConfig = new()
        {
            VocabSize = 256,
            ContextLength = 128,
            DModel = 64,
            NHeads = 4,
            NLayers = 2,
            DFF = 256,
            TieWeights = true,
            PreLayerNorm = true,
        };

        private GPT1Model _model = null!;
        private SlmInferenceEngine _legacyEngine = null!;
        private CachedSlmInferenceEngine _cachedEngine = null!;
        private int[] _prompt = null!;
        private int[] _outputBuffer = null!;
        private SamplingOptions _sampling;
        private GenerationOptions _legacyOptions;
        private int _checksum;
        private bool _disposed;

        [Params(1, 10, 100)]
        public int MaxNewTokens { get; set; }

        [GlobalSetup]
        public void Setup()
        {
            _model = new GPT1Model(SmallConfig);
            _model.Eval();

            _legacyEngine = SlmInferenceEngine.FromGpt1(_model);
            _cachedEngine = CachedSlmInferenceEngine.FromGpt1(_model);

            _prompt = CreatePrompt(PromptLength, SmallConfig.VocabSize, seed: 42);
            _outputBuffer = new int[MaxNewTokens];
            _sampling = SamplingOptions.Greedy;

            _legacyOptions = new GenerationOptions(
                maxNewTokens: MaxNewTokens,
                maxContextLength: SmallConfig.ContextLength,
                sampling: _sampling,
                stopOnEndOfTextToken: false);

            // Warmup — ensure JIT and internal caches are hot
            using var warmupLegacy = _legacyEngine.CreateSession(SmallConfig.ContextLength);
            warmupLegacy.Reset(_prompt.AsSpan());
            _checksum ^= warmupLegacy.GenerateNextToken(in _sampling);

            using var warmupCached = _cachedEngine.CreateSession();
            warmupCached.Reset(_prompt.AsSpan());
            _checksum ^= warmupCached.GenerateNextToken(in _sampling);
        }

        /// <summary>
        /// Legacy path: full forward pass per token.
        /// Allocates ~context.Length floats + ComputationGraph per step.
        /// Complexity: O(N²) in sequence length.
        /// </summary>
        [Benchmark(Baseline = true)]
        public void Legacy_SlmSession_GenerateNextToken()
        {
            using var session = _legacyEngine.CreateSession(SmallConfig.ContextLength);
            session.Reset(_prompt.AsSpan());

            for (var i = 0; i < MaxNewTokens; i++)
            {
                _checksum ^= session.GenerateNextToken(in _sampling);
            }
        }

        /// <summary>
        /// KV-cache path: decode only the new token, reuse K/V from previous steps.
        /// Should allocate 0 bytes per GenerateNextToken after session creation.
        /// Complexity: O(N) in sequence length.
        /// </summary>
        [Benchmark]
        public void Cached_CachedSlmSession_GenerateNextToken()
        {
            using var session = _cachedEngine.CreateSession();
            session.Reset(_prompt.AsSpan());

            for (var i = 0; i < MaxNewTokens; i++)
            {
                _checksum ^= session.GenerateNextToken(in _sampling);
            }
        }

        [GlobalCleanup]
        public void Cleanup()
        {
            if (_disposed)
            {
                return;
            }
            _disposed = true;

            _cachedEngine.Dispose();
            _legacyEngine.Dispose();
            _model.Dispose();

            GC.KeepAlive(_checksum);
        }

        public void Dispose() => Cleanup();

        private static int[] CreatePrompt(int length, int vocabularySize, int seed)
        {
            var prompt = new int[length];
            var rng = new Random(seed);
            for (var i = 0; i < prompt.Length; i++)
            {
                prompt[i] = rng.Next(0, vocabularySize);
            }
            return prompt;
        }
    }
}
