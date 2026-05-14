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
    /// End-to-end GPT-2 Small inference benchmark using real HuggingFace weights.
    ///
    /// Measures the full generation loop:
    ///   prompt prefill → N × GenerateNextToken → tokens/sec
    ///
    /// Three paths compared:
    ///
    ///   Legacy   — SlmSession: full O(N²) forward pass per token.
    ///              Allocates per step (new context array + ComputationGraph).
    ///
    ///   Cached   — CachedSlmSession: KV-cache O(N) decode.
    ///              Zero allocations per token after session creation.
    ///              Weight storage: zero-copy TensorStorage references.
    ///
    ///   Prefill  — CachedSlmSession prefill only (prompt → first token).
    ///              Isolates the one-time prompt cost vs steady-state decode.
    ///
    /// Requires:
    ///   test_fixtures/gpt2_small.bin
    ///   python3 Scripts/convert_gpt2.py --size small --out test_fixtures/
    ///
    /// Run:
    ///   dotnet run -c Release --project Sources/Benchmark --filter "*Gpt2Tokens*"
    ///
    /// Key metric: Mean / MaxNewTokens = ms per token.
    /// </summary>
    [Config(typeof(BenchmarkConfig))]
    [MemoryDiagnoser]
    public class Gpt2TokensPerSecondBenchmark : IDisposable
    {
        private const string CheckpointPath = "test_fixtures/gpt2_small.bin";

        // "The future of software development is" — GPT-2 BPE token IDs
        private static readonly int[] Prompt = [464, 2003, 286, 3788, 2478, 318];

        private GPT1Model                _model        = null!;
        private SlmInferenceEngine       _legacyEngine = null!;
        private CachedSlmInferenceEngine _cachedEngine = null!;
        private SamplingOptions          _sampling;
        private GenerationOptions        _legacyOptions;
        private int                      _checksum;
        private bool                     _disposed;

        [Params(16, 64, 128)]
        public int MaxNewTokens { get; set; }

        [GlobalSetup]
        public void Setup()
        {
            if (!File.Exists(CheckpointPath))
            {
                throw new FileNotFoundException(
                $"GPT-2 checkpoint not found at '{CheckpointPath}'. " +
                "Run: python3 Scripts/convert_gpt2.py --size small --out test_fixtures/");
            }

            _model = new GPT1Model(Gpt2Config.Small);
            _model.Eval();

            using var fs = File.OpenRead(CheckpointPath);
            using var br = new BinaryReader(fs);
            _model.Load(br);

            _legacyEngine = SlmInferenceEngine.FromGpt1(_model);
            _cachedEngine = CachedSlmInferenceEngine.FromGpt1(_model);
            _sampling     = SamplingOptions.Greedy;

            _legacyOptions = new GenerationOptions(
                maxNewTokens: MaxNewTokens,
                maxContextLength: Gpt2Config.Small.ContextLength,
                sampling: _sampling,
                stopOnEndOfTextToken: false);

            // Warmup — JIT + weight data into L3 cache
            using var warmup = _cachedEngine.CreateSession();
            warmup.Reset(Prompt.AsSpan());
            _checksum ^= warmup.GenerateNextToken(in _sampling);
        }

        /// <summary>
        /// Naive path: full forward pass per token. O(N²) allocations.
        /// Included as baseline only — not intended for production use.
        /// </summary>
        [Benchmark(Baseline = true)]
        public int Legacy_FullForwardPerToken()
        {
            using var session = _legacyEngine.CreateSession(Gpt2Config.Small.ContextLength);
            session.Reset(Prompt.AsSpan());

            for (var i = 0; i < MaxNewTokens; i++)
            {
                _checksum ^= session.GenerateNextToken(in _sampling);
            }

            return _checksum;
        }

        /// <summary>
        /// KV-cache path: decode only the new token per step. O(N) allocations.
        /// Zero allocations per GenerateNextToken call.
        /// </summary>
        [Benchmark]
        public int Cached_KvCacheDecode()
        {
            using var session = _cachedEngine.CreateSession();
            session.Reset(Prompt.AsSpan());

            for (var i = 0; i < MaxNewTokens; i++)
            {
                _checksum ^= session.GenerateNextToken(in _sampling);
            }

            return _checksum;
        }

        /// <summary>
        /// Measures prompt prefill cost only (first token).
        /// Use to separate prefill latency from steady-state decode latency.
        /// </summary>
        [Benchmark]
        public int Cached_PrefillOnly()
        {
            using var session = _cachedEngine.CreateSession();
            session.Reset(Prompt.AsSpan());
            _checksum ^= session.GenerateNextToken(in _sampling);
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

            _cachedEngine.Dispose();
            _legacyEngine.Dispose();
            _model.Dispose();

            GC.KeepAlive(_checksum);
        }

        public void Dispose() => Cleanup();
    }
}
