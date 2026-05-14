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
    /// GPT-2 Small KV-cache benchmark — real model, real numbers.
    ///
    /// Compares:
    ///   Legacy:  full forward pass per token — O(N²), high GC pressure
    ///   Cached:  KV-cache decode per token — O(N), zero GC per token
    ///
    /// Requires:
    ///   test_fixtures/gpt2_small.bin  (generate with Scripts/convert_gpt2.py)
    ///
    /// Run:
    ///   dotnet run -c Release --project Sources/Benchmark --filter "*Gpt2KvCache*"
    ///
    /// Expected results (Ryzen 9 9950X3D):
    ///   Legacy  100 tokens: ~3-5s, ~500MB allocated, Gen2 collections
    ///   Cached  100 tokens: ~0.5-1s, ~0 B per token, 0 Gen collections
    /// </summary>
    [Config(typeof(BenchmarkConfig))]
    public class Gpt2KvCacheBenchmark : IDisposable
    {
        private const string CheckpointPath = "D:\\Overfit\\Tests\\test_fixtures\\gpt2_small.bin";
        private const int PromptLength = 8;

        private GPT1Model _model = null!;
        private SlmInferenceEngine _legacyEngine = null!;
        private CachedSlmInferenceEngine _cachedEngine = null!;
        private int[] _prompt = null!;
        private SamplingOptions _sampling;
        private GenerationOptions _legacyOptions;
        private int _checksum;
        private bool _disposed;

        [Params(16, 64, 128)]
        public int MaxNewTokens { get; set; }

        [GlobalSetup]
        public void Setup()
        {
            if (!File.Exists(CheckpointPath))
            {
                throw new FileNotFoundException(
                    $"GPT-2 checkpoint not found: {CheckpointPath}\n" +
                    "Run: python3 Scripts/convert_gpt2.py --size small --out test_fixtures/");
            }

            _model = new GPT1Model(Gpt2Config.Small);
            _model.Eval();

            using var fs = File.OpenRead(CheckpointPath);
            using var br = new BinaryReader(fs);
            _model.Load(br);

            _legacyEngine = SlmInferenceEngine.FromGpt1(_model);
            _cachedEngine = CachedSlmInferenceEngine.FromGpt1(_model);

            // Prompt: "The future of software development is"
            // Using hand-coded BPE token ids from GPT-2 tokenizer
            _prompt = [464, 2003, 286, 3788, 2478, 318];

            _sampling = SamplingOptions.Greedy;

            _legacyOptions = new GenerationOptions(
                maxNewTokens: MaxNewTokens,
                maxContextLength: Gpt2Config.Small.ContextLength,
                sampling: _sampling,
                stopOnEndOfTextToken: false);

            // Warmup — JIT + weight loading into cache
            using var warmupSession = _cachedEngine.CreateSession();
            warmupSession.Reset(_prompt.AsSpan());
            _checksum ^= warmupSession.GenerateNextToken(in _sampling);
        }

        /// <summary>
        /// Full forward pass per token. Context grows each step → O(N²).
        /// </summary>
        [Benchmark(Baseline = true)]
        public void Legacy_FullForwardPerToken()
        {
            using var session = _legacyEngine.CreateSession(
                Gpt2Config.Small.ContextLength);

            session.Reset(_prompt.AsSpan());

            for (var i = 0; i < MaxNewTokens; i++)
            {
                _checksum ^= session.GenerateNextToken(in _sampling);
            }
        }

        /// <summary>
        /// KV-cache decode. Only new token is processed each step → O(N).
        /// Should allocate 0 bytes per GenerateNextToken call.
        /// </summary>
        [Benchmark]
        public void Cached_KvCacheDecodePerToken()
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
    }
}
