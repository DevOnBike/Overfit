// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using BenchmarkDotNet.Attributes;
using Benchmarks.Helpers;
using DevOnBike.Overfit.DeepLearning;

namespace Benchmarks
{
    /// <summary>
    /// Autoregressive generation benchmarks for GPT-style models.
    ///
    /// This benchmark is intentionally separate from GPT1InferenceBenchmark:
    ///
    /// - GPT1InferenceBenchmark measures full forward / GenerateLogits cost.
    /// - GPT1GenerationBenchmark measures repeated autoregressive decoding cost.
    ///
    /// This is the baseline we need before implementing KV cache.
    ///
    /// Current GPT1Model.Generate(...) recomputes the whole context for every new
    /// token and allocates intermediate arrays. That is expected before the SLM
    /// session / KV-cache runtime exists.
    ///
    /// Run:
    ///   dotnet run -c Release --project Sources/Benchmark --filter "*GPT1GenerationBenchmark*"
    /// </summary>
    [Config(typeof(BenchmarkConfig))]
    public class GPT1GenerationBenchmark : IDisposable
    {
        private const int FullPromptLength = 16;
        private const int SmallPromptLength = 8;

        private GPT1Model _smallModel = null!;
        private GPT1Model _fullModel = null!;

        private int[] _smallPrompt = null!;
        private int[] _fullPrompt = null!;

        private int _checksum;
        private bool _disposed;

        private static readonly GPT1Config SmallConfig = new()
        {
            VocabSize = 256,
            ContextLength = 16,
            DModel = 64,
            NHeads = 4,
            NLayers = 2,
            DFF = 256,
            TieWeights = true,
            PreLayerNorm = true
        };

        private static readonly GPT1Config FullConfig = new()
        {
            VocabSize = 40478,
            ContextLength = 512,
            DModel = 768,
            NHeads = 12,
            NLayers = 12,
            DFF = 3072,
            TieWeights = true,
            PreLayerNorm = true
        };

        [GlobalSetup]
        public void Setup()
        {
            _smallModel = new GPT1Model(SmallConfig);
            _smallModel.Eval();

            _fullModel = new GPT1Model(FullConfig);
            _fullModel.Eval();

            _smallPrompt = CreatePrompt(
                SmallPromptLength,
                SmallConfig.VocabSize,
                seed: 123);

            _fullPrompt = CreatePrompt(
                FullPromptLength,
                FullConfig.VocabSize,
                seed: 456);

            // Warmup. Keep this small; full GPT-1 GenerateLogits is already heavy.
            Consume(_smallModel.GenerateLogits(_smallPrompt));
            Consume(_smallModel.Generate(_smallPrompt, maxNewTokens: 1));
            Consume(_fullModel.GenerateLogits(_fullPrompt));
        }

        /// <summary>
        /// Small GPT config, one next-token logits call.
        ///
        /// Useful for isolating the current non-cached single-step cost on a cheap
        /// model before adding SlmSession / KV cache.
        /// </summary>
        [Benchmark]
        public void Small_GenerateLogits_Context8()
        {
            Consume(_smallModel.GenerateLogits(_smallPrompt));
        }

        /// <summary>
        /// Small GPT config, generate 1 token.
        ///
        /// This includes:
        /// - tokens.ToArray() / context construction in GPT1Model.Generate(...)
        /// - GenerateLogits(...)
        /// - greedy argmax
        /// - generated token array allocation
        /// </summary>
        [Benchmark]
        public void Small_Generate_1Token()
        {
            Consume(_smallModel.Generate(_smallPrompt, maxNewTokens: 1));
        }

        /// <summary>
        /// Small GPT config, generate 10 tokens.
        ///
        /// This is the first practical autoregressive baseline before KV cache.
        /// The current implementation recomputes the full context on every token.
        /// </summary>
        [Benchmark]
        public void Small_Generate_10Tokens()
        {
            Consume(_smallModel.Generate(_smallPrompt, maxNewTokens: 10));
        }

        /// <summary>
        /// Small GPT config, generate 100 tokens.
        ///
        /// This intentionally stresses repeated decode and context-window truncation.
        /// It should show why KV cache and a stateful session are required.
        /// </summary>
        [Benchmark]
        public void Small_Generate_100Tokens()
        {
            Consume(_smallModel.Generate(_smallPrompt, maxNewTokens: 100));
        }

        /// <summary>
        /// Full GPT-1 style config, one next-token logits call.
        ///
        /// This corresponds to the heavy path from GPT1InferenceBenchmark but uses
        /// a fixed prompt length for generation-oriented comparison.
        /// </summary>
        [Benchmark]
        public void Full_GenerateLogits_Context16()
        {
            Consume(_fullModel.GenerateLogits(_fullPrompt));
        }

        /// <summary>
        /// Full GPT-1 style config, generate only 1 token.
        ///
        /// Do not add 10/100-token full GPT-1 benchmarks yet. Without KV cache they
        /// are too slow and mostly repeat the same known bottleneck.
        /// </summary>
        [Benchmark]
        public void Full_Generate_1Token()
        {
            Consume(_fullModel.Generate(_fullPrompt, maxNewTokens: 1));
        }

        [GlobalCleanup]
        public void Cleanup()
        {
            if (_disposed)
            {
                return;
            }

            _disposed = true;

            _smallModel.Dispose();
            _fullModel.Dispose();

            // Make sure benchmark results cannot be optimized away completely.
            GC.KeepAlive(_checksum);
        }

        public void Dispose()
        {
            Cleanup();
        }

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

        private void Consume(float[] values)
        {
            if (values.Length > 0)
            {
                _checksum ^= (int)values[0];
                _checksum ^= values.Length;
            }
        }

        private void Consume(int[] values)
        {
            if (values.Length > 0)
            {
                _checksum ^= values[0];
                _checksum ^= values.Length;
            }
        }
    }
}
