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
    [Config(typeof(BenchmarkConfig))]
    public class GPT1SlmRuntimeSmallGenerationBenchmark : IDisposable
    {
        private const int PromptLength = 8;

        private static readonly GPT1Config Config = new()
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

        private GPT1Model _model = null!;
        private SlmInferenceEngine _engine = null!;
        private int[] _prompt = null!;
        private int[] _output = null!;
        private GenerationOptions _options;
        private SamplingOptions _sampling;
        private int _checksum;
        private bool _disposed;

        [Params(1, 10, 100)]
        public int MaxNewTokens { get; set; }

        [GlobalSetup]
        public void Setup()
        {
            _model = new GPT1Model(Config);
            _model.Eval();

            _engine = SlmInferenceEngine.FromGpt1(_model);

            _prompt = CreatePrompt(
                PromptLength,
                Config.VocabSize,
                seed: 123);

            _output = new int[MaxNewTokens];

            _sampling = SamplingOptions.Greedy;

            _options = new GenerationOptions(
                maxNewTokens: MaxNewTokens,
                maxContextLength: Config.ContextLength,
                sampling: _sampling,
                stopOnEndOfTextToken: false);

            Consume(_model.Generate(_prompt, maxNewTokens: 1));

            _ = _engine.Generate(
                _prompt,
                _output,
                in _options);
        }

        [Benchmark(Baseline = true)]
        public void Legacy_GPT1Model_Generate()
        {
            Consume(_model.Generate(
                _prompt,
                maxNewTokens: MaxNewTokens));
        }

        [Benchmark]
        public void RuntimeEngine_Generate()
        {
            Array.Clear(_output);

            var generated = _engine.Generate(
                _prompt,
                _output,
                in _options);

            Consume(_output, generated);
        }

        [Benchmark]
        public void RuntimeSession_GenerateNextToken_Loop()
        {
            using var session = _engine.CreateSession(Config.ContextLength);
            session.Reset(_prompt);

            for (var i = 0; i < MaxNewTokens; i++)
            {
                var token = session.GenerateNextToken(in _sampling);
                _checksum ^= token;
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

            _engine.Dispose();
            _model.Dispose();

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

        private void Consume(int[] values)
        {
            if (values.Length == 0)
            {
                return;
            }

            _checksum ^= values[0];
            _checksum ^= values.Length;
        }

        private void Consume(ReadOnlySpan<int> values, int length)
        {
            if (length <= 0)
            {
                return;
            }

            _checksum ^= values[0];
            _checksum ^= length;
        }
    }
}

