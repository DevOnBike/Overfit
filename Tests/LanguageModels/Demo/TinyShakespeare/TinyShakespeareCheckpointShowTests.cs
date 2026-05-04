// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
//
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Diagnostics;
using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.LanguageModels.Contracts;
using DevOnBike.Overfit.LanguageModels.Runtime;
using DevOnBike.Overfit.Tokenization;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests.LanguageModels.Demo.TinyShakespeare
{
    /// <summary>
    /// Display-oriented checkpoint load demo.
    ///
    /// This test does not train anything. It loads:
    ///
    /// test_fixtures/checkpoint.bin
    ///
    /// and shows cached KV generation with TopK + temperature + repetition penalty.
    /// Repetition penalty is display-only; it does not change checkpoint weights.
    /// </summary>
    public class TinyShakespeareCheckpointShowTests
    {
        private const string FixturePath = "test_fixtures/tiny_shakespeare.txt";
        private const string DemoCheckpointPath = "test_fixtures/checkpoint.bin";

        private const string DemoPrompt = "ROMEO:\n";

        private const int DemoSeqLen = 128;
        private const int DemoRequestedNewTokens = 120;
        private const int DemoParityRequestedNewTokens = 32;
        private const int DemoAllocationMeasuredTokenCount = 8;

        private const int DisplayTopK = 16;
        private const float DisplayTemperature = 0.85f;
        private const float DisplayRepetitionPenalty = 1.15f;
        private const int DisplayRepetitionWindow = 64;
        private const int DisplaySeed = 42;

        private readonly ITestOutputHelper _output;

        public TinyShakespeareCheckpointShowTests(
            ITestOutputHelper output)
        {
            _output = output;
        }

        [Fact]
        [Trait("Category", "Demo")]
        public void Demo_LoadCheckpoint_AndShowCachedRuntimeGeneration_RepetitionAware()
        {
            SkipIfMissing(FixturePath);

            if (!File.Exists(DemoCheckpointPath))
            {
                throw new InvalidOperationException(
                    $"Checkpoint '{DemoCheckpointPath}' not found. Run a TinyShakespeare checkpoint training demo first.");
            }

            var text = File.ReadAllText(FixturePath);
            var tokenizer = CharacterTokenizer.FromCorpus(text);
            var config = CreateDemoConfig(tokenizer.VocabSize);

            using var model = new GPT1Model(config);

            using (var stream = File.OpenRead(DemoCheckpointPath))
            using (var reader = new BinaryReader(stream))
            {
                model.Load(reader);
            }

            model.Eval();

            var promptTokens = tokenizer.Encode(DemoPrompt);
            var safeGeneratedTokenCount = GetSafeGeneratedTokenCount(
                config.ContextLength,
                promptTokens.Length,
                DemoRequestedNewTokens);

            var stopwatch = Stopwatch.StartNew();

            var generatedText = GenerateDisplaySampleWithRepetitionPenalty(
                model,
                tokenizer,
                DemoPrompt,
                safeGeneratedTokenCount,
                out var generatedTokenCount);

            stopwatch.Stop();

            var timePerToken =
                generatedTokenCount <= 0
                    ? 0.0
                    : stopwatch.Elapsed.TotalMilliseconds / generatedTokenCount;

            _output.WriteLine("=== Overfit TinyShakespeare Cached Runtime Demo ===");
            _output.WriteLine("");
            _output.WriteLine("Prompt:");
            _output.WriteLine(DemoPrompt);
            _output.WriteLine("");
            _output.WriteLine("Generated text:");
            _output.WriteLine(generatedText);
            _output.WriteLine("");
            _output.WriteLine("Runtime:");
            _output.WriteLine("Cached KV runtime + TopK + repetition penalty for display");
            _output.WriteLine("");
            _output.WriteLine($"Requested generated tokens: {DemoRequestedNewTokens}");
            _output.WriteLine($"Safe generated tokens: {safeGeneratedTokenCount}");
            _output.WriteLine($"Generated tokens: {generatedTokenCount}");
            _output.WriteLine($"ContextLength: {config.ContextLength}");
            _output.WriteLine($"Prompt tokens: {promptTokens.Length}");
            _output.WriteLine($"Sampling: TopK(k={DisplayTopK}), temperature={DisplayTemperature}, repetitionPenalty={DisplayRepetitionPenalty}, window={DisplayRepetitionWindow}, seed={DisplaySeed}");
            _output.WriteLine($"Cached generation time: {stopwatch.Elapsed.TotalMilliseconds:F3} ms");
            _output.WriteLine($"Time per token: {timePerToken:F3} ms/token");

            using var runtime = SlmRuntimeFactory.CreateGpt1(
                model,
                SlmRuntimeMode.Cached);

            _output.WriteLine($"Has KV cache: {runtime.HasKeyValueCache}");

            Assert.True(generatedTokenCount > 0);
            AssertDisplayTextLooksValid(generatedText);

            AssertDemoCachedMatchesLegacyGreedy(
                model,
                tokenizer,
                DemoPrompt,
                GetSafeGeneratedTokenCount(
                    config.ContextLength,
                    promptTokens.Length,
                    DemoParityRequestedNewTokens));

            AssertDemoCachedContinuationDoesNotAllocate(
                model,
                tokenizer,
                DemoPrompt,
                DemoAllocationMeasuredTokenCount);

            _output.WriteLine("");
            _output.WriteLine("Validation:");
            _output.WriteLine("Legacy parity: OK");
            _output.WriteLine($"Continuation allocation check: 0 B for {DemoAllocationMeasuredTokenCount} tokens");
            _output.WriteLine($"Checksum: {ComputeChecksum(generatedText)}");
        }

        private static GPT1Config CreateDemoConfig(
            int vocabSize)
        {
            return new GPT1Config
            {
                VocabSize = vocabSize,
                ContextLength = DemoSeqLen,
                DModel = 128,
                NHeads = 4,
                NLayers = 4,
                DFF = 512,
                TieWeights = false,
                PreLayerNorm = true
            };
        }

        private static string GenerateDisplaySampleWithRepetitionPenalty(
            GPT1Model model,
            CharacterTokenizer tokenizer,
            string prompt,
            int maxTokens,
            out int generatedTokenCount)
        {
            model.Eval();

            var promptTokens = tokenizer.Encode(prompt);
            var generatedTokens = new List<int>(promptTokens.Length + maxTokens);

            generatedTokens.AddRange(promptTokens);

            using var adapter = new CachedGpt1ModelAdapter(model);

            var logits = new float[model.Config.VocabSize];
            var adjustedLogits = new float[model.Config.VocabSize];
            var indexScratch = new int[model.Config.VocabSize];
            var scoreScratch = new float[model.Config.VocabSize];
            var random = new Random(DisplaySeed);

            for (var i = 0; i < promptTokens.Length; i++)
            {
                adapter.DecodeNextToken(
                    promptTokens[i],
                    logits);
            }

            var sampling = new SamplingOptions(
                SamplingStrategy.TopK,
                temperature: DisplayTemperature,
                topK: DisplayTopK,
                topP: 1.0f,
                seed: DisplaySeed);

            generatedTokenCount = 0;

            for (var i = 0; i < maxTokens; i++)
            {
                if (adapter.IsFull)
                {
                    break;
                }

                logits.AsSpan().CopyTo(adjustedLogits);

                ApplyRepetitionPenalty(
                    adjustedLogits,
                    generatedTokens,
                    DisplayRepetitionWindow,
                    DisplayRepetitionPenalty);

                var nextToken = TokenSampler.Sample(
                    adjustedLogits,
                    in sampling,
                    random,
                    indexScratch,
                    scoreScratch);

                generatedTokens.Add(nextToken);
                generatedTokenCount++;

                adapter.DecodeNextToken(
                    nextToken,
                    logits);
            }

            return tokenizer.Decode(generatedTokens.ToArray());
        }

        private static void ApplyRepetitionPenalty(
            Span<float> logits,
            IReadOnlyList<int> generatedTokens,
            int window,
            float penalty)
        {
            if (penalty <= 1f || generatedTokens.Count == 0)
            {
                return;
            }

            var start = Math.Max(
                0,
                generatedTokens.Count - window);

            for (var i = start; i < generatedTokens.Count; i++)
            {
                var token = generatedTokens[i];

                if ((uint)token >= (uint)logits.Length)
                {
                    continue;
                }

                if (logits[token] >= 0f)
                {
                    logits[token] /= penalty;
                }
                else
                {
                    logits[token] *= penalty;
                }
            }
        }

        private static void AssertDemoCachedMatchesLegacyGreedy(
            GPT1Model model,
            CharacterTokenizer tokenizer,
            string prompt,
            int maxNewTokens)
        {
            var promptTokens = tokenizer.Encode(prompt);

            var legacyTokens = new int[maxNewTokens];
            var cachedTokens = new int[maxNewTokens];

            using var legacy = SlmRuntimeFactory.CreateGpt1(
                model,
                SlmRuntimeMode.Legacy);

            using var cached = SlmRuntimeFactory.CreateGpt1(
                model,
                SlmRuntimeMode.Cached);

            var legacyGenerated = legacy.GenerateGreedy(
                promptTokens,
                legacyTokens,
                maxNewTokens);

            var cachedGenerated = cached.GenerateGreedy(
                promptTokens,
                cachedTokens,
                maxNewTokens);

            Assert.Equal(legacyGenerated, cachedGenerated);
            Assert.Equal(
                legacyTokens.AsSpan(0, legacyGenerated).ToArray(),
                cachedTokens.AsSpan(0, cachedGenerated).ToArray());
        }

        private static void AssertDemoCachedContinuationDoesNotAllocate(
            GPT1Model model,
            CharacterTokenizer tokenizer,
            string prompt,
            int measuredTokenCount)
        {
            var promptTokens = tokenizer.Encode(prompt);
            var sampling = SamplingOptions.Greedy;

            using var runtime = SlmRuntimeFactory.CreateGpt1(
                model,
                SlmRuntimeMode.Cached);

            runtime.Session.Reset(promptTokens);

            // Warm up the continuation path before allocation measurement.
            _ = runtime.Session.GenerateNextToken(in sampling);

            var before = GC.GetAllocatedBytesForCurrentThread();

            for (var i = 0; i < measuredTokenCount; i++)
            {
                _ = runtime.Session.GenerateNextToken(in sampling);
            }

            var after = GC.GetAllocatedBytesForCurrentThread();

            Assert.Equal(0, after - before);
        }

        private static void AssertDisplayTextLooksValid(
            string text)
        {
            Assert.False(string.IsNullOrWhiteSpace(text));
            AssertNoNullCharacters(text);
            Assert.Contains("ROMEO", text, StringComparison.OrdinalIgnoreCase);
        }

        private static void AssertNoNullCharacters(
            string text)
        {
            for (var i = 0; i < text.Length; i++)
            {
                if (text[i] == '\0')
                {
                    throw new InvalidOperationException(
                        $"Generated text contains a null character at index {i}.");
                }
            }
        }

        private static int GetSafeGeneratedTokenCount(
            int contextLength,
            int promptTokenCount,
            int requestedGeneratedTokenCount)
        {
            var available = contextLength - promptTokenCount;

            if (available <= 0)
            {
                throw new InvalidOperationException(
                    $"Prompt length {promptTokenCount} does not fit context length {contextLength}.");
            }

            return Math.Min(
                requestedGeneratedTokenCount,
                available);
        }

        private static int ComputeChecksum(
            string text)
        {
            unchecked
            {
                var checksum = 17;

                foreach (var ch in text)
                {
                    checksum = checksum * 31 + ch;
                }

                return Math.Abs(checksum % 10_000);
            }
        }

        private static void SkipIfMissing(
            string path)
        {
            if (!File.Exists(path))
            {
                throw new InvalidOperationException(
                    $"Required fixture is missing: {Path.GetFullPath(path)}");
            }
        }
    }
}
