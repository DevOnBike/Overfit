// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Diagnostics;
using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.LanguageModels.Contracts;
using DevOnBike.Overfit.LanguageModels.Runtime;
using DevOnBike.Overfit.Tests.TestSupport;
using DevOnBike.Overfit.Tokenization;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests.LanguageModels.Demo.Gpt2
{
    /// <summary>
    /// xUnit mirror of <c>Demo/Gpt2ConsoleDemo</c> — verifies the GPT-2 Small
    /// pipeline end-to-end (load → tokenize → KV-cache decode) and **asserts the
    /// headline 0 B / generated token claim** from the README.
    ///
    /// Defends the inference contract on every <c>dotnet test -c Release</c>:
    /// if a future change adds an allocation to <c>GenerateNextToken</c> (or one
    /// of its kernels), this test fails immediately with a precise byte count.
    ///
    /// Companion to the user-facing console demo:
    ///   dotnet run -c Release --project Demo/Gpt2ConsoleDemo
    /// They use the same engine + session API; this one asserts numbers the
    /// console demo only prints.
    ///
    /// Fixtures resolved via <c>TestModelPaths.Gpt2Small</c> (override with
    /// <c>OVERFIT_GPT2_DIR</c>). Throws <see cref="FileNotFoundException"/>
    /// with an actionable hint if the fixture is missing.
    /// </summary>
    [Trait("Category", "Demo")]
    [Trait("Category", "Gpt2")]
    public sealed class Gpt2GenerationDemoTests
    {
        private readonly ITestOutputHelper _output;

        public Gpt2GenerationDemoTests(ITestOutputHelper output)
        {
            _output = output;
        }

        [Fact]
        public void Demo_Gpt2Small_KvCacheDecode_AllocatesZeroBytesPerToken()
        {
            // ── Resolve fixtures (throw with env-var hint if missing) ──────────
            var modelPath  = TestModelPaths.Gpt2Small.RequireBinaryPath();
            var vocabPath  = TestModelPaths.Gpt2Small.RequireVocabPath();
            var mergesPath = TestModelPaths.Gpt2Small.RequireMergesPath();

            const string prompt = "The future of software development is";
            const int maxTokens = 32;

            // ── Load tokenizer + model ─────────────────────────────────────────
            var tokenizer = BytePairEncoder.Load(vocabPath, mergesPath);
            Assert.Equal(50257, tokenizer.VocabSize);

            using var model = new GPT1Model(Gpt2Config.Small);
            model.Eval();
            using (var fs = File.OpenRead(modelPath))
            using (var br = new BinaryReader(fs))
            {
                model.Load(br);
            }

            using var engine  = CachedSlmInferenceEngine.FromGpt1(model);
            using var session = engine.CreateSession();
            session.Reset(tokenizer.Encode(prompt));

            // ── Warmup token outside the measurement window ────────────────────
            // The first GenerateNextToken pays for JIT + arena warm-up; excluding
            // it gives a clean steady-state measurement.
            var sampling = SamplingOptions.Greedy;
            _ = session.GenerateNextToken(in sampling);

            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();

            // ── Inference-only measurement zone ────────────────────────────────
            // Per-iter alloc delta via GC.GetAllocatedBytesForCurrentThread().
            // CRITICAL: this is current-thread-only — the console demo can use
            // GC.GetTotalAllocatedBytes because it runs in an otherwise-empty
            // process, but xUnit has background threads (parallel discovery,
            // output sinks) that pollute process-wide counters. Current-thread
            // sampling is the only honest measurement inside a test runner.
            //
            // String decode happens OUTSIDE this loop so the measurement
            // isolates the inference contract.
            var generated = new int[maxTokens - 1];
            long inferTicks = 0;
            long inferAlloc = 0;

            for (var i = 0; i < generated.Length; i++)
            {
                var allocBefore = GC.GetAllocatedBytesForCurrentThread();
                var tBefore = Stopwatch.GetTimestamp();

                generated[i] = session.GenerateNextToken(in sampling);

                inferTicks += Stopwatch.GetTimestamp() - tBefore;
                inferAlloc += GC.GetAllocatedBytesForCurrentThread() - allocBefore;
            }

            // ── Report ─────────────────────────────────────────────────────────
            var elapsedMs    = inferTicks / (double)Stopwatch.Frequency * 1000.0;
            var tokensPerSec = generated.Length * 1000.0 / elapsedMs;
            var bytesPerTok  = (double)inferAlloc / generated.Length;
            var text         = prompt + tokenizer.Decode(generated);

            _output.WriteLine($"Prompt:               \"{prompt}\"");
            _output.WriteLine($"Output:               \"{text}\"");
            _output.WriteLine(string.Empty);
            _output.WriteLine($"Tokens generated:     {generated.Length}");
            _output.WriteLine($"Elapsed (inference):  {elapsedMs:F1} ms");
            _output.WriteLine($"Tokens/sec:           {tokensPerSec:F1}");
            _output.WriteLine($"Bytes / token:        {bytesPerTok:F1}  (total: {inferAlloc:N0} B)");

            // ── Hard assertions ────────────────────────────────────────────────
            Assert.NotEmpty(generated);
            Assert.All(generated, t => Assert.InRange(t, 0, tokenizer.VocabSize - 1));

            // Headline claim from the README: KV-cache decode allocates zero
            // managed bytes per generated token. If this regresses, the README
            // is lying — fail loudly here, not from a benchmark run.
            Assert.True(
                inferAlloc == 0,
                $"GenerateNextToken allocated {inferAlloc} B across {generated.Length} tokens " +
                $"({bytesPerTok:F1} B/token). README claims 0 B/token. " +
                "Run BenchmarkDotNet (*Gpt2TokensPerSecond*) to pinpoint the source.");
        }
    }
}
