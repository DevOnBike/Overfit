// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Diagnostics;
using DevOnBike.Overfit.LanguageModels.Contracts;
using DevOnBike.Overfit.LanguageModels.Runtime;
using DevOnBike.Overfit.LanguageModels.Tokenizers;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests.LanguageModels.Loading
{
    /// <summary>
    /// Steady-state decode throughput for Qwen3-0.6B Q8_0 — load once, warm up, then best-of-N fixed-length
    /// decode (no early stop) so each run generates the same token count. Reports min/median/max tok/s,
    /// ns/token, and per-token alloc (must stay 0 for the zero-alloc decode claim). [LongFact] — needs
    /// C:\qwen3-06b. Flip to [LongFact] to run.
    /// </summary>
    public sealed class Qwen3PerfTests
    {
        private const string Path = @"C:\qwen3-06b\Qwen3-0.6B-Q8_0.gguf";
        private const int DecodeTokens = 128;
        private const int Runs = 3;
        private readonly ITestOutputHelper _out;

        public Qwen3PerfTests(ITestOutputHelper output) => _out = output;

        [LongFact]
        public void Qwen3_DecodeThroughput_BestOfN()
        {
            if (!File.Exists(Path)) { _out.WriteLine("missing Qwen3-0.6B gguf"); return; }

            using var engine = CachedLlamaInferenceEngine.LoadGguf(Path);
            var tok = GgufTokenizer.Load(Path);
            var prompt = tok.Encode("The history of computing began");
            var sampling = SamplingOptions.GreedyWithPenalty(1.1f);

            _out.WriteLine($"workers={OverfitParallelForWorkerCountReflectionFree()} cpu={Environment.ProcessorCount}");

            using var session = engine.CreateSession(512);

            // Warm-up (JIT + cache fill) — not timed.
            DecodeFixed(session, prompt, 16, in sampling, out _);

            var perRun = new double[Runs];
            long allocPerToken = -1;
            for (var r = 0; r < Runs; r++)
            {
                var before = GC.GetAllocatedBytesForCurrentThread();
                var ns = DecodeFixed(session, prompt, DecodeTokens, in sampling, out var produced);
                var alloc = GC.GetAllocatedBytesForCurrentThread() - before;
                allocPerToken = produced == 0 ? alloc : alloc / produced;
                var tps = produced / (ns / 1_000_000_000.0);
                perRun[r] = tps;
                _out.WriteLine($"run {r + 1}: {produced} tok in {ns / 1_000_000.0:F1} ms → {tps:F2} tok/s, {alloc / produced} B/tok");
            }

            Array.Sort(perRun);
            _out.WriteLine($"=== Qwen3-0.6B Q8_0 decode: min {perRun[0]:F2} | median {perRun[Runs / 2]:F2} | max {perRun[^1]:F2} tok/s ===");
            Assert.True(perRun[^1] > 0);
        }

        private static long DecodeFixed(
            CachedLlamaSession session,
            int[] prompt,
            int count,
            in SamplingOptions sampling,
            out int produced)
        {
            session.Reset(prompt);
            var sw = Stopwatch.StartNew();
            produced = 0;
            for (var i = 0; i < count && !session.IsFull; i++)
            {
                _ = session.GenerateNextToken(in sampling);
                produced++;
            }
            sw.Stop();
            return sw.Elapsed.Ticks * 100; // ticks → ns
        }

        // Avoid touching the internal worker-count field via reflection (banned); just report ProcessorCount-derived note.
        private static string OverfitParallelForWorkerCountReflectionFree()
            => $"~min({Environment.ProcessorCount},10) decode";
    }
}
