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
    /// Steady-state decode throughput for Phi-4 (14B) Q4_K_M — load once, warm up, then best-of-N fixed-length
    /// decode. Prefill (Reset) is the batched path and ALLOCATES, so it sits OUTSIDE the alloc + time measurement
    /// (otherwise it pollutes the per-token figure — see GemmaPerfTests). Reports min/median/max tok/s and per-token
    /// alloc (must stay 0 for the zero-alloc decode claim). 14B on CPU is slow (~2-3 tok/s) and uses ~9 GB RAM.
    /// [LongFact] — needs C:\phi\phi-4-Q4_K_M.gguf. Flip to [Fact] to run.
    /// </summary>
    public sealed class Phi4PerfTests
    {
        private const string Path = @"C:\phi\phi-4-Q4_K_M.gguf";
        private const int DecodeTokens = 64;
        private const int Runs = 3;
        private readonly ITestOutputHelper _out;

        public Phi4PerfTests(ITestOutputHelper output) => _out = output;

        [LongFact]
        public void Phi4_DecodeThroughput_BestOfN()
        {
            if (!File.Exists(Path))
            {
                _out.WriteLine("missing Phi-4 gguf");
                return;
            }

            using var engine = CachedLlamaInferenceEngine.LoadGguf(Path);
            var tok = GgufTokenizer.Load(Path);
            var prompt = tok.Encode("The history of computing began");
            var sampling = SamplingOptions.GreedyWithPenalty(1.1f);

            _out.WriteLine($"cpu={Environment.ProcessorCount}");

            using var session = engine.CreateSession(512);
            DecodeFixed(session, prompt, 8, in sampling, out _, out _);  // warm-up (not timed)

            var perRun = new double[Runs];
            long bytesPerTok = 0;
            for (var r = 0; r < Runs; r++)
            {
                var ns = DecodeFixed(session, prompt, DecodeTokens, in sampling, out var produced, out var alloc);
                bytesPerTok = alloc / produced;
                var tps = produced / (ns / 1_000_000_000.0);
                perRun[r] = tps;
                _out.WriteLine($"run {r + 1}: {produced} tok in {ns / 1_000_000.0:F1} ms → {tps:F2} tok/s, {bytesPerTok} B/tok");
            }

            Array.Sort(perRun);
            _out.WriteLine($"=== Phi-4 14B Q4_K_M decode: min {perRun[0]:F2} | median {perRun[Runs / 2]:F2} | max {perRun[^1]:F2} tok/s, {bytesPerTok} B/tok ===");
            Assert.True(perRun[^1] > 0);
        }

        private static long DecodeFixed(
            CachedLlamaSession session,
            int[] prompt,
            int count,
            in SamplingOptions sampling,
            out int produced,
            out long decodeAlloc)
        {
            // Prefill (Reset) allocates (batched path) — keep it OUTSIDE the alloc + time window.
            session.Reset(prompt);
            var before = GC.GetAllocatedBytesForCurrentThread();
            var sw = Stopwatch.StartNew();
            produced = 0;
            for (var i = 0; i < count && !session.IsFull; i++)
            {
                _ = session.GenerateNextToken(in sampling);
                produced++;
            }
            sw.Stop();
            decodeAlloc = GC.GetAllocatedBytesForCurrentThread() - before;
            return sw.Elapsed.Ticks * 100; // ticks → ns
        }
    }
}
