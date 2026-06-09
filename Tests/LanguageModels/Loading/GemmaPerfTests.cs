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
    /// Steady-state decode throughput for Gemma-2-2B Q4_K_M — load once, warm up, then best-of-N fixed-length
    /// decode (no early stop) so each run generates the same token count. Reports min/median/max tok/s,
    /// ns/token, and per-token alloc (must stay 0 for the zero-alloc decode claim). Soft-caps engaged
    /// (attn 50 / final 30). [LongFact] — needs C:\gemma. Flip to [Fact] to run.
    /// </summary>
    public sealed class GemmaPerfTests
    {
        private const string Path = @"C:\gemma\gemma-2-2b-it-Q4_K_M.gguf";
        private const int DecodeTokens = 128;
        private const int Runs = 3;
        private readonly ITestOutputHelper _out;

        public GemmaPerfTests(ITestOutputHelper output) => _out = output;

        [LongFact]
        public void Gemma2_DecodeThroughput_BestOfN()
        {
            if (!File.Exists(Path))
            {
                _out.WriteLine("missing Gemma-2-2B gguf");
                return;
            }

            using var engine = CachedLlamaInferenceEngine.LoadGguf(Path);
            var tok = GgufTokenizer.Load(Path);
            var prompt = tok.Encode("<start_of_turn>user\nThe history of computing began<end_of_turn>\n<start_of_turn>model\n");
            var sampling = SamplingOptions.GreedyWithPenalty(1.1f);

            _out.WriteLine($"cpu={Environment.ProcessorCount}");

            var greedy = SamplingOptions.Greedy;

            using var session = engine.CreateSession(512);

            // Warm-up (JIT + cache fill) — not timed.
            DecodeFixed(session, prompt, 16, in sampling, out _, out _);

            // Isolate prefill (Reset) alloc from steady-state decode alloc.
            var beforeReset = GC.GetAllocatedBytesForCurrentThread();
            session.Reset(prompt);
            var resetAlloc = GC.GetAllocatedBytesForCurrentThread() - beforeReset;
            var beforeDecode = GC.GetAllocatedBytesForCurrentThread();
            for (var i = 0; i < DecodeTokens && !session.IsFull; i++)
            {
                _ = session.GenerateNextToken(in sampling);
            }
            var decodeAlloc = GC.GetAllocatedBytesForCurrentThread() - beforeDecode;
            _out.WriteLine($"ISOLATED: prefill(Reset {prompt.Length} tok)={resetAlloc} B | decode {DecodeTokens} tok={decodeAlloc} B = {decodeAlloc / DecodeTokens} B/tok");

            Measure("GreedyWithPenalty(1.1)", session, prompt, in sampling);
            Measure("Greedy (no penalty)", session, prompt, in greedy);
        }

        private void Measure(string label, CachedLlamaSession session, int[] prompt, in SamplingOptions sampling)
        {
            var perRun = new double[Runs];
            long bytesPerTok = 0;
            for (var r = 0; r < Runs; r++)
            {
                var ns = DecodeFixed(session, prompt, DecodeTokens, in sampling, out var produced, out var alloc);
                bytesPerTok = alloc / produced;
                var tps = produced / (ns / 1_000_000_000.0);
                perRun[r] = tps;
                _out.WriteLine($"  [{label}] run {r + 1}: {produced} tok in {ns / 1_000_000.0:F1} ms → {tps:F2} tok/s, {bytesPerTok} B/tok");
            }

            Array.Sort(perRun);
            _out.WriteLine($"=== Gemma-2-2B Q4_K_M [{label}]: min {perRun[0]:F2} | median {perRun[Runs / 2]:F2} | max {perRun[^1]:F2} tok/s, {bytesPerTok} B/tok ===");
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
            // Prefill (Reset) is the batched path and allocates; it is NOT the zero-alloc decode hot path,
            // so it must sit OUTSIDE the alloc + time measurement — otherwise it pollutes the per-token figure.
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
