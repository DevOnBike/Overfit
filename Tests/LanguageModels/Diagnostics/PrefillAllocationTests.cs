// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Text;
using DevOnBike.Overfit.LanguageModels.Contracts;
using DevOnBike.Overfit.LanguageModels.Runtime;
using DevOnBike.Overfit.LanguageModels.Tokenizers;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests.LanguageModels.Diagnostics
{
    /// <summary>
    /// Measures managed allocations of the BATCHED PREFILL path (the Runtime-85 OVERFIT001 backlog:
    /// per-request scratch arrays in DecodeBatched/DecodeBatchedQuant/BatchedQuantProjection…).
    /// Decode is 0 B/token by contract; prefill allocates per REQUEST — this quantifies how much,
    /// and pins the greedy output so the pooling sweep can prove bit-identical behaviour.
    /// [LongFact] — needs C:\qwen3-06b.
    /// </summary>
    public sealed class PrefillAllocationTests
    {
        private const string Path = @"C:\qwen3-06b\Qwen3-0.6B-Q4_K_M.gguf";
        private readonly ITestOutputHelper _out;
        public PrefillAllocationTests(ITestOutputHelper output) => _out = output;

        [LongFact]
        public void Prefill_AllocationsPerRequest_AndGreedyPin()
        {
            if (!File.Exists(Path)) { _out.WriteLine("missing gguf"); return; }

            using var engine = CachedLlamaInferenceEngine.LoadGguf(Path);
            var tok = GgufTokenizer.Load(Path);

            // ~200-token prompt — long enough that batched prefill dominates.
            var prompt = string.Join(" ", Enumerable.Repeat(
                "The quick brown fox jumps over the lazy dog near the quiet river bank at dawn.", 16));
            var tokens = tok.Encode(prompt);
            _out.WriteLine($"prompt tokens: {tokens.Length}");

            using var session = engine.CreateSession(tokens.Length + 64);

            // Warm-up prefill (JIT, pools, lazies), then measure steady-state requests.
            session.Reset(tokens);

            const int rounds = 5;
            long total = 0;
            for (var r = 0; r < rounds; r++)
            {
                var before = GC.GetAllocatedBytesForCurrentThread();
                session.Reset(tokens);
                var delta = GC.GetAllocatedBytesForCurrentThread() - before;
                total += delta;
                _out.WriteLine($"prefill #{r}: {delta:N0} B allocated (this thread)");
            }
            _out.WriteLine($"AVG per prefill: {total / rounds:N0} B");

            // Greedy pin: 24 tokens — the pooling sweep must reproduce this exactly.
            var sampling = SamplingOptions.Greedy;
            var sb = new StringBuilder();
            for (var i = 0; i < 24; i++)
            {
                sb.Append(session.GenerateNextToken(in sampling)).Append(',');
            }
            _out.WriteLine($"GREEDY PIN: {sb}");
        }
    }
}
