// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels.Contracts;
using DevOnBike.Overfit.LanguageModels.Runtime;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests.LanguageModels.Runtime.Parity
{
    /// <summary>
    /// Prefix / system-prompt KV reuse: restoring a snapshotted prefix and appending a turn must produce
    /// the SAME generation as prefilling prefix+turn together (the restored KV is bit-identical — causal,
    /// the prefix never attends to the turn). Verified on real Qwen2.5-3B Q4_K_M. The win is skipping the
    /// prefix forward pass (a memcpy restore vs re-encoding the system prompt every request). [LongFact].
    /// </summary>
    [Trait("Category", "Qwen")]
    [Trait("Category", "Parity")]
    public sealed class PrefixKvCacheParityTests
    {
        private const string ModelPath = @"C:\qwen3b\qwen.q4km.gguf";

        private readonly ITestOutputHelper _out;
        public PrefixKvCacheParityTests(ITestOutputHelper output) => _out = output;

        [LongFact]
        public void RestoredPrefix_MatchesFullPrefill_AndReusesAcrossRequests()
        {
            if (!File.Exists(ModelPath)) { _out.WriteLine($"missing {ModelPath}"); return; }

            using var engine = CachedLlamaInferenceEngine.LoadGguf(ModelPath);

            var prefix = new int[24];   // "system prompt"
            for (var i = 0; i < prefix.Length; i++) { prefix[i] = 50 + i * 17; }
            var turn = new int[8];      // "user turn"
            for (var i = 0; i < turn.Length; i++) { turn[i] = 900 + i * 5; }
            const int generate = 16;
            var sampling = SamplingOptions.Greedy;

            // Reference: prefill prefix+turn together, then generate.
            var reference = new List<int>();
            using (var s = engine.CreateSession(128))
            {
                var both = new int[prefix.Length + turn.Length];
                prefix.CopyTo(both, 0);
                turn.CopyTo(both, prefix.Length);
                s.Reset(both);
                for (var i = 0; i < generate; i++) { reference.Add(s.GenerateNextToken(in sampling)); }
            }

            // Prefix reuse: prefill the prefix ONCE, snapshot it, then restore + append the turn.
            using var session = engine.CreateSession(128);
            session.Reset(prefix);
            var snapshot = session.SavePrefix();
            Assert.Equal(prefix.Length, snapshot.Length);

            var viaPrefix = new List<int>();
            session.RestorePrefix(snapshot);
            session.Prefill(turn);
            for (var i = 0; i < generate; i++) { viaPrefix.Add(session.GenerateNextToken(in sampling)); }

            for (var i = 0; i < generate; i++)
            {
                Assert.Equal(reference[i], viaPrefix[i]);
            }

            // Second request reuses the SAME prefix without re-encoding it — a different turn.
            var turn2 = new int[6];
            for (var i = 0; i < turn2.Length; i++) { turn2[i] = 1200 + i * 9; }
            using var session2 = engine.CreateSession(128);
            var reference2 = new List<int>();
            {
                var both = new int[prefix.Length + turn2.Length];
                prefix.CopyTo(both, 0);
                turn2.CopyTo(both, prefix.Length);
                session2.Reset(both);
                for (var i = 0; i < generate; i++) { reference2.Add(session2.GenerateNextToken(in sampling)); }
            }

            using var reuse = engine.CreateSession(128);
            reuse.RestorePrefix(snapshot);   // fresh session, no prefix prefill at all
            reuse.Prefill(turn2);
            for (var i = 0; i < generate; i++)
            {
                Assert.Equal(reference2[i], reuse.GenerateNextToken(in sampling));
            }

            _out.WriteLine($"prefix reuse bit-identical across 2 requests (prefix={prefix.Length} tokens reused via memcpy)");
        }
    }
}
