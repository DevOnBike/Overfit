// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels.Contracts;
using DevOnBike.Overfit.LanguageModels.Runtime;
using DevOnBike.Overfit.Tests.TestSupport;

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

        [LongFact]  // runtime unmeasured — the test failed after 4s (2026-08-07)
        public void RestoredPrefix_MatchesFullPrefill_AndReusesAcrossRequests()
        {
            if (!File.Exists(ModelPath))
            {
                _out.WriteLine($"missing {ModelPath}");
                return;
            }

            // HOLD THE KERNEL LAYOUT CONSTANT, or this test silently stops testing what it claims.
            //
            // It compares prefilling 32 tokens in one go against 24 + 8 in two, and asserts the generated
            // tokens match. Those two shapes can dispatch to different kernels, and the repacked
            // `block_q*_Kx8` GEMMs associate their reduction differently from the per-row ones — measured
            // elsewhere in this repository at `maxAbsLogitDiff ~ 0.44`, which is enough to flip an argmax.
            // A `*.gguf.repack` sidecar sets `IsPrepacked` and switches the repacked path on regardless of
            // any env flag, so on a box with a sidecar both halves ran repacked and disagreed.
            //
            // Measured 2026-08-07: without this scope the very first generated token differed (expected 34,
            // got 322); with it, the test passes. The failure was never about prefix reuse — the thing this
            // test exists to verify — and it took the first-ever [LongFact] run to surface it at all.
            // `BatchedPrefillParityTests` learned the same lesson two days earlier and carries the same scope.
            // Scope BEFORE load, and the order is load-bearing since 2026-08-21: the loader reads this flag to
            // decide whether to build the per-head attention output weights at all
            // (GgufLlamaLoader.UseWholeOutputOnly). Reordering these two lines throws a named error.
            using var layout = new NonRepackedKernelScope();
            using var engine = CachedLlamaInferenceEngine.LoadGguf(ModelPath);

            var prefix = new int[24];   // "system prompt"
            for (var i = 0; i < prefix.Length; i++)
            {
                prefix[i] = 50 + i * 17;
            }
            var turn = new int[8];      // "user turn"
            for (var i = 0; i < turn.Length; i++)
            {
                turn[i] = 900 + i * 5;
            }
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
                for (var i = 0; i < generate; i++)
                {
                    reference.Add(s.GenerateNextToken(in sampling));
                }
            }

            // Prefix reuse: prefill the prefix ONCE, snapshot it, then restore + append the turn.
            using var session = engine.CreateSession(128);
            session.Reset(prefix);
            var snapshot = session.SavePrefix();
            Assert.Equal(prefix.Length, snapshot.Length);

            var viaPrefix = new List<int>();
            session.RestorePrefix(snapshot);
            session.Prefill(turn);
            for (var i = 0; i < generate; i++)
            {
                viaPrefix.Add(session.GenerateNextToken(in sampling));
            }

            for (var i = 0; i < generate; i++)
            {
                Assert.Equal(reference[i], viaPrefix[i]);
            }

            // Second request reuses the SAME prefix without re-encoding it — a different turn.
            var turn2 = new int[6];
            for (var i = 0; i < turn2.Length; i++)
            {
                turn2[i] = 1200 + i * 9;
            }
            using var session2 = engine.CreateSession(128);
            var reference2 = new List<int>();
            {
                var both = new int[prefix.Length + turn2.Length];
                prefix.CopyTo(both, 0);
                turn2.CopyTo(both, prefix.Length);
                session2.Reset(both);
                for (var i = 0; i < generate; i++)
                {
                    reference2.Add(session2.GenerateNextToken(in sampling));
                }
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
