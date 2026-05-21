// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Diagnostics;
using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.LanguageModels.Runtime;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests.LanguageModels.Runtime
{
    /// <summary>
    /// Prefill Phase 3 (session delegation): the batched prefill path
    /// (<see cref="CachedGpt1ModelAdapter.PrefillBatched"/>) must leave the session in
    /// exactly the state the single-token prefill loop would — the last prompt token's
    /// logits must be bit-identical. Plus a TTFT measurement ([LongFact]).
    /// </summary>
    public sealed class CachedSlmPrefillBatchedTests
    {
        private readonly ITestOutputHelper _output;

        public CachedSlmPrefillBatchedTests(ITestOutputHelper output)
        {
            _output = output;
        }

        [Theory]
        [InlineData(8, 32, 2, 2, 64)]
        [InlineData(12, 64, 4, 3, 128)]
        [InlineData(20, 96, 4, 2, 200)]
        public void PrefillBatched_LastTokenLogits_MatchSingleTokenLoop(
            int promptLen, int dModel, int heads, int layers, int vocab)
        {
            using var model = NewModel(dModel, heads, layers, vocab, contextLength: promptLen + 8);
            var rng = new Random(promptLen * 7 + dModel + heads);
            var tokens = new int[promptLen];
            for (var i = 0; i < promptLen; i++)
            {
                tokens[i] = rng.Next(0, vocab);
            }

            // Reference: single-token prefill loop (PrefillToken × N-1, then full decode).
            var single = new float[vocab];
            using (var adapterS = new CachedGpt1ModelAdapter(model))
            {
                for (var i = 0; i < promptLen - 1; i++)
                {
                    adapterS.PrefillToken(tokens[i]);
                }
                adapterS.DecodeNextToken(tokens[promptLen - 1], single);
            }

            // Batched prefill.
            var batched = new float[vocab];
            using (var adapterB = new CachedGpt1ModelAdapter(model))
            {
                Assert.True(adapterB.SupportsBatchedPrefill);
                adapterB.PrefillBatched(tokens, batched);
            }

            for (var i = 0; i < vocab; i++)
            {
                Assert.Equal(single[i], batched[i]);
            }
        }

        [LongFact]
        public void Ttft_BatchedVsSingleToken_Gpt2SmallDims()
        {
            // GPT-2-Small dimensions, random weights (timing doesn't need real weights).
            const int dModel = 768;
            const int heads = 12;
            const int layers = 12;
            const int vocab = 50257;
            const int promptLen = 64;

            using var model = NewModel(dModel, heads, layers, vocab, contextLength: 256);
            var rng = new Random(1);
            var tokens = new int[promptLen];
            for (var i = 0; i < promptLen; i++)
            {
                tokens[i] = rng.Next(0, vocab);
            }

            var logits = new float[vocab];

            double SingleToken()
            {
                using var a = new CachedGpt1ModelAdapter(model);
                var sw = Stopwatch.StartNew();
                for (var i = 0; i < promptLen - 1; i++)
                {
                    a.PrefillToken(tokens[i]);
                }
                a.DecodeNextToken(tokens[promptLen - 1], logits);
                sw.Stop();
                return sw.Elapsed.TotalMilliseconds;
            }

            double Batched()
            {
                using var a = new CachedGpt1ModelAdapter(model);
                var sw = Stopwatch.StartNew();
                a.PrefillBatched(tokens, logits);
                sw.Stop();
                return sw.Elapsed.TotalMilliseconds;
            }

            // Warm up (JIT) then best-of-3.
            SingleToken(); Batched();
            var single = Math.Min(Math.Min(SingleToken(), SingleToken()), SingleToken());
            var batched = Math.Min(Math.Min(Batched(), Batched()), Batched());

            _output.WriteLine($"TTFT prefill {promptLen} tokens, GPT-2-Small dims ({layers}L d={dModel}):");
            _output.WriteLine($"  single-token loop : {single:F1} ms");
            _output.WriteLine($"  batched           : {batched:F1} ms");
            _output.WriteLine($"  speedup           : {single / batched:F2}x");

            Assert.True(batched > 0 && single > 0);
        }

        private static GPT1Model NewModel(int dModel, int heads, int layers, int vocab, int contextLength)
        {
            var model = new GPT1Model(new GPT1Config
            {
                VocabSize = vocab,
                ContextLength = contextLength,
                DModel = dModel,
                NHeads = heads,
                NLayers = layers,
                DFF = dModel * 4,
                TieWeights = false,
                PreLayerNorm = true,
            });
            model.Eval();
            return model;
        }
    }
}
