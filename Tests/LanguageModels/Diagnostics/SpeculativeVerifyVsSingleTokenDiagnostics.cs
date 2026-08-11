// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels.Contracts;
using DevOnBike.Overfit.LanguageModels.Runtime;

namespace DevOnBike.Overfit.Tests.LanguageModels.Diagnostics
{
    /// <summary>
    /// Why greedy speculative decoding does not reproduce greedy single-token decoding — measured rather
    /// than argued.
    ///
    /// <para><b>The observation, 2026-08-07.</b> Three tests assert that the two paths emit an identical
    /// token sequence, and all three fail the first time they are ever executed:
    /// <c>BielikDraftSpeculativeBench</c> (Bielik 1.5B→4.5B, expected 17258 got 31940),
    /// <c>DraftModelSpeculativeBench</c> (Qwen 0.5B→3B, expected 23327 got 279) and
    /// <c>SpeculativeDecodeParityTests.Speculative_ProducesIdenticalSequence_ToGreedy</c> (Qwen 3B with the
    /// prompt-lookup drafter, expected 16 got 11930). Three models and two drafters, so it is not a
    /// property of any one of them. The failure is deterministic to the token across repeated runs.</para>
    ///
    /// <para><b>What is already ruled out.</b> Not randomness: under
    /// <see cref="SamplingOptions.Greedy"/>, <c>TokenSampler.ComputeProbabilities</c> writes a point mass on
    /// the argmax, so <c>SpeculativeSampler.AcceptOrResample</c> accepts exactly when the draft equals the
    /// argmax and <c>Sample</c> returns that argmax — the <c>_random</c> threaded through both is inert on
    /// this path. The fourth test, <c>ChatSession_SpeculativePath_MatchesSingleToken_Greedy</c>, passes, but
    /// it never checks that speculation engaged and its "speculative" arm measured SLOWER than its
    /// single-token control — the signature of a step that fell back to plain decoding, where identity is a
    /// tautology.</para>
    ///
    /// <para><b>What is left, and what separates the two answers.</b> The verify runs
    /// <c>PrefillBatchedQuantAllRows</c> + <c>ProjectLogitsBatched</c>; the reference runs the single-token
    /// path. Different summation order, and floating-point addition is not associative. So either the
    /// committed token is the argmax of a slightly different logit vector — arithmetic, and the tests'
    /// premise is wrong — or it is not the argmax of anything, which is a defect in
    /// <c>GenerateSpeculativeCore</c>. <b>The rank and the deficit separate these.</b> A reassociation flip
    /// takes the RUNNER-UP by a hair: rank 2, deficit on the order of the measured batched-vs-single-token
    /// difference (~0.44 for this family). A logic error has no reason to land on rank 2 at all.</para>
    ///
    /// <para>Both tests here REPORT; neither asserts a threshold. A number invented before it is measured is
    /// the trap this repository has already paid for twice — most recently by carrying a 0.44 taken from a
    /// different comparison into an unrelated one.</para>
    /// </summary>
    [Trait("Category", "Qwen")]
    [Trait("Category", "Diagnostics")]
    public sealed class SpeculativeVerifyVsSingleTokenDiagnostics
    {
        private const string ModelPath = @"C:\qwen3b\qwen.q4km.gguf";

        private readonly ITestOutputHelper _out;

        public SpeculativeVerifyVsSingleTokenDiagnostics(ITestOutputHelper output) => _out = output;

        /// <summary>
        /// Finds the first position where the two paths disagree, then asks where the speculative choice sat
        /// in the single-token path's own ranking. Rank 2 with a small deficit says arithmetic; anything else
        /// says the commit logic picked a token no ranking justifies.
        /// </summary>
        [ModelFact(ModelPath)]
        public void WhereDoTheSpeculativeAndGreedyPathsFirstDisagree()
        {
            using var engine = CachedLlamaInferenceEngine.LoadGguf(ModelPath);

            // The same repetitive prompt the failing parity test uses, so this diagnoses THAT failure rather
            // than a nearby one of my own construction.
            var prompt = new List<int>();

            for (var r = 0; r < 8; r++)
            {
                prompt.AddRange([10, 11, 12, 13, 14, 15]);
            }

            var promptArr = prompt.ToArray();
            const int generate = 40;

            var greedy = new List<int>(generate);

            using (var s = engine.CreateSession(256))
            {
                s.Reset(promptArr);
                var sampling = SamplingOptions.Greedy;

                for (var i = 0; i < generate; i++)
                {
                    greedy.Add(s.GenerateNextToken(in sampling));
                }
            }

            var spec = new List<int>(generate + 8);
            var multiCommits = 0;

            using (var s = engine.CreateSession(256))
            {
                s.Reset(promptArr);
                var history = new List<int>(promptArr);
                var committed = new int[6];

                while (spec.Count < generate)
                {
                    var n = s.GenerateSpeculative(
                        System.Runtime.InteropServices.CollectionsMarshal.AsSpan(history), committed, maxDraft: 4);

                    if (n > 1)
                    {
                        multiCommits++;
                    }

                    for (var c = 0; c < n; c++)
                    {
                        spec.Add(committed[c]);
                        history.Add(committed[c]);
                    }
                }
            }

            _out.WriteLine($"multi-token commits: {multiCommits}   (0 would mean speculation never engaged)");
            _out.WriteLine($"greedy[0..12] = [{string.Join(",", greedy.GetRange(0, Math.Min(12, greedy.Count)))}]");
            _out.WriteLine($"spec  [0..12] = [{string.Join(",", spec.GetRange(0, Math.Min(12, spec.Count)))}]");

            var divergence = -1;

            for (var i = 0; i < generate; i++)
            {
                if (greedy[i] != spec[i])
                {
                    divergence = i;

                    break;
                }
            }

            if (divergence < 0)
            {
                _out.WriteLine("\nthe two sequences agree over all 40 tokens in this run.");

                return;
            }

            _out.WriteLine($"\nfirst divergence at index {divergence}: greedy {greedy[divergence]}, "
                + $"speculative {spec[divergence]}");

            // Replay the single-token path to exactly that position. Both paths agree on everything before
            // `divergence`, so the two contexts are identical there and the logits are comparable.
            using var replay = engine.CreateSession(256);
            replay.Reset(promptArr);
            var greedySampling = SamplingOptions.Greedy;

            for (var i = 0; i < divergence; i++)
            {
                replay.GenerateNextToken(in greedySampling);
            }

            var logits = new float[replay.VocabularySize];
            replay.GetLastLogits(logits);

            var ranked = Enumerable.Range(0, logits.Length)
                .OrderByDescending(t => logits[t])
                .ToArray();

            var rankOfSpeculative = Array.IndexOf(ranked, spec[divergence]);

            _out.WriteLine("\nsingle-token logits at that position:");
            _out.WriteLine($"   rank 1: token {ranked[0]}  logit {logits[ranked[0]]:F6}");
            _out.WriteLine($"   rank 2: token {ranked[1]}  logit {logits[ranked[1]]:F6}"
                + $"   (gap to rank 1: {logits[ranked[0]] - logits[ranked[1]]:F6})");
            _out.WriteLine($"   rank 3: token {ranked[2]}  logit {logits[ranked[2]]:F6}");
            _out.WriteLine($"\n   the SPECULATIVE choice, token {spec[divergence]}: "
                + $"rank {rankOfSpeculative + 1}, logit {logits[spec[divergence]]:F6}, "
                + $"deficit {logits[ranked[0]] - logits[spec[divergence]]:F6} below rank 1");
            _out.WriteLine("\n   rank 2 with a deficit near the batched-vs-single-token difference => arithmetic.");
            _out.WriteLine("   a distant rank, or a large deficit                                 => commit logic.");
        }

        /// <summary>
        /// Measures the batched-versus-single-token logit difference directly, without speculation in the
        /// picture at all: the same context reached two ways.
        ///
        /// <para><c>Reset</c> prefills a whole token span in one batched pass, which is the same class of
        /// kernel the verify uses; stepping with <c>GenerateNextToken</c> reaches the identical context one
        /// token at a time. Any difference between the resulting logit vectors is summation order, since
        /// the weights and the context are the same. That magnitude is what decides whether an argmax flip
        /// at a near-tie is expected behaviour or a symptom.</para>
        /// </summary>
        [ModelFact(ModelPath)]
        public void HowFarApartAreBatchedAndSingleTokenLogitsOnTheSameContext()
        {
            using var engine = CachedLlamaInferenceEngine.LoadGguf(ModelPath);

            var prompt = new List<int>();

            for (var r = 0; r < 8; r++)
            {
                prompt.AddRange([10, 11, 12, 13, 14, 15]);
            }

            var promptArr = prompt.ToArray();

            foreach (var steps in new[] { 1, 4, 12 })
            {
                var stepped = new int[steps];
                float[] singleTokenLogits;

                using (var s = engine.CreateSession(256))
                {
                    s.Reset(promptArr);
                    var sampling = SamplingOptions.Greedy;

                    for (var i = 0; i < steps; i++)
                    {
                        stepped[i] = s.GenerateNextToken(in sampling);
                    }

                    singleTokenLogits = new float[s.VocabularySize];
                    s.GetLastLogits(singleTokenLogits);
                }

                float[] batchedLogits;

                using (var s = engine.CreateSession(256))
                {
                    s.Reset([.. promptArr, .. stepped]);
                    batchedLogits = new float[s.VocabularySize];
                    s.GetLastLogits(batchedLogits);
                }

                var maxAbsolute = 0f;
                var argmaxSingle = 0;
                var argmaxBatched = 0;

                for (var t = 0; t < singleTokenLogits.Length; t++)
                {
                    var difference = MathF.Abs(singleTokenLogits[t] - batchedLogits[t]);

                    if (difference > maxAbsolute)
                    {
                        maxAbsolute = difference;
                    }

                    if (singleTokenLogits[t] > singleTokenLogits[argmaxSingle])
                    {
                        argmaxSingle = t;
                    }

                    if (batchedLogits[t] > batchedLogits[argmaxBatched])
                    {
                        argmaxBatched = t;
                    }
                }

                var sorted = Enumerable.Range(0, singleTokenLogits.Length)
                    .OrderByDescending(t => singleTokenLogits[t])
                    .Take(2)
                    .ToArray();
                var topTwoGap = singleTokenLogits[sorted[0]] - singleTokenLogits[sorted[1]];

                _out.WriteLine($"after {steps,2} decoded token(s): max|delta logit| {maxAbsolute:F6}   "
                    + $"top-2 gap {topTwoGap:F6}   argmax {argmaxSingle} vs {argmaxBatched}"
                    + (argmaxSingle == argmaxBatched ? "  (agree)" : "  (DISAGREE — a flip)"));
            }

            _out.WriteLine("\nA max|delta| that exceeds the top-2 gap is a flip waiting to happen; over a long");
            _out.WriteLine("generation it eventually does, and from there the two sequences are unrelated.");
        }
    }
}
