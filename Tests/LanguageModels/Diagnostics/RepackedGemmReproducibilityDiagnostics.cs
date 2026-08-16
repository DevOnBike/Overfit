// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels.Contracts;
using DevOnBike.Overfit.LanguageModels.Runtime;
using DevOnBike.Overfit.LanguageModels.Tokenizers;

namespace DevOnBike.Overfit.Tests.LanguageModels.Diagnostics
{
    /// <summary>
    /// Does the repacked Q4_K prefill GEMM produce the same answer twice? (T8)
    ///
    /// <para><b>Where the question comes from.</b> <c>TinyBlasTiledPrefillE2EPhase3Tests</c> believes it
    /// compares the tiled kernel against the weight-stationary one by flipping
    /// <c>UseTiledPrefillQ4K</c>. It does not: the dispatch is
    /// <c>(w.IsPrepacked || UseTiledPrefillQ4K)</c>, and a <c>*.gguf.repack</c> sidecar sits beside this
    /// model, so <b>both arms run the repacked kernel</b>. Its own timing agrees — 1203 ms versus 1156 ms,
    /// a 1.04x tie. That test is therefore an accidental A/A comparison, and on 2026-08-07 it failed one:
    /// two runs of the same kernel diverged from the very first token (<c>matched 0/24</c>). A later run
    /// of the same test passed. One failure and one pass is flakiness, not a verdict.</para>
    ///
    /// <para><b>What this measures that the other cannot.</b> Repetitions <i>inside one process</i>, on one
    /// engine, with nothing changed between them. That removes every explanation except the kernel itself:
    /// no reload, no flag, no separate process, no cold cache. If the outputs differ here, the repacked
    /// GEMM is not deterministic; the likeliest mechanism is a reduction whose order follows the parallel
    /// work split, which varies with scheduling.</para>
    ///
    /// <para><b>Why it matters beyond one test.</b> <c>IsPrepacked</c> makes this the DEFAULT path wherever
    /// a repack sidecar exists, which is ordinary usage here. Every coherence assertion in the repository
    /// that runs over a prepacked model inherits whatever this answers.</para>
    ///
    /// <para>Reports; asserts nothing. A verdict on determinism from a single run of a suspected-flaky
    /// path would be the same mistake in a new place.</para>
    /// </summary>
    [Trait("Category", "Qwen")]
    [Trait("Category", "Diagnostics")]
    public sealed class RepackedGemmReproducibilityDiagnostics
    {
        private const string ModelPath = @"C:\qwen3b\qwen.q4km.gguf";
        private const int Repetitions = 6;
        private const int GenerateTokens = 24;
        private const int Context = 2048;

        private readonly ITestOutputHelper _out;

        public RepackedGemmReproducibilityDiagnostics(ITestOutputHelper output) => _out = output;

        /// <summary>
        /// Does <c>UseTiledPrefillQ4K</c> change the output at all — and if so, is the difference stable?
        ///
        /// <para><b>Why this exists after the repetition test came back clean.</b> Twelve identical runs
        /// said the kernel reproduces itself, which weakens "non-deterministic" — but every one of them ran
        /// with the flag OFF, exactly as the sibling test's first arm does. I had claimed the flag was
        /// inert because <c>IsPrepacked</c> overrides it. That is only true for tensors the sidecar
        /// actually contains. For any Q4_K weight NOT in it, <c>(w.IsPrepacked || UseTiledPrefillQ4K)</c>
        /// really does turn on the tiled kernel — so the A/B may be partly live, and the divergence a
        /// genuine tiled-versus-weight-stationary difference rather than noise.</para>
        ///
        /// <para>Three runs per setting, in one process: the within-setting rows say whether each side is
        /// self-consistent, and the across-setting row says whether the flag does anything.</para>
        /// </summary>
        [ModelFact(ModelPath)]
        public void DoesTheTiledFlagChangeAnything()
        {
            using var engine = CachedLlamaInferenceEngine.LoadGguf(ModelPath);
            var tokenizer = GgufTokenizer.Load(ModelPath);
            var original = BatchedQuantProjection.UseTiledPrefillQ4K;

            var paragraph =
                "The history of computing is a long and winding road that begins with mechanical calculators, "
                + "passes through vacuum tubes and transistors, and arrives at the integrated circuits that power "
                + "modern processors. Each generation made machines smaller, faster, and far more capable. ";
            var prompt = tokenizer.Encode(string.Concat(Enumerable.Repeat(paragraph, 6)));

            try
            {
                var off = Generate(engine, prompt, tiled: false, times: 3);
                var on = Generate(engine, prompt, tiled: true, times: 3);

                _out.WriteLine($"prompt {prompt.Length} tokens, 3 runs per setting");
                _out.WriteLine("");
                _out.WriteLine($"flag OFF, self-consistent: {AllEqual(off)}");
                _out.WriteLine($"flag ON,  self-consistent: {AllEqual(on)}");

                var matched = 0;

                while (matched < GenerateTokens && off[0][matched] == on[0][matched])
                {
                    matched++;
                }

                _out.WriteLine($"OFF vs ON: {matched}/{GenerateTokens} tokens match");
                _out.WriteLine("");
                _out.WriteLine($"OFF: {tokenizer.Decode(off[0])[..Math.Min(80, tokenizer.Decode(off[0]).Length)]}");
                _out.WriteLine($"ON : {tokenizer.Decode(on[0])[..Math.Min(80, tokenizer.Decode(on[0]).Length)]}");
                _out.WriteLine("");

                if (matched == GenerateTokens)
                {
                    _out.WriteLine("READING: the flag changes nothing. Either the sidecar covers every Q4_K "
                                   + "weight this prompt touches — so IsPrepacked really does make the A/B "
                                   + "inert — or the two kernels agree exactly. The sibling test's 1.04x "
                                   + "timing says the former.");
                }
                else if (AllEqual(off) && AllEqual(on))
                {
                    _out.WriteLine("READING: each setting is self-consistent but they DISAGREE with each "
                                   + "other. That is not non-determinism — it is a real, reproducible "
                                   + "difference between the tiled and weight-stationary kernels, which is "
                                   + "a correctness question about one of them.");
                }
                else
                {
                    _out.WriteLine("READING: a setting disagreed with itself. Non-determinism after all — "
                                   + "and the repetition test above simply had not hit it.");
                }

                _out.WriteLine("");
                _out.WriteLine("(diagnostic — reports, does not assert)");
            }
            finally
            {
                BatchedQuantProjection.UseTiledPrefillQ4K = original;
            }
        }

        private List<int[]> Generate(
            CachedLlamaInferenceEngine engine, int[] prompt, bool tiled, int times)
        {
            BatchedQuantProjection.UseTiledPrefillQ4K = tiled;
            var runs = new List<int[]>(times);

            for (var attempt = 0; attempt < times; attempt++)
            {
                using var session = engine.CreateSession(Context);
                session.Reset(prompt);
                var sampling = SamplingOptions.Greedy;
                var produced = new int[GenerateTokens];

                for (var i = 0; i < GenerateTokens && !session.IsFull; i++)
                {
                    produced[i] = session.GenerateNextToken(in sampling);
                }

                runs.Add(produced);
            }

            return runs;
        }

        private static bool AllEqual(List<int[]> runs)
        {
            foreach (var run in runs)
            {
                if (!run.SequenceEqual(runs[0]))
                {
                    return false;
                }
            }

            return true;
        }

        [ModelFact(ModelPath)]
        public void SameConfigurationRepeated_DoesItReproduce()
        {
            using var engine = CachedLlamaInferenceEngine.LoadGguf(ModelPath);
            var tokenizer = GgufTokenizer.Load(ModelPath);

            // The prompt from the tiled test: long enough to reach the batched prefill and the NR=8 tile
            // regime, which is where the reassociation being investigated actually happens.
            var paragraph =
                "The history of computing is a long and winding road that begins with mechanical calculators, "
                + "passes through vacuum tubes and transistors, and arrives at the integrated circuits that power "
                + "modern processors. Each generation made machines smaller, faster, and far more capable. ";
            var text = string.Concat(Enumerable.Repeat(paragraph, 6));
            var prompt = tokenizer.Encode(text);
            _out.WriteLine($"prompt {prompt.Length} tokens, {Repetitions} repetitions, "
                           + $"{GenerateTokens} greedy tokens each");
            _out.WriteLine($"IsPrepacked path in use: a .repack sidecar makes the repacked kernel the "
                           + $"default regardless of UseTiledPrefillQ4K (currently "
                           + $"{BatchedQuantProjection.UseTiledPrefillQ4K})");
            _out.WriteLine("");

            var runs = new List<int[]>(Repetitions);

            for (var repetition = 0; repetition < Repetitions; repetition++)
            {
                using var session = engine.CreateSession(Context);
                session.Reset(prompt);
                var sampling = SamplingOptions.Greedy;
                var produced = new int[GenerateTokens];

                for (var i = 0; i < GenerateTokens && !session.IsFull; i++)
                {
                    produced[i] = session.GenerateNextToken(in sampling);
                }

                runs.Add(produced);
            }

            var first = runs[0];
            var identical = 0;

            _out.WriteLine("run   matches run 0   first divergence");

            for (var repetition = 0; repetition < runs.Count; repetition++)
            {
                var matched = 0;

                while (matched < GenerateTokens && runs[repetition][matched] == first[matched])
                {
                    matched++;
                }

                if (matched == GenerateTokens)
                {
                    identical++;
                }

                _out.WriteLine($"{repetition,3}   {matched,13}/{GenerateTokens}   "
                               + (matched == GenerateTokens ? "—" : $"token {matched}"));
            }

            _out.WriteLine("");
            _out.WriteLine($"identical to run 0: {identical}/{runs.Count}");
            _out.WriteLine($"run 0: {tokenizer.Decode(first)[..Math.Min(90, tokenizer.Decode(first).Length)]}");

            if (identical == runs.Count)
            {
                _out.WriteLine("");
                _out.WriteLine("READING: every repetition agreed. This run shows no non-determinism — which "
                               + "is NOT the same as proving there is none: the failure that raised T8 was "
                               + "intermittent, and an intermittent fault can hide behind any number of "
                               + "clean runs. It bounds the rate, it does not clear the kernel.");
            }
            else
            {
                _out.WriteLine("");
                _out.WriteLine("READING: repetitions of an IDENTICAL configuration, in one process, on one "
                               + "engine, produced different tokens. Nothing varies between them except "
                               + "execution, so the repacked GEMM is not deterministic. Every coherence "
                               + "assertion over a prepacked model is measuring luck.");
            }

            _out.WriteLine("");
            _out.WriteLine("(diagnostic — reports, does not assert)");
        }
    }
}
