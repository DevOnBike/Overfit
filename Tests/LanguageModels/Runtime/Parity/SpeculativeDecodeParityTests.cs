// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Collections.Generic;
using DevOnBike.Overfit.LanguageModels.Contracts;
using DevOnBike.Overfit.LanguageModels.Runtime;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests.LanguageModels.Runtime.Parity
{
    /// <summary>
    /// Greedy speculative decoding (prompt-lookup) must produce the EXACT same token sequence as plain
    /// greedy single-token decoding — verification only accepts what greedy would emit, so it's a pure
    /// speedup. Verified on real Qwen2.5-3B Q4_K_M, with a deliberately repetitive prompt so the
    /// prompt-lookup drafter actually fires (and accepts) tokens. [LongFact].
    /// </summary>
    [Trait("Category", "Qwen")]
    [Trait("Category", "Parity")]
    public sealed class SpeculativeDecodeParityTests
    {
        private const string ModelPath = @"C:\qwen3b\qwen.q4km.gguf";

        private readonly ITestOutputHelper _out;
        public SpeculativeDecodeParityTests(ITestOutputHelper output) => _out = output;

        [LongFact]
        public void Speculative_ProducesIdenticalSequence_ToGreedy()
        {
            if (!File.Exists(ModelPath)) { _out.WriteLine($"missing {ModelPath}"); return; }

            using var engine = CachedLlamaInferenceEngine.LoadGguf(ModelPath);

            // Repetitive prompt → the n-gram drafter finds matches and accepts drafts.
            var prompt = new List<int>();
            for (var r = 0; r < 8; r++) { prompt.AddRange([10, 11, 12, 13, 14, 15]); }
            var promptArr = prompt.ToArray();
            const int generate = 40;

            // Reference: plain greedy, single token at a time.
            var greedy = new List<int>();
            using (var s = engine.CreateSession(256))
            {
                s.Reset(promptArr);
                var sampling = SamplingOptions.Greedy;
                for (var i = 0; i < generate; i++) { greedy.Add(s.GenerateNextToken(in sampling)); }
            }

            // Speculative greedy: commits ≥1 token per batched verify.
            var spec = new List<int>();
            var anyMultiCommit = false;
            using (var s = engine.CreateSession(256))
            {
                s.Reset(promptArr);
                var history = new List<int>(promptArr);
                var committed = new int[6];   // maxDraft(4) + 2 (t0 + drafts + bonus)
                while (spec.Count < generate)
                {
                    var n = s.GenerateSpeculative(System.Runtime.InteropServices.CollectionsMarshal.AsSpan(history), committed, maxDraft: 4);
                    if (n > 1) { anyMultiCommit = true; }
                    for (var c = 0; c < n; c++) { spec.Add(committed[c]); history.Add(committed[c]); }
                }
            }

            _out.WriteLine($"greedy[0..8]=[{string.Join(",", greedy.GetRange(0, 8))}]  multiCommit={anyMultiCommit}");

            // Identical token sequence (speculative is exact for greedy), and the drafter actually fired.
            for (var i = 0; i < generate; i++)
            {
                Assert.Equal(greedy[i], spec[i]);
            }
            Assert.True(anyMultiCommit, "speculative never committed >1 token — drafter/verify not exercised.");
        }

        [LongFact]
        public void Speculative_DecodeSpeedup_OnRepetitiveText()
        {
            if (!File.Exists(ModelPath)) { _out.WriteLine($"missing {ModelPath}"); return; }
            if (!File.Exists(@"C:\qwen3b\tokenizer.json")) { _out.WriteLine("no tokenizer"); return; }

            using var engine = CachedLlamaInferenceEngine.LoadGguf(ModelPath);
            var tok = DevOnBike.Overfit.LanguageModels.Tokenizers.QwenTokenizer.Load(@"C:\qwen3b");

            // A genuinely-echoing prompt: the model continues the repeated sentence and the n-gram
            // drafter (matching the earlier repeats) proposes exactly what greedy emits → high acceptance.
            const string text = "The quick brown fox jumps over the lazy dog. " +
                                 "The quick brown fox jumps over the lazy dog. " +
                                 "The quick brown fox jumps over the lazy dog. " +
                                 "The quick brown fox jumps over the lazy dog. ";
            var promptArr = tok.Encode(text, addBos: false);
            const int generate = 96;

            // Acceptance signal: average tokens committed per speculative step.
            {
                using var s = engine.CreateSession(512);
                s.Reset(promptArr);
                var hist = new List<int>(promptArr);
                var buf = new int[6];
                int steps = 0, produced = 0;
                while (produced < generate)
                {
                    var n = s.GenerateSpeculative(System.Runtime.InteropServices.CollectionsMarshal.AsSpan(hist), buf, maxDraft: 4);
                    for (var c = 0; c < n; c++) { hist.Add(buf[c]); }
                    produced += n; steps++;
                }
                _out.WriteLine($"avg tokens/step = {(double)produced / steps:F2} ({produced} in {steps} steps)");
            }

            double Single()
            {
                using var s = engine.CreateSession(512);
                s.Reset(promptArr);
                var sampling = SamplingOptions.Greedy;
                var sw = System.Diagnostics.Stopwatch.StartNew();
                for (var i = 0; i < generate; i++) { s.GenerateNextToken(in sampling); }
                sw.Stop();
                return generate / sw.Elapsed.TotalSeconds;
            }

            double Speculative()
            {
                using var s = engine.CreateSession(512);
                s.Reset(promptArr);
                var history = new List<int>(promptArr);
                var committed = new int[6];
                var produced = 0;
                var sw = System.Diagnostics.Stopwatch.StartNew();
                while (produced < generate)
                {
                    var n = s.GenerateSpeculative(System.Runtime.InteropServices.CollectionsMarshal.AsSpan(history), committed, maxDraft: 4);
                    for (var c = 0; c < n; c++) { history.Add(committed[c]); }
                    produced += n;
                }
                sw.Stop();
                return produced / sw.Elapsed.TotalSeconds;
            }

            var single = Single();
            var spec = Speculative();
            _out.WriteLine($"decode tok/s: single={single:F2}  speculative={spec:F2}  speedup={spec / single:F2}×");
        }
    }
}
