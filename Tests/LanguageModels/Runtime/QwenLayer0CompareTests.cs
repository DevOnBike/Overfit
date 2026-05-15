// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels.Runtime;
using DevOnBike.Overfit.LanguageModels.Tokenizers;
using DevOnBike.Overfit.Tests.TestSupport;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests.LanguageModels.Runtime
{
    [Trait("Category", "QwenL0")]
    [Trait("Category", "Qwen")]
    public sealed class QwenLayer0CompareTests
    {
        private readonly ITestOutputHelper _out;
        public QwenLayer0CompareTests(ITestOutputHelper output) => _out = output;
        private static string ModelPath => TestModelPaths.Qwen3B.BinaryPath;
        // Original code had TokenizerDir = "c:/qwen/" (typo — pointed at a non-existent
        // sibling dir). The model + tokenizer live under the same Qwen3B root.
        private static string TokenizerDir => TestModelPaths.Qwen3B.Dir;

        /// <summary>
        /// Position-0 (BOS): logity C# muszą zgadzać się z Python forward_multitoken.py.
        /// Aktualna fixture: Qwen2.5-3B-Instruct FP16 (36 layers, head_dim=128).
        /// Python TEST 1 → top-1 = [33975] 15.5608.
        /// </summary>
        [LongFact]
        public void L0_LogitsAfterReset_NotAfterGenerate()
        {
            TestModelPaths.Qwen3B.RequireBinaryPath();

            var engine = CachedLlamaInferenceEngine.Load(ModelPath);
            using (engine)
            {
                using var session = engine.CreateSession(64);
                session.Reset([151643]);
                var logits = session.LastLogits.ToArray();

                _out.WriteLine("=== LOGITS AFTER RESET (position 0) ===");
                var top5 = logits.Select((v, i) => (v, i)).OrderByDescending(x => x.v).Take(5).ToArray();
                _out.WriteLine("C# TOP-5:");
                foreach (var (v, i) in top5)
                {
                    _out.WriteLine($"  [{i,7}]  {v,8:F4}");
                }

                // Top-1 expected: [33975] 15.5608 (from forward_multitoken.py TEST 1 for 3B FP16)
                var top1 = top5[0];
                _out.WriteLine($"C# top-1   = [{top1.i}] {top1.v:F4}");
                _out.WriteLine("Python top-1 = [33975] 15.5608");

                Assert.Equal(33975, top1.i);
                Assert.True(Math.Abs(top1.v - 15.5608f) < 0.1f,
                    $"logit[33975]={top1.v:F4} should be ≈15.5608 (got diff={top1.v - 15.5608f:F4})");
            }
        }

        /// <summary>
        /// Porównanie hidden state i logitów C# vs Python dla 2 tokenów [BOS, im_start].
        ///
        /// Python (forward_multitoken.py post-fix, grouped GQA, Qwen2.5-3B FP16):
        ///   top-1 = [198] 12.3511
        ///   hidden[:4] = [0.14059, 0.84549, 1.01591, -1.83366]
        ///   logit[198] = 12.3511
        /// </summary>
        [LongFact]
        public void L0_TwoToken_HiddenStateVsPython()
        {
            TestModelPaths.Qwen3B.RequireBinaryPath();

            var engine = CachedLlamaInferenceEngine.Load(ModelPath);
            using (engine)
            {
                using var session = engine.CreateSession(64);
                session.Reset([151643, 151644]);

                var logits = session.LastLogits.ToArray();
                var top1 = logits.Select((v, i) => (v, i)).OrderByDescending(x => x.v).First();

                _out.WriteLine("=== 2-TOKEN [BOS, im_start] ===");
                _out.WriteLine($"C# top-1 = [{top1.i}] {top1.v:F4}");
                _out.WriteLine("  Python:  [198] 12.3511  (grouped GQA)");
                _out.WriteLine($"  Match:   {(top1.i == 198 ? "✓ SAME TOKEN" : $"✗ DIFFERENT (got {top1.i})")}");
                _out.WriteLine(string.Empty);

                // HIDDEN STATE comparison
                var hidden = session.LastHiddenState.ToArray();
                _out.WriteLine("=== HIDDEN STATE (before final RMSNorm) ===");
                _out.WriteLine($"C# hidden[:4] = [{string.Join(", ", hidden.Take(4).Select(v => v.ToString("F5")))}]");
                _out.WriteLine("Py hidden[:4] = [0.14059, 0.84549, 1.01591, -1.83366]");
                _out.WriteLine(string.Empty);

                float[] pyHidden = [0.14059351f, 0.8454995f, 1.0159123f, -1.8336577f];
                _out.WriteLine("Hidden state diff (C# - Python):");
                var maxDiff = 0f;
                for (var i = 0; i < 4; i++)
                {
                    var diff = hidden[i] - pyHidden[i];
                    _out.WriteLine($"  [{i}]: C#={hidden[i]:F5}  Py={pyHidden[i]:F5}  diff={diff:+0.00000;-0.00000}");
                    maxDiff = Math.Max(maxDiff, Math.Abs(diff));
                }
                _out.WriteLine($"Max |diff| hidden[:4] = {maxDiff:F5}");

                if (maxDiff < 0.05f)
                {
                    _out.WriteLine("→ HIDDEN STATE MATCHES Python ✓");
                }
                else
                {
                    _out.WriteLine($"→ HIDDEN STATE DIFFERS — bug in transformer forward (diff={maxDiff:F5})");
                }

                Assert.Equal(198, top1.i);
                Assert.True(maxDiff < 0.05f, $"Hidden state must match Python within float32 noise, got {maxDiff:F5}");
            }
        }

        /// <summary>
        /// Pełny chat prompt (36 tokenów z poprawnym system message).
        /// C# i Python zgadzają się: top-1 = [36366] ≈ 11.9
        /// </summary>
        [LongFact]
        public void L0_ChatPromptLogits()
        {
            TestModelPaths.Qwen3B.RequireBinaryPath();
            TestModelPaths.Qwen3B.RequireTokenizerJsonPath();

            var engine = CachedLlamaInferenceEngine.Load(ModelPath);
            var tok = QwenTokenizer.Load(TokenizerDir);
            using (engine)
            {
                using var session = engine.CreateSession(64);
                var fullPrompt = tok.BuildChatPrompt("What is 2+2?");
                session.Reset(fullPrompt);
                var logits = session.LastLogits.ToArray();

                _out.WriteLine($"=== CHAT PROMPT LOGITS (position {fullPrompt.Length - 1}) ===");
                _out.WriteLine($"Prompt: {fullPrompt.Length} tokens");
                _out.WriteLine("C# TOP-10:");
                var top10 = logits.Select((v, i) => (v, i)).OrderByDescending(x => x.v).Take(10).ToArray();
                for (var r = 0; r < top10.Length; r++)
                {
                    var (v, id) = top10[r];
                    var dec = tok.DecodeToken(id).Replace("\n", "\\n");
                    _out.WriteLine($"  #{r + 1,2}  [{id,7}]  {v,8:F3}  '{dec}'");
                }
                _out.WriteLine($"logit[19]=' 4'  = {logits[19],8:F4}");
                _out.WriteLine($"logit[220]=' ' = {logits[220],8:F4}");
                _out.WriteLine(string.Empty);
                _out.WriteLine("Python top-1: [151644] 22.36 (post-fix 3B FP16 grouped GQA)");
            }
        }

        /// <summary>
        /// Progressive prefix: kiedy pojawia się token '4' jako top-1.
        /// </summary>
        [LongFact]
        public void Multitoken_ProgressivePrefixTest()
        {
            TestModelPaths.Qwen3B.RequireBinaryPath();
            TestModelPaths.Qwen3B.RequireTokenizerJsonPath();

            var engine = CachedLlamaInferenceEngine.Load(ModelPath);
            var tok = QwenTokenizer.Load(TokenizerDir);
            using (engine)
            {
                using var session = engine.CreateSession(64);
                var fullPrompt = tok.BuildChatPrompt("What is 2+2?");
                _out.WriteLine($"Full chat prompt: {fullPrompt.Length} tokens");
                _out.WriteLine($"Tokens: [{string.Join(", ", fullPrompt)}]");
                _out.WriteLine(string.Empty);

                _out.WriteLine($"{"Len",4}  {"Top-1 Token",8}  {"Top-1 Logit",10}  {"Top-1 Decoded",-20}");
                _out.WriteLine(new string('-', 55));

                for (var n = 1; n <= Math.Min(fullPrompt.Length, 36); n++)
                {
                    session.Reset(fullPrompt.Take(n).ToArray());
                    var logits = session.LastLogits.ToArray();
                    var top1 = logits.Select((v, i) => (v, i)).OrderByDescending(x => x.v).First();
                    var dec = tok.DecodeToken(top1.i).Replace("\n", "\\n");
                    _out.WriteLine($"{n,4}  [{top1.i,7}]  {top1.v,10:F3}  '{dec}'");
                }
            }
        }
    }
}
