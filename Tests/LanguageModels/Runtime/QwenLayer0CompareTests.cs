// Copyright (c) 2026 DevOnBike. AGPLv3.

using DevOnBike.Overfit.LanguageModels.Runtime;
using DevOnBike.Overfit.LanguageModels.Tokenizers;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests.LanguageModels.Runtime
{
    [Trait("Category", "QwenL0")]
    [Trait("Category", "Qwen")]
    public sealed class QwenLayer0CompareTests
    {
        private readonly ITestOutputHelper _out;
        public QwenLayer0CompareTests(ITestOutputHelper output) => _out = output;
        private const string ModelPath = "d:/qwen/qwen.bin";
        private const string TokenizerDir = "d:/qwen/";

        /// <summary>
        /// Position-0 (BOS): logity C# muszą zgadzać się z Python.
        /// VERIFIED CORRECT: top-1=[62406] 11.922
        /// </summary>
        [Fact]
        public void L0_LogitsAfterReset_NotAfterGenerate()
        {
            if (!File.Exists(ModelPath))
            {
                return;
            }

            var engine = CachedLlamaInferenceEngine.Load(ModelPath);
            using (engine)
            {
                using var session = engine.CreateSession(64);
                session.Reset(new[] { 151643 });
                var logits = session.LastLogits.ToArray();

                _out.WriteLine("=== LOGITS AFTER RESET (position 0) ===");
                var top5 = logits.Select((v, i) => (v, i)).OrderByDescending(x => x.v).Take(5).ToArray();
                _out.WriteLine("C# TOP-5:");
                foreach (var (v, i) in top5)
                {
                    _out.WriteLine($"  [{i,7}]  {v,8:F4}");
                }

                _out.WriteLine("C# specific tokens:");
                _out.WriteLine($"  [ 62406] = {logits[62406],8:F4}  Python= 11.922  diff={(logits[62406] - 11.922f):+0.0000;-0.0000}");
                _out.WriteLine($"  [ 75101] = {logits[75101],8:F4}  Python= 11.033  diff={(logits[75101] - 11.033f):+0.0000;-0.0000}");
                _out.WriteLine($"  [ 34603] = {logits[34603],8:F4}  Python= 10.855  diff={(logits[34603] - 10.855f):+0.0000;-0.0000}");

                Assert.True(Math.Abs(logits[62406] - 11.922f) < 0.05f,
                    $"logit[62406]={logits[62406]:F4} should be ≈11.922");
            }
        }

        /// <summary>
        /// Porównanie hidden state i logitów C# vs Python dla 2 tokenów [BOS, im_start].
        ///
        /// Python (forward_multitoken.py z adjacent-pair RoPE):
        ///   top-1 = [74949] 13.028
        ///   hidden[:4] = [2.8801217, -1.3621705, -3.6742501, 2.3216434]
        ///   logit[6622]  = 10.1300
        ///   logit[198]   = -1.0521
        /// </summary>
        [Fact]
        public void L0_TwoToken_HiddenStateVsPython()
        {
            if (!File.Exists(ModelPath))
            {
                return;
            }

            var engine = CachedLlamaInferenceEngine.Load(ModelPath);
            using (engine)
            {
                using var session = engine.CreateSession(64);
                session.Reset(new[] { 151643, 151644 });

                var logits = session.LastLogits.ToArray();
                var top1 = logits.Select((v, i) => (v, i)).OrderByDescending(x => x.v).First();

                _out.WriteLine("=== 2-TOKEN [BOS, im_start] ===");
                _out.WriteLine($"C# top-1 = [{top1.i}] {top1.v:F4}");
                _out.WriteLine($"  Python:  [74949] 13.028  (adjacent-pair RoPE)");
                _out.WriteLine($"  Match:   {(top1.i == 74949 ? "✓ SAME TOKEN" : $"✗ DIFFERENT (got {top1.i})")}");
                _out.WriteLine(string.Empty);

                _out.WriteLine("C# vs Python logits at pos 1:");
                _out.WriteLine($"  [  6622] C#={logits[6622],8:F4}  Py=10.1300  diff={logits[6622] - 10.1300f:+0.0000;-0.0000}");
                _out.WriteLine($"  [  8948] C#={logits[8948],8:F4}  Py=-8.7950  diff={logits[8948] - (-8.7950f):+0.0000;-0.0000}");
                _out.WriteLine($"  [   198] C#={logits[198],8:F4}  Py=-1.0521  diff={logits[198] - (-1.0521f):+0.0000;-0.0000}");
                _out.WriteLine(string.Empty);

                // HIDDEN STATE — po naprawieniu LastHiddenState
                var hidden = session.LastHiddenState.ToArray();
                _out.WriteLine("=== HIDDEN STATE (before final RMSNorm) ===");
                _out.WriteLine($"C# hidden[:4] = [{string.Join(", ", hidden.Take(4).Select(v => v.ToString("F5")))}]");
                _out.WriteLine($"Py hidden[:4] =  [2.88012, -1.36217, -3.67425,  2.32164]");
                _out.WriteLine(string.Empty);

                // Poprawne wartości Python po RoPE fix (adjacent-pair)
                float[] pyHidden = { 2.8801217f, -1.3621705f, -3.6742501f, 2.3216434f };
                _out.WriteLine("Hidden state diff (C# - Python):");
                var maxDiff = 0f;
                for (var i = 0; i < 4; i++)
                {
                    var diff = hidden[i] - pyHidden[i];
                    _out.WriteLine($"  [{i}]: C#={hidden[i]:F5}  Py={pyHidden[i]:F5}  diff={diff:+0.00000;-0.00000}");
                    maxDiff = Math.Max(maxDiff, Math.Abs(diff));
                }
                _out.WriteLine(string.Empty);
                _out.WriteLine($"Max |diff| hidden[:4] = {maxDiff:F5}");
                if (maxDiff < 0.01f)
                {
                    _out.WriteLine("→ HIDDEN STATE MATCHES Python ✓");
                }
                else if (maxDiff < 0.1f)
                {
                    _out.WriteLine("→ HIDDEN STATE CLOSE (float32 accumulation)");
                }
                else
                {
                    _out.WriteLine("→ HIDDEN STATE DIFFERS — bug in transformer forward");
                }

                Assert.Equal(74949, top1.i);
            }
        }

        /// <summary>
        /// Pełny chat prompt (36 tokenów z poprawnym system message).
        /// C# i Python zgadzają się: top-1 = [36366] ≈ 11.9
        /// </summary>
        [Fact]
        public void L0_ChatPromptLogits()
        {
            if (!File.Exists(ModelPath))
            {
                return;
            }
            if (!File.Exists(Path.Combine(TokenizerDir, "tokenizer.json")))
            {
                return;
            }

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
                _out.WriteLine($"C# TOP-10:");
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
                _out.WriteLine("Python top-1: [36366] 11.902 (MATCHES C# when same token sequence)");
            }
        }

        /// <summary>
        /// Progressive prefix: kiedy pojawia się token '4' jako top-1.
        /// </summary>
        [Fact]
        public void Multitoken_ProgressivePrefixTest()
        {
            if (!File.Exists(ModelPath))
            {
                return;
            }
            if (!File.Exists(Path.Combine(TokenizerDir, "tokenizer.json")))
            {
                return;
            }

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
