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
        private const string ModelPath    = "d:/qwen/qwen.bin";
        private const string TokenizerDir = "d:/qwen/";

        /// <summary>
        /// Porównanie position-0: logity C# vs Python.
        /// VERIFIED CORRECT: top-1=[62406] 11.922 ← matches Python.
        /// </summary>
        [Fact]
        public void L0_LogitsAfterReset_NotAfterGenerate()
        {
            if (!File.Exists(ModelPath)) return;

            var engine = CachedLlamaInferenceEngine.Load(ModelPath);
            using (engine)
            {
                using var session = engine.CreateSession(64);

                session.Reset(new[] { 151643 });
                var logits = session.LastLogits.ToArray();

                _out.WriteLine("=== LOGITS AFTER RESET (position 0 — correct comparison) ===");
                _out.WriteLine("C# TOP-5:");
                var top5 = logits.Select((v, i) => (v, i)).OrderByDescending(x => x.v).Take(5).ToArray();
                foreach (var (v, i) in top5)
                    _out.WriteLine($"  [{i,7}]  {v,8:F4}");

                _out.WriteLine("C# specific tokens:");
                _out.WriteLine($"  [ 62406] = {logits[62406],8:F4}  Python= 11.922  diff={(logits[62406]-11.922f):+0.0000;-0.0000}");
                _out.WriteLine($"  [ 75101] = {logits[75101],8:F4}  Python= 11.033  diff={(logits[75101]-11.033f):+0.0000;-0.0000}");
                _out.WriteLine($"  [ 34603] = {logits[34603],8:F4}  Python= 10.855  diff={(logits[34603]-10.855f):+0.0000;-0.0000}");
            }
        }

        /// <summary>
        /// KLUCZOWY TEST: porównanie hidden state C# vs Python dla 2 tokenów.
        ///
        /// Python (forward_multitoken.py) daje:
        ///   position 1 hidden x[:4] = [1.1300201, -0.91244936, -3.6715374, 4.397436]
        ///   top-1 = [6622] 11.677
        ///   logit[8948]='system' = -6.78
        ///
        /// C# (TwoTokenCheck) daje:
        ///   top-1 = [6622] 13.457   ← inny moduł! (+1.78 vs Python)
        ///   logit[8948]='system' = -7.93
        ///
        /// Ten test sprawdza czy hidden state przed finalnym RMSNorm zgadza się z Python.
        /// </summary>
        [Fact]
        public void L0_TwoToken_HiddenStateVsPython()
        {
            if (!File.Exists(ModelPath)) return;

            var engine = CachedLlamaInferenceEngine.Load(ModelPath);
            using (engine)
            {
                using var session = engine.CreateSession(64);

                // Process [BOS, im_start]
                session.Reset(new[] { 151643, 151644 });

                var logits = session.LastLogits.ToArray();
                var top1 = logits.Select((v, i) => (v, i)).OrderByDescending(x => x.v).First();

                _out.WriteLine("=== 2-TOKEN [BOS, im_start] HIDDEN STATE ===");
                _out.WriteLine($"C# top-1 = [{top1.i}] {top1.v:F4}");
                _out.WriteLine($"  Python:  [{6622}] 11.677   (should be same token, similar magnitude)");
                _out.WriteLine($"  Delta:   {top1.v - 11.677f:+0.000;-0.000} logit units");

                _out.WriteLine("C# vs Python logits at pos 1:");
                _out.WriteLine($"  [  6622] C#={logits[6622],8:F4}  Py=11.6773  diff={logits[6622]-11.6773f:+0.0000;-0.0000}");
                _out.WriteLine($"  [  8948] C#={logits[8948],8:F4}  Py=-6.7819  diff={logits[8948]-(-6.7819f):+0.0000;-0.0000}");
                _out.WriteLine($"  [   198] C#={logits[198],8:F4}  Py= 4.9264  diff={logits[198]-4.9264f:+0.0000;-0.0000}");
                _out.WriteLine($"  [   271] C#={logits[271],8:F4}  Py=10.9256  diff={logits[271]-10.9256f:+0.0000;-0.0000}");
                _out.WriteLine($"  [  3407] C#={logits[3407],8:F4}  Py=10.9029  diff={logits[3407]-10.9029f:+0.0000;-0.0000}");

                // PORÓWNANIE HIDDEN STATE
                // Python hidden x[:8] at position 1 (before final RMSNorm):
                //   [1.1300201, -0.91244936, -3.6715374, 4.397436, ?, ?, ?, ?]
                // session.LastHiddenState musi być dostępny
                var hidden = session.LastHiddenState.ToArray();
                _out.WriteLine("=== HIDDEN STATE (before final RMSNorm) ===");
                _out.WriteLine($"C# hidden[:8] = [{string.Join(", ", hidden.Take(8).Select(v => v.ToString("F5")))}]");
                _out.WriteLine($"Py hidden[:4] =  [1.13002, -0.91245, -3.67154,  4.39744]");
                _out.WriteLine("Hidden state diff (C# - Python) for first 4 elements:");
                float[] pyHidden = { 1.1300201f, -0.91244936f, -3.6715374f, 4.397436f };
                for (var i = 0; i < 4; i++)
                    _out.WriteLine($"  [{i}]: C#={hidden[i]:F5}  Py={pyHidden[i]:F5}  diff={hidden[i]-pyHidden[i]:+0.00000;-0.00000}");

                // Analiza: jeżeli hidden state się zgadza, bug jest w LM head projekcji
                // Jeżeli nie, bug jest w transformer forward pass dla pos>0
                var maxDiff = pyHidden.Select((v, i) => Math.Abs(hidden[i] - v)).Max();
                _out.WriteLine($"Max |diff| hidden[:4] = {maxDiff:F5}");
                if (maxDiff < 0.01f)
                    _out.WriteLine("→ HIDDEN STATE MATCHES Python → bug in LM head projection");
                else
                    _out.WriteLine("→ HIDDEN STATE DIFFERS from Python → bug in transformer forward (attention/FFN)");
            }
        }

        /// <summary>
        /// Pełny chat prompt (26 tokenów) — sprawdzamy czy Python i C# zgadzają się na pozycji 25.
        /// Python reference musi być uruchomiony z forward_multitoken.py z pełnym promptem.
        /// </summary>
        [Fact]
        public void L0_ChatPromptLogits()
        {
            if (!File.Exists(ModelPath)) return;
            if (!File.Exists(Path.Combine(TokenizerDir, "tokenizer.json"))) return;

            var engine = CachedLlamaInferenceEngine.Load(ModelPath);
            var tok    = QwenTokenizer.Load(TokenizerDir);
            using (engine)
            {
                using var session = engine.CreateSession(64);

                var fullPrompt = tok.BuildChatPrompt("What is 2+2?");
                session.Reset(fullPrompt);
                var logits = session.LastLogits.ToArray();

                _out.WriteLine($"=== CHAT PROMPT LOGITS (position {fullPrompt.Length - 1}) ===");
                _out.WriteLine($"C# TOP-10 (after 'What is 2+2?<|im_start|>assistant\\n'):");
                var top10 = logits.Select((v, i) => (v, i)).OrderByDescending(x => x.v).Take(10).ToArray();
                for (var r = 0; r < top10.Length; r++)
                {
                    var (v, id) = top10[r];
                    var dec = tok.DecodeToken(id).Replace("\n", "\\n");
                    _out.WriteLine($"  #{r+1,2}  [{id,7}]  {v,8:F3}  '{dec}'");
                }
                _out.WriteLine($"Working model should predict: '4' or ' ' or 'The' etc.");
                _out.WriteLine($"logit[19]=' 4'  = {logits[19],8:F4}");
                _out.WriteLine($"logit[220]=' ' = {logits[220],8:F4}");
            }
        }

        /// <summary>
        /// Progressive prefix: każdy prefix o 1 token dłuższy → kiedy pojawia się rozbieżność.
        /// </summary>
        [Fact]
        public void Multitoken_ProgressivePrefixTest()
        {
            if (!File.Exists(ModelPath)) return;
            if (!File.Exists(Path.Combine(TokenizerDir, "tokenizer.json"))) return;

            var engine = CachedLlamaInferenceEngine.Load(ModelPath);
            var tok    = QwenTokenizer.Load(TokenizerDir);
            using (engine)
            {
                using var session = engine.CreateSession(64);

                var fullPrompt = tok.BuildChatPrompt("What is 2+2?");
                _out.WriteLine($"Full chat prompt: {fullPrompt.Length} tokens");
                _out.WriteLine($"Tokens: [{string.Join(", ", fullPrompt)}]");
                _out.WriteLine(string.Empty);

                _out.WriteLine($"{"Len",4}  {"Top-1 Token",8}  {"Top-1 Logit",10}  {"Top-1 Decoded",-20}");
                _out.WriteLine(new string('-', 55));

                for (var n = 1; n <= Math.Min(fullPrompt.Length, 28); n++)
                {
                    var prefix = fullPrompt.Take(n).ToArray();
                    session.Reset(prefix);
                    var logits = session.LastLogits.ToArray();

                    var top1 = logits.Select((v, i) => (v, i)).OrderByDescending(x => x.v).First();
                    var topDec = tok.DecodeToken(top1.i).Replace("\n", "\\n");
                    _out.WriteLine($"{n,4}  [{top1.i,7}]  {top1.v,10:F3}  '{topDec}'");
                }
            }
        }
    }
}
