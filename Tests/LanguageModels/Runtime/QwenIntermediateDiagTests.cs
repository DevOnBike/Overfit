// Copyright (c) 2026 DevOnBike. AGPLv3.

using DevOnBike.Overfit.LanguageModels.Contracts;
using DevOnBike.Overfit.LanguageModels.Runtime;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests.LanguageModels.Runtime
{
    [Trait("Category", "QwenIntDiag")]
    [Trait("Category", "Qwen")]
    public sealed class QwenIntermediateDiagTests
    {
        private readonly ITestOutputHelper _out;
        public QwenIntermediateDiagTests(ITestOutputHelper output) => _out = output;

        private const string ModelPath = "c:/qwen3b/qwen.bin";

        /// <summary>
        /// Compares specific token logits between C# and Python forward_pass_reference.py.
        ///
        /// Python (GGUF reference) top-1 = token 3352 (logit 14.206)
        /// C# top-1 = token 50560 (logit 14.210)
        ///
        /// Key check: C# logit[3352] vs Python logit[3352] = 14.206
        ///   - If C# logit[3352] ≈ 14.206  → hidden state same, token mapping wrong (LM head?)
        ///   - If C# logit[3352] << 14.206 → hidden state different (layer computation bug)
        /// </summary>
        [Fact]
        public void IntDiag_LogitTrace()
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
                var s = SamplingOptions.Greedy;
                session.GenerateNextToken(in s);

                var logits = session.LastLogits.ToArray();

                _out.WriteLine("=== SPECIFIC TOKEN LOGIT COMPARISON (C# vs Python reference) ===");
                _out.WriteLine($"{"Token",-8} {"C# logit",10} {"Python ref",10} {"Diff",8} Note");
                _out.WriteLine(new string('-', 60));

                var checks = new (int tok, float pyLogit, string note)[]
                {
                    (3352,   14.206f,  "Python #1"),
                    (35167,  13.082f,  "Python #2"),
                    (87253,  12.215f,  "Python #3"),
                    (101448, 11.914f,  "Python #4"),
                    (50560,  0f,       "C# #1 (not in Python top-20)"),
                    (96402,  0f,       "C# #2"),
                    (151644, -1.424f,  "<|im_start|>"),
                    (198,     2.691f,  "newline"),
                };

                foreach (var (tok, pyLogit, note) in checks)
                {
                    if (tok >= logits.Length)
                    {
                        continue;
                    }
                    var cs = logits[tok];
                    var diff = pyLogit != 0f ? cs - pyLogit : float.NaN;
                    var mark = !float.IsNaN(diff) && MathF.Abs(diff) < 0.5f ? "✓" : "?";
                    _out.WriteLine($"{tok,-8} {cs,10:F4} {pyLogit,10:F4} {diff,8:F4}  {mark} {note}");
                }

                _out.WriteLine(string.Empty);
                _out.WriteLine($"C# logit[3352]  = {logits[3352]:F6}   (Python: 14.206000)");
                _out.WriteLine($"C# logit[50560] = {logits[50560]:F6}   (C# top-1)");
                _out.WriteLine(string.Empty);
                _out.WriteLine("VERDICT:");
                var ratio = logits[3352] / 14.206f;
                if (MathF.Abs(ratio - 1f) < 0.05f)
                {
                    _out.WriteLine("  ✓ logit[3352] ≈ 14.206 → hidden state is CORRECT, LM head has token-mapping bug");
                }
                else if (logits[3352] > 10f)
                {
                    _out.WriteLine("  ~ logit[3352] is high but off → hidden state slightly wrong OR LM head off");
                }
                else
                {
                    _out.WriteLine("  ✗ logit[3352] << 14.206 → hidden state is WRONG (layer computation bug)");
                }
            }
        }
    }
}
