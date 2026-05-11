// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels.Contracts;
using DevOnBike.Overfit.LanguageModels.Runtime;
using DevOnBike.Overfit.LanguageModels.Tokenizers;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests.LanguageModels.Runtime
{
    [Trait("Category", "QwenNumDiag")]
    [Trait("Category", "Qwen")]
    public sealed class QwenNumericalDiagnosticTests
    {
        private readonly ITestOutputHelper _out;
        public QwenNumericalDiagnosticTests(ITestOutputHelper output) => _out = output;

        private const string ModelPath = "c:/qwen3b/qwen.bin";
        private const string TokenizerDir = "c:/qwen3b/";

        [Fact]
        public void NumDiag_BosToken_Top20_Logits()
        {
            if (!File.Exists(ModelPath))
            {
                return;
            }
            var engine = CachedLlamaInferenceEngine.Load(ModelPath);
            var tok = File.Exists(Path.Combine(TokenizerDir, "tokenizer.json"))
                ? QwenTokenizer.Load(TokenizerDir) : null;
            using (engine)
            {
                using var session = engine.CreateSession(64);
                session.Reset(new[] { 151643 });
                var s = SamplingOptions.Greedy;
                session.GenerateNextToken(in s);
                var logits = session.LastLogits.ToArray();
                var top20 = logits.Select((v, i) => (v, i))
                    .OrderByDescending(x => x.v).Take(20).ToArray();

                _out.WriteLine($"Logit range: [{logits.Min():F3}, {logits.Max():F3}]  std={Std(logits):F3}");
                _out.WriteLine("TOP-20 after BOS:");
                foreach (var ((v, id), rank) in top20.Select((x, r) => (x, r)))
                {
                    _out.WriteLine($"  #{rank + 1,2}  [{id,7}]  {v,8:F3}  '{tok?.DecodeToken(id) ?? ""}' ");
                }

                _out.WriteLine($"\nlogits[151644]=<|im_start|> = {logits[151644]:F4}");
                _out.WriteLine($"logits[198]=\\n              = {logits[198]:F4}");
                _out.WriteLine($"logits[358]=' I'           = {logits[358]:F4}");
            }
        }

        [Fact]
        public void NumDiag_AssistantPrompt_Top20_Logits()
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
                // Full chat prompt ending with <|im_start|>assistant\n
                var prompt = tok.BuildChatPrompt("What is 2+2?");
                session.Reset(prompt);
                var s = SamplingOptions.Greedy;
                session.GenerateNextToken(in s);
                var logits = session.LastLogits.ToArray();
                var top20 = logits.Select((v, i) => (v, i))
                    .OrderByDescending(x => x.v).Take(20).ToArray();

                _out.WriteLine($"Prompt: {prompt.Length} tokens");
                _out.WriteLine($"Logit range: [{logits.Min():F3}, {logits.Max():F3}]  std={Std(logits):F3}");
                _out.WriteLine("TOP-20 after full chat prompt:");
                foreach (var ((v, id), rank) in top20.Select((x, r) => (x, r)))
                {
                    _out.WriteLine($"  #{rank + 1,2}  [{id,7}]  {v,8:F3}  '{tok.DecodeToken(id)}'");
                }

                _out.WriteLine("\nWorking Qwen2.5-Instruct should predict:");
                _out.WriteLine("  token 220 (' ') or 19/56 (digit) or similar short answer tokens");
                _out.WriteLine($"Logit[220]=' '  = {logits[220]:F4}");
                _out.WriteLine($"Logit[19]='2'   = {logits[19]:F4}");
                _out.WriteLine($"Logit[19]='56'? = {logits[20]:F4}");
            }
        }

        private static float Std(float[] arr)
        {
            var mean = arr.Average();
            return MathF.Sqrt(arr.Select(x => (x - mean) * (x - mean)).Average());
        }
    }
}
