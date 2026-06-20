// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels.Contracts;
using DevOnBike.Overfit.LanguageModels.Runtime;
using DevOnBike.Overfit.LanguageModels.Tokenizers;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests.LanguageModels.Loading
{
    /// <summary>
    /// End-to-end on the FULL Qwen1.5-MoE <b>Q4_K_M</b> GGUF — the smaller, mixed-quant variant
    /// llama.cpp's "_M" strategy produces. It puts <c>Q5_0</c> on ~half the layers'
    /// <c>ffn_down_exps</c> (and Q6_K/Q4_K on the shared-down), which previously made the file
    /// unloadable. With Q5_0 dequant + per-expert re-quant to Q8 in place, the file now loads and
    /// must still answer "capital of France?" → "Paris". (Distinct from
    /// <see cref="Qwen2MoeEndToEndTests"/>, which prefers the uniform Q8_0 variant.)
    /// </summary>
    [Trait("Category", "Qwen")]
    [Trait("Category", "MoE")]
    public sealed class Qwen2MoeQ4KMTests
    {
        private const string Q4KmPath = @"C:\qwen-moe\Qwen1.5-MoE-A2.7B-Chat.Q4_K_M.gguf";

        private readonly ITestOutputHelper _out;
        public Qwen2MoeQ4KMTests(ITestOutputHelper output) => _out = output;

        [LongFact]
        public void LoadsMixedQuantQ4KM_AndDecodesCoherently()
        {
            if (!File.Exists(Q4KmPath))
            {
                _out.WriteLine($"missing {Q4KmPath}");
                return;
            }

            using var engine = CachedLlamaInferenceEngine.LoadGguf(Q4KmPath);   // must NOT throw on Q5_0
            Assert.True(engine.Config.IsMixtureOfExperts);
            _out.WriteLine($"loaded Q4_K_M: {engine.Config.NLayers}L dModel={engine.Config.DModel} " +
                           $"experts={engine.Config.ExpertCount}/{engine.Config.ExpertUsedCount}");

            using var session = engine.CreateSession(256);

            QwenTokenizer? tok = TryLoadTokenizer();
            int[] prompt = tok is not null
                ? tok.Encode("<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n")
                : [151643, 151644, 198];

            session.Reset(prompt);

            var generated = new List<int>();
            var sampling = SamplingOptions.Greedy;
            for (var i = 0; i < 24 && !session.IsFull; i++)
            {
                var token = session.GenerateNextToken(in sampling);
                Assert.InRange(token, 0, engine.Config.VocabSize - 1);
                if (token == QwenTokenizer.ImEnd || token == QwenTokenizer.EndOfText)
                {
                    break;
                }
                generated.Add(token);
            }

            Assert.NotEmpty(generated);
            _out.WriteLine($"tokens: [{string.Join(", ", generated)}]");

            if (tok is not null)
            {
                var text = tok.Decode(System.Runtime.InteropServices.CollectionsMarshal.AsSpan(generated));
                _out.WriteLine($"--- decoded ---\n{text}");
                Assert.Contains("Paris", text, StringComparison.OrdinalIgnoreCase);
            }
        }

        private static QwenTokenizer? TryLoadTokenizer()
        {
            foreach (var dir in new[] { @"C:\qwen3b", @"C:\qwen-moe" })
            {
                if (File.Exists(Path.Combine(dir, "tokenizer.json")) || File.Exists(Path.Combine(dir, "vocab.json")))
                {
                    try
                    {
                        return QwenTokenizer.Load(dir);
                    }
                    catch { /* fall through */ }
                }
            }
            return null;
        }
    }
}
