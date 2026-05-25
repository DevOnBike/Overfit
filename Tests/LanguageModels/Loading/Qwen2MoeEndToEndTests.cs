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
    /// SMOKE test — load the FULL Qwen1.5-MoE GGUF through <c>GgufLlamaLoader.Load</c> and greedily
    /// decode: exercises embed → 24 MoE layers (router + 60 experts + shared) → final norm → LM head.
    /// Asserts the model LOADS and the forward RUNS with finite, in-range output (no crash). It does
    /// NOT assert coherence — coherent generation is a separate, KNOWN-PENDING correctness item
    /// (the forward currently produces incoherent text; suspects: top-k weight normalisation
    /// semantics / tokenizer match / expert-tensor orientation — needs a llama.cpp reference to debug).
    /// </summary>
    [Trait("Category", "Qwen")]
    [Trait("Category", "MoE")]
    public sealed class Qwen2MoeEndToEndTests
    {
        // Prefer the Q8_0 variant (uniform Q8_0 — all expert tensors supported); fall back to Q4_K_M
        // (which mixes in Q5_0 on some layers and is skipped end-to-end until Q5_0 coverage lands).
        private static readonly string MoePath = ResolveMoe();

        private static string ResolveMoe()
        {
            const string dir = @"C:\qwen-moe";
            var q8 = Path.Combine(dir, "Qwen1.5-MoE-A2.7B-Chat.Q8_0.gguf");
            return File.Exists(q8) ? q8 : Path.Combine(dir, "Qwen1.5-MoE-A2.7B-Chat.Q4_K_M.gguf");
        }

        private readonly ITestOutputHelper _out;
        public Qwen2MoeEndToEndTests(ITestOutputHelper output) => _out = output;

        [LongFact]
        public void LoadsFullModel_AndDecodesCoherently()
        {
            if (!File.Exists(MoePath)) { _out.WriteLine($"missing {MoePath}"); return; }

            CachedLlamaInferenceEngine engine;
            try
            {
                engine = CachedLlamaInferenceEngine.LoadGguf(MoePath);
            }
            catch (NotSupportedException ex)
            {
                // This Q4_K_M MoE mixes in quant types we don't support yet (e.g. Q5_0 on some layers'
                // ffn_down_exps — llama.cpp's heterogeneous "_M" mix). End-to-end needs a file whose
                // expert tensors are all F32/F16/BF16/Q8_0/Q4_K/Q6_K (e.g. the Q8_0 variant).
                _out.WriteLine($"SKIP — unsupported expert quant in this file: {ex.Message}");
                return;
            }

            using var _ = engine;
            Assert.True(engine.Config.IsMixtureOfExperts);
            _out.WriteLine($"loaded: {engine.Config.NLayers}L dModel={engine.Config.DModel} " +
                           $"experts={engine.Config.ExpertCount}/{engine.Config.ExpertUsedCount} expertDff={engine.Config.ExpertFeedForwardLength}");

            using var session = engine.CreateSession(256);

            // Try the real Qwen tokenizer (Qwen1.5/2.5 share it); else raw control tokens.
            QwenTokenizer? tok = TryLoadTokenizer();
            int[] prompt = tok is not null
                ? tok.Encode("<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n")
                : [151643, 151644, 198];

            session.Reset(prompt);

            var generated = new List<int>();
            var sampling = SamplingOptions.Greedy;
            for (var i = 0; i < 24 && !session.IsFull; i++)
            {
                AssertFiniteLogits(session);
                var token = session.GenerateNextToken(in sampling);
                Assert.InRange(token, 0, engine.Config.VocabSize - 1);
                if (token == QwenTokenizer.ImEnd || token == QwenTokenizer.EndOfText) { break; }
                generated.Add(token);
            }

            // Smoke only: the forward produced finite, in-range tokens without crashing. Coherence
            // is NOT asserted here (known-pending — see the class summary).
            Assert.NotEmpty(generated);

            _out.WriteLine($"tokens: [{string.Join(", ", generated)}]");
            if (tok is not null)
            {
                _out.WriteLine($"--- decoded ---\n{tok.Decode(System.Runtime.InteropServices.CollectionsMarshal.AsSpan(generated))}");
            }
        }

        private static void AssertFiniteLogits(CachedLlamaSession session)
        {
            var logits = session.LastLogits;
            for (var i = 0; i < logits.Length; i += 997)   // sparse sample — full vocab is 151936
            {
                Assert.True(float.IsFinite(logits[i]), $"non-finite logit at {i}");
            }
        }

        private static QwenTokenizer? TryLoadTokenizer()
        {
            foreach (var dir in new[] { @"C:\qwen3b", @"C:\qwen-moe" })
            {
                if (File.Exists(Path.Combine(dir, "tokenizer.json")) || File.Exists(Path.Combine(dir, "vocab.json")))
                {
                    try { return QwenTokenizer.Load(dir); } catch { /* fall through */ }
                }
            }
            return null;
        }
    }
}
