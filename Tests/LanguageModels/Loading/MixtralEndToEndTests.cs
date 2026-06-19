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
    /// End-to-end on the FULL Mixtral-8x7B-Instruct GGUF — the <b>routed-only</b> MoE shape (8 experts,
    /// top-2, no shared expert; <c>llama</c> arch with expert metadata). Validates that the loader
    /// recognises Mixtral (sets <c>HasSharedExpert=false</c>, <c>NormalizeExpertWeights=true</c>),
    /// skips the absent <c>ffn_*_shexp</c> tensors, and that <see cref="Qwen2MoeFeedForwardBlock"/>
    /// runs the routed sum alone. The GGUF-embedded SentencePiece vocab is read by
    /// <see cref="GgufTokenizer"/>, so this now asserts <b>text coherence</b>: "capital of France?"
    /// → an answer naming Paris.
    /// </summary>
    [Trait("Category", "Mixtral")]
    [Trait("Category", "MoE")]
    public sealed class MixtralEndToEndTests
    {
        private const string MixtralPath = @"C:\mixtral\mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf";

        private readonly ITestOutputHelper _out;
        public MixtralEndToEndTests(ITestOutputHelper output) => _out = output;

        [LongFact]
        public void LoadsRoutedOnlyMoE_AndDecodesCoherently()
        {
            if (!File.Exists(MixtralPath))
            {
                _out.WriteLine($"missing {MixtralPath}");
                return;
            }

            using var engine = CachedLlamaInferenceEngine.LoadGguf(MixtralPath);

            Assert.True(engine.Config.IsMixtureOfExperts);
            Assert.False(engine.Config.HasSharedExpert);          // routed-only
            Assert.True(engine.Config.NormalizeExpertWeights);    // Mixtral renormalises top-k
            Assert.Equal(8, engine.Config.ExpertCount);
            Assert.Equal(2, engine.Config.ExpertUsedCount);
            _out.WriteLine($"loaded Mixtral: {engine.Config.NLayers}L dModel={engine.Config.DModel} " +
                           $"experts={engine.Config.ExpertCount}/{engine.Config.ExpertUsedCount} " +
                           $"kvHeads={engine.Config.KvHeads} expertDff={engine.Config.ExpertFeedForwardLength}");

            // Tokenizer straight from the GGUF's embedded SentencePiece vocab — no side-loaded file.
            var tok = GgufTokenizer.Load(MixtralPath);
            Assert.Equal(engine.Config.VocabSize, tok.VocabSize);

            // Mixtral-Instruct prompt format: "<s>[INST] ... [/INST]" (BOS added by the tokenizer).
            var prompt = tok.Encode("[INST] What is the capital of France? [/INST]");

            using var session = engine.CreateSession(128);
            session.Reset(prompt);

            var generated = new List<int>();
            var sampling = SamplingOptions.Greedy;
            for (var i = 0; i < 24 && !session.IsFull; i++)
            {
                var logits = session.LastLogits;
                for (var j = 0; j < logits.Length; j += 991)   // sparse finite check over 32000 vocab
                {
                    Assert.True(float.IsFinite(logits[j]), $"non-finite logit at {j} (step {i})");
                }

                var token = session.GenerateNextToken(in sampling);
                Assert.InRange(token, 0, engine.Config.VocabSize - 1);
                if (token == tok.EosId)
                {
                    break;
                }
                generated.Add(token);
            }

            Assert.NotEmpty(generated);
            var text = tok.Decode(System.Runtime.InteropServices.CollectionsMarshal.AsSpan(generated));
            _out.WriteLine($"tokens: [{string.Join(", ", generated)}]");
            _out.WriteLine($"--- decoded ---\n{text}");

            Assert.Contains("Paris", text, StringComparison.OrdinalIgnoreCase);
        }
    }
}
