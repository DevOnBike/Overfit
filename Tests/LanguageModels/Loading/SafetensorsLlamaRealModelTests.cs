// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Runtime.InteropServices;
using DevOnBike.Overfit.LanguageModels.Contracts;
using DevOnBike.Overfit.LanguageModels.Loading;
using DevOnBike.Overfit.LanguageModels.Tokenizers;
using DevOnBike.Overfit.Tests.TestSupport;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests.LanguageModels.Loading
{
    /// <summary>
    /// End-to-end validation of the native HuggingFace safetensors path on a REAL model:
    /// Qwen2.5-0.5B (<c>model.safetensors</c> + <c>config.json</c> + <c>tokenizer.json</c>
    /// at <c>C:\qwen3b\</c>). Loads config → weights → tokenizer with zero Python and checks
    /// the model produces a coherent, near-deterministic completion — which only holds if the
    /// per-head / FFN / GQA / tied-LM-head mapping AND the RoPE row-permute (HF rotate-half →
    /// adjacent-pair, the bit that distinguishes a coherent model from word-salad) are correct.
    /// [LongFact] — loads a ~1 GB model.
    /// </summary>
    [Trait("Category", "Qwen")]
    public sealed class SafetensorsLlamaRealModelTests
    {
        private readonly ITestOutputHelper _out;
        public SafetensorsLlamaRealModelTests(ITestOutputHelper output) => _out = output;

        private static string SafetensorsPath => Path.Combine(TestModelPaths.Qwen3B.Dir, "model.safetensors");

        [LongFact]
        public void LoadConfig_RealQwen05B_MatchesArchitecture()
        {
            if (!File.Exists(SafetensorsPath))
            {
                _out.WriteLine("model.safetensors not present — skipping.");
                return;
            }

            var cfg = LlamaConfigReader.ReadFromDirectory(TestModelPaths.Qwen3B.Dir);

            Assert.Equal(24, cfg.NLayers);
            Assert.Equal(896, cfg.DModel);
            Assert.Equal(14, cfg.NHeads);
            Assert.Equal(2, cfg.KvHeads);          // GQA
            Assert.Equal(4864, cfg.DFF);
            Assert.Equal(151936, cfg.VocabSize);
            Assert.Equal(8192, cfg.ContextLength); // capped from 32768
            Assert.Equal(1_000_000f, cfg.RoPETheta);
            Assert.True(cfg.TieWeights);
        }

        [LongFact]
        public void Generate_RealQwen05B_FromSafetensors_CompletesCoherently()
        {
            if (!File.Exists(SafetensorsPath))
            {
                _out.WriteLine("model.safetensors not present — skipping.");
                return;
            }

            using var engine = SafetensorsLlamaLoader.Load(TestModelPaths.Qwen3B.Dir, quantize: false);
            var tok = QwenTokenizer.Load(TestModelPaths.Qwen3B.Dir);
            using var session = engine.CreateSession(256);

            session.Reset(tok.Encode("The capital of France is"));

            var sampling = SamplingOptions.Greedy;
            var generated = new List<int>();
            for (var i = 0; i < 6 && !session.IsFull; i++)
            {
                var t = session.GenerateNextToken(in sampling);
                if (t == QwenTokenizer.EndOfText)
                {
                    break;
                }
                generated.Add(t);
            }

            var completion = tok.Decode(CollectionsMarshal.AsSpan(generated));
            _out.WriteLine($"completion: '{completion}'");

            // Logits stay finite — no NaN/Inf from a mis-mapped weight.
            var logits = new float[engine.Config.VocabSize];
            session.GetLastLogits(logits);
            for (var i = 0; i < logits.Length; i++)
            {
                Assert.False(float.IsNaN(logits[i]) || float.IsInfinity(logits[i]));
            }

            // Greedy completion of a near-deterministic factual prompt: a correctly mapped
            // model says "Paris"; a RoPE-permute or head-split bug yields word-salad.
            Assert.Contains("Paris", completion, StringComparison.OrdinalIgnoreCase);
        }
    }
}
