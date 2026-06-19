// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Text;
using DevOnBike.Overfit.LanguageModels.Contracts;
using DevOnBike.Overfit.LanguageModels.Loading;
using DevOnBike.Overfit.LanguageModels.Runtime;
using DevOnBike.Overfit.LanguageModels.Tokenizers;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests.LanguageModels.Loading
{
    /// <summary>
    /// Loads the Gemma-2 2B GGUF and asserts coherent English output. Gemma-2 is the most arch-divergent loader yet:
    /// GeGLU FFN, (1+w) RMSNorm, embedding ×√d_model, explicit head_dim (256) + GQA, and — the structural bit —
    /// SANDWICH norm (post_attention/post_ffw RMSNorm before each residual). Soft-caps are deferred (final cap is a
    /// no-op for greedy argmax; attn cap is mild). A coherent answer here is the end-to-end gate. [LongFact] — needs C:\gemma.
    /// </summary>
    public sealed class GemmaSmokeTests
    {
        private const string Path = @"C:\gemma\gemma-2-2b-it-Q4_K_M.gguf";
        private readonly ITestOutputHelper _out;
        public GemmaSmokeTests(ITestOutputHelper output) => _out = output;

        [LongFact]
        public void Gemma2_Loads_And_Generates_Coherent_English()
        {
            if (!File.Exists(Path))
            {
                _out.WriteLine("missing Gemma-2 gguf");
                return;
            }

            using (var reader = new GgufReader(Path))
            {
                _out.WriteLine("arch       : " + reader.GetMeta("general.architecture", "?"));
                _out.WriteLine("attn_cap   : " + reader.GetMeta("gemma2.attn_logit_softcapping", 0f));
                _out.WriteLine("final_cap  : " + reader.GetMeta("gemma2.final_logit_softcapping", 0f));
                _out.WriteLine("head_dim   : " + reader.GetMeta("gemma2.attention.key_length", -1));
            }

            using var engine = CachedLlamaInferenceEngine.LoadGguf(Path);
            var tok = GgufTokenizer.Load(Path);

            // Gemma instruct chat turn (the IT model expects this; raw completion is out-of-distribution).
            var prompt = tok.Encode("<start_of_turn>user\nWhat is the capital of France? Answer with just the city name.<end_of_turn>\n<start_of_turn>model\n");

            using var session = engine.CreateSession(512);
            session.Reset(prompt);
            var sampling = SamplingOptions.GreedyWithPenalty(1.1f);
            var sb = new StringBuilder();
            for (var i = 0; i < 32 && !session.IsFull; i++)
            {
                var t = session.GenerateNextToken(in sampling);
                if (t == 1 || t == 107)
                {
                    break;
                } // <eos> / <end_of_turn>
                sb.Append(tok.DecodeToken(t));
            }

            var reply = sb.ToString();
            _out.WriteLine("REPLY: " + reply);
            Assert.False(string.IsNullOrWhiteSpace(reply));
            Assert.Contains("Paris", reply, StringComparison.OrdinalIgnoreCase);
        }
    }
}
