// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Runtime.InteropServices;
using DevOnBike.Overfit.LanguageModels.Contracts;
using DevOnBike.Overfit.LanguageModels.Loading;
using DevOnBike.Overfit.LanguageModels.Tokenizers;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests.LanguageModels.Loading
{
    /// <summary>
    /// Greedy (temp=0) parity for Bielik against HuggingFace transformers loading the SAME fp16 GGUF.
    /// Prints the prompt token ids + the generated token ids for a raw continuation prompt; compare
    /// against <c>Scripts/diag_bielik_parity.py</c> (which dequantizes the identical GGUF). Loaded with
    /// quantize:false (pure F16→F32) so the only difference from the reference is Overfit's forward
    /// pass — matching continuations ⇒ the tokenizer, RoPE, attention/GQA and tensor mapping are correct.
    /// [LongFact] — needs the ~9 GB fp16 GGUF at C:\bielik. Flip to [Fact] to run.
    /// </summary>
    public sealed class BielikParityTests
    {
        private const string Fp16Path = @"C:\bielik\Bielik-4.5B-v3.0-Instruct-fp16.gguf";
        private const string Prompt = "Stolicą Polski jest";   // raw continuation — isolates the forward pass
        private const int MaxNew = 30;

        private readonly ITestOutputHelper _out;
        public BielikParityTests(ITestOutputHelper output) => _out = output;

        [LongFact]
        public void Bielik_GreedyParity_PrintTokens()
        {
            if (!File.Exists(Fp16Path))
            {
                _out.WriteLine($"missing {Fp16Path}");
                return;
            }

            using var engine = GgufLlamaLoader.Load(Fp16Path, quantize: false, mmap: false);
            var tok = GgufTokenizer.Load(Fp16Path);

            var prompt = tok.Encode(Prompt);
            _out.WriteLine("PROMPT_IDS " + string.Join(",", prompt));

            using var session = engine.CreateSession(256);
            session.Reset(prompt);

            var gen = new List<int>();
            var sampling = SamplingOptions.Greedy;
            for (var i = 0; i < MaxNew && !session.IsFull; i++)
            {
                var t = session.GenerateNextToken(in sampling);
                if (t == tok.EosId)
                {
                    break;
                }
                gen.Add(t);
            }

            _out.WriteLine("GEN_IDS " + string.Join(",", gen));
            _out.WriteLine("GEN_TEXT " + tok.Decode(CollectionsMarshal.AsSpan(gen)));
            Assert.NotEmpty(gen);
        }
    }
}
