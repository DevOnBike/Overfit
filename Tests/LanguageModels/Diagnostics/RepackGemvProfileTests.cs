// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels.Contracts;
using DevOnBike.Overfit.LanguageModels.Runtime;
using DevOnBike.Overfit.LanguageModels.Tokenizers;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests.LanguageModels.Diagnostics
{
    /// <summary>
    /// Per-component decode profile for the repacked 8×8 GEMV work (OVERFIT_REPACK_GEMV): where do the
    /// remaining milliseconds live (Attention / FFN / LM-head / Sampler) on Qwen-3B Q4_K_M? Toggle the env
    /// var across runs to see the shift. [LongFact] — needs C:\qwen3b\qwen.q4km.gguf.
    /// </summary>
    public sealed class RepackGemvProfileTests
    {
        private const string Path = @"C:\qwen3b\qwen.q4km.gguf";
        private readonly ITestOutputHelper _out;

        public RepackGemvProfileTests(ITestOutputHelper output) => _out = output;

        [LongFact]
        public void Profile_Decode_PerComponent()
        {
            if (!File.Exists(Path)) { _out.WriteLine("missing gguf"); return; }

            using var engine = CachedLlamaInferenceEngine.LoadGguf(Path);
            var tok = GgufTokenizer.Load(Path);
            var prompt = tok.Encode("The history of computing began");
            var sampling = SamplingOptions.Greedy;

            using var session = engine.CreateSession(256);
            session.Reset(prompt);
            for (var i = 0; i < 8; i++) { session.GenerateNextToken(in sampling); }   // warm-up

            DecodeProfiler.Enabled = true;
            try
            {
                for (var i = 0; i < 64 && !session.IsFull; i++)
                {
                    session.GenerateNextToken(in sampling);
                }
                _out.WriteLine($"repack GEMV enabled: {Q4KGemvKernel.Enabled}");
                _out.WriteLine(DecodeProfiler.Report());
            }
            finally
            {
                DecodeProfiler.Enabled = false;
            }
        }
    }
}
