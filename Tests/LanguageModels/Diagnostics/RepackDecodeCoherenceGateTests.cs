// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Diagnostics;
using System.Text;
using DevOnBike.Overfit.LanguageModels.Contracts;
using DevOnBike.Overfit.LanguageModels.Runtime;
using DevOnBike.Overfit.LanguageModels.Tokenizers;

namespace DevOnBike.Overfit.Tests.LanguageModels.Diagnostics
{
    /// <summary>
    /// Step 0 of the repack-and-replace sprint — the DECODE coherence gate that must be green before the
    /// repacked block_q4_Kx8 layout can become the sole resident layout (dropping the original). Deliberately
    /// does NOT touch the hot path: it prints a decode fingerprint (greedy token ids + text + tok/s + which
    /// repack flags are live) for the CURRENT process env, so the same [Fact] run twice —
    /// once with OVERFIT_REPACK_GEMV/ATTN unset, once with them =1 — reveals whether the repacked kernels
    /// (which reassociate the reduction) keep the greedy continuation identical and how much faster decode is.
    /// Matching ids across the two runs ⇒ replace is safe; a divergence ⇒ investigate before dropping the
    /// original. [LongFact] — needs C:\qwen3b\qwen.q4km.gguf.
    /// </summary>
    public sealed class RepackDecodeCoherenceGateTests
    {
        private const string Path = @"C:\qwen3b\qwen.q4km.gguf";
        private const int Context = 512;
        private const int GenTokens = 32;

        private readonly ITestOutputHelper _out;

        public RepackDecodeCoherenceGateTests(ITestOutputHelper output) => _out = output;

        [ModelFact(Path, "4s")]
        public void Decode_Fingerprint_ForCurrentRepackEnv()
        {

            using var engine = CachedLlamaInferenceEngine.LoadGguf(Path);
            var tok = GgufTokenizer.Load(Path);
            var sampling = SamplingOptions.Greedy;

            _out.WriteLine($"REPACK_GEMV(Q4KGemvKernel.Enabled)={Q4KGemvKernel.Enabled}  REPACK_ATTN(AttnEnabled)={Q4KGemvKernel.AttnEnabled}");

            var prompt = tok.Encode("The capital of France is Paris. The history of computing began with");
            using var session = engine.CreateSession(Context);
            session.Reset(prompt);

            // warm
            for (var i = 0; i < 4; i++)
            {
                session.GenerateNextToken(in sampling);
            }

            using var warmSession = engine.CreateSession(Context);
            warmSession.Reset(prompt);
            var ids = new List<int>(GenTokens);
            var sw = Stopwatch.StartNew();
            for (var i = 0; i < GenTokens && !warmSession.IsFull; i++)
            {
                ids.Add(warmSession.GenerateNextToken(in sampling));
            }
            sw.Stop();

            var tps = ids.Count / sw.Elapsed.TotalSeconds;
            var idStr = new StringBuilder();
            for (var i = 0; i < ids.Count; i++)
            {
                if (i > 0)
                {
                    idStr.Append(',');
                }
                idStr.Append(ids[i]);
            }

            _out.WriteLine($"tok/s: {tps:F1}");
            _out.WriteLine($"ids: {idStr}");
            _out.WriteLine($"text: {tok.Decode(ids.ToArray())}");
        }
    }
}
