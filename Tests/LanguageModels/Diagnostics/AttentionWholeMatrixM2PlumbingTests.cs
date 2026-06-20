// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests.LanguageModels.Diagnostics
{
    /// <summary>
    /// M2 plumbing check for the whole-matrix Q4_K attention lever (ROADMAP #4): confirms the loader actually
    /// BUILT the whole-matrix Q/K/V/O handles end-to-end (GgufLlamaLoader → LayerWeightBuffers → BlockWeights)
    /// for a real Q4_K_M GGUF loaded with mmap — not silently left empty. Q/K/V/O on Qwen-3B Q4_K_M are Q4_K
    /// on disk with dims 2048 (÷256 and ÷8), so every block must report <c>HasWholeAttnQ4K</c>. The handles
    /// stay DORMANT (M3 consumes them); this only asserts they exist so M3 doesn't start from a silent no-op.
    /// [LongFact] — needs C:\qwen3b\qwen.q4km.gguf.
    /// </summary>
    public sealed class AttentionWholeMatrixM2PlumbingTests
    {
        private const string Model = @"C:\qwen3b\qwen.q4km.gguf";

        private readonly ITestOutputHelper _out;
        public AttentionWholeMatrixM2PlumbingTests(ITestOutputHelper output) => _out = output;

        [LongFact]
        public void Qwen3B_Q4KM_Mmap_BuildsWholeAttnHandles()
        {
            if (!File.Exists(Model))
            {
                _out.WriteLine($"missing {Model}");
                return;
            }

            using var client = OverfitClient.LoadGguf(Model, maxContextLength: 256, mmap: true, maxNewTokens: 4);
            var engine = client.Engine;

            // Q4_K_M is a MIXED quant — Q/K are Q4_K, V and O are typically Q6_K — so we assert per-projection,
            // not all-four. Q is the guaranteed-Q4_K, largest (2048×2048) projection: it must carry a whole handle.
            var layers = engine.Config.NLayers;
            var p0 = engine.BlockWholeAttnPresence(0);
            _out.WriteLine($"block 0 whole-matrix Q4_K presence — Q:{p0.Q} K:{p0.K} V:{p0.V} O:{p0.O}");

            var qAll = true;
            (bool Q, bool K, bool V, bool O) seen = (true, true, true, true);
            for (var l = 0; l < layers; l++)
            {
                var p = engine.BlockWholeAttnPresence(l);
                qAll &= p.Q;
                seen.Q &= p.Q;
                seen.K &= p.K;
                seen.V &= p.V;
                seen.O &= p.O;
            }

            _out.WriteLine(
                $"M2 plumbing across all {layers} blocks — Q:{seen.Q} K:{seen.K} V:{seen.V} O:{seen.O} " +
                "(Q4_K_M stores V/O as Q6_K → those stay per-head; Q/K get the whole-matrix GEMV in M3).");

            Assert.True(qAll, "the Q projection (guaranteed Q4_K in Q4_K_M) is missing its whole-matrix handle — M2 plumbing did not populate");
        }
    }
}
