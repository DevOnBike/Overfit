// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels.Loading;
using DevOnBike.Overfit.LanguageModels.Runtime;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests.LanguageModels.Loading
{
    /// <summary>
    /// Numerical parity for the quantized MoE decode path: run real layer-0 of the Qwen-MoE GGUF
    /// through <see cref="Qwen2MoeFeedForwardBlock"/> twice — once with the file's quantized expert /
    /// shared weights, once with an F32 dequantization of the SAME weights — and assert the outputs
    /// agree to within quantization noise. Same router weights ⇒ identical top-k selection, so the
    /// only difference is the quant error in the expert/shared projections. (A whole-model F32
    /// baseline is infeasible — Qwen1.5-MoE in F32 is ~57 GB — so parity is validated per-block.)
    /// [LongFact].
    /// </summary>
    [Trait("Category", "Qwen")]
    [Trait("Category", "MoE")]
    public sealed class Qwen2MoeQuantParityTests
    {
        private static string MoePath()
        {
            const string dir = @"C:\qwen-moe";
            var q8 = Path.Combine(dir, "Qwen1.5-MoE-A2.7B-Chat.Q8_0.gguf");
            return File.Exists(q8) ? q8 : Path.Combine(dir, "Qwen1.5-MoE-A2.7B-Chat.Q4_K_M.gguf");
        }

        private readonly ITestOutputHelper _out;
        public Qwen2MoeQuantParityTests(ITestOutputHelper output) => _out = output;

        [LongFact]
        public void QuantizedBlock_MatchesF32Dequant_Layer0()
        {
            var path = MoePath();
            if (!File.Exists(path))
            {
                _out.WriteLine($"missing {path}");
                return;
            }

            using var reader = new GgufReader(path);
            var dModel = reader.GetMeta("qwen2moe.embedding_length", 0);
            var expertCount = reader.GetMeta("qwen2moe.expert_count", 0);
            var expertUsed = reader.GetMeta("qwen2moe.expert_used_count", 0);
            var gateInfo = reader.Tensors["blk.0.ffn_gate_exps.weight"];
            var upInfo = reader.Tensors["blk.0.ffn_up_exps.weight"];
            var downInfo = reader.Tensors["blk.0.ffn_down_exps.weight"];
            var expertDff = (int)gateInfo.Dims[1];
            var sharedDff = (int)reader.Tensors["blk.0.ffn_gate_shexp.weight"].Dims[1];

            // Shared router inputs (F32 — identical for both runs ⇒ identical routing).
            var router = GgufLlamaLoader.LoadRouter(reader, reader.Tensors["blk.0.ffn_gate_inp.weight"], dModel, expertCount);
            using var sharedGateInp = GgufLlamaLoader.AllocAndLoad(reader, "blk.0.ffn_gate_inp_shexp.weight", dModel);

            // Quantized weights (as the model ships).
            var qGate = GgufLlamaLoader.LoadExperts(reader, gateInfo);
            var qUp = GgufLlamaLoader.LoadExperts(reader, upInfo);
            var qDown = GgufLlamaLoader.LoadExperts(reader, downInfo);
            var qShGate = GgufLlamaLoader.AllocAndLoadResident(reader, "blk.0.ffn_gate_shexp.weight", dModel, sharedDff, null);
            var qShUp = GgufLlamaLoader.AllocAndLoadResident(reader, "blk.0.ffn_up_shexp.weight", dModel, sharedDff, null);
            var qShDown = GgufLlamaLoader.AllocAndLoadResident(reader, "blk.0.ffn_down_shexp.weight", sharedDff, dModel, null);

            // F32 dequantization of the same weights (input-major).
            var fGate = GgufLlamaLoader.LoadExpertsF32(reader, gateInfo);
            var fUp = GgufLlamaLoader.LoadExpertsF32(reader, upInfo);
            var fDown = GgufLlamaLoader.LoadExpertsF32(reader, downInfo);
            using var fShGate = GgufLlamaLoader.AllocAndLoadTransposed(reader, "blk.0.ffn_gate_shexp.weight", dModel, sharedDff);
            using var fShUp = GgufLlamaLoader.AllocAndLoadTransposed(reader, "blk.0.ffn_up_shexp.weight", dModel, sharedDff);
            using var fShDown = GgufLlamaLoader.AllocAndLoadTransposed(reader, "blk.0.ffn_down_shexp.weight", sharedDff, dModel);

            var block = new Qwen2MoeFeedForwardBlock(dModel, expertDff, sharedDff, expertCount, expertUsed, normalizeExpertWeights: false);

            var hidden = new float[dModel];
            for (var d = 0; d < dModel; d++)
            {
                hidden[d] = MathF.Sin(d * 0.05f) * 0.3f;
            }

            var qOut = new float[dModel];
            var fOut = new float[dModel];
            block.Decode(hidden, router, qGate, qUp, qDown, sharedGateInp.AsReadOnlySpan(), qShGate, qShUp, qShDown, qOut);
            block.Decode(hidden, router, fGate, fUp, fDown, sharedGateInp.AsReadOnlySpan(), fShGate, fShUp, fShDown, fOut);

            var maxAbs = 0f;
            var maxDiff = 0f;
            for (var d = 0; d < dModel; d++)
            {
                maxAbs = MathF.Max(maxAbs, MathF.Abs(fOut[d]));
                maxDiff = MathF.Max(maxDiff, MathF.Abs(qOut[d] - fOut[d]));
            }
            var rel = maxAbs > 0 ? maxDiff / maxAbs : 0;
            _out.WriteLine($"maxAbs(F32)={maxAbs:G4}  maxAbsDiff={maxDiff:G4}  relative={rel:P2}");

            // Q8_0 is near-lossless; the quantized block must track the F32 reference to within a
            // few percent (same routing, only projection quant error differs).
            Assert.True(rel < 0.05f, $"quantized MoE block diverges from F32 by {rel:P2} (> 5%).");

            foreach (var w in fGate)
            {
                w.Dispose();
            }
            foreach (var w in fUp)
            {
                w.Dispose();
            }
            foreach (var w in fDown)
            {
                w.Dispose();
            }
            foreach (var w in qGate)
            {
                w.Dispose();
            }
            foreach (var w in qUp)
            {
                w.Dispose();
            }
            foreach (var w in qDown)
            {
                w.Dispose();
            }
            qShGate.Dispose();
            qShUp.Dispose();
            qShDown.Dispose();
        }
    }
}
