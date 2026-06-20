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
    /// Validates the qwen2moe expert-tensor slicing against the REAL Qwen1.5-MoE GGUF: loads layer-0
    /// router + 60 routed experts (Q4_K gate/up, Q8_0 down) + the sigmoid-gated shared expert, runs a
    /// hidden vector through <see cref="Qwen2MoeFeedForwardBlock"/>, and asserts a finite, non-trivial
    /// output. De-risks the 3-D slicing + per-expert Q8_0 de-interleave (numerical parity vs an F32
    /// baseline is the separate decode-parity step). [LongFact] — loads the model.
    /// </summary>
    [Trait("Category", "Qwen")]
    [Trait("Category", "MoE")]
    public sealed class Qwen2MoeLoaderTests
    {
        private const string MoePath = @"C:\qwen-moe\Qwen1.5-MoE-A2.7B-Chat.Q4_K_M.gguf";

        private readonly ITestOutputHelper _out;
        public Qwen2MoeLoaderTests(ITestOutputHelper output) => _out = output;

        [LongFact]
        public void Layer0_LoadsAndDecodes_ThroughQwen2MoeBlock()
        {
            if (!File.Exists(MoePath))
            {
                _out.WriteLine($"missing {MoePath}");
                return;
            }

            using var reader = new GgufReader(MoePath);
            Assert.Equal("qwen2moe", reader.GetMeta("general.architecture", ""));

            var dModel = reader.GetMeta("qwen2moe.embedding_length", 0);
            var expertCount = reader.GetMeta("qwen2moe.expert_count", 0);
            var expertUsed = reader.GetMeta("qwen2moe.expert_used_count", 0);

            var gateInfo = reader.Tensors["blk.0.ffn_gate_exps.weight"];
            var upInfo = reader.Tensors["blk.0.ffn_up_exps.weight"];
            var downInfo = reader.Tensors["blk.0.ffn_down_exps.weight"];
            var expertDff = (int)gateInfo.Dims[1];
            var sharedDff = (int)reader.Tensors["blk.0.ffn_gate_shexp.weight"].Dims[1];

            _out.WriteLine($"dModel={dModel} experts={expertCount} used={expertUsed} expertDff={expertDff} sharedDff={sharedDff}");

            // Routed experts (the 3-D slicing under test).
            var gate = GgufLlamaLoader.LoadExperts(reader, gateInfo);
            var up = GgufLlamaLoader.LoadExperts(reader, upInfo);
            var down = GgufLlamaLoader.LoadExperts(reader, downInfo);
            Assert.Equal(expertCount, gate.Length);
            Assert.Equal(expertCount, up.Length);
            Assert.Equal(expertCount, down.Length);
            Assert.True(gate[0].IsQ4K && down[0].IsQuantized, "expected Q4_K gate + Q8_0 down");

            var router = GgufLlamaLoader.LoadRouter(reader, reader.Tensors["blk.0.ffn_gate_inp.weight"], dModel, expertCount);

            // Sigmoid-gated shared expert.
            var sGate = GgufLlamaLoader.AllocAndLoadResident(reader, "blk.0.ffn_gate_shexp.weight", dModel, sharedDff, null);
            var sUp = GgufLlamaLoader.AllocAndLoadResident(reader, "blk.0.ffn_up_shexp.weight", dModel, sharedDff, null);
            var sDown = GgufLlamaLoader.AllocAndLoadResident(reader, "blk.0.ffn_down_shexp.weight", sharedDff, dModel, null);
            using var sGateInp = GgufLlamaLoader.AllocAndLoad(reader, "blk.0.ffn_gate_inp_shexp.weight", dModel);

            var block = new Qwen2MoeFeedForwardBlock(dModel, expertDff, sharedDff, expertCount, expertUsed);

            var hidden = new float[dModel];
            for (var d = 0; d < dModel; d++)
            {
                hidden[d] = MathF.Sin(d * 0.05f) * 0.3f;
            }
            var output = new float[dModel];

            block.Decode(hidden, router, gate, up, down, sGateInp.AsReadOnlySpan(), sGate, sUp, sDown, output);

            var nonZero = 0;
            var max = 0f;
            foreach (var v in output)
            {
                Assert.True(float.IsFinite(v), "non-finite output element");
                if (v != 0f)
                {
                    nonZero++;
                }
                max = MathF.Max(max, MathF.Abs(v));
            }
            _out.WriteLine($"output: nonZero={nonZero}/{dModel}  maxAbs={max:G4}");
            Assert.True(nonZero > dModel / 2, "output is mostly zero — slicing/decoding likely wrong");
        }
    }
}
