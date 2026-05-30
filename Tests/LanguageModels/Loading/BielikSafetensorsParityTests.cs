// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Linq;
using System.Text.Json;
using DevOnBike.Overfit.LanguageModels.Contracts;
using DevOnBike.Overfit.LanguageModels.Loading;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests.LanguageModels.Loading
{
    /// <summary>
    /// True forward-pass parity for Bielik on the REAL safetensors weights (source of truth, not a
    /// GGUF re-quant). <c>Scripts/diag_bielik_safetensors_parity.py</c> loads the HF safetensors
    /// (`C:\bielik-st`), greedy-decodes a raw prompt, and writes <c>bielik_st_ref.json</c>
    /// (prompt_ids + gen_ids). This loads the SAME safetensors via <see cref="SafetensorsLlamaLoader"/>
    /// (quantize:false → pure bf16→F32), feeds the IDENTICAL prompt ids (bypassing any tokenizer
    /// difference) and greedy-decodes — so the only variable is Overfit's forward pass. Leading-token
    /// agreement ⇒ RoPE, attention/GQA and tensor mapping match the reference.
    /// [LongFact] — needs C:\bielik-st + bielik_st_ref.json. Flip to [Fact] to run.
    /// </summary>
    public sealed class BielikSafetensorsParityTests
    {
        private const string Dir = @"C:\bielik-st";
        private const string RefJson = @"D:\Overfit\bielik_st_ref.json";

        private readonly ITestOutputHelper _out;
        public BielikSafetensorsParityTests(ITestOutputHelper output) => _out = output;

        [LongFact]
        public void Bielik_Safetensors_GreedyParity_vs_HF()
        {
            if (!Directory.Exists(Dir) || !File.Exists(RefJson))
            {
                _out.WriteLine("missing C:\\bielik-st or bielik_st_ref.json (run the python ref first)");
                return;
            }

            using var doc = JsonDocument.Parse(File.ReadAllText(RefJson));
            var promptIds = doc.RootElement.GetProperty("prompt_ids").EnumerateArray().Select(e => e.GetInt32()).ToArray();
            var hfGen = doc.RootElement.GetProperty("gen_ids").EnumerateArray().Select(e => e.GetInt32()).ToArray();
            _out.WriteLine("PROMPT_IDS  " + string.Join(",", promptIds));
            _out.WriteLine("HF_GEN      " + string.Join(",", hfGen));

            using var engine = SafetensorsLlamaLoader.Load(Dir, quantize: false);
            using var session = engine.CreateSession(256);
            session.Reset(promptIds);

            var gen = new List<int>();
            var sampling = SamplingOptions.Greedy;
            for (var i = 0; i < hfGen.Length && !session.IsFull; i++)
            {
                gen.Add(session.GenerateNextToken(in sampling));
            }
            _out.WriteLine("OVERFIT_GEN " + string.Join(",", gen));

            var match = 0;
            while (match < gen.Count && match < hfGen.Length && gen[match] == hfGen[match]) { match++; }
            _out.WriteLine($"MATCH {match}/{hfGen.Length} leading tokens identical");

            Assert.True(match >= 1, $"first token diverges (Overfit {(gen.Count > 0 ? gen[0] : -1)} vs HF {hfGen[0]})");
        }
    }
}
