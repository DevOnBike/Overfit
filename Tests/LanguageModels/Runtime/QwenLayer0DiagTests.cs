// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels.Contracts;
using DevOnBike.Overfit.LanguageModels.Runtime;
using DevOnBike.Overfit.Tests.TestSupport;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests.LanguageModels.Runtime
{
    [Trait("Category", "QwenL0Diag")]
    [Trait("Category", "Qwen")]
    public sealed class QwenLayer0DiagTests
    {
        private readonly ITestOutputHelper _out;
        public QwenLayer0DiagTests(ITestOutputHelper output) => _out = output;
        private static string ModelPath => TestModelPaths.Qwen3B.BinaryPath;

        /// <summary>
        /// Compare intermediate values after layer 0 with Python forward_from_bin.py.
        /// Run Python first: python3 Scripts/forward_from_bin.py --bin test_fixtures/qwen.bin --token 151643
        ///
        /// Expected from Python (fp16 model):
        ///   Layer0 ln1_gamma[:3]   = [some values]
        ///   Layer0 x_norm[:5]      = [some values]
        ///   Layer0 head0 V[:5]     = [some values]
        ///   Layer0 attn_out[:5]    = [some values]
        ///   Layer0 x_norm2[:5]     = [some values]
        ///   Layer 0 after attn_residual: [-0.0097559, -0.002254, 0.00501522, ...]
        ///   Layer 0 after ffn_residual:  [0.18736684, 0.03360228, -0.11243583, ...]
        /// </summary>
        [LongFact]
        public void L0Diag_IntermediateValues()
        {
            TestModelPaths.Qwen3B.RequireBinaryPath();

            // Use CachedLlamaInferenceEngine diagnostics
            var engine = CachedLlamaInferenceEngine.Load(ModelPath);
            var D = engine.Config.DModel;
            var HD = engine.Config.DModel / engine.Config.NHeads;
            using (engine)
            {
                using var session = engine.CreateSession(64);
                var bos = new[] { 151643 };

                // We need to expose internal state. Since CachedTransformerBlock has
                // GetLastLayerNorm1Output and GetLastLayerNorm2Output, let's use them.
                // But they're on the block, not the session. Use engine's test exposure.

                // Alternative: just print logits and compare specific values
                session.Reset(bos);
                var sampling = SamplingOptions.Greedy;
                session.GenerateNextToken(in sampling);
                var logits = session.LastLogits.ToArray();

                // Print logit for specific tokens that Python showed
                _out.WriteLine("=== LOGIT COMPARISON (BOS token 151643) ===");
                _out.WriteLine("Run Python: python3 Scripts/forward_from_bin.py --bin test_fixtures/qwen.bin --token 151643");
                _out.WriteLine("Then compare these values:");
                _out.WriteLine(string.Empty);

                var topN = logits.Select((v, i) => (v, i))
                               .OrderByDescending(x => x.v)
                               .Take(5)
                               .ToArray();

                _out.WriteLine("C# TOP-5:");
                foreach (var (v, id) in topN)
                {
                    _out.WriteLine($"  [{id,7}]  {v,9:F4}");
                }

                _out.WriteLine(string.Empty);
                _out.WriteLine("Python TOP-5 (expected from forward_from_bin.py):");
                _out.WriteLine("  [  62406]    11.9220   ← if C# matches this → logits are same");
                _out.WriteLine("  [  75101]    11.0330");
                _out.WriteLine("  [  34603]    10.8550");
                _out.WriteLine(string.Empty);

                // Check specific tokens
                var checkTokens = new[] { 62406, 75101, 34603, 101394, 151644, 198 };
                _out.WriteLine("C# logit for specific tokens:");
                foreach (var t in checkTokens)
                {
                    _out.WriteLine($"  [{t,7}] = {logits[t],9:F4}");
                }

                _out.WriteLine(string.Empty);
                _out.WriteLine("Python logit for same tokens:");
                _out.WriteLine("  [  62406] =    11.9220  (Python top-1)");
                _out.WriteLine("  [  75101] =    11.0330");
                _out.WriteLine("  [  34603] =    10.8550");
                _out.WriteLine("  [ 101394] = ??? (C# top-1)");
                _out.WriteLine("  [ 151644] =    -1.4688");
                _out.WriteLine("  [    198] =     0.4527");

                _out.WriteLine(string.Empty);
                _out.WriteLine($"VERDICT: C# top-1 is [{topN[0].i}] = {topN[0].v:F4}");
                _out.WriteLine($"         Python top-1 is [62406] = 11.9220");
                _out.WriteLine(topN[0].i == 62406 ? "✓ MATCH! Logits are correct." :
                                                    "✗ MISMATCH — inference bug exists");
            }
        }
    }
}
