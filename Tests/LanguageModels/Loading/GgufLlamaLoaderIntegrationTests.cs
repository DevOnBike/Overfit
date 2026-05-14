// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels.Runtime;
using DevOnBike.Overfit.Tests.TestSupport;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests.LanguageModels.Loading
{
    /// <summary>
    /// Integration tests verifying GGUF loader produces identical results to the legacy binary loader.
    /// Both files must exist on disk. Tests are skipped silently if either file is missing,
    /// so this works in environments without local model checkpoints.
    ///
    /// To run locally:
    ///   1. Download a Qwen2.5 model with `ollama pull qwen2.5:3b`
    ///   2. Convert to legacy binary with `python Scripts/convert_gguf.py`
    ///   3. Set BinaryModelPath and GgufModelPath below.
    /// </summary>
    [Trait("Category", "Gguf")]
    [Trait("Category", "Integration")]
    public sealed class GgufLlamaLoaderIntegrationTests
    {
        // ── Paths resolved via TestModelPaths — override via OVERFIT_QWEN3B_DIR ─
        private static string GgufModelPath   => TestModelPaths.Qwen3B.GgufPath;
        private static string BinaryModelPath => TestModelPaths.Qwen3B.BinaryPath;

        private readonly ITestOutputHelper _output;

        public GgufLlamaLoaderIntegrationTests(ITestOutputHelper output)
        {
            _output = output;
        }

        [LongFact]
        public void LoadGguf_ProducesSameLogitsAsBinaryLoader_For3B()
        {
            TestModelPaths.Qwen3B.RequireGgufPath();
            TestModelPaths.Qwen3B.RequireBinaryPath();

            // Known-good 3-token prompt
            int[] prompt = [151643, 151644, 198];

            // Load engines SEQUENTIALLY to keep peak RAM low.
            // For 3B FP32, each engine is ~13 GB; holding both at once requires ~30 GB+.
            // By disposing the first before loading the second, peak stays around ~14 GB.

            float[] logitsGguf;
            int nLayersGguf, dModelGguf, nHeadsGguf, nKvHeadsGguf, vocabGguf, dFFGguf;

            // ─── Phase 1: GGUF ────────────────────────────────────────────
            using (var engineGguf = CachedLlamaInferenceEngine.LoadGguf(GgufModelPath))
            {
                nLayersGguf = engineGguf.Config.NLayers;
                dModelGguf = engineGguf.Config.DModel;
                nHeadsGguf = engineGguf.Config.NHeads;
                nKvHeadsGguf = engineGguf.Config.NKvHeads;
                vocabGguf = engineGguf.Config.VocabSize;
                dFFGguf = engineGguf.Config.DFF;

                using var sessionGguf = engineGguf.CreateSession(64);
                sessionGguf.Reset(prompt);
                logitsGguf = sessionGguf.LastLogits.ToArray();
            }

            // Force GC between engines so RAM headroom is available for the binary load
            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();

            float[] logitsBin;

            // ─── Phase 2: Binary ──────────────────────────────────────────
            using (var engineBin = CachedLlamaInferenceEngine.Load(BinaryModelPath))
            {
                // Verify config matches (same model, just different loader paths)
                Assert.Equal(engineBin.Config.NLayers, nLayersGguf);
                Assert.Equal(engineBin.Config.DModel, dModelGguf);
                Assert.Equal(engineBin.Config.NHeads, nHeadsGguf);
                Assert.Equal(engineBin.Config.NKvHeads, nKvHeadsGguf);
                Assert.Equal(engineBin.Config.VocabSize, vocabGguf);
                Assert.Equal(engineBin.Config.DFF, dFFGguf);

                using var sessionBin = engineBin.CreateSession(64);
                sessionBin.Reset(prompt);
                logitsBin = sessionBin.LastLogits.ToArray();
            }

            Assert.Equal(logitsBin.Length, logitsGguf.Length);

            // Compute max abs diff
            var maxDiff = 0f;
            var maxDiffIdx = 0;  // init to 0 so the failure message is safe even when diff is zero
            var sumDiff = 0.0;
            for (var i = 0; i < logitsGguf.Length; i++)
            {
                var diff = MathF.Abs(logitsGguf[i] - logitsBin[i]);
                sumDiff += diff;
                if (diff > maxDiff)
                {
                    maxDiff = diff;
                    maxDiffIdx = i;
                }
            }
            var meanDiff = sumDiff / logitsGguf.Length;

            _output.WriteLine($"Logit comparison ({logitsGguf.Length} vocab):");
            _output.WriteLine($"  Max diff: {maxDiff:F6} at token {maxDiffIdx}");
            _output.WriteLine($"  Mean diff: {meanDiff:F6}");
            _output.WriteLine($"  Top-1 (GGUF): {ArgMax(logitsGguf)}");
            _output.WriteLine($"  Top-1 (BIN):  {ArgMax(logitsBin)}");

            // Both go through identical FP32 kernels — only loader differs
            // FP16 → FP32 conversion is the same in both paths so diff should be ~0
            Assert.True(maxDiff < 0.001f,
                $"GGUF and binary loaders produced different logits. " +
                $"Max diff = {maxDiff:F6} at vocab[{maxDiffIdx}] " +
                $"(GGUF={logitsGguf[maxDiffIdx]:F4}, BIN={logitsBin[maxDiffIdx]:F4})");

            // Top-1 token MUST match (argmax preserved even if floats wobble in noise)
            Assert.Equal(ArgMax(logitsBin), ArgMax(logitsGguf));
        }

        private static int ArgMax(float[] arr)
        {
            var best = 0;
            var bestVal = arr[0];
            for (var i = 1; i < arr.Length; i++)
            {
                if (arr[i] > bestVal) { bestVal = arr[i]; best = i; }
            }
            return best;
        }
    }
}
