// Copyright (c) 2026 DevOnBike. AGPLv3.

using DevOnBike.Overfit.LanguageModels.Runtime;
using Xunit;
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
        // ── Configure these paths to your local checkpoints ────────────────
        private const string GgufModelPath = @"c:\qwen3b\qwen.gguf";
        private const string BinaryModelPath = @"c:\qwen3b\qwen.bin";

        private readonly ITestOutputHelper _output;

        public GgufLlamaLoaderIntegrationTests(ITestOutputHelper output)
        {
            _output = output;
        }

        [Fact]
        public void LoadGguf_ProducesSameLogitsAsBinaryLoader_For3B()
        {
            if (!File.Exists(GgufModelPath))
            {
                _output.WriteLine($"SKIPPED: GGUF model not found at {GgufModelPath}");
                return;
            }
            if (!File.Exists(BinaryModelPath))
            {
                _output.WriteLine($"SKIPPED: binary model not found at {BinaryModelPath}");
                return;
            }

            // Load both engines from different sources
            using var engineGguf = CachedLlamaInferenceEngine.LoadGguf(GgufModelPath);
            using var engineBin = CachedLlamaInferenceEngine.Load(BinaryModelPath);

            // Verify config matches (same model, just different loader paths)
            Assert.Equal(engineBin.Config.NLayers, engineGguf.Config.NLayers);
            Assert.Equal(engineBin.Config.DModel, engineGguf.Config.DModel);
            Assert.Equal(engineBin.Config.NHeads, engineGguf.Config.NHeads);
            Assert.Equal(engineBin.Config.NKvHeads, engineGguf.Config.NKvHeads);
            Assert.Equal(engineBin.Config.VocabSize, engineGguf.Config.VocabSize);
            Assert.Equal(engineBin.Config.DFF, engineGguf.Config.DFF);

            // Forward the same prompt through both engines
            using var sessionGguf = engineGguf.CreateSession(64);
            using var sessionBin = engineBin.CreateSession(64);

            // Known-good 3-token prompt from session work
            int[] prompt = { 151643, 151644, 198 };

            sessionGguf.Reset(prompt);
            sessionBin.Reset(prompt);

            var logitsGguf = sessionGguf.LastLogits.ToArray();
            var logitsBin = sessionBin.LastLogits.ToArray();

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
