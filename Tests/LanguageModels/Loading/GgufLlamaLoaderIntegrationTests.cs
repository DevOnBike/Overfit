// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels.Loading;
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
        private static string GgufModelPath => TestModelPaths.Qwen3B.GgufPath;
        private static string BinaryModelPath => TestModelPaths.Qwen3B.BinaryPath;

        private readonly ITestOutputHelper _output;

        public GgufLlamaLoaderIntegrationTests(ITestOutputHelper output)
        {
            _output = output;
        }

        [LongFact]  // runtime unmeasured — the test failed after 21s (2026-08-07)
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
            // quantize: false — WITHOUT IT THIS TEST CANNOT PASS, and its own comment said otherwise.
            //
            // `LoadGguf` defaults to `quantize: true`, which makes attention, FFN and the LM head
            // Q8_0-resident. The binary path has no such thing: "the binary path stays F32 — Q8 is
            // GGUF-only". So the two sides were never running identical kernels, and the assertion below
            // demands agreement to 0.001. Measured 2026-08-07 after the RoPE permute was fixed: cosine
            // 0.9910 on the logits with max|d| 2.23 — exactly the shape of Q8 error accumulating through
            // 36 layers, and nowhere near 0.001.
            //
            // The loader already provides the right entry point and documents it as such: "When false
            // every weight loads as F32 — the pre-quantization decode path, used as the parity reference".
            using (var engineGguf = GgufLlamaLoader.Load(GgufModelPath, quantize: false))
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

            // THE TWO FILES ARE STRUCTURALLY DIFFERENT — reported, because it is true and because the
            // config assertions above cannot see it (layers, width and head counts are identical; whether
            // the head is tied is not part of them).
            //
            // Read from the GGUF's own tensor table on 2026-08-07: 3,085,938,688 parameters at 2.00 bytes
            // each (FP16), and NO `output.weight` — its head is tied to `token_embd`. The .bin is
            // 3,398,432,781 parameters at FP32; the difference of 312,494,093 is 1.004x one
            // 2048 x 151936 matrix, so the .bin carries a separate head.
            //
            // **THIS IS NOT WHY THE LOGITS DIFFER, and the first version of this check said it was.**
            // GgufVsBinaryLayerDivergenceDiagnostics compared the residual stream layer by layer and the
            // two stacks already disagree after LAYER 0 (cosine 0.9789, and 0.8639 by layer 35). The head
            // is applied once, at the end, so it cannot produce a difference that exists before it. FP16
            // versus FP32 rounding cannot either — that would leave cosine above 0.9999.
            //
            // What remains is a real disagreement between the two loaders, from the first block onward.
            // See T10 in docs/test-gate-backlog.md. Do not let this assertion stand in for that: it fires
            // first, and an earlier version of it hid the layer-0 finding behind a tidy explanation.
            var ggufParameters = new FileInfo(GgufModelPath).Length / 2L;      // FP16
            var binaryParameters = new FileInfo(BinaryModelPath).Length / 4L;  // FP32
            var head = (long)vocabGguf * dModelGguf;
            var extra = binaryParameters - ggufParameters;

            // REPORTED, NOT ASSERTED — and the earlier version of this, which asserted, was wrong twice
            // over. It was added on the belief that the size difference explained the logit divergence;
            // the per-layer diagnostic then showed the divergence starts at layer 0, before any head can
            // act. Worse, it then blocked the test from ever reaching the comparison it exists for.
            //
            // The difference is by design. `Scripts/convert_llama.py` documents its own format:
            // "lm_head [vocab_size, d_model] (written even if tie_weights=1, Overfit resolves at load)".
            // The .bin always materialises the head; the GGUF ties it. Same weights, different
            // representation — exactly one head's worth, which is what the ratio below shows.
            _output.WriteLine($"  GGUF ~{ggufParameters:N0} params, .bin ~{binaryParameters:N0} — "
                              + $"difference {extra:N0} = {(double)extra / head:F3}x one "
                              + $"{dModelGguf} x {vocabGguf} head (expected: the .bin materialises the "
                              + "tied head)");

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
                if (arr[i] > bestVal)
                {
                    bestVal = arr[i];
                    best = i;
                }
            }
            return best;
        }
    }
}
