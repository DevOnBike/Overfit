// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels.Runtime;
using DevOnBike.Overfit.Tests.TestSupport;

namespace DevOnBike.Overfit.Tests.LanguageModels.Diagnostics
{
    /// <summary>
    /// WHERE the GGUF and binary loaders start to disagree — per layer, not at the end.
    ///
    /// <para><b>The question this settles.</b> <c>GgufLlamaLoaderIntegrationTests</c> compares only final
    /// logits, and measured 2026-08-07 they differ enormously: max 7.83, <b>mean 1.387</b> across 151936
    /// entries, with a completely different top-1. Reading the GGUF's tensor table showed the two files are
    /// structurally different — it holds 3,085,938,688 parameters at FP16 with <b>no</b> <c>output.weight</c>
    /// (the head is tied to <c>token_embd</c>), while the .bin is 3,398,432,781 at FP32, a difference of
    /// 1.004x one 2048 x 151936 matrix. That is consistent with "the .bin carries a separate head", and it
    /// would explain the divergence.</para>
    ///
    /// <para><b>Consistent is not the same as demonstrated, and the alternative reverses the conclusion.</b>
    /// If the converter simply <i>duplicated</i> the tied embedding into a separate head tensor, the two
    /// files are mathematically equivalent — more bytes, same function — and the logits ought to match. The
    /// divergence would then be a genuine defect in a loader, not a mismatched pair of files, and the
    /// integration test would be right to fail.</para>
    ///
    /// <para><b>Comparing per layer separates the two in one run.</b> The head is applied once, after the
    /// last block. So:</para>
    /// <list type="bullet">
    ///   <item>hidden states agree through every layer and only the logits differ → the head is the whole
    ///     story, and the file-structure explanation stands;</item>
    ///   <item>they already differ after layer 0 → the head is irrelevant and something upstream (weight
    ///     layout, RoPE, norms, quantisation) is wrong in one of the loaders;</item>
    ///   <item>they agree at first and drift from some layer N → that layer names the suspect.</item>
    /// </list>
    ///
    /// <para>Cosine similarity per layer, not max absolute difference: the two paths run at different
    /// precisions (FP16 source vs FP32), so small magnitude differences are expected and uninteresting.
    /// A rotation of the residual stream is not.</para>
    ///
    /// <para>Loads the engines <b>sequentially</b> — each is multi-gigabyte and holding both would need
    /// ~30 GB.</para>
    /// </summary>
    [Trait("Category", "Gguf")]
    [Trait("Category", "Diagnostics")]
    public sealed class GgufVsBinaryLayerDivergenceDiagnostics
    {
        private static string GgufPath => TestModelPaths.Qwen3B.GgufPath;
        private static string BinaryPath => TestModelPaths.Qwen3B.BinaryPath;

        private readonly ITestOutputHelper _out;

        public GgufVsBinaryLayerDivergenceDiagnostics(ITestOutputHelper output) => _out = output;

        /// <summary>
        /// Splits the layer-0 disagreement by prompt length, which separates position-dependent causes
        /// from position-independent ones without reading a line of kernel code.
        ///
        /// <para><b>RoPE at position 0 is the identity rotation.</b> So a single-token prompt exercises the
        /// embedding, the norms, the projections and the FFN, but not the rotation. If the two loaders
        /// agree on one token and disagree on three, the rotary convention is implicated — this repository
        /// has already paid for exactly that once, in the HF <c>rotate-half</c> versus GGUF adjacent-pair
        /// permutation. If they disagree on a single token, RoPE is innocent and the cause is upstream of
        /// it: weights, layout, or the norm.</para>
        /// </summary>
        [LongFact]
        public void DoesTheDisagreementDependOnPosition()
        {
            TestModelPaths.Qwen3B.RequireGgufPath();
            TestModelPaths.Qwen3B.RequireBinaryPath();

            int[][] prompts = [[151643], [151643, 151644], [151643, 151644, 198]];
            var results = new List<(int Length, float Layer0, float Last)>();

            foreach (var prompt in prompts)
            {
                var (gguf, _, layers, _) = Capture(
                    () => CachedLlamaInferenceEngine.LoadGguf(GgufPath), prompt);

                GC.Collect();
                GC.WaitForPendingFinalizers();
                GC.Collect();

                var (binary, _, _, _) = Capture(() => CachedLlamaInferenceEngine.Load(BinaryPath), prompt);

                results.Add((prompt.Length,
                    Compare(gguf[0], binary[0]).Cosine,
                    Compare(gguf[layers - 1], binary[layers - 1]).Cosine));
            }

            _out.WriteLine("tokens   cosine layer 0   cosine last layer");

            foreach (var (length, first, last) in results)
            {
                _out.WriteLine($"{length,6}   {first,13:F6}   {last,17:F6}");
            }

            _out.WriteLine("");

            // The reading compares ACROSS lengths rather than looking at the single-token case alone. An
            // earlier version tested only `results[0]`, and once the permute fix landed it kept printing
            // "position-dependent" at a run where all three lengths agreed to 0.9996 — a diagnostic
            // asserting the opposite of its own table. Position dependence means layer 0 gets WORSE with
            // length; that is a comparison, so it takes at least two numbers.
            const float Agreement = 0.999f;
            var single = results[0].Layer0;
            var worstLonger = results.Skip(1).Min(r => r.Layer0);

            if (single < Agreement)
            {
                _out.WriteLine("READING: they disagree on a SINGLE token at position 0, where RoPE is the "
                               + "identity. The rotary convention is therefore NOT the cause — look at the "
                               + "embedding, the norms, or the block-0 projections.");
            }
            else if (worstLonger < single - 0.001f)
            {
                _out.WriteLine("READING: one token agrees and longer prompts are measurably worse, so the "
                               + "disagreement is POSITION-DEPENDENT — the rotary convention is the first "
                               + "suspect. This repository has had that exact bug before (HF rotate-half "
                               + "vs GGUF adjacent-pair).");
            }
            else
            {
                _out.WriteLine($"READING: layer 0 agrees at every length ({worstLonger:F6} at worst), so "
                               + "there is NO position-dependent disagreement — whatever remains does not "
                               + "come from the rotary convention. Any residual drift accumulates with "
                               + "DEPTH instead, which points at numeric precision rather than layout.");
            }

            _out.WriteLine("");
            _out.WriteLine("(diagnostic — reports, does not assert)");
        }

        [LongFact]
        public void WhereDoTheTwoLoadersFirstDisagree()
        {
            TestModelPaths.Qwen3B.RequireGgufPath();
            TestModelPaths.Qwen3B.RequireBinaryPath();

            // The same three-token prompt the integration test uses, so the two diagnostics describe the
            // same event rather than two similar ones.
            int[] prompt = [151643, 151644, 198];

            var (fromGguf, ggufLogits, layers, width) = Capture(
                () => CachedLlamaInferenceEngine.LoadGguf(GgufPath), prompt);

            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();

            var (fromBinary, binaryLogits, binaryLayers, binaryWidth) = Capture(
                () => CachedLlamaInferenceEngine.Load(BinaryPath), prompt);

            Assert.Equal(layers, binaryLayers);
            Assert.Equal(width, binaryWidth);

            _out.WriteLine($"{layers} layers, width {width}, prompt {prompt.Length} tokens");
            _out.WriteLine("");
            _out.WriteLine("layer   cosine      max|d|      note");

            var firstDivergent = -1;

            for (var layer = 0; layer < layers; layer++)
            {
                var (cosine, maximum) = Compare(fromGguf[layer], fromBinary[layer]);

                if (firstDivergent < 0 && cosine < 0.999f)
                {
                    firstDivergent = layer;
                }

                if (layer < 4 || layer == layers - 1 || layer == firstDivergent)
                {
                    _out.WriteLine($"{layer,5}   {cosine,8:F6}   {maximum,9:F4}"
                                   + (layer == firstDivergent ? "   <== first below 0.999" : ""));
                }
            }

            var (logitCosine, logitMax) = Compare(ggufLogits, binaryLogits);
            _out.WriteLine("");
            _out.WriteLine($"final logits   cosine {logitCosine:F6}   max|d| {logitMax:F4}");
            _out.WriteLine("");

            if (firstDivergent < 0)
            {
                _out.WriteLine("READING: every layer agrees. The two stacks compute the same residual "
                               + "stream, so any logit difference comes from the LM HEAD alone — which is "
                               + "what the tied-vs-separate head difference between the files predicts.");
            }
            else if (firstDivergent == 0)
            {
                _out.WriteLine("READING: they already disagree after layer 0. The head is applied once, at "
                               + "the end, so it cannot be the cause — this is upstream, and one of the "
                               + "two loaders is reading weights the other is not.");
            }
            else
            {
                _out.WriteLine($"READING: agreement holds until layer {firstDivergent}, which is where to "
                               + "look. Not the head, and not the embedding.");
            }

            // Reported, not asserted. This exists to locate a disagreement that is already known to be
            // there; failing on it would only restate what the integration test already says, and would
            // stop the output being printed at the point it becomes useful.
            _out.WriteLine("");
            _out.WriteLine("(diagnostic — reports, does not assert)");
        }

        private static (float[][] Layers, float[] Logits, int LayerCount, int Width) Capture(
            Func<CachedLlamaInferenceEngine> load, int[] prompt)
        {
            using var engine = load();
            var layers = engine.Config.NLayers;
            var width = engine.Config.DModel;

            engine.EnableActivationCapture(true);

            using var session = engine.CreateSession(64);
            session.Reset(prompt);

            var captured = new float[layers][];

            for (var layer = 0; layer < layers; layer++)
            {
                captured[layer] = new float[width];
                engine.GetLayerActivation(layer, captured[layer]);
            }

            return (captured, session.LastLogits.ToArray(), layers, width);
        }

        /// <summary>Cosine similarity and max absolute difference between two equal-length vectors.</summary>
        private static (float Cosine, float Max) Compare(ReadOnlySpan<float> left, ReadOnlySpan<float> right)
        {
            double dot = 0, leftNorm = 0, rightNorm = 0;
            var maximum = 0f;

            for (var i = 0; i < left.Length; i++)
            {
                dot += (double)left[i] * right[i];
                leftNorm += (double)left[i] * left[i];
                rightNorm += (double)right[i] * right[i];
                maximum = MathF.Max(maximum, MathF.Abs(left[i] - right[i]));
            }

            var denominator = Math.Sqrt(leftNorm) * Math.Sqrt(rightNorm);

            return (denominator > 0 ? (float)(dot / denominator) : 0f, maximum);
        }
    }
}
