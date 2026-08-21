// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels.Loading;
using DevOnBike.Overfit.LanguageModels.Runtime;

namespace DevOnBike.Overfit.Tests.LanguageModels.Runtime.Parity
{
    /// <summary>
    /// Pins the numerical size of the change <c>CachedMultiHeadAttention.TryDecodeWholeOutput</c> makes to the
    /// attention output projection, and it is a WEIGHT question, not a kernel question.
    /// <para>
    /// Per head, Wo is [headDim -> dModel] and <c>headDim</c> (128 on Qwen2.5-3B, 64 on Phi-3) is smaller than
    /// the 256-element K-quant super-block, so no per-head K-quant representation exists.
    /// <c>GgufLlamaLoader.LoadOutputHeads</c> therefore DEQUANTIZES the Q4_K tensor to F32 and RE-QUANTIZES each
    /// head to Q8. The whole matrix is [nHeads*headDim -> dModel], which does divide by 256, so the whole-output
    /// decode path reads those Q4_K bytes directly and never pays the re-quantize.
    /// </para>
    /// <para>
    /// So the entire weight-side difference between the two decode paths is the Q8 re-quantize, and this test
    /// measures exactly that: project one activation through the dequantized Q4_K weights (what the whole path
    /// uses) and through the per-head Q8 re-quantization of those same weights (what the per-head path uses),
    /// in F32 both times so no kernel or activation rounding is mixed in. The repacked GEMV's own reassociation
    /// is a separate question and is pinned by <c>AttentionWholeMatrixSplitAfterParityTests</c>.
    /// </para>
    /// <para>
    /// The bound is measured, not chosen: see the recorded figures in the assertion below. It exists so that a
    /// future change to <c>SplitOutput</c> or to the Q8 quantizer that widens this gap fails here rather than
    /// surfacing as a model that answers slightly differently.
    /// </para>
    /// </summary>
    public sealed class AttentionWholeOutputWeightFidelityTests
    {
        private readonly ITestOutputHelper _out;

        public AttentionWholeOutputWeightFidelityTests(ITestOutputHelper output)
        {
            _out = output;
        }

        [Theory]
        [InlineData(2048, 16, 128)]  // Qwen2.5-3B: dModel 2048, 16 heads, headDim 128
        [InlineData(3072, 48, 64)]   // Phi-3     : dModel 3072, 48 heads, headDim 64
        public void PerHeadQ8Requantize_AgreesWithDirectQ4K_OnTheOutputProjection(int dModel, int nHeads, int headDim)
        {
            var inputSize = nHeads * headDim;
            var outputSize = dModel;

            var weight = new float[(long)outputSize * inputSize];
            FillDeterministic(weight, 4242);

            // The on-disk representation. Everything below approximates THIS, not the F32 above.
            var q4kBytes = GgmlQuant.QuantizeQ4_K(weight, inputSize, outputSize);
            var whole = new Q4KWeight(q4kBytes, inputSize, outputSize);

            // What the whole-output decode path effectively multiplies by.
            var dequantized = new float[(long)outputSize * inputSize];

            for (var row = 0; row < outputSize; row++)
            {
                whole.DecodeRow(row, dequantized.AsSpan(row * inputSize, inputSize));
            }

            // What the per-head path builds — the gather in GgufLlamaLoader.SplitOutput, verbatim: head h's
            // contraction run for output row o is dequantized[o * inputSize + h * headDim ..].
            var perHead = new Q8Weight[nHeads];

            for (var h = 0; h < nHeads; h++)
            {
                var gather = new float[(long)outputSize * headDim];

                for (var o = 0; o < outputSize; o++)
                {
                    for (var i = 0; i < headDim; i++)
                    {
                        gather[o * headDim + i] = dequantized[o * inputSize + h * headDim + i];
                    }
                }

                perHead[h] = Q8Weight.QuantizeRows(gather, outputSize, headDim);
            }

            var activation = new float[inputSize];
            FillDeterministic(activation, 8888);

            // Reference: the dequantized Q4_K weights, projected exactly.
            var reference = new float[outputSize];

            for (var o = 0; o < outputSize; o++)
            {
                var sum = 0.0;

                for (var i = 0; i < inputSize; i++)
                {
                    sum += (double)dequantized[o * inputSize + i] * activation[i];
                }

                reference[o] = (float)sum;
            }

            // Per-head: each head's Q8 weights projected over its own activation band, summed in ascending
            // head order — the accumulation order CachedMultiHeadAttention's per-head path uses.
            var projected = new float[outputSize];
            var decodedRow = new float[headDim];

            for (var h = 0; h < nHeads; h++)
            {
                for (var o = 0; o < outputSize; o++)
                {
                    perHead[h].DecodeRow(o, decodedRow);

                    var sum = 0.0;

                    for (var i = 0; i < headDim; i++)
                    {
                        sum += (double)decodedRow[i] * activation[h * headDim + i];
                    }

                    projected[o] += (float)sum;
                }
            }

            var diff = 0.0;
            var norm = 0.0;

            for (var o = 0; o < outputSize; o++)
            {
                var d = (double)projected[o] - reference[o];
                diff += d * d;
                norm += (double)reference[o] * reference[o];
            }

            var relative = Math.Sqrt(diff) / Math.Sqrt(norm);

            _out.WriteLine($"dModel={dModel} nHeads={nHeads} headDim={headDim}");
            _out.WriteLine($"relative L2 difference introduced by the per-head Q8 re-quantize: {relative:E3}");

            // Measured 2026-08-21 on this deterministic input: 3.702E-003 at 2048/16/128 and 3.637E-003 at
            // 3072/48/64. The bound is ~3x the larger of the two. It is deliberately NOT set just above the
            // reading: a bound 8% clear of its own measurement goes red on rounding rather than on a change
            // of kind, and a test that cries wolf gets loosened by the next person rather than read.
            Assert.True(
                relative < 1.2e-2,
                $"per-head Q8 re-quantize now differs from the direct Q4_K weights by {relative:E3} relative L2");
        }

        private static void FillDeterministic(Span<float> destination, int seed)
        {
            var state = (uint)seed;

            for (var i = 0; i < destination.Length; i++)
            {
                state = (state * 1664525u) + 1013904223u;
                destination[i] = ((state >> 8) / (float)(1 << 24) * 2.0f) - 1.0f;
            }
        }
    }
}
