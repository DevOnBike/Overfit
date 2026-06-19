// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Intrinsics;
using DevOnBike.Overfit.LanguageModels.Loading;
using DevOnBike.Overfit.LanguageModels.Runtime;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests.LanguageModels.Diagnostics
{
    /// <summary>
    /// M2 de-risk for the whole-matrix Q4_K attention refactor (ROADMAP #4). The whole refactor rests on one
    /// assumption: projecting Q/K/V/O as ONE whole-matrix Q4_K repacked GEMV and slicing the output into heads
    /// AFTERWARDS yields the same result as today's per-head Q4_K projections. This pins that numerically before
    /// any loader / runtime surgery.
    /// <para>
    /// The per-head split is byte-for-byte a slice of the whole matrix (output-major Q4_K rows), so per-head
    /// <see cref="Q4KDotKernel.ProjectPreQuantized"/> is bit-identical to the whole-matrix reference on the same
    /// rows. The only thing that reassociates the reduction is the <b>repacked GEMV</b> (<see cref="Q4KGemvKernel"/>,
    /// 8 rows interleaved) the decode path will use — so we assert (a) the repacked whole GEMV matches the per-row
    /// reference, and (b) its output, sliced per head, matches the per-head projections, both within float tolerance.
    /// </para>
    /// </summary>
    public sealed class AttentionWholeMatrixSplitAfterParityTests
    {
        private readonly ITestOutputHelper _out;
        public AttentionWholeMatrixSplitAfterParityTests(ITestOutputHelper output) => _out = output;

        [Theory]
        [InlineData(2048, 2048, 16, 128)]  // Qwen-3B Q / O : dModel 2048, nHeads 16, headDim 128
        [InlineData(3072, 3072, 48, 64)]   // Phi-3 Q / O   : dModel 3072, nHeads 48, headDim 64
        public void WholeRepackedGemv_SplitPerHead_MatchesPerHeadProjection(int inputSize, int outputSize, int nHeads, int headDim)
        {
            if (!CpuFeatures.HasAvx2)
            {
                _out.WriteLine("AVX2 not available — the repacked GEMV is AVX2-only; skipping.");
                return;
            }

            Assert.Equal(outputSize, nHeads * headDim);

            // Whole projection weight [outputSize × inputSize] and a single token's hidden row.
            var weight = new float[outputSize * inputSize];
            FillDeterministic(weight, 1234);
            var hidden = new float[inputSize];
            FillDeterministic(hidden, 5678);

            // Whole-matrix Q4_K — the exact bytes the loader keeps before it slices per head.
            var wholeBytes = GgmlQuant.QuantizeQ4_K(weight, inputSize, outputSize);
            var whole = new Q4KWeight(wholeBytes, inputSize, outputSize);
            var repacked = whole.EnsureRepacked().ToArray();

            // One shared Q8_K activation, exactly as decode quantizes the hidden once for all projections.
            var actQuants = new sbyte[inputSize];
            var actScales = new float[(inputSize + Q4KDotKernel.SuperBlockElements - 1) / Q4KDotKernel.SuperBlockElements];
            var actBsums = new short[(inputSize + Q4KDotKernel.GroupSize - 1) / Q4KDotKernel.GroupSize];
            Q4KDotKernel.QuantizeActivationQ8K(hidden, actQuants, actScales, actBsums);

            // (a) Repacked whole GEMV (decode path) vs per-row reference over the whole matrix.
            var gemvOut = new float[outputSize];
            Q4KGemvKernel.Gemv(repacked, outputSize, inputSize, actQuants, actScales, actBsums, gemvOut);

            var refOut = new float[outputSize];
            Q4KDotKernel.ProjectPreQuantized(whole, [], refOut, actQuants, actScales, actBsums);

            var gemvVsRef = MaxAbsDiff(gemvOut, refOut);

            // (b) The same per-head projection the runtime does today: slice the whole Q4_K bytes into head rows.
            // The bytes are output-major, so the nHeads contiguous head-row groups divide the buffer evenly.
            var headBytes = wholeBytes.Length / nHeads;
            var headOut = new float[headDim];
            var maxSplitDiff = 0f;
            for (var h = 0; h < nHeads; h++)
            {
                var slice = new byte[headBytes];
                Array.Copy(wholeBytes, (long)h * headBytes, slice, 0, headBytes);
                var head = new Q4KWeight(slice, inputSize, headDim);
                Q4KDotKernel.ProjectPreQuantized(head, [], headOut, actQuants, actScales, actBsums);

                for (var j = 0; j < headDim; j++)
                {
                    var d = MathF.Abs(headOut[j] - gemvOut[h * headDim + j]);
                    if (d > maxSplitDiff) { maxSplitDiff = d; }
                }
            }

            var scale = MaxAbs(refOut);
            _out.WriteLine($"dims {outputSize}×{inputSize}, {nHeads}×{headDim} | |out|max {scale:F3} | GEMV-vs-ref {gemvVsRef:E2} | split-vs-perHead {maxSplitDiff:E2}");

            // Repacked GEMV reassociates the reduction → tolerance, not bit-identity. A few 1e-3 relative is ample.
            var tol = 1e-2f * MathF.Max(1f, scale);
            Assert.True(gemvVsRef < tol, $"repacked GEMV diverges from per-row reference: {gemvVsRef} (tol {tol})");
            Assert.True(maxSplitDiff < tol, $"whole-GEMV split-after diverges from per-head projection: {maxSplitDiff} (tol {tol})");
        }

        private static float MaxAbsDiff(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
        {
            var m = 0f;
            for (var i = 0; i < a.Length; i++)
            {
                var d = MathF.Abs(a[i] - b[i]);
                if (d > m) { m = d; }
            }
            return m;
        }

        private static float MaxAbs(ReadOnlySpan<float> a)
        {
            var m = 0f;
            for (var i = 0; i < a.Length; i++)
            {
                var v = MathF.Abs(a[i]);
                if (v > m) { m = v; }
            }
            return m;
        }

        private static void FillDeterministic(float[] a, int seed)
        {
            var state = (uint)seed | 1u;
            for (var i = 0; i < a.Length; i++)
            {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                a[i] = (state / (float)uint.MaxValue) * 2f - 1f;
            }
        }
    }
}
