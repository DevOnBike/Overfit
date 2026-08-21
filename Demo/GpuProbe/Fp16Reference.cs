// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Numerics.Tensors;

namespace DevOnBike.Overfit.GpuProbe
{
    /// <summary>
    /// Measures, on the host, how much accuracy an FP16 arm loses at a real shape — so the parity
    /// ceiling for arm X1 is a measured number rather than a hopeful constant.
    /// <para>
    /// TWO bounds are produced, because FP16 GEMM is not one thing and the difference between them is
    /// larger than either:
    /// </para>
    /// <list type="bullet">
    /// <item><b>FP32 accumulate</b> — inputs rounded to FP16, products accumulated in FP32. This is what
    /// a TENSOR CORE actually does, and what <c>cublasGemmEx</c> with <c>CUBLAS_COMPUTE_32F</c> gives.
    /// The error is one rounding of each input and nothing more.</item>
    /// <item><b>FP16 accumulate</b> — the running sum is rounded back to FP16 at every step. This is what
    /// <c>cublasHgemm</c> gives, which is the only FP16 entry point ILGPU's wrapper exposes. The error
    /// grows with the contraction length and is far larger.</item>
    /// </list>
    /// <para>
    /// The FP16-accumulate figure is estimated from a random SUBSAMPLE of output elements, because the
    /// running sum has to be rounded once per multiply-add and that cannot be vectorised. A relative L2
    /// over a random subsample is an unbiased estimator of the whole-tensor one, and the subsample size
    /// is reported so a reader can judge it rather than trust it.
    /// </para>
    /// </summary>
    internal static class Fp16Reference
    {
        /// <summary>Output elements sampled for the FP16-accumulate estimate.</summary>
        public const int AccumulateSampleCount = 4096;

        /// <summary>
        /// Rounds <paramref name="input"/> and <paramref name="weight"/> to FP16 and recomputes the
        /// forward in FP32, filling <paramref name="output"/>. Row-major throughout:
        /// <c>output[b, o] = sum_i input[b, i] * weight[o, i]</c>.
        /// </summary>
        public static void ForwardFp32Accumulate(
            ReadOnlySpan<float> input,
            ReadOnlySpan<float> weight,
            Span<float> output,
            int n,
            int k,
            int m)
        {
            var inputHalf = new float[(long)n * k];
            var weightHalf = new float[(long)m * k];
            RoundToFp16(input, inputHalf);
            RoundToFp16(weight, weightHalf);

            for (var b = 0; b < n; b++)
            {
                var row = inputHalf.AsSpan(b * k, k);
                for (var o = 0; o < m; o++)
                {
                    output[(b * m) + o] = TensorPrimitives.Dot(row, weightHalf.AsSpan(o * k, k));
                }
            }
        }

        /// <summary>
        /// Relative L2 of an FP16-ACCUMULATE forward against <paramref name="reference"/>, estimated
        /// over <see cref="AccumulateSampleCount"/> random output elements.
        /// </summary>
        public static double RelativeL2WithFp16Accumulate(
            ReadOnlySpan<float> input,
            ReadOnlySpan<float> weight,
            ReadOnlySpan<float> reference,
            int n,
            int k,
            int m,
            int seed,
            out int sampled)
        {
            var inputHalf = new float[(long)n * k];
            var weightHalf = new float[(long)m * k];
            RoundToFp16(input, inputHalf);
            RoundToFp16(weight, weightHalf);

            var total = (long)n * m;
            sampled = (int)Math.Min(AccumulateSampleCount, total);

            var rnd = new Random(seed);
            double diff = 0, norm = 0;

            for (var s = 0; s < sampled; s++)
            {
                var index = (long)(rnd.NextDouble() * total);
                if (index >= total)
                {
                    index = total - 1;
                }

                var b = (int)(index / m);
                var o = (int)(index % m);

                var acc = (Half)0f;
                for (var i = 0; i < k; i++)
                {
                    // One rounding to FP16 per multiply-add: the defining property of cublasHgemm, and
                    // the reason its error grows with k rather than staying at one ulp.
                    acc = (Half)((float)acc + (inputHalf[(b * k) + i] * weightHalf[(o * k) + i]));
                }

                double a = reference[(b * m) + o];
                double d = a - (float)acc;
                diff += d * d;
                norm += a * a;
            }

            return norm > 0 ? Math.Sqrt(diff) / Math.Sqrt(norm) : 0;
        }

        private static void RoundToFp16(ReadOnlySpan<float> source, Span<float> destination)
        {
            for (var i = 0; i < source.Length; i++)
            {
                destination[i] = (float)(Half)source[i];
            }
        }
    }
}
