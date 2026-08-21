// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.GpuProbe
{
    /// <summary>
    /// The outcome of comparing one arm's output against the CPU F32 reference. A fast wrong kernel is
    /// the failure mode this probe is most exposed to, so a failed check withholds the timing rather
    /// than sitting beside it (plan, section 3.6).
    /// <para>
    /// THE GATE IS COSINE PLUS RELATIVE L2. Maximum relative error is measured and printed, but it does
    /// NOT gate, and that changed on 2026-08-21 when the FP16 arm arrived. Max-rel is an extreme-value
    /// statistic over millions of elements: it is decided by the single worst element in the output, so
    /// on an FP16 arm one near-cancellation - where the reference itself is the small difference of two
    /// large numbers - fails a kernel that is correct everywhere else. Relative L2,
    /// <c>||a-b||_2 / ||a||_2</c>, is the whole-tensor error the numerics literature uses for exactly
    /// this reason, and it is what a training run would actually feel.
    /// </para>
    /// <para>
    /// Max-rel is kept and printed because it is the better DIAGNOSTIC of the two. A transposed index or
    /// a wrong tile boundary shows up in max-rel long before it moves an L2 norm, so the number is worth
    /// having in front of a human even though it is the wrong thing to gate on.
    /// </para>
    /// </summary>
    internal sealed class ParityResult
    {
        private ParityResult(
            bool passed,
            double cosine,
            double relativeL2,
            double maxRelative,
            double refMax,
            double ceiling,
            string detail)
        {
            Passed = passed;
            Cosine = cosine;
            RelativeL2 = relativeL2;
            MaxRelative = maxRelative;
            ReferenceMax = refMax;
            Ceiling = ceiling;
            Detail = detail;
        }

        /// <summary>Cosine must reach this. Below it the kernel is computing something else.</summary>
        public const double CosineFloor = 0.9999;

        /// <summary>
        /// Relative L2 ceiling for an arm computed in F32 throughout. Not bit-parity: a serial dot
        /// product and a parallel tree reduction accumulate in different orders, so bit-parity would
        /// fail a correct kernel.
        /// </summary>
        public const double F32RelativeL2Ceiling = 1e-4;

        /// <summary>
        /// Relative L2 ceiling for an FP16 arm that accumulates in FP32 - a tensor core, or
        /// <c>cublasGemmEx</c> with <c>CUBLAS_COMPUTE_32F</c>. Measured, not assumed: 2.87e-4 to 2.97e-4
        /// across every real shape and batch on 2026-08-21 (<c>--fp16-bound</c>), independent of k, m
        /// and n, because the only error is one rounding of each input. Three times the worst measured
        /// value.
        /// </summary>
        public const double Fp16Fp32AccumulateCeiling = 1e-3;

        /// <summary>
        /// Relative L2 ceiling for an FP16 arm that accumulates in FP16 - <c>cublasHgemm</c>, which is
        /// the only FP16 entry point ILGPU's cuBLAS wrapper exposes. This one DEPENDS ON K, because the
        /// running sum is rounded once per multiply-add and the error of a random walk of k roundings
        /// grows as sqrt(k).
        /// <para>
        /// Measured on 2026-08-21 with <c>--fp16-bound</c>, 4096 sampled output elements per shape:
        /// 6.53e-3, 6.58e-3 and 6.91e-3 at k = 2048, and 1.58e-2 at k = 11008. Fitting
        /// <c>C * sqrt(k)</c> to the k = 11008 point gives C = 1.51e-4, which PREDICTS 6.8e-3 at
        /// k = 2048 against 6.5e-3 to 6.9e-3 measured - so the mechanism is confirmed, not curve-fitted.
        /// The ceiling is three times that fit.
        /// </para>
        /// </summary>
        public static double Fp16Fp16AccumulateCeiling(int k) => 3.0 * 1.51e-4 * Math.Sqrt(k);

        public bool Passed { get; }

        public double Cosine { get; }

        /// <summary>The gate: <c>||reference - candidate||_2 / ||reference||_2</c>.</summary>
        public double RelativeL2 { get; }

        /// <summary>
        /// Diagnostic only. Maximum relative error over elements whose REFERENCE magnitude exceeds 1.
        /// Does not gate - see the type comment.
        /// </summary>
        public double MaxRelative { get; }

        public double ReferenceMax { get; }

        /// <summary>The relative-L2 ceiling this comparison was judged against.</summary>
        public double Ceiling { get; }

        public string Detail { get; }

        public static ParityResult NotRun(string reason) => new(false, 0, 0, 0, 0, 0, reason);

        /// <summary>
        /// Compares against the F32 ceiling. Use <see cref="Compare(ReadOnlySpan{float},
        /// ReadOnlySpan{float}, double)"/> with <see cref="Fp16Fp16AccumulateCeiling(int)"/> or
        /// <see cref="Fp16Fp32AccumulateCeiling"/> for an FP16 arm.
        /// </summary>
        public static ParityResult Compare(ReadOnlySpan<float> reference, ReadOnlySpan<float> candidate)
            => Compare(reference, candidate, F32RelativeL2Ceiling);

        public static ParityResult Compare(
            ReadOnlySpan<float> reference,
            ReadOnlySpan<float> candidate,
            double relativeL2Ceiling)
        {
            if (reference.Length != candidate.Length)
            {
                return NotRun($"length mismatch: reference {reference.Length}, candidate {candidate.Length}");
            }

            double dot = 0, na = 0, nb = 0, diff = 0, maxRel = 0, refMax = 0;
            var nonFinite = 0;

            for (var i = 0; i < reference.Length; i++)
            {
                double a = reference[i], b = candidate[i];
                if (!double.IsFinite(b))
                {
                    nonFinite++;
                    continue;
                }

                dot += a * b;
                na += a * a;
                nb += b * b;
                diff += (a - b) * (a - b);
                refMax = Math.Max(refMax, Math.Abs(a));

                if (Math.Abs(a) > 1.0)
                {
                    maxRel = Math.Max(maxRel, Math.Abs(a - b) / Math.Abs(a));
                }
            }

            var cosine = na > 0 && nb > 0 ? dot / (Math.Sqrt(na) * Math.Sqrt(nb)) : 0;
            var relativeL2 = na > 0 ? Math.Sqrt(diff) / Math.Sqrt(na) : 0;
            var passed = nonFinite == 0 && cosine >= CosineFloor && relativeL2 < relativeL2Ceiling;

            var detail = nonFinite > 0
                ? $"{nonFinite} non-finite output element(s)"
                : passed
                    ? "ok"
                    : cosine < CosineFloor
                        ? $"cosine {cosine:F7} below the {CosineFloor} floor"
                        : $"relative L2 {relativeL2:E2} at or above the {relativeL2Ceiling:E1} ceiling";

            return new ParityResult(passed, cosine, relativeL2, maxRel, refMax, relativeL2Ceiling, detail);
        }
    }
}
