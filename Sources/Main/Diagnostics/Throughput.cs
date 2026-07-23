// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Diagnostics
{
    /// <summary>
    /// Turns "this much work in this much time" into a rate — TFLOP/s, GFLOP/s, GB/s — so throughput is
    /// computed the same way everywhere instead of being re-derived per caller.
    ///
    /// <para><b>Why this is a type and not a formula at each call site.</b> Rates are trivial arithmetic and
    /// exactly for that reason they get written ad hoc, in scripts, next to the numbers they describe. On
    /// 2026-07-22 that produced a reported <c>29.6 TFLOP/s</c> for a routine that performs no multiply-add at
    /// all: the matmul FLOP formula had been applied to an activation-quantization pass, so memory bandwidth
    /// was published as arithmetic throughput. Naming the work — <see cref="MatmulFlops"/> versus
    /// <see cref="GigabytesPerSecond"/> — makes that category error visible at the call site.</para>
    ///
    /// <para><b>Counting convention.</b> A multiply-accumulate is <b>two</b> operations, matching llama.cpp's
    /// <c>test-backend-ops</c>: its printed "60.13 GFLOP" for <c>m=4096, k=14336, n=512</c> is exactly
    /// <c>2·512·14336·4096</c>, so figures produced here are directly comparable with that project's. For
    /// quantized kernels count the <i>logical</i> MACs of the matmul being performed, not the machine
    /// instructions retired — a Q4_K matmul and an F32 matmul of the same shape are credited identically,
    /// which is the only way a quantized kernel can be placed against a dense roofline.</para>
    ///
    /// <para><b>Reading a rate needs a ceiling.</b> A bare "1.70 TFLOP/s" says nothing; the same kernel was 78%
    /// of one ceiling and 15% of another, and only the second was the ceiling that applied. Compare against the
    /// instruction mix the code actually issues, not against a peak it could never reach.</para>
    /// </summary>
    public static class Throughput
    {
        /// <summary>Operations charged to one multiply-accumulate.</summary>
        public const int OperationsPerMultiplyAccumulate = 2;

        /// <summary>
        /// Logical operations in a matmul of <paramref name="rows"/>×<paramref name="inner"/> by
        /// <paramref name="inner"/>×<paramref name="columns"/>.
        /// </summary>
        public static long MatmulFlops(long rows, long inner, long columns)
        {
            return OperationsPerMultiplyAccumulate * rows * inner * columns;
        }

        /// <summary>Teraflops sustained by <paramref name="flops"/> operations over <paramref name="elapsed"/>.</summary>
        public static double TeraflopsPerSecond(long flops, TimeSpan elapsed)
        {
            return RatePerSecond(flops, elapsed) / 1e12;
        }

        /// <summary>Gigaflops sustained by <paramref name="flops"/> operations over <paramref name="elapsed"/>.</summary>
        public static double GigaflopsPerSecond(long flops, TimeSpan elapsed)
        {
            return RatePerSecond(flops, elapsed) / 1e9;
        }

        /// <summary>
        /// Gigabytes per second moved by <paramref name="bytes"/> over <paramref name="elapsed"/>. Count reads
        /// plus writes, following the STREAM convention: a copy of N bytes is 2N, and the read-for-ownership
        /// traffic a write implies is not counted.
        /// </summary>
        public static double GigabytesPerSecond(long bytes, TimeSpan elapsed)
        {
            return RatePerSecond(bytes, elapsed) / 1e9;
        }

        /// <summary>
        /// What fraction of <paramref name="ceiling"/> a measured rate reaches, as a value in [0, ∞).
        /// Both arguments must be in the same unit; the point of the helper is to make the pairing explicit.
        /// </summary>
        public static double FractionOfCeiling(double achieved, double ceiling)
        {
            return ceiling <= 0.0 ? 0.0 : achieved / ceiling;
        }

        private static double RatePerSecond(long amount, TimeSpan elapsed)
        {
            var seconds = elapsed.TotalSeconds;

            return amount <= 0L || seconds <= 0.0 ? 0.0 : amount / seconds;
        }
    }
}
