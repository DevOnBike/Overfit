// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace Benchmarks.Helpers
{
    /// <summary>
    /// How much work one invocation of a benchmark performs, so BenchmarkDotNet's measured time can be
    /// turned into a rate (TFLOP/s, GB/s) <b>in the repository</b> rather than in a throwaway script.
    ///
    /// <para><b>Why this type exists.</b> On 2026-07-22 a prefill comparison against llama.cpp was reported in
    /// TFLOP/s computed ad hoc outside the repo. The matmul rows were right, but the same
    /// <c>2·rows·k·n</c> formula was also applied to a benchmark that performs <i>no multiply-accumulate at
    /// all</i> (activation quantization), producing a nonsense "29.6 TFLOP/s" that was really memory
    /// bandwidth wearing a FLOP costume. Encoding the work amount next to the benchmark makes that class of
    /// error impossible: a benchmark that does no arithmetic declares <see cref="Flops"/> = 0 and simply gets
    /// no TFLOP/s column.</para>
    ///
    /// <para><b>Conventions.</b> <see cref="Flops"/> counts a multiply-accumulate as <b>2</b> operations, which
    /// is what llama.cpp's own <c>test-backend-ops</c> does — its printed "60.13 GFLOP/run" for
    /// <c>q4_K m=4096 k=14336 n=512</c> is exactly <c>2·512·14336·4096</c>, so the two projects' numbers are
    /// directly comparable. For quantized kernels the count is the <i>logical</i> MAC count of the matmul, not
    /// the number of machine instructions retired — a Q4_K matmul and an F32 matmul of the same shape are
    /// credited identically, which is the only way a quantized kernel can be compared to a dense roofline.</para>
    ///
    /// <para><see cref="Bytes"/> counts bytes that must cross the memory bus, reads plus writes. Follow the
    /// STREAM convention and count the write itself but not the read-for-ownership traffic it implies, so
    /// a copy of N bytes is 2N, not 3N.</para>
    /// </summary>
    /// <param name="Flops">Logical floating-point operations per invocation; 0 when the benchmark does no arithmetic.</param>
    /// <param name="Bytes">Bytes moved to/from memory per invocation; 0 when the benchmark is not bandwidth-bound.</param>
    public readonly record struct WorkAmount(long Flops, long Bytes)
    {
        /// <summary>A matmul of <paramref name="rows"/>×<paramref name="k"/> by <paramref name="k"/>×<paramref name="n"/>.</summary>
        public static WorkAmount Matmul(long rows, long k, long n)
        {
            return new WorkAmount(2L * rows * k * n, 0L);
        }

        /// <summary>Pure memory traffic, no arithmetic worth counting.</summary>
        public static WorkAmount Memory(long bytes)
        {
            return new WorkAmount(0L, bytes);
        }
    }
}
