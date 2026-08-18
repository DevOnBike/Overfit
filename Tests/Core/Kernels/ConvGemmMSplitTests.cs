// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Kernels;

namespace DevOnBike.Overfit.Tests.Core.Kernels
{
    /// <summary>
    /// The convolution GEMM matches a naive triple loop at the shapes where its work decomposition changes.
    ///
    /// <para><b>Why these shapes and not others (`XC-78`).</b> The GEMM dispatches one work item per N-panel,
    /// 32 columns wide on AVX-512. VGG-16's last three convolutions have <c>N = 196</c>, which is seven
    /// panels — fewer items than the machine has workers, so most of the machine sits idle whatever the
    /// micro-kernel does. Those layers measured 2.84x scaling against the machine's own 14.30x ceiling. The
    /// fix splits the M sweep as well, so one panel can occupy several workers.</para>
    ///
    /// <para><b>What made this test necessary is a mutation that escaped the whole suite.</b> Replacing the
    /// M-block's end bound with <c>rowBlockStart + 1</c> — which computes only the first row block of each
    /// M-block and drops the rest — left all 2707 tests green. Every conv shape the suite exercises is small
    /// enough that each M-block holds exactly one row block, so the bound was never read. <b>A green suite
    /// says nothing about a branch no shape reaches</b>, and the shape, not the code, was what was
    /// missing.</para>
    ///
    /// <para>The oracle is a naive triple loop, not a second configuration of the same kernel, so a
    /// decomposition that is self-consistently wrong cannot agree with it.</para>
    ///
    /// <para><b>One class of defect here is invisible to any output test, and that is worth knowing.</b>
    /// Dropping the M-block's START bound — so every block begins at row 0 — leaves these tests green.
    /// It is not a coverage gap: the workers then recompute each other's rows and store identical values
    /// to identical addresses, so the result is still right and only the cost multiplies. <b>A lost start
    /// bound presents as a slowdown, never as a wrong answer</b>, so it is the benchmark and not the suite
    /// that would catch it.</para>
    /// </summary>
    public sealed class ConvGemmMSplitTests
    {
        /// <summary>
        /// Shapes chosen against the decomposition rather than against convolution: <c>N</c> below one panel
        /// per worker, and <c>M</c> large enough that an M-block spans several 8-row blocks.
        /// </summary>
        public static TheoryData<int, int, int> Shapes()
        {
            return new TheoryData<int, int, int>
            {
                // VGG-16 conv11-13 proportions: 7 panels, 64 row blocks. This is the case the escaped
                // mutation lived in — several row blocks per M-block.
                { 512, 196, 64 },

                // One panel exactly, so the split has only M to work with.
                { 256, 32, 48 },

                // A partial last panel (196 = 6*32 + 4) AND a partial last row block (100 = 12*8 + 4),
                // so both edges are exercised in the same call.
                { 100, 196, 33 },

                // Fewer row blocks than the split would like, which must clamp rather than produce empty work.
                { 8, 64, 16 },

                // Wide N, so no split happens at all and this is the original dispatch.
                { 64, 2048, 32 },
            };
        }

        [Theory]
        [MemberData(nameof(Shapes))]
        public void Gemm_MatchesANaiveTripleLoop(int m, int n, int k)
        {
            var a = Deterministic(m * k, seed: 11);
            var b = Deterministic(k * n, seed: 29);
            var actual = new float[m * n];

            Conv2DGemmKernels.Gemm(a, b, actual, m, n, k);

            var expected = new float[m * n];

            // The error bound is carried per element, not chosen as a constant. A dot product of signed
            // values cancels, so |expected| is not the size of the arithmetic that produced it: here a
            // result of 0.0021 came out of partial sums near 2.7, and the two engines' last bits differed
            // by 1.0e-6 — ordinary rounding that a relative test on the cancelled result reads as a 4.7e-4
            // failure. The textbook bound for a K-term dot product is |error| <= gamma_K * sum|a_i * b_i|
            // with gamma_K = K*u and u = 2^-24, so the sum of absolute products is what is accumulated
            // alongside the value. Doubled for headroom, and no looser: the defects this test exists for
            // (a dropped row block, a swapped index) miss by whole result magnitudes, not by last bits.
            var bound = new float[m * n];

            const float Unit = 5.9604645e-8f;

            for (var i = 0; i < m; i++)
            {
                for (var j = 0; j < n; j++)
                {
                    var sum = 0f;
                    var absSum = 0f;

                    for (var kk = 0; kk < k; kk++)
                    {
                        var product = a[(i * k) + kk] * b[(kk * n) + j];

                        sum += product;
                        absSum += MathF.Abs(product);
                    }

                    expected[(i * n) + j] = sum;
                    bound[(i * n) + j] = MathF.Max(1e-7f, 2f * k * Unit * absSum);
                }
            }

            for (var i = 0; i < expected.Length; i++)
            {
                Assert.True(
                    MathF.Abs(actual[i] - expected[i]) <= bound[i],
                    $"m={m} n={n} k={k}, element {i} (row {i / n}, col {i % n}): "
                    + $"kernel {actual[i]}, naive reference {expected[i]}, "
                    + $"difference {MathF.Abs(actual[i] - expected[i])} over a bound of {bound[i]}");
            }
        }

        /// <summary>
        /// Nothing outside the result is written. The decomposition indexes C by (row block, panel), and an
        /// off-by-one in either would land outside the matrix — which a value test on C alone cannot see.
        /// </summary>
        [Fact]
        public void Gemm_WritesNothingPastTheResult()
        {
            const int M = 512;
            const int N = 196;
            const int K = 64;
            const int Guard = 64;
            const float Sentinel = -12345.5f;

            var a = Deterministic(M * K, seed: 11);
            var b = Deterministic(K * N, seed: 29);
            var buffer = new float[(M * N) + Guard];

            for (var i = 0; i < buffer.Length; i++)
            {
                buffer[i] = Sentinel;
            }

            Conv2DGemmKernels.Gemm(a, b, buffer.AsSpan(0, M * N), M, N, K);

            for (var i = M * N; i < buffer.Length; i++)
            {
                Assert.True(
                    buffer[i] == Sentinel,
                    $"the GEMM wrote {buffer[i]} at index {i}, {i - (M * N)} elements past the result");
            }
        }

        /// <summary>
        /// Every element of the result is written. An M-block whose row range is computed wrongly leaves rows
        /// untouched, and an untouched row holding a plausible number reads as a correct answer.
        /// </summary>
        [Fact]
        public void Gemm_WritesEveryElementOfTheResult()
        {
            const int M = 512;
            const int N = 196;
            const int K = 64;
            const float Sentinel = -12345.5f;

            var a = Deterministic(M * K, seed: 11);
            var b = Deterministic(K * N, seed: 29);
            var c = new float[M * N];

            for (var i = 0; i < c.Length; i++)
            {
                c[i] = Sentinel;
            }

            Conv2DGemmKernels.Gemm(a, b, c, M, N, K);

            for (var i = 0; i < c.Length; i++)
            {
                Assert.True(
                    c[i] != Sentinel,
                    $"row {i / N}, column {i % N} was never written: the decomposition skipped it");
            }
        }

        private static float[] Deterministic(int length, int seed)
        {
            var values = new float[length];
            var state = (uint)(0x9E3779B9 + seed);

            for (var i = 0; i < length; i++)
            {
                state = (state * 1664525u) + 1013904223u;
                values[i] = (((state & 0x00FFFFFF) / 16777216f) * 2f) - 1f;
            }

            return values;
        }
    }
}
