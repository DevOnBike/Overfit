// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Intrinsics;

namespace DevOnBike.Overfit.Tests.Core.Kernels
{
    /// <summary>
    /// Correctness pin for <see cref="Simd.Dot"/>, written because it did not have one.
    ///
    /// <para><b>Why this had to exist before the kernel could be touched.</b> The measured improvement to
    /// that method is a second accumulator, and a second accumulator changes the ORDER in which floats are
    /// summed. Floating-point addition is not associative, so the returned value legitimately changes — which
    /// means "the number moved" cannot be used as a failure signal, and the only thing that can is a
    /// reference computed at higher precision with a tolerance sized to the accumulation error. Without that,
    /// a genuine regression and a harmless reordering look identical.</para>
    ///
    /// <para>The lengths deliberately straddle every boundary in the kernel: the 32-float dual-accumulator
    /// step, the 16-float single-accumulator step, the 8-float AVX2 step, and the scalar tail — plus one
    /// past each, because an off-by-one in a tail loop is the failure this shape of code actually has.</para>
    /// </summary>
    public sealed class SimdDotTests
    {
        /// <summary>
        /// Straddles 8 / 16 / 32-float blocks and their tails, then two sizes large enough that the
        /// accumulator count matters.
        /// </summary>
        public static TheoryData<int> Lengths =>
            [0, 1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 512, 1023, 1024, 1025, 4096, 65_537];

        /// <summary>
        /// The reference: the same products summed in <see cref="double"/>, sequentially. Not a second
        /// float implementation — comparing float against float would agree on a shared mistake.
        /// </summary>
        private static double ReferenceDot(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
        {
            var sum = 0.0;

            for (var i = 0; i < a.Length; i++)
            {
                sum += (double)a[i] * b[i];
            }

            return sum;
        }

        /// <summary>
        /// The error bound for summing n floats is proportional to the sum of the ABSOLUTE terms, not to the
        /// result — a dot product of mixed signs can cancel to near zero while every partial sum along the
        /// way was large. Scaling the tolerance by the result would make this test vacuous for exactly the
        /// inputs where cancellation makes it most valuable.
        /// </summary>
        private static double AbsoluteMagnitude(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
        {
            var sum = 0.0;

            for (var i = 0; i < a.Length; i++)
            {
                sum += Math.Abs((double)a[i] * b[i]);
            }

            return sum;
        }

        private static float[] Random(int length, uint seed)
        {
            var data = new float[length];

            for (var i = 0; i < length; i++)
            {
                seed = seed * 1664525u + 1013904223u;
                data[i] = ((seed & 0x00FFFFFF) / 16777216f) * 2f - 1f;
            }

            return data;
        }

        [Theory]
        [MemberData(nameof(Lengths))]
        public void MatchesADoublePrecisionReference(int length)
        {
            var a = Random(length, 11);
            var b = Random(length, 23);

            var actual = Simd.Dot(a, b);
            var expected = ReferenceDot(a, b);
            var tolerance = 1e-5 * AbsoluteMagnitude(a, b) + 1e-6;

            Assert.True(
                Math.Abs(actual - expected) <= tolerance,
                $"length {length}: got {actual}, reference {expected}, tolerance {tolerance}");
        }

        /// <summary>
        /// Cancellation case: b is -a, so every product is negative and the true answer is a large negative
        /// number — but the terms do not cancel, which keeps the reference meaningful while still exercising
        /// a sign the random case may not reach.
        /// </summary>
        [Theory]
        [MemberData(nameof(Lengths))]
        public void MatchesTheReferenceWhenEveryProductIsNegative(int length)
        {
            var a = Random(length, 31);
            var b = new float[length];

            for (var i = 0; i < length; i++)
            {
                b[i] = -a[i];
            }

            var actual = Simd.Dot(a, b);
            var expected = ReferenceDot(a, b);
            var tolerance = 1e-5 * AbsoluteMagnitude(a, b) + 1e-6;

            Assert.True(
                Math.Abs(actual - expected) <= tolerance,
                $"length {length}: got {actual}, reference {expected}, tolerance {tolerance}");
        }

        /// <summary>
        /// Exact for small integer inputs: every partial sum is representable, so there is no rounding to
        /// hide behind and the answer must be bit-exact regardless of how many accumulators the kernel uses.
        /// This is the arm that would catch a tail loop that skips or double-counts an element — a tolerance
        /// wide enough for float error is wide enough to swallow one missing term at these magnitudes.
        /// </summary>
        [Theory]
        [MemberData(nameof(Lengths))]
        public void IsExactForSmallIntegerInputs(int length)
        {
            var a = new float[length];
            var b = new float[length];

            for (var i = 0; i < length; i++)
            {
                a[i] = (i % 7) - 3;
                b[i] = (i % 5) - 2;
            }

            var expected = 0f;

            for (var i = 0; i < length; i++)
            {
                expected += a[i] * b[i];
            }

            Assert.Equal(expected, Simd.Dot(a, b));
        }

        /// <summary>Same input, same answer — bit for bit. A kernel that is not deterministic is not testable.</summary>
        [Fact]
        public void IsDeterministic()
        {
            var a = Random(4096, 5);
            var b = Random(4096, 6);

            Assert.Equal(Simd.Dot(a, b), Simd.Dot(a, b));
        }

        [Fact]
        public void RefusesMismatchedLengths()
        {
            Assert.Throws<ArgumentException>(() => Simd.Dot(new float[8], new float[9]));
        }

        /// <summary>
        /// A slice starting off a vector boundary. The kernel takes a reference to the first element and
        /// indexes from there, so an unaligned start must not change the answer — and the call sites in
        /// <c>TensorMath.Algebra</c> pass exactly this shape (a row slice out of a packed matrix).
        /// </summary>
        [Fact]
        public void HandlesUnalignedSlices()
        {
            var a = Random(4096 + 3, 7);
            var b = Random(4096 + 3, 8);

            var actual = Simd.Dot(a.AsSpan(3), b.AsSpan(3));
            var expected = ReferenceDot(a.AsSpan(3), b.AsSpan(3));
            var tolerance = 1e-5 * AbsoluteMagnitude(a.AsSpan(3), b.AsSpan(3)) + 1e-6;

            Assert.True(Math.Abs(actual - expected) <= tolerance, $"got {actual}, reference {expected}");
        }
    }
}
