// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Kernels;

namespace DevOnBike.Overfit.Tests.Kernels
{
    public sealed class ElementwiseKernelsAliasingTests
    {
        [Fact]
        public void MultiplyAdd_AllowsAddendDestinationAliasing_SmallScalarPath()
        {
            var input = new float[]
            {
                1f,
                2f,
                3f,
                -4f
            };

            var destination = new float[]
            {
                10f,
                20f,
                30f,
                40f
            };

            ElementwiseKernels.MultiplyAdd(
                input,
                -0.1f,
                destination,
                destination);

            Assert.Equal(9.9f, destination[0], precision: 6);
            Assert.Equal(19.8f, destination[1], precision: 6);
            Assert.Equal(29.7f, destination[2], precision: 6);
            Assert.Equal(40.4f, destination[3], precision: 6);
        }

        [Fact]
        public void MultiplyAdd_AllowsAddendDestinationAliasing_LargeTensorPrimitivesPath()
        {
            const int length = 4096;

            var input = new float[length];
            var expected = new float[length];
            var destination = new float[length];

            FillDeterministic(
                input,
                seedOffset: 1);

            FillDeterministic(
                destination,
                seedOffset: 2);

            destination.AsSpan().CopyTo(expected);

            const float scalar = -0.03125f;

            for (var i = 0; i < expected.Length; i++)
            {
                expected[i] = (input[i] * scalar) + expected[i];
            }

            ElementwiseKernels.MultiplyAdd(
                input,
                scalar,
                destination,
                destination);

            AssertClose(
                expected,
                destination,
                tolerance: 0f);
        }

        [Fact]
        public void MultiplyAdd_AddendDestinationAliasing_AllocatesZeroBytes()
        {
            const int length = 4096;

            var input = new float[length];
            var destination = new float[length];

            FillDeterministic(
                input,
                seedOffset: 1);

            FillDeterministic(
                destination,
                seedOffset: 2);

            ElementwiseKernels.MultiplyAdd(
                input,
                -0.01f,
                destination,
                destination);

            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();

            var before = GC.GetAllocatedBytesForCurrentThread();

            for (var i = 0; i < 10_000; i++)
            {
                ElementwiseKernels.MultiplyAdd(
                    input,
                    -0.01f,
                    destination,
                    destination);
            }

            var allocated = GC.GetAllocatedBytesForCurrentThread() - before;

            Assert.Equal(
                0,
                allocated);
        }

        private static void AssertClose(
            ReadOnlySpan<float> expected,
            ReadOnlySpan<float> actual,
            float tolerance)
        {
            Assert.Equal(
                expected.Length,
                actual.Length);

            for (var i = 0; i < expected.Length; i++)
            {
                var diff = MathF.Abs(expected[i] - actual[i]);

                Assert.True(
                    diff <= tolerance,
                    $"Mismatch at {i}: expected={expected[i]}, actual={actual[i]}, diff={diff}, tolerance={tolerance}");
            }
        }

        private static void FillDeterministic(
            float[] data,
            int seedOffset)
        {
            var seed = 0x12345678u + (uint)(seedOffset * 7919);

            for (var i = 0; i < data.Length; i++)
            {
                seed = seed * 1664525u + 1013904223u;

                var normalized = (seed & 0x00FFFFFF) / 16777216f;
                data[i] = normalized * 2f - 1f;
            }
        }
    }
}