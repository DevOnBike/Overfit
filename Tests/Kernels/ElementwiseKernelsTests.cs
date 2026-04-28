// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Kernels;

namespace DevOnBike.Overfit.Tests.Kernels
{
    public class ElementwiseKernelsTests
    {
        [Fact]
        public void Add_MatchesReference()
        {
            var left = new float[128];
            var right = new float[128];
            var actual = new float[128];

            FillDeterministic(left, 1);
            FillDeterministic(right, 2);

            ElementwiseKernels.Add(
                left,
                right,
                actual);

            for (var i = 0; i < actual.Length; i++)
            {
                Assert.Equal(
                    left[i] + right[i],
                    actual[i],
                    precision: 6);
            }
        }

        [Fact]
        public void Subtract_MatchesReference()
        {
            var left = new float[128];
            var right = new float[128];
            var actual = new float[128];

            FillDeterministic(left, 1);
            FillDeterministic(right, 2);

            ElementwiseKernels.Subtract(
                left,
                right,
                actual);

            for (var i = 0; i < actual.Length; i++)
            {
                Assert.Equal(
                    left[i] - right[i],
                    actual[i],
                    precision: 6);
            }
        }

        [Fact]
        public void Multiply_MatchesReference()
        {
            var left = new float[128];
            var right = new float[128];
            var actual = new float[128];

            FillDeterministic(left, 1);
            FillDeterministic(right, 2);

            ElementwiseKernels.Multiply(
                left,
                right,
                actual);

            for (var i = 0; i < actual.Length; i++)
            {
                Assert.Equal(
                    left[i] * right[i],
                    actual[i],
                    precision: 6);
            }
        }

        [Fact]
        public void MultiplyScalar_MatchesReference()
        {
            var input = new float[128];
            var actual = new float[128];
            const float scalar = -0.25f;

            FillDeterministic(input, 1);

            ElementwiseKernels.Multiply(
                input,
                scalar,
                actual);

            for (var i = 0; i < actual.Length; i++)
            {
                Assert.Equal(
                    input[i] * scalar,
                    actual[i],
                    precision: 6);
            }
        }

        [Fact]
        public void MultiplyAdd_MatchesReference()
        {
            var input = new float[128];
            var addend = new float[128];
            var actual = new float[128];
            const float scalar = 0.125f;

            FillDeterministic(input, 1);
            FillDeterministic(addend, 2);

            ElementwiseKernels.MultiplyAdd(
                input,
                scalar,
                addend,
                actual);

            for (var i = 0; i < actual.Length; i++)
            {
                Assert.Equal(
                    (input[i] * scalar) + addend[i],
                    actual[i],
                    precision: 6);
            }
        }

        [Fact]
        public void ReLU_MatchesReference()
        {
            var input = new float[]
            {
                -3f, -0.5f, 0f, 0.25f, 4f
            };

            var actual = new float[input.Length];

            ElementwiseKernels.ReLU(
                input,
                actual);

            Assert.Equal(0f, actual[0]);
            Assert.Equal(0f, actual[1]);
            Assert.Equal(0f, actual[2]);
            Assert.Equal(0.25f, actual[3]);
            Assert.Equal(4f, actual[4]);
        }

        [Fact]
        public void ReLUBackward_AccumulatesGradient()
        {
            var input = new float[]
            {
                -1f, 0f, 2f, 3f
            };

            var outputGradient = new float[]
            {
                10f, 20f, 30f, 40f
            };

            var inputGradient = new float[]
            {
                1f, 1f, 1f, 1f
            };

            ElementwiseKernels.ReLUBackward(
                input,
                outputGradient,
                inputGradient);

            Assert.Equal(1f, inputGradient[0]);
            Assert.Equal(1f, inputGradient[1]);
            Assert.Equal(31f, inputGradient[2]);
            Assert.Equal(41f, inputGradient[3]);
        }

        [Fact]
        public void Dot_MatchesReference()
        {
            var left = new float[257];
            var right = new float[257];

            FillDeterministic(left, 1);
            FillDeterministic(right, 2);

            var expected = 0f;

            for (var i = 0; i < left.Length; i++)
            {
                expected += left[i] * right[i];
            }

            var actual = ElementwiseKernels.Dot(
                left,
                right);

            Assert.Equal(
                expected,
                actual,
                precision: 3);
        }

        [Fact]
        public void SumOfSquares_MatchesReference()
        {
            var input = new float[257];

            FillDeterministic(input, 1);

            var expected = 0f;

            for (var i = 0; i < input.Length; i++)
            {
                expected += input[i] * input[i];
            }

            var actual = ElementwiseKernels.SumOfSquares(input);

            Assert.Equal(
                expected,
                actual,
                precision: 3);
        }

        [Fact]
        public void MeanSquaredError_MatchesReference()
        {
            var prediction = new float[64];
            var target = new float[64];

            FillDeterministic(prediction, 1);
            FillDeterministic(target, 2);

            var expected = 0f;

            for (var i = 0; i < prediction.Length; i++)
            {
                var diff = prediction[i] - target[i];
                expected += diff * diff;
            }

            expected /= prediction.Length;

            var actual = ElementwiseKernels.MeanSquaredError(
                prediction,
                target);

            Assert.Equal(
                expected,
                actual,
                precision: 5);
        }

        [Fact]
        public void Add_AllocatesZeroBytes()
        {
            var left = new float[1024];
            var right = new float[1024];
            var destination = new float[1024];

            FillDeterministic(left, 1);
            FillDeterministic(right, 2);

            ElementwiseKernels.Add(
                left,
                right,
                destination);

            var before = GC.GetAllocatedBytesForCurrentThread();

            for (var i = 0; i < 1024; i++)
            {
                ElementwiseKernels.Add(
                    left,
                    right,
                    destination);
            }

            var after = GC.GetAllocatedBytesForCurrentThread();

            Assert.Equal(
                0,
                after - before);
        }

        [Fact]
        public void MultiplyAdd_AllocatesZeroBytes()
        {
            var input = new float[1024];
            var addend = new float[1024];
            var destination = new float[1024];

            FillDeterministic(input, 1);
            FillDeterministic(addend, 2);

            ElementwiseKernels.MultiplyAdd(
                input,
                0.5f,
                addend,
                destination);

            var before = GC.GetAllocatedBytesForCurrentThread();

            for (var i = 0; i < 1024; i++)
            {
                ElementwiseKernels.MultiplyAdd(
                    input,
                    0.5f,
                    addend,
                    destination);
            }

            var after = GC.GetAllocatedBytesForCurrentThread();

            Assert.Equal(
                0,
                after - before);
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
