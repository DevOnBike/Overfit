// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.Optimizers;
using DevOnBike.Overfit.Tensors;
using DevOnBike.Overfit.Tensors.Core;

namespace DevOnBike.Overfit.Tests.Optimizers
{
    public sealed class SgdOptimizerBehaviorTests
    {
        [Fact]
        public void Step_UpdatesWeightsUsingGradientAndLearningRate()
        {
            using var parameter = CreateParameter(
                data:
                [
                    10f,
                    -20f,
                    30f,
                    -40f
                ],
                grad:
                [
                    1f,
                    -2f,
                    3f,
                    -4f
                ]);

            var optimizer = new SGD(
                new[]
                {
                    parameter
                },
                learningRate: 0.1f);

            optimizer.Step();

            var actual = parameter.DataView.AsReadOnlySpan();

            Assert.Equal(9.9f, actual[0], precision: 6);
            Assert.Equal(-19.8f, actual[1], precision: 6);
            Assert.Equal(29.7f, actual[2], precision: 6);
            Assert.Equal(-39.6f, actual[3], precision: 6);
        }

        [Fact]
        public void Step_IsEquivalentToManualSgd_ForLargeParameter()
        {
            const int length = 4096;
            const float learningRate = 0.03125f;

            var data = new float[length];
            var grad = new float[length];
            var expected = new float[length];

            FillDeterministic(
                data,
                seedOffset: 1);

            FillDeterministic(
                grad,
                seedOffset: 2);

            data.AsSpan().CopyTo(expected);

            for (var i = 0; i < expected.Length; i++)
            {
                expected[i] -= learningRate * grad[i];
            }

            using var parameter = CreateParameter(
                data,
                grad);

            var optimizer = new SGD(
                new[]
                {
                    parameter
                },
                learningRate);

            optimizer.Step();

            AssertClose(
                expected,
                parameter.DataView.AsReadOnlySpan(),
                tolerance: 0f);
        }

        [Fact]
        public void Step_IgnoresParametersThatDoNotRequireGrad()
        {
            using var trainable = CreateParameter(
                data:
                [
                    10f,
                    20f
                ],
                grad:
                [
                    1f,
                    2f
                ]);

            using var frozen = CreateNodeWithoutGrad(
                data:
                [
                    100f,
                    200f
                ]);

            var optimizer = new SGD(
                new[]
                {
                    trainable,
                    frozen
                },
                learningRate: 0.5f);

            optimizer.Step();

            var trainableData = trainable.DataView.AsReadOnlySpan();
            var frozenData = frozen.DataView.AsReadOnlySpan();

            Assert.Equal(9.5f, trainableData[0], precision: 6);
            Assert.Equal(19f, trainableData[1], precision: 6);

            Assert.Equal(100f, frozenData[0], precision: 6);
            Assert.Equal(200f, frozenData[1], precision: 6);
        }

        [Fact]
        public void ZeroGrad_ClearsTrackedParameterGradients()
        {
            using var parameter = CreateParameter(
                data:
                [
                    1f,
                    2f,
                    3f
                ],
                grad:
                [
                    10f,
                    20f,
                    30f
                ]);

            var optimizer = new SGD(
                new[]
                {
                    parameter
                },
                learningRate: 0.1f);

            optimizer.ZeroGrad();

            var grad = parameter.GradView.AsReadOnlySpan();

            Assert.Equal(0f, grad[0]);
            Assert.Equal(0f, grad[1]);
            Assert.Equal(0f, grad[2]);
        }

        [Fact]
        public void Step_AllocatesZeroBytesAfterConstructionAndWarmup()
        {
            const int length = 4096;

            var data = new float[length];
            var grad = new float[length];

            FillDeterministic(
                data,
                seedOffset: 1);

            FillDeterministic(
                grad,
                seedOffset: 2);

            using var parameter = CreateParameter(
                data,
                grad);

            var optimizer = new SGD(
                new[]
                {
                    parameter
                },
                learningRate: 0.001f);

            optimizer.Step();

            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();

            var before = GC.GetAllocatedBytesForCurrentThread();

            for (var i = 0; i < 10_000; i++)
            {
                optimizer.Step();
            }

            var allocated = GC.GetAllocatedBytesForCurrentThread() - before;

            Assert.Equal(
                0,
                allocated);
        }

        private static AutogradNode CreateParameter(
            ReadOnlySpan<float> data,
            ReadOnlySpan<float> grad)
        {
            if (data.Length != grad.Length)
            {
                throw new ArgumentException(
                    "Data and grad lengths must match.");
            }

            var storage = new TensorStorage<float>(
                data.Length,
                clearMemory: false);

            data.CopyTo(
                storage.AsSpan());

            var node = new AutogradNode(
                storage,
                TensorShape.Vector(data.Length),
                requiresGrad: true);

            grad.CopyTo(
                node.GradView.AsSpan());

            return node;
        }

        private static AutogradNode CreateNodeWithoutGrad(
            ReadOnlySpan<float> data)
        {
            var storage = new TensorStorage<float>(
                data.Length,
                clearMemory: false);

            data.CopyTo(
                storage.AsSpan());

            return new AutogradNode(
                storage,
                TensorShape.Vector(data.Length),
                requiresGrad: false);
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