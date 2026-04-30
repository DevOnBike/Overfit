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
    public sealed class AdamAllocationProbeTests
    {
        [Fact]
        public void ManualAdamSmallStep_AllocatesZeroBytes()
        {
            const int length = 128;
            const int iterations = 10_000;

            var weights = new float[length];
            var grad = new float[length];
            var m = new float[length];
            var v = new float[length];

            FillDeterministic(weights, seedOffset: 1);
            FillDeterministic(grad, seedOffset: 2);

            var t = 1;

            ManualAdamWStep(
                weights,
                grad,
                m,
                v,
                t,
                learningRate: 0.001f,
                beta1: 0.9f,
                beta2: 0.999f,
                epsilon: 1e-8f,
                weightDecay: 0.0001f);

            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();

            var before = GC.GetAllocatedBytesForCurrentThread();

            for (var i = 0; i < iterations; i++)
            {
                t++;

                ManualAdamWStep(
                    weights,
                    grad,
                    m,
                    v,
                    t,
                    learningRate: 0.001f,
                    beta1: 0.9f,
                    beta2: 0.999f,
                    epsilon: 1e-8f,
                    weightDecay: 0.0001f);
            }

            var allocated = GC.GetAllocatedBytesForCurrentThread() - before;

            Assert.Equal(0, allocated);
        }

        [Fact]
        public void AdamStep_CurrentImplementation_StillAllocatesBaseline()
        {
            const int length = 128;
            const int iterations = 10_000;

            var data = new float[length];
            var grad = new float[length];

            FillDeterministic(data, seedOffset: 1);
            FillDeterministic(grad, seedOffset: 2);

            using var parameter = CreateParameter(data, grad);

            using var optimizer = new Adam(
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

            for (var i = 0; i < iterations; i++)
            {
                optimizer.Step();
            }

            var allocated = GC.GetAllocatedBytesForCurrentThread() - before;
            var allocatedPerStep = allocated / iterations;

            Assert.True(
                allocatedPerStep <= 128,
                $"Adam.Step allocation baseline changed. Total={allocated} B, per step={allocatedPerStep} B.");
        }

        [Fact]
        public void AdamStep_WithSinglePreallocatedParameterArray_StillAllocatesBaseline()
        {
            const int length = 128;
            const int iterations = 10_000;

            var data = new float[length];
            var grad = new float[length];

            FillDeterministic(data, seedOffset: 1);
            FillDeterministic(grad, seedOffset: 2);

            using var parameter = CreateParameter(data, grad);

            var parameters = new[]
            {
                parameter
            };

            using var optimizer = new Adam(
                parameters,
                learningRate: 0.001f);

            optimizer.Step();

            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();

            var before = GC.GetAllocatedBytesForCurrentThread();

            for (var i = 0; i < iterations; i++)
            {
                optimizer.Step();
            }

            var allocated = GC.GetAllocatedBytesForCurrentThread() - before;
            var allocatedPerStep = allocated / iterations;

            Assert.True(
                allocatedPerStep <= 128,
                $"Adam.Step allocation baseline changed. Total={allocated} B, per step={allocatedPerStep} B.");
        }

        private static void ManualAdamWStep(
            Span<float> weights,
            ReadOnlySpan<float> grad,
            Span<float> m,
            Span<float> v,
            int step,
            float learningRate,
            float beta1,
            float beta2,
            float epsilon,
            float weightDecay)
        {
            var bc1 = 1f - MathF.Pow(beta1, step);
            var bc2 = 1f - MathF.Pow(beta2, step);

            var invBc1 = 1f / bc1;
            var invBc2 = 1f / bc2;

            var beta1Inv = 1f - beta1;
            var beta2Inv = 1f - beta2;

            for (var i = 0; i < weights.Length; i++)
            {
                var gw = grad[i];
                var mw = m[i];
                var vw = v[i];
                var ww = weights[i];

                mw = beta1 * mw + beta1Inv * gw;
                vw = beta2 * vw + beta2Inv * (gw * gw);

                var mHat = mw * invBc1;
                var vHat = MathF.Sqrt(vw * invBc2) + epsilon;

                ww -= learningRate * (mHat / vHat);
                ww -= ww * weightDecay * learningRate;

                m[i] = mw;
                v[i] = vw;
                weights[i] = ww;
            }
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

            data.CopyTo(storage.AsSpan());

            var node = new AutogradNode(
                storage,
                TensorShape.Vector(data.Length),
                requiresGrad: true);

            grad.CopyTo(node.GradView.AsSpan());

            return node;
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