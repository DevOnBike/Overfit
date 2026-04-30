// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.Tensors;
using DevOnBike.Overfit.Tensors.Core;

namespace DevOnBike.Overfit.Tests.Optimizers
{
    public sealed class AdamMutableSpanProbeTests
    {
        [Fact]
        public void DataAndGradMutableSpanAccess_AllocatesZeroBytes()
        {
            const int length = 128;
            const int iterations = 10_000;

            var data = new float[length];
            var grad = new float[length];

            FillDeterministic(data, seedOffset: 1);
            FillDeterministic(grad, seedOffset: 2);

            using var parameter = CreateParameter(data, grad);

            // warmup
            _ = parameter.DataView.AsSpan()[0];
            _ = parameter.GradView.AsSpan()[0];

            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();

            var before = GC.GetAllocatedBytesForCurrentThread();

            var checksum = 0f;

            for (var i = 0; i < iterations; i++)
            {
                var dataSpan = parameter.DataView.AsSpan();
                var gradSpan = parameter.GradView.AsSpan();

                checksum += dataSpan[0] + gradSpan[0];

                dataSpan[0] += 0.000001f;
                gradSpan[0] += 0.000001f;
            }

            var allocated = GC.GetAllocatedBytesForCurrentThread() - before;

            Assert.Equal(0, allocated);
            Assert.False(float.IsNaN(checksum));
        }

        [Fact]
        public void DataAndGradMutableSpanSlice_AllocatesZeroBytes()
        {
            const int length = 128;
            const int iterations = 10_000;

            var data = new float[length];
            var grad = new float[length];

            FillDeterministic(data, seedOffset: 1);
            FillDeterministic(grad, seedOffset: 2);

            using var parameter = CreateParameter(data, grad);

            // warmup
            _ = parameter.DataView.AsSpan().Slice(0, length)[0];
            _ = parameter.GradView.AsSpan().Slice(0, length)[0];

            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();

            var before = GC.GetAllocatedBytesForCurrentThread();

            var checksum = 0f;

            for (var i = 0; i < iterations; i++)
            {
                var dataSpan = parameter.DataView.AsSpan().Slice(0, length);
                var gradSpan = parameter.GradView.AsSpan().Slice(0, length);

                checksum += dataSpan[0] + gradSpan[0];

                dataSpan[0] += 0.000001f;
                gradSpan[0] += 0.000001f;
            }

            var allocated = GC.GetAllocatedBytesForCurrentThread() - before;

            Assert.Equal(0, allocated);
            Assert.False(float.IsNaN(checksum));
        }

        [Fact]
        public void ManualAdamUsingAutogradNodeViews_AllocatesZeroBytes()
        {
            const int length = 128;
            const int iterations = 10_000;

            var data = new float[length];
            var grad = new float[length];

            FillDeterministic(data, seedOffset: 1);
            FillDeterministic(grad, seedOffset: 2);

            using var parameter = CreateParameter(data, grad);

            var m = new float[length];
            var v = new float[length];

            var step = 1;

            ManualAdamWStepUsingNodeViews(
                parameter,
                m,
                v,
                step);

            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();

            var before = GC.GetAllocatedBytesForCurrentThread();

            for (var i = 0; i < iterations; i++)
            {
                step++;

                ManualAdamWStepUsingNodeViews(
                    parameter,
                    m,
                    v,
                    step);
            }

            var allocated = GC.GetAllocatedBytesForCurrentThread() - before;

            Assert.Equal(0, allocated);
        }

        private static void ManualAdamWStepUsingNodeViews(
            AutogradNode parameter,
            Span<float> m,
            Span<float> v,
            int step)
        {
            const float learningRate = 0.001f;
            const float beta1 = 0.9f;
            const float beta2 = 0.999f;
            const float epsilon = 1e-8f;
            const float weightDecay = 0.0001f;

            var bc1 = 1f - MathF.Pow(beta1, step);
            var bc2 = 1f - MathF.Pow(beta2, step);

            var invBc1 = 1f / bc1;
            var invBc2 = 1f / bc2;

            var beta1Inv = 1f - beta1;
            var beta2Inv = 1f - beta2;

            var gSpan = parameter.GradView.AsSpan();
            var wSpan = parameter.DataView.AsSpan();

            for (var i = 0; i < wSpan.Length; i++)
            {
                var gw = gSpan[i];
                var mw = m[i];
                var vw = v[i];
                var ww = wSpan[i];

                mw = beta1 * mw + beta1Inv * gw;
                vw = beta2 * vw + beta2Inv * (gw * gw);

                var mHat = mw * invBc1;
                var vHat = MathF.Sqrt(vw * invBc2) + epsilon;

                ww -= learningRate * (mHat / vHat);
                ww -= ww * weightDecay * learningRate;

                m[i] = mw;
                v[i] = vw;
                wSpan[i] = ww;
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