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
    public sealed class AdamStructuralAllocationProbeTests
    {
        [Fact]
        public void MinimalAdamLikeClass_AllocatesZeroBytes()
        {
            const int length = 128;
            const int iterations = 10_000;

            var data = new float[length];
            var grad = new float[length];

            FillDeterministic(data, seedOffset: 1);
            FillDeterministic(grad, seedOffset: 2);

            using var parameter = CreateParameter(data, grad);
            using var optimizer = new MinimalAdamLike(parameter);

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

            Assert.Equal(0, allocated);
        }

        [Fact]
        public void RealAdam_StillHasKnownAllocationBaseline()
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
                $"Adam allocation baseline changed. Total={allocated} B, per step={allocatedPerStep} B.");
        }

        private sealed class MinimalAdamLike : IDisposable
        {
            private readonly State[] _states;
            private int _t;

            public MinimalAdamLike(
                AutogradNode parameter)
            {
                _states =
                [
                    new State(parameter)
                ];
            }

            public void Step()
            {
                _t++;

                const float beta1 = 0.9f;
                const float beta2 = 0.999f;
                const float epsilon = 1e-8f;
                const float learningRate = 0.001f;
                const float weightDecay = 0.0001f;

                var bc1 = 1f - MathF.Pow(beta1, _t);
                var bc2 = 1f - MathF.Pow(beta2, _t);

                var invBc1 = 1f / bc1;
                var invBc2 = 1f / bc2;

                var beta1Inv = 1f - beta1;
                var beta2Inv = 1f - beta2;

                for (var s = 0; s < _states.Length; s++)
                {
                    var state = _states[s];

                    if (!state.Node.RequiresGrad)
                    {
                        continue;
                    }

                    StepState(
                        state,
                        beta1,
                        beta2,
                        beta1Inv,
                        beta2Inv,
                        invBc1,
                        invBc2,
                        epsilon,
                        learningRate,
                        weightDecay);
                }
            }

            public void Dispose()
            {
                for (var i = 0; i < _states.Length; i++)
                {
                    _states[i].M.Dispose();
                    _states[i].V.Dispose();
                }
            }

            private static void StepState(
                State state,
                float beta1,
                float beta2,
                float beta1Inv,
                float beta2Inv,
                float invBc1,
                float invBc2,
                float epsilon,
                float learningRate,
                float weightDecay)
            {
                var grad = state.Node.GradView.AsSpan();
                var weights = state.Node.DataView.AsSpan();
                var m = state.M.AsSpan();
                var v = state.V.AsSpan();

                for (var i = 0; i < state.Size; i++)
                {
                    var gw = grad[i];
                    var mw = m[i];
                    var vw = v[i];
                    var ww = weights[i];

                    mw = (beta1 * mw) + (beta1Inv * gw);
                    vw = (beta2 * vw) + (beta2Inv * (gw * gw));

                    var mHat = mw * invBc1;
                    var vHat = MathF.Sqrt(vw * invBc2) + epsilon;

                    ww -= learningRate * (mHat / vHat);
                    ww -= ww * weightDecay * learningRate;

                    m[i] = mw;
                    v[i] = vw;
                    weights[i] = ww;
                }
            }

            private readonly struct State
            {
                public readonly AutogradNode Node;
                public readonly FastTensor<float> M;
                public readonly FastTensor<float> V;
                public readonly int Size;

                public State(
                    AutogradNode node)
                {
                    Node = node;
                    Size = node.DataView.Size;
                    M = new FastTensor<float>(Size, clearMemory: true);
                    V = new FastTensor<float>(Size, clearMemory: true);
                }
            }
        }

        private static AutogradNode CreateParameter(
            ReadOnlySpan<float> data,
            ReadOnlySpan<float> grad)
        {
            if (data.Length != grad.Length)
            {
                throw new ArgumentException("Data and grad lengths must match.");
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