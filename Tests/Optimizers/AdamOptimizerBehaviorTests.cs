// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.Optimizers;
using DevOnBike.Overfit.Tensors;
using DevOnBike.Overfit.Tensors.Core;
using Xunit;

namespace DevOnBike.Overfit.Tests.Optimizers
{
    public sealed class AdamOptimizerBehaviorTests
    {
        [Fact]
        public void Step_AdamW_MatchesManualReference_BitExact_ForThreeSteps()
        {
            const float learningRate = 0.001f;
            const float beta1 = 0.9f;
            const float beta2 = 0.999f;
            const float epsilon = 1e-8f;
            const float weightDecay = 0.0001f;

            var initialWeights = new float[]
            {
                0.25f,
                -0.75f,
                1.5f,
                -2.0f
            };

            var gradients = new float[]
            {
                0.125f,
                -0.25f,
                0.5f,
                -1.0f
            };

            var expectedWeights = new float[initialWeights.Length];
            var expectedM = new float[initialWeights.Length];
            var expectedV = new float[initialWeights.Length];

            initialWeights.AsSpan().CopyTo(expectedWeights);

            using var parameter = CreateParameter(
                initialWeights,
                gradients);

            using var optimizer = new Adam(
                new[]
                {
                    parameter
                },
                learningRate);

            optimizer.Beta1 = beta1;
            optimizer.Beta2 = beta2;
            optimizer.Epsilon = epsilon;
            optimizer.WeightDecay = weightDecay;
            optimizer.UseAdamW = true;

            for (var step = 1; step <= 3; step++)
            {
                ManualAdamStep(
                    expectedWeights,
                    gradients,
                    expectedM,
                    expectedV,
                    step,
                    learningRate,
                    beta1,
                    beta2,
                    epsilon,
                    weightDecay,
                    useAdamW: true);

                optimizer.Step();

                AssertBitExact(
                    expectedWeights,
                    parameter.DataView.AsReadOnlySpan());
            }
        }

        [Fact]
        public void Step_CoupledWeightDecayAdam_MatchesManualReference_BitExact_ForThreeSteps()
        {
            const float learningRate = 0.001f;
            const float beta1 = 0.9f;
            const float beta2 = 0.999f;
            const float epsilon = 1e-8f;
            const float weightDecay = 0.0001f;

            var initialWeights = new float[]
            {
                0.25f,
                -0.75f,
                1.5f,
                -2.0f
            };

            var gradients = new float[]
            {
                0.125f,
                -0.25f,
                0.5f,
                -1.0f
            };

            var expectedWeights = new float[initialWeights.Length];
            var expectedM = new float[initialWeights.Length];
            var expectedV = new float[initialWeights.Length];

            initialWeights.AsSpan().CopyTo(expectedWeights);

            using var parameter = CreateParameter(
                initialWeights,
                gradients);

            using var optimizer = new Adam(
                new[]
                {
                    parameter
                },
                learningRate);

            optimizer.Beta1 = beta1;
            optimizer.Beta2 = beta2;
            optimizer.Epsilon = epsilon;
            optimizer.WeightDecay = weightDecay;
            optimizer.UseAdamW = false;

            for (var step = 1; step <= 3; step++)
            {
                ManualAdamStep(
                    expectedWeights,
                    gradients,
                    expectedM,
                    expectedV,
                    step,
                    learningRate,
                    beta1,
                    beta2,
                    epsilon,
                    weightDecay,
                    useAdamW: false);

                optimizer.Step();

                AssertBitExact(
                    expectedWeights,
                    parameter.DataView.AsReadOnlySpan());
            }
        }

        [Fact]
        public void Step_IgnoresParametersThatDoNotRequireGrad()
        {
            using var trainable = CreateParameter(
                data:
                [
                    1f,
                    2f,
                    3f
                ],
                grad:
                [
                    0.1f,
                    0.2f,
                    0.3f
                ]);

            using var frozen = CreateNodeWithoutGrad(
                data:
                [
                    10f,
                    20f,
                    30f
                ]);

            using var optimizer = new Adam(
                new[]
                {
                    trainable,
                    frozen
                },
                learningRate: 0.001f);

            optimizer.Step();

            var frozenData = frozen.DataView.AsReadOnlySpan();

            Assert.Equal(10f, frozenData[0]);
            Assert.Equal(20f, frozenData[1]);
            Assert.Equal(30f, frozenData[2]);
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

            using var optimizer = new Adam(
                new[]
                {
                    parameter
                },
                learningRate: 0.001f);

            optimizer.ZeroGrad();

            var grad = parameter.GradView.AsReadOnlySpan();

            Assert.Equal(0f, grad[0]);
            Assert.Equal(0f, grad[1]);
            Assert.Equal(0f, grad[2]);
        }

        [Fact]
        public void ResetTime_RestartsBiasCorrectionBehavior()
        {
            var initialWeights = new float[]
            {
                0.5f,
                -1.25f,
                2.5f,
                -3.75f
            };

            var gradients = new float[]
            {
                0.05f,
                -0.1f,
                0.15f,
                -0.2f
            };

            using var first = CreateParameter(
                initialWeights,
                gradients);

            using var second = CreateParameter(
                initialWeights,
                gradients);

            using var optimizerA = CreateConfiguredAdam(first);
            using var optimizerB = CreateConfiguredAdam(second);

            optimizerA.Step();
            optimizerA.ResetTime();
            optimizerA.Step();

            optimizerB.Step();
            optimizerB.ResetTime();
            optimizerB.Step();

            AssertBitExact(
                first.DataView.AsReadOnlySpan(),
                second.DataView.AsReadOnlySpan());
        }

        [Fact]
        public void Step_CurrentImplementation_AllocationBaseline_IsStable()
        {
            const int length = 128;
            const int iterations = 10_000;

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

            // Current Adam baseline observed on .NET 10:
            // ~96 B per Step.
            //
            // This test documents the current baseline before refactoring Adam.
            // Do not tighten this to 0 until Adam.Step() is intentionally refactored.
            Assert.True(
                allocatedPerStep <= 128,
                $"Adam.Step allocation baseline regressed. Allocated {allocated} B total, {allocatedPerStep} B/step.");
        }

        [Fact(Skip = "Current Adam.Step allocates about 96 B per Step. Enable this after Adam is explicitly refactored to a zero-allocation path.")]
        public void Step_AllocatesZeroBytesAfterConstructionAndWarmup_Target()
        {
            const int length = 128;
            const int iterations = 10_000;

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

            Assert.Equal(
                0,
                allocated);
        }

        [Fact(Skip = "Current Adam vector path allocates ~96 B per Step. Keep as a pending optimization guard.")]
        public void Step_VectorPath_AllocatesZeroBytesAfterConstructionAndWarmup_Pending()
        {
            // Keep above Adam.VectorThreshold=512 in the current implementation.
            // This documents the current known allocation in the vector path.
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

            for (var i = 0; i < 1_000; i++)
            {
                optimizer.Step();
            }

            var allocated = GC.GetAllocatedBytesForCurrentThread() - before;

            Assert.Equal(
                0,
                allocated);
        }

        private static Adam CreateConfiguredAdam(
            AutogradNode parameter)
        {
            return new Adam(
                new[]
                {
                    parameter
                },
                learningRate: 0.001f)
            {
                Beta1 = 0.9f,
                Beta2 = 0.999f,
                Epsilon = 1e-8f,
                WeightDecay = 0.0001f,
                UseAdamW = true
            };
        }

        private static void ManualAdamStep(
            Span<float> weights,
            ReadOnlySpan<float> gradients,
            Span<float> m,
            Span<float> v,
            int step,
            float learningRate,
            float beta1,
            float beta2,
            float epsilon,
            float weightDecay,
            bool useAdamW)
        {
            var bc1 = 1f - MathF.Pow(beta1, step);
            var bc2 = 1f - MathF.Pow(beta2, step);

            var invBc1 = 1f / bc1;
            var invBc2 = 1f / bc2;

            var beta1Inv = 1f - beta1;
            var beta2Inv = 1f - beta2;

            for (var i = 0; i < weights.Length; i++)
            {
                var gw = gradients[i];
                var ww = weights[i];
                var mw = m[i];
                var vw = v[i];

                if (useAdamW)
                {
                    mw = beta1 * mw + beta1Inv * gw;
                    vw = beta2 * vw + beta2Inv * (gw * gw);

                    var mHat = mw * invBc1;
                    var vHat = MathF.Sqrt(vw * invBc2) + epsilon;

                    ww -= learningRate * (mHat / vHat);

                    if (weightDecay > 0f)
                    {
                        ww -= ww * weightDecay * learningRate;
                    }
                }
                else
                {
                    var gl2 = gw + weightDecay * ww;

                    mw = beta1 * mw + beta1Inv * gl2;
                    vw = beta2 * vw + beta2Inv * (gl2 * gl2);

                    var mHat = mw * invBc1;
                    var vHat = MathF.Sqrt(vw * invBc2) + epsilon;

                    ww -= learningRate * (mHat / vHat);
                }

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

        private static void AssertBitExact(
            ReadOnlySpan<float> expected,
            ReadOnlySpan<float> actual)
        {
            Assert.Equal(
                expected.Length,
                actual.Length);

            for (var i = 0; i < expected.Length; i++)
            {
                var expectedBits = BitConverter.SingleToUInt32Bits(expected[i]);
                var actualBits = BitConverter.SingleToUInt32Bits(actual[i]);

                Assert.True(
                    expectedBits == actualBits,
                    $"Mismatch at {i}: expected={expected[i]} bits=0x{expectedBits:X8}, actual={actual[i]} bits=0x{actualBits:X8}");
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