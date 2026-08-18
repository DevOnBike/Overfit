// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.Optimizers;
using DevOnBike.Overfit.Tensors;
using DevOnBike.Overfit.Tensors.Core;

namespace DevOnBike.Overfit.Tests.Core.Optimizers
{
    /// <summary>
    /// Several Adam / AdamW steps match a scalar reference, on a parameter long enough to reach the
    /// vectorised loop in <c>StepAdamWRange</c> / <c>StepAdamRange</c>.
    ///
    /// <para><b>Why this exists, and why the FIRST version of it was worthless (`XC-81`).</b> An explicit
    /// 512-bit AdamW loop was added and the whole suite stayed green — then a mutation that TRIPLED the
    /// weight decay inside that loop left it green too. A parity test was written and <b>three mutations
    /// escaped that as well</b>, which was read as "the vector loop is unreachable". <b>That reading was
    /// wrong.</b> A probe — an unconditional throw at the top of the vector branch — reddened the test
    /// immediately, so the loop runs. The defect was in the test, and it was arithmetic, not coverage.</para>
    ///
    /// <para><b>A single step makes this update degenerate, and the degeneracy hides the coefficients.</b>
    /// At <c>t = 1</c> the bias corrections are exactly the factors that were just applied:
    /// <c>mHat = (1-b1)g / (1-b1) = g</c> and <c>vHat = sqrt((1-b2)g^2 / (1-b2)) = |g|</c>. The whole update
    /// collapses to <c>lr * sign(g)</c>, which does not depend on <c>Beta1</c> or <c>Beta2</c> at all — so
    /// swapping them is a mathematical no-op that no tolerance could ever catch. Hence
    /// <c>Steps = 5</c>.</para>
    ///
    /// <para><b>Two more values were sized against the tolerance rather than copied from a tutorial.</b>
    /// With <c>WeightDecay * LearningRate = 1e-5</c>, tripling the decay moves the weight by 2e-5 relative —
    /// <i>under</i> a 1e-4 assertion, so the mutation is invisible by construction. Here the product is 1e-3
    /// per step. Likewise <c>Epsilon = 1e-8</c> against gradients near 0.5 is a 2e-8 perturbation; this uses
    /// 1e-4, which is also the value this project's QLoRA fine-tuning actually needs. <b>A tolerance must be
    /// smaller than the effect of every coefficient the test claims to pin.</b></para>
    ///
    /// <para>The oracle is a scalar re-implementation of the update rule, not a second run of the same code,
    /// so a kernel that is self-consistently wrong cannot agree with it.</para>

    /// <para><b>The two modes are two different update rules, and the test found that out the hard way.</b>
    /// <c>UseAdamW = true</c> decouples the decay and applies it to the WEIGHT after the Adam update.
    /// <c>UseAdamW = false</c> couples it into the GRADIENT before the moments — <c>g' = g + wd*w</c>,
    /// classic L2 — so the decay also feeds the second moment. The first version of this reference decayed
    /// the same way in both modes; the AdamW arm passed and the Adam arm failed by 3e-3, which is 5 steps of
    /// <c>wd * lr</c>. <b>The reference was wrong, not the kernel.</b></para>
    /// </summary>
    public sealed class AdamWVectorParityTests
    {
        // Past the 512-element threshold that gates the vector loops, and a multiple of neither vector width,
        // so the scalar tail runs in the same call. Below the 65_536 parallel threshold, so one range is used.
        private const int Length = 2053;

        // More than one, because at t = 1 the bias correction cancels the coefficient it corrects and the
        // update stops depending on Beta1 / Beta2 entirely. See the class remarks.
        private const int Steps = 5;

        private const float LearningRate = 0.01f;
        private const float Beta1 = 0.9f;
        private const float Beta2 = 0.999f;

        // Sized so it is visible above Tolerance: 1e-4 against gradients near 0.5 perturbs by ~2e-4.
        private const float Epsilon = 1e-4f;

        // Sized so it is visible above Tolerance: WeightDecay * LearningRate = 1e-3 per step.
        private const float WeightDecay = 0.1f;

        private const float Tolerance = 1e-5f;

        [Fact]
        public void AdamW_MatchesAScalarReference_PastTheVectorThreshold()
        {
            RunParity(useAdamW: true);
        }

        [Fact]
        public void Adam_MatchesAScalarReference_PastTheVectorThreshold()
        {
            RunParity(useAdamW: false);
        }

        private static void RunParity(bool useAdamW)
        {
            var weights = Deterministic(Length, seed: 7);

            using var storage = new TensorStorage<float>(Length, clearMemory: false);
            weights.CopyTo(storage.AsSpan());

            using var parameter = new AutogradNode(storage, TensorShape.Vector(Length), requiresGrad: true);

            using var optimizer = new Adam([parameter], LearningRate)
            {
                Beta1 = Beta1,
                Beta2 = Beta2,
                Epsilon = Epsilon,
                WeightDecay = WeightDecay,
                UseAdamW = useAdamW,
            };

            // The scalar reference carries its own moments across steps, exactly as the optimiser does.
            var expected = (float[])weights.Clone();
            var refM = new float[Length];
            var refV = new float[Length];

            for (var step = 1; step <= Steps; step++)
            {
                // A fresh gradient each step, so the moments actually accumulate a history. A repeated
                // gradient would let a wrong Beta still converge to the same fixed point.
                var grads = Deterministic(Length, seed: 13 + step);
                grads.CopyTo(parameter.GradView.AsSpan());

                optimizer.Step();

                var invBc1 = 1f / (1f - MathF.Pow(Beta1, step));
                var invBc2 = 1f / (1f - MathF.Pow(Beta2, step));

                for (var i = 0; i < Length; i++)
                {
                    // Plain Adam couples the decay into the GRADIENT before the moments (classic L2);
                    // AdamW decouples it and applies it to the weight after the update. Two different
                    // update rules, not one rule with a flag — this test failed until the reference said so.
                    var g = useAdamW ? grads[i] : grads[i] + (WeightDecay * expected[i]);

                    refM[i] = (Beta1 * refM[i]) + ((1f - Beta1) * g);
                    refV[i] = (Beta2 * refV[i]) + ((1f - Beta2) * g * g);

                    var mHat = refM[i] * invBc1;
                    var vHat = MathF.Sqrt(refV[i] * invBc2) + Epsilon;

                    expected[i] -= LearningRate * (mHat / vHat);

                    if (useAdamW)
                    {
                        // Decoupled decay, AFTER the Adam update — the ordering this optimiser implements.
                        expected[i] -= expected[i] * WeightDecay * LearningRate;
                    }
                }
            }

            var actual = parameter.DataView.AsSpan();

            for (var i = 0; i < Length; i++)
            {
                var scale = MathF.Max(1e-3f, MathF.Abs(expected[i]));

                Assert.True(
                    MathF.Abs(actual[i] - expected[i]) / scale < Tolerance,
                    $"element {i} of {Length} after {Steps} steps (AdamW={useAdamW}): "
                    + $"kernel {actual[i]}, scalar reference {expected[i]}");
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
