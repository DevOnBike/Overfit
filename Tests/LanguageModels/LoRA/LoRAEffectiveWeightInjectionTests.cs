// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.Optimizers;
using DevOnBike.Overfit.Parameters;
using DevOnBike.Overfit.Tensors;
using DevOnBike.Overfit.Tensors.Core;

namespace DevOnBike.Overfit.Tests.LanguageModels.LoRA
{
    /// <summary>
    /// Risk PoC for the GPT1 LoRA fine-tune design ("effective-weight injection").
    ///
    /// The design feeds each targeted projection a COMPUTED weight node
    ///   W_eff = W_base (frozen) + (A @ B)
    /// where A and B are trainable LoRA factors. For LoRA training to work,
    /// graph.Linear must backpropagate through a weight node that is NOT a leaf
    /// Parameter — the gradient has to flow W_eff -> (A@B) -> A and B.
    ///
    /// This test proves exactly that:
    ///   1. gradient reaches both A and B through the computed W_eff node,
    ///   2. it matches a numeric finite-difference gradient (mechanism is correct,
    ///      not merely non-zero),
    ///   3. an optimizer step over {A, B} leaves the frozen base bit-identical.
    ///
    /// (Production LoRA inits B = 0 so only B moves on step 1; here both factors
    /// are seeded non-zero so the test exercises gradient flow into both.)
    /// </summary>
    public sealed class LoRAEffectiveWeightInjectionTests
    {
        private const int InDim = 4;
        private const int Rank = 2;
        private const int OutDim = 3;
        private const int Batch = 2;

        [Fact]
        public void EffectiveWeightInjection_GradientReachesLoRAFactors_AndBaseStaysFrozen()
        {
            using var graph = new ComputationGraph { IsRecording = true };

            // LoRA factors — trainable.
            using var a = new Parameter(new TensorShape(InDim, Rank), requiresGrad: true, clearData: true);
            using var b = new Parameter(new TensorShape(Rank, OutDim), requiresGrad: true, clearData: true);
            Fill(a.DataSpan, seed: 1);
            Fill(b.DataSpan, seed: 2);

            // Frozen base weight + frozen inputs/biases (requiresGrad: false).
            using var wBase = Frozen(new TensorShape(InDim, OutDim), seed: 3);
            using var input = Frozen(new TensorShape(Batch, InDim), seed: 4);
            using var abBias = Frozen(new TensorShape(OutDim), seed: 0);
            using var outBias = Frozen(new TensorShape(OutDim), seed: 0);

            var aNode = a.AsNode();
            var bNode = b.AsNode();

            // Forward: W_eff = W_base + (A @ B); output = input @ W_eff + outBias.
            AutogradNode BuildForward()
            {
                var ab = graph.Linear(aNode, bNode, abBias);   // [InDim, OutDim]
                var wEff = graph.Add(wBase, ab);                // [InDim, OutDim] — computed weight
                return graph.Linear(input, wEff, outBias);      // [Batch, OutDim]
            }

            float ForwardSum()
            {
                graph.Reset();
                var outNode = BuildForward();
                var data = outNode.DataView.AsReadOnlySpan();
                var sum = 0f;
                for (var i = 0; i < data.Length; i++)
                {
                    sum += data[i];
                }
                return sum;
            }

            // --- Analytic backward through the computed weight node ---
            graph.Reset();
            a.ZeroGrad();
            b.ZeroGrad();

            var output = BuildForward();
            output.GradView.AsSpan().Fill(1f);   // d(sum of outputs)/d(output) = 1
            graph.BackwardFromGrad(output);

            var gradA = a.GradSpan.ToArray();
            var gradB = b.GradSpan.ToArray();

            // (1) Gradient actually reached both LoRA factors.
            Assert.True(SumAbs(gradA) > 1e-6f, "No gradient reached A through the computed W_eff node.");
            Assert.True(SumAbs(gradB) > 1e-6f, "No gradient reached B through the computed W_eff node.");

            // (2) Analytic gradient matches numeric finite differences.
            const float eps = 1e-3f;
            VerifyNumeric(a.DataSpan, gradA, ForwardSum, eps);
            VerifyNumeric(b.DataSpan, gradB, ForwardSum, eps);

            // (3) An optimizer step over {A, B} only must leave the frozen base untouched.
            var baseBefore = wBase.DataView.AsReadOnlySpan().ToArray();
            var aBefore = a.DataReadOnlySpan.ToArray();

            using (var optimizer = new Adam(new[] { a, b }, learningRate: 0.05f))
            {
                optimizer.Step();
            }

            var baseAfter = wBase.DataView.AsReadOnlySpan().ToArray();
            Assert.Equal(baseBefore, baseAfter);                       // base frozen — bit-identical
            Assert.NotEqual<float[]>(aBefore, a.DataReadOnlySpan.ToArray()); // A actually updated
        }

        private static void VerifyNumeric(
            Span<float> data,
            float[] analyticGrad,
            Func<float> forwardSum,
            float eps)
        {
            for (var i = 0; i < analyticGrad.Length; i++)
            {
                var original = data[i];

                data[i] = original + eps;
                var plus = forwardSum();

                data[i] = original - eps;
                var minus = forwardSum();

                data[i] = original;

                var numeric = (plus - minus) / (2f * eps);
                var analytic = analyticGrad[i];

                var absErr = MathF.Abs(numeric - analytic);
                var relErr = absErr / MathF.Max(1e-4f, MathF.Abs(numeric) + MathF.Abs(analytic));

                Assert.True(
                    absErr < 5e-3f || relErr < 3e-2f,
                    $"Gradient mismatch at index {i}: analytic={analytic:E4}, numeric={numeric:E4}, " +
                    $"absErr={absErr:E4}, relErr={relErr:E4}.");
            }
        }

        private static AutogradNode Frozen(TensorShape shape, int seed)
        {
            var storage = new TensorStorage<float>(shape.Size, clearMemory: true);
            if (seed != 0)
            {
                Fill(storage.AsSpan(), seed);
            }
            return new AutogradNode(storage, shape, requiresGrad: false);
        }

        private static void Fill(Span<float> span, int seed)
        {
            // Deterministic, non-trivial, mixed-sign values.
            for (var i = 0; i < span.Length; i++)
            {
                span[i] = 0.1f * ((((i + seed) * 7) % 11) - 5);
            }
        }

        private static float SumAbs(float[] values)
        {
            var sum = 0f;
            foreach (var v in values)
            {
                sum += MathF.Abs(v);
            }
            return sum;
        }
    }
}
