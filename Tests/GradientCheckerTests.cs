using DevOnBike.Overfit.Core;
using DevOnBike.Overfit.DeepLearning;
using Xunit;

namespace DevOnBike.Overfit.Tests
{
    public sealed class GradientCheckerTests
    {
        [Fact]
        public void BatchNorm1D_NumericalGradient_MatchesAnalytical()
        {
            using var bn = new BatchNorm1D(numFeatures: 4);

            using var inputTensor = new FastTensor<float>(3, 4, clearMemory: true);
            using var input = new AutogradNode(inputTensor, requiresGrad: true);

            var x = input.DataView.AsSpan();
            for (int i = 0; i < x.Length; i++)
            {
                x[i] = (i - 5) * 0.1f;
            }

            var runningMeanSnapshot = new float[bn.RunningMean.Size];
            var runningVarSnapshot = new float[bn.RunningVar.Size];

            bn.RunningMean.GetView().AsReadOnlySpan().CopyTo(runningMeanSnapshot);
            bn.RunningVar.GetView().AsReadOnlySpan().CopyTo(runningVarSnapshot);

            GradientChecker.Verify(
                bn,
                graph =>
                {
                    runningMeanSnapshot.CopyTo(bn.RunningMean.GetView().AsSpan());
                    runningVarSnapshot.CopyTo(bn.RunningVar.GetView().AsSpan());

                    var y = bn.Forward(graph, input);

                    var targetTensor = new FastTensor<float>(3, 4, clearMemory: true);
                    var target = new AutogradNode(targetTensor, requiresGrad: false);

                    var t = target.DataView.AsSpan();
                    for (int i = 0; i < t.Length; i++)
                    {
                        t[i] = 0.05f * i;
                    }

                    return TensorMath.MSELoss(graph, y, target);
                },
                epsilon: 1e-3f,
                tolerance: 2e-2f,
                maxChecksPerParameter: 16);
        }

        [Fact]
        public void ResidualBlock_NumericalGradient_MatchesAnalytical()
        {
            using var block = new ResidualBlock(8);

            using var inputTensor = new FastTensor<float>(2, 8, clearMemory: true);
            using var input = new AutogradNode(inputTensor, requiresGrad: true);

            var x = input.DataView.AsSpan();
            for (int i = 0; i < x.Length; i++)
            {
                x[i] = (i - 4) * 0.07f;
            }

            GradientChecker.Verify(
                block,
                graph =>
                {
                    var y = block.Forward(graph, input);

                    var targetTensor = new FastTensor<float>(2, 8, clearMemory: true);
                    var target = new AutogradNode(targetTensor, requiresGrad: false);

                    var t = target.DataView.AsSpan();
                    for (int i = 0; i < t.Length; i++)
                    {
                        t[i] = ((i % 3) - 1) * 0.2f;
                    }

                    return TensorMath.MSELoss(graph, y, target);
                },
                epsilon: 1e-3f,
                tolerance: 2e-2f,
                maxChecksPerParameter: 12);
        }

        [Fact]
        public void Sigmoid_Backward_NumericalGradient_MatchesAnalytical()
        {
            using var inputTensor = new FastTensor<float>(2, 3, clearMemory: true);
            using var input = new AutogradNode(inputTensor, requiresGrad: true);

            var x = input.DataView.AsSpan();
            x[0] = -1.2f;
            x[1] = -0.5f;
            x[2] = 0.0f;
            x[3] = 0.3f;
            x[4] = 0.9f;
            x[5] = 1.7f;

            var graph = new ComputationGraph();
            using var y = TensorMath.Sigmoid(graph, input);

            using var zeroTensor = new FastTensor<float>(2, 3, clearMemory: true);
            using var zero = new AutogradNode(zeroTensor, requiresGrad: false);

            using var loss = TensorMath.MSELoss(graph, y, zero);
            graph.Backward(loss);

            var analytical = input.GradView.AsReadOnlySpan().ToArray();
            const float eps = 1e-3f;

            for (int i = 0; i < x.Length; i++)
            {
                float original = x[i];

                x[i] = original + eps;
                float lossPlus = EvalSigmoidSquaredMean(input);

                x[i] = original - eps;
                float lossMinus = EvalSigmoidSquaredMean(input);

                x[i] = original;

                float numerical = (lossPlus - lossMinus) / (2f * eps);
                float absError = MathF.Abs(analytical[i] - numerical);
                float relError = absError / MathF.Max(1e-6f, MathF.Abs(analytical[i]) + MathF.Abs(numerical));

                Assert.True(
                    relError < 1e-2f || absError < 1e-3f,
                    $"Sigmoid grad mismatch at {i}: analytical={analytical[i]}, numerical={numerical}, rel={relError}");
            }

            static float EvalSigmoidSquaredMean(AutogradNode inputNode)
            {
                var g = new ComputationGraph();
                using var y2 = TensorMath.Sigmoid(g, inputNode);

                float sum = 0f;
                var s = y2.DataView.AsReadOnlySpan();
                for (int j = 0; j < s.Length; j++)
                {
                    sum += s[j] * s[j];
                }

                return sum / s.Length;
            }
        }

        [Fact]
        public void Tanh_Backward_NumericalGradient_MatchesAnalytical()
        {
            using var inputTensor = new FastTensor<float>(2, 3, clearMemory: true);
            using var input = new AutogradNode(inputTensor, requiresGrad: true);

            var x = input.DataView.AsSpan();
            x[0] = -1.3f;
            x[1] = -0.4f;
            x[2] = 0.0f;
            x[3] = 0.25f;
            x[4] = 0.8f;
            x[5] = 1.4f;

            var graph = new ComputationGraph();
            using var y = TensorMath.Tanh(graph, input);

            using var zeroTensor = new FastTensor<float>(2, 3, clearMemory: true);
            using var zero = new AutogradNode(zeroTensor, requiresGrad: false);

            using var loss = TensorMath.MSELoss(graph, y, zero);
            graph.Backward(loss);

            var analytical = input.GradView.AsReadOnlySpan().ToArray();
            const float eps = 1e-3f;

            for (int i = 0; i < x.Length; i++)
            {
                float original = x[i];

                x[i] = original + eps;
                float lossPlus = EvalTanhSquaredMean(input);

                x[i] = original - eps;
                float lossMinus = EvalTanhSquaredMean(input);

                x[i] = original;

                float numerical = (lossPlus - lossMinus) / (2f * eps);
                float absError = MathF.Abs(analytical[i] - numerical);
                float relError = absError / MathF.Max(1e-6f, MathF.Abs(analytical[i]) + MathF.Abs(numerical));

                Assert.True(
                    relError < 1e-2f || absError < 1e-3f,
                    $"Tanh grad mismatch at {i}: analytical={analytical[i]}, numerical={numerical}, rel={relError}");
            }

            static float EvalTanhSquaredMean(AutogradNode inputNode)
            {
                var g = new ComputationGraph();
                using var y2 = TensorMath.Tanh(g, inputNode);

                float sum = 0f;
                var s = y2.DataView.AsReadOnlySpan();
                for (int j = 0; j < s.Length; j++)
                {
                    sum += s[j] * s[j];
                }

                return sum / s.Length;
            }
        }
    }
}