using DevOnBike.Overfit.Core;
using DevOnBike.Overfit.DeepLearning;

namespace DevOnBike.Overfit.Tests
{
    public sealed class GradientCheckerDeepLearningTests
    {
        [Fact]
        public void RepeatVector_InputGradient_NumericalGradient_MatchesAnalytical()
        {
            using var repeat = new RepeatVector(seqLen: 4);

            using var inputTensor = new FastTensor<float>(2, 3, clearMemory: true);
            using var input = new AutogradNode(inputTensor, requiresGrad: true);

            var x = input.DataView.AsSpan();
            for (int i = 0; i < x.Length; i++)
            {
                x[i] = (i - 2) * 0.15f;
            }

            var graph = new ComputationGraph();
            using var y = repeat.Forward(graph, input);

            using var zeroTensor = new FastTensor<float>(2, 4, 3, clearMemory: true);
            using var zero = new AutogradNode(zeroTensor, requiresGrad: false);

            using var loss = TensorMath.MSELoss(graph, y, zero);
            graph.Backward(loss);

            var analytical = input.GradView.AsReadOnlySpan().ToArray();
            const float eps = 1e-3f;

            for (int i = 0; i < x.Length; i++)
            {
                float original = x[i];

                x[i] = original + eps;
                float lossPlus = EvalRepeatSquaredMean(input, repeat);

                x[i] = original - eps;
                float lossMinus = EvalRepeatSquaredMean(input, repeat);

                x[i] = original;

                float numerical = (lossPlus - lossMinus) / (2f * eps);
                float absError = MathF.Abs(analytical[i] - numerical);
                float relError = absError / MathF.Max(1e-6f, MathF.Abs(analytical[i]) + MathF.Abs(numerical));

                Assert.True(
                    relError < 1e-2f || absError < 1e-3f,
                    $"RepeatVector grad mismatch at {i}: analytical={analytical[i]}, numerical={numerical}, rel={relError}");
            }

            static float EvalRepeatSquaredMean(AutogradNode inputNode, RepeatVector repeatModule)
            {
                var g = new ComputationGraph();
                using var y2 = repeatModule.Forward(g, inputNode);

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
        public void LSTMLayer_NumericalGradient_MatchesAnalytical()
        {
            using var layer = new LSTMLayer(inputSize: 3, hiddenSize: 2, returnSequences: false);

            using var inputTensor = new FastTensor<float>(2, 3, 3, clearMemory: true);
            using var input = new AutogradNode(inputTensor, requiresGrad: true);

            var x = input.DataView.AsSpan();
            for (int i = 0; i < x.Length; i++)
            {
                x[i] = (i - 4) * 0.08f;
            }

            GradientChecker.Verify(
                layer,
                graph =>
                {
                    var y = layer.Forward(graph, input);

                    var targetTensor = new FastTensor<float>(2, 2, clearMemory: true);
                    var target = new AutogradNode(targetTensor, requiresGrad: false);

                    var t = target.DataView.AsSpan();
                    for (int i = 0; i < t.Length; i++)
                    {
                        t[i] = ((i % 3) - 1) * 0.1f;
                    }

                    return TensorMath.MSELoss(graph, y, target);
                },
                epsilon: 1e-3f,
                tolerance: 3e-2f,
                maxChecksPerParameter: 12);
        }

        [Fact]
        public void LinearLayer_TrainAndEval_ProduceMatchingOutputs_ForBatch1()
        {
            using var layer = new LinearLayer(inputSize: 5, outputSize: 4);

            using var inputTensor = new FastTensor<float>(1, 5, clearMemory: true);
            using var input = new AutogradNode(inputTensor, requiresGrad: false);

            var x = input.DataView.AsSpan();
            x[0] = -1.0f;
            x[1] = -0.25f;
            x[2] = 0.5f;
            x[3] = 1.25f;
            x[4] = 2.0f;

            layer.Train();
            using var trainOut = layer.Forward(new ComputationGraph(), input);
            var expected = trainOut.DataView.AsReadOnlySpan().ToArray();

            layer.Eval();
            using var evalOut = layer.Forward(null, input);
            var actual = evalOut.DataView.AsReadOnlySpan();

            Assert.Equal(expected.Length, actual.Length);

            for (int i = 0; i < expected.Length; i++)
            {
                Assert.True(
                    MathF.Abs(expected[i] - actual[i]) < 1e-5f,
                    $"LinearLayer train/eval mismatch at {i}: train={expected[i]}, eval={actual[i]}");
            }
        }

        [Fact]
        public void Sequential_SmallNetwork_NumericalGradient_MatchesAnalytical()
        {
            using var seq = new Sequential(
                new LinearLayer(4, 5),
                new ReluActivation(),
                new LinearLayer(5, 3));

            using var inputTensor = new FastTensor<float>(2, 4, clearMemory: true);
            using var input = new AutogradNode(inputTensor, requiresGrad: true);

            var x = input.DataView.AsSpan();
            for (int i = 0; i < x.Length; i++)
            {
                x[i] = (i - 3) * 0.09f;
            }

            GradientChecker.Verify(
                seq,
                graph =>
                {
                    var y = seq.Forward(graph, input);

                    var targetTensor = new FastTensor<float>(2, 3, clearMemory: true);
                    var target = new AutogradNode(targetTensor, requiresGrad: false);

                    var t = target.DataView.AsSpan();
                    for (int i = 0; i < t.Length; i++)
                    {
                        t[i] = ((i % 4) - 1.5f) * 0.12f;
                    }

                    return TensorMath.MSELoss(graph, y, target);
                },
                epsilon: 1e-3f,
                tolerance: 2e-2f,
                maxChecksPerParameter: 10);
        }

        [Fact]
        public void Sequential_TrainAndEval_FlagsPropagateToChildren()
        {
            using var l1 = new LinearLayer(3, 4);
            using var relu = new ReluActivation();
            using var l2 = new LinearLayer(4, 2);
            using var seq = new Sequential(l1, relu, l2);

            seq.Eval();

            Assert.False(seq.IsTraining);
            Assert.False(l1.IsTraining);
            Assert.False(relu.IsTraining);
            Assert.False(l2.IsTraining);

            seq.Train();

            Assert.True(seq.IsTraining);
            Assert.True(l1.IsTraining);
            Assert.True(relu.IsTraining);
            Assert.True(l2.IsTraining);
        }
    }
}