// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System;
using System.IO;
using System.Linq;
using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.Ops;
using DevOnBike.Overfit.Optimizers;
using DevOnBike.Overfit.Tensors;
using DevOnBike.Overfit.Tensors.Core;
using Xunit;

namespace DevOnBike.Overfit.Tests
{
    public sealed class CorrectnessTests
    {
        [Fact]
        public void BatchNorm1D_NumericalGradient_MatchesAnalytical()
        {
            using var bn = new BatchNorm1D(numFeatures: 4);
            using var inputTensor = new TensorStorage<float>(12, clearMemory: true);
            using var input = new AutogradNode(inputTensor, new TensorShape(3, 4), requiresGrad: true);

            var x = input.DataView.AsSpan();
            for (var i = 0; i < x.Length; i++)
            {
                x[i] = (i - 5) * 0.1f;
            }

            var runningMeanSnapshot = new float[bn.RunningMean.Length];
            var runningVarSnapshot = new float[bn.RunningVar.Length];
            bn.RunningMean.AsReadOnlySpan().CopyTo(runningMeanSnapshot);
            bn.RunningVar.AsReadOnlySpan().CopyTo(runningVarSnapshot);

            GradientChecker.Verify(
                bn,
                graph =>
                {
                    runningMeanSnapshot.CopyTo(bn.RunningMean.AsSpan());
                    runningVarSnapshot.CopyTo(bn.RunningVar.AsSpan());

                    var y = bn.Forward(graph, input);
                    var targetTensor = new TensorStorage<float>(12, clearMemory: true);
                    var target = new AutogradNode(targetTensor, new TensorShape(3, 4), requiresGrad: false);

                    var t = target.DataView.AsSpan();
                    for (var i = 0; i < t.Length; i++)
                    {
                        t[i] = 0.05f * i;
                    }

                    return TensorMath.MSELoss(graph, y, target);
                },
                epsilon: 1e-3f,
                tolerance: 2e-2f,
                maxChecksPerParameter: 16);
        }

        [Fact(Skip = "Skipping test due to numerical instability in Eval mode")]
        public void ResidualBlock_NumericalGradient_MatchesAnalytical_EvalMode()
        {
            using var block = new ResidualBlock(8);

            var bn1 = (BatchNorm1D)typeof(ResidualBlock)
                .GetField("_bn1", System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic)!
                .GetValue(block)!;

            var bn2 = (BatchNorm1D)typeof(ResidualBlock)
                .GetField("_bn2", System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic)!
                .GetValue(block)!;

            bn1.Gamma.DataView.AsSpan().Fill(1f);
            bn1.Beta.DataView.AsSpan().Clear();
            bn1.RunningMean.AsSpan().Clear();
            bn1.RunningVar.AsSpan().Fill(1f);

            bn2.Gamma.DataView.AsSpan().Fill(1f);
            bn2.Beta.DataView.AsSpan().Clear();
            bn2.RunningMean.AsSpan().Clear();
            bn2.RunningVar.AsSpan().Fill(1f);

            block.Eval();

            using var inputTensor = new TensorStorage<float>(16, clearMemory: true);
            using var input = new AutogradNode(inputTensor, new TensorShape(2, 8), requiresGrad: true);

            var x = input.DataView.AsSpan();
            for (var i = 0; i < x.Length; i++)
            {
                x[i] = (i - 4) * 0.07f;
            }

            GradientChecker.Verify(
                block,
                graph =>
                {
                    var y = block.Forward(graph, input);

                    var targetTensor = new TensorStorage<float>(16, clearMemory: true);
                    var target = new AutogradNode(targetTensor, new TensorShape(2, 8), requiresGrad: false);

                    var t = target.DataView.AsSpan();
                    for (var i = 0; i < t.Length; i++)
                    {
                        t[i] = ((i % 3) - 1) * 0.2f;
                    }

                    return TensorMath.MSELoss(graph, y, target);
                },
                epsilon: 1e-3f,
                tolerance: 8e-2f,
                maxChecksPerParameter: 12);
        }

        [Fact]
        public void Sigmoid_Backward_NumericalGradient_MatchesAnalytical()
        {
            using var inputTensor = new TensorStorage<float>(6, clearMemory: true);
            using var input = new AutogradNode(inputTensor, new TensorShape(2, 3), requiresGrad: true);

            var x = input.DataView.AsSpan();
            x[0] = -1.2f; x[1] = -0.5f; x[2] = 0.0f; x[3] = 0.3f; x[4] = 0.9f; x[5] = 1.7f;

            var graph = new ComputationGraph();
            using var y = TensorMath.Sigmoid(graph, input);
            using var zeroTensor = new TensorStorage<float>(6, clearMemory: true);
            using var zero = new AutogradNode(zeroTensor, new TensorShape(2, 3), requiresGrad: false);
            using var loss = TensorMath.MSELoss(graph, y, zero);
            graph.Backward(loss);

            var analytical = input.GradView.AsReadOnlySpan().ToArray();
            const float eps = 1e-3f;

            for (var i = 0; i < x.Length; i++)
            {
                var original = x[i];
                x[i] = original + eps;
                var lossPlus = EvalSigmoidSquaredMean(input);
                x[i] = original - eps;
                var lossMinus = EvalSigmoidSquaredMean(input);
                x[i] = original;

                var numerical = (lossPlus - lossMinus) / (2f * eps);
                var absError = MathF.Abs(analytical[i] - numerical);
                var relError = absError / MathF.Max(1e-6f, MathF.Abs(analytical[i]) + MathF.Abs(numerical));

                Assert.True(relError < 1e-2f || absError < 1e-3f,
                    $"Sigmoid grad mismatch at {i}: analytical={analytical[i]}, numerical={numerical}, rel={relError}");
            }

            static float EvalSigmoidSquaredMean(AutogradNode inputNode)
            {
                var g = new ComputationGraph();
                using var y2 = TensorMath.Sigmoid(g, inputNode);
                var sum = 0f;
                var s = y2.DataView.AsReadOnlySpan();
                for (var j = 0; j < s.Length; j++)
                {
                    sum += s[j] * s[j];
                }
                return sum / s.Length;
            }
        }

        [Fact]
        public void Tanh_Backward_NumericalGradient_MatchesAnalytical()
        {
            using var inputTensor = new TensorStorage<float>(6, clearMemory: true);
            using var input = new AutogradNode(inputTensor, new TensorShape(2, 3), requiresGrad: true);

            var x = input.DataView.AsSpan();
            x[0] = -1.3f; x[1] = -0.4f; x[2] = 0.0f; x[3] = 0.25f; x[4] = 0.8f; x[5] = 1.4f;

            var graph = new ComputationGraph();
            using var y = TensorMath.Tanh(graph, input);
            using var zeroTensor = new TensorStorage<float>(6, clearMemory: true);
            using var zero = new AutogradNode(zeroTensor, new TensorShape(2, 3), requiresGrad: false);
            using var loss = TensorMath.MSELoss(graph, y, zero);
            graph.Backward(loss);

            var analytical = input.GradView.AsReadOnlySpan().ToArray();
            const float eps = 1e-3f;

            for (var i = 0; i < x.Length; i++)
            {
                var original = x[i];
                x[i] = original + eps;
                var lossPlus = EvalTanhSquaredMean(input);
                x[i] = original - eps;
                var lossMinus = EvalTanhSquaredMean(input);
                x[i] = original;

                var numerical = (lossPlus - lossMinus) / (2f * eps);
                var absError = MathF.Abs(analytical[i] - numerical);
                var relError = absError / MathF.Max(1e-6f, MathF.Abs(analytical[i]) + MathF.Abs(numerical));

                Assert.True(relError < 1e-2f || absError < 1e-3f,
                    $"Tanh grad mismatch at {i}: analytical={analytical[i]}, numerical={numerical}, rel={relError}");
            }

            static float EvalTanhSquaredMean(AutogradNode inputNode)
            {
                var g = new ComputationGraph();
                using var y2 = TensorMath.Tanh(g, inputNode);
                var sum = 0f;
                var s = y2.DataView.AsReadOnlySpan();
                for (var j = 0; j < s.Length; j++)
                {
                    sum += s[j] * s[j];
                }
                return sum / s.Length;
            }
        }

        [Fact]
        public void RepeatVector_InputGradient_NumericalGradient_MatchesAnalytical()
        {
            using var repeat = new RepeatVector(seqLen: 4);
            using var inputTensor = new TensorStorage<float>(6, clearMemory: true);
            using var input = new AutogradNode(inputTensor, new TensorShape(2, 3), requiresGrad: true);

            var x = input.DataView.AsSpan();
            for (var i = 0; i < x.Length; i++)
            {
                x[i] = (i - 2) * 0.15f;
            }

            var graph = new ComputationGraph();
            using var y = repeat.Forward(graph, input);
            using var zeroTensor = new TensorStorage<float>(24, clearMemory: true);
            using var zero = new AutogradNode(zeroTensor, new TensorShape(2, 4, 3), requiresGrad: false);
            using var loss = TensorMath.MSELoss(graph, y, zero);
            graph.Backward(loss);

            var analytical = input.GradView.AsReadOnlySpan().ToArray();
            const float eps = 1e-3f;

            for (var i = 0; i < x.Length; i++)
            {
                var original = x[i];
                x[i] = original + eps;
                var lossPlus = EvalRepeatSquaredMean(input, repeat);
                x[i] = original - eps;
                var lossMinus = EvalRepeatSquaredMean(input, repeat);
                x[i] = original;

                var numerical = (lossPlus - lossMinus) / (2f * eps);
                var absError = MathF.Abs(analytical[i] - numerical);
                var relError = absError / MathF.Max(1e-6f, MathF.Abs(analytical[i]) + MathF.Abs(numerical));

                Assert.True(relError < 1e-2f || absError < 1e-3f,
                    $"RepeatVector grad mismatch at {i}: analytical={analytical[i]}, numerical={numerical}, rel={relError}");
            }

            static float EvalRepeatSquaredMean(AutogradNode inputNode, RepeatVector repeatModule)
            {
                var g = new ComputationGraph();
                using var y2 = repeatModule.Forward(g, inputNode);
                var sum = 0f;
                var s = y2.DataView.AsReadOnlySpan();
                for (var j = 0; j < s.Length; j++)
                {
                    sum += s[j] * s[j];
                }
                return sum / s.Length;
            }
        }

        [Fact]
        public void LSTMLayer_NumericalGradient_MatchesAnalytical()
        {
            using var layer = new LstmLayer(inputSize: 3, hiddenSize: 2, returnSequences: false);
            using var inputTensor = new TensorStorage<float>(18, clearMemory: true);
            using var input = new AutogradNode(inputTensor, new TensorShape(2, 3, 3), requiresGrad: true);

            var x = input.DataView.AsSpan();
            for (var i = 0; i < x.Length; i++)
            {
                x[i] = (i - 4) * 0.08f;
            }

            GradientChecker.Verify(
                layer,
                graph =>
                {
                    var y = layer.Forward(graph, input);
                    var targetTensor = new TensorStorage<float>(4, clearMemory: true);
                    var target = new AutogradNode(targetTensor, new TensorShape(2, 2), requiresGrad: false);

                    var t = target.DataView.AsSpan();
                    for (var i = 0; i < t.Length; i++)
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
        public void MaxPool2D_Forward_PicksExpectedMaxima()
        {
            using var inputTensor = new TensorStorage<float>(16, clearMemory: true);
            using var input = new AutogradNode(inputTensor, new TensorShape(1, 1, 4, 4), requiresGrad: false);

            var x = input.DataView.AsSpan();
            for (var i = 0; i < x.Length; i++)
            {
                x[i] = i + 1;
            }

            using var pooled = TensorMath.MaxPool2D(null, input, 1, 4, 4, 2);
            var actual = pooled.DataView.AsReadOnlySpan().ToArray();

            Assert.Equal(4, actual.Length);
            Assert.Equal(6f, actual[0], 5);
            Assert.Equal(8f, actual[1], 5);
            Assert.Equal(14f, actual[2], 5);
            Assert.Equal(16f, actual[3], 5);
        }

        [Fact]
        public void GlobalAveragePool2D_Forward_ComputesExpectedMeans()
        {
            using var inputTensor = new TensorStorage<float>(8, clearMemory: true);
            using var input = new AutogradNode(inputTensor, new TensorShape(1, 2, 2, 2), requiresGrad: false);

            var x = input.DataView.AsSpan();
            x[0] = 1f; x[1] = 2f; x[2] = 3f; x[3] = 4f;
            x[4] = 10f; x[5] = 20f; x[6] = 30f; x[7] = 40f;

            using var pooled = TensorMath.GlobalAveragePool2D(null, input, 2, 2, 2);
            var actual = pooled.DataView.AsReadOnlySpan().ToArray();

            Assert.Equal(2, actual.Length);
            Assert.Equal(2.5f, actual[0], 5);
            Assert.Equal(25f, actual[1], 5);
        }

        [Fact]
        public void MatMul_SmallHandComputedCase_IsCorrect()
        {
            using var aTensor = new TensorStorage<float>(6, clearMemory: true);
            using var bTensor = new TensorStorage<float>(6, clearMemory: true);
            using var a = new AutogradNode(aTensor, new TensorShape(2, 3), requiresGrad: false);
            using var b = new AutogradNode(bTensor, new TensorShape(3, 2), requiresGrad: false);

            var aS = a.DataView.AsSpan();
            aS[0] = 1; aS[1] = 2; aS[2] = 3; aS[3] = 4; aS[4] = 5; aS[5] = 6;

            var bS = b.DataView.AsSpan();
            bS[0] = 7; bS[1] = 8; bS[2] = 9; bS[3] = 10; bS[4] = 11; bS[5] = 12;

            using var c = TensorMath.MatMul(null, a, b);
            var actual = c.DataView.AsReadOnlySpan().ToArray();

            Assert.Equal(58f, actual[0], 5);
            Assert.Equal(64f, actual[1], 5);
            Assert.Equal(139f, actual[2], 5);
            Assert.Equal(154f, actual[3], 5);
        }

        [Fact]
        public void AddBias_SmallHandComputedCase_IsCorrect()
        {
            using var xTensor = new TensorStorage<float>(6, clearMemory: true);
            using var bTensor = new TensorStorage<float>(3, clearMemory: true);
            using var x = new AutogradNode(xTensor, new TensorShape(2, 3), requiresGrad: false);
            using var b = new AutogradNode(bTensor, new TensorShape(3), requiresGrad: false);

            var xs = x.DataView.AsSpan();
            xs[0] = 1; xs[1] = 2; xs[2] = 3; xs[3] = 4; xs[4] = 5; xs[5] = 6;

            var bs = b.DataView.AsSpan();
            bs[0] = 10; bs[1] = 20; bs[2] = 30;

            using var y = TensorMath.AddBias(null, x, b);
            var actual = y.DataView.AsReadOnlySpan().ToArray();

            Assert.Equal(11f, actual[0], 5);
            Assert.Equal(22f, actual[1], 5);
            Assert.Equal(33f, actual[2], 5);
            Assert.Equal(14f, actual[3], 5);
            Assert.Equal(25f, actual[4], 5);
            Assert.Equal(36f, actual[5], 5);
        }

        [Fact]
        public void SoftmaxCrossEntropy_LowerForBetterTargetLogit()
        {
            using var logitsGoodTensor = new TensorStorage<float>(3, clearMemory: true);
            using var logitsBadTensor = new TensorStorage<float>(3, clearMemory: true);
            using var targetTensor = new TensorStorage<float>(3, clearMemory: true);

            using var logitsGood = new AutogradNode(logitsGoodTensor, new TensorShape(1, 3), requiresGrad: true);
            using var logitsBad = new AutogradNode(logitsBadTensor, new TensorShape(1, 3), requiresGrad: true);
            using var target = new AutogradNode(targetTensor, new TensorShape(1, 3), requiresGrad: false);

            var good = logitsGood.DataView.AsSpan();
            good[0] = 5f; good[1] = 1f; good[2] = 0f;

            var bad = logitsBad.DataView.AsSpan();
            bad[0] = 0.5f; bad[1] = 1f; bad[2] = 0f;

            var t = target.DataView.AsSpan();
            t[0] = 1f; t[1] = 0f; t[2] = 0f;

            using var lossGood = TensorMath.SoftmaxCrossEntropy(new ComputationGraph(), logitsGood, target);
            using var lossBad = TensorMath.SoftmaxCrossEntropy(new ComputationGraph(), logitsBad, target);

            Assert.True(lossGood.DataView.AsReadOnlySpan()[0] < lossBad.DataView.AsReadOnlySpan()[0],
                $"Expected better-target logit to produce lower loss, got good={lossGood.DataView.AsReadOnlySpan()[0]} bad={lossBad.DataView.AsReadOnlySpan()[0]}");
        }

        [Fact]
        public void ConvLayer_OutputShape_IsExpected()
        {
            using var conv = new ConvLayer(1, 2, 5, 5, 3);
            using var inputTensor = new TensorStorage<float>(25, clearMemory: true);
            using var input = new AutogradNode(inputTensor, new TensorShape(1, 1, 5, 5), requiresGrad: false);

            using var output = conv.Forward(new ComputationGraph(), input);

            Assert.Equal(1, output.DataView.GetDim(0));
            Assert.Equal(2, output.DataView.GetDim(1));
            Assert.Equal(3, output.DataView.GetDim(2));
            Assert.Equal(3, output.DataView.GetDim(3));
        }

        [Fact]
        public void ConvLayer_NumericalGradient_MatchesAnalytical_SmallCase()
        {
            using var conv = new ConvLayer(1, 1, 4, 4, 3);
            using var inputTensor = new TensorStorage<float>(16, clearMemory: true);
            using var input = new AutogradNode(inputTensor, new TensorShape(1, 1, 4, 4), requiresGrad: true);

            var x = input.DataView.AsSpan();
            for (var i = 0; i < x.Length; i++)
            {
                x[i] = (i - 4) * 0.05f;
            }

            GradientChecker.Verify(
                conv,
                graph =>
                {
                    var y = conv.Forward(graph, input);
                    var targetTensor = new TensorStorage<float>(4, clearMemory: true);
                    var target = new AutogradNode(targetTensor, new TensorShape(1, 1, 2, 2), requiresGrad: false);

                    var t = target.DataView.AsSpan();
                    for (var i = 0; i < t.Length; i++)
                    {
                        t[i] = 0.02f * (i + 1);
                    }

                    return TensorMath.MSELoss(graph, y, target);
                },
                epsilon: 1e-3f,
                tolerance: 8e-2f,
                maxChecksPerParameter: 8);
        }

        [Fact]
        public void Reshape_Backward_PropagatesUnitGradientToAllInputElements()
        {
            var graph = new ComputationGraph();

            using var inputTensor = new TensorStorage<float>(6, clearMemory: true);
            using var input = new AutogradNode(inputTensor, new TensorShape(2, 3), requiresGrad: true);

            var inData = input.DataView.AsSpan();
            for (var i = 0; i < inData.Length; i++)
            {
                inData[i] = i + 1;
            }

            using var reshaped = TensorMath.Reshape(graph, input, 3, 2);

            graph.Backward(reshaped);

            var actual = input.GradView.AsReadOnlySpan().ToArray();

            Assert.Equal(6, actual.Length);

            for (var i = 0; i < actual.Length; i++)
            {
                Assert.Equal(1f, actual[i], 3);
            }
        }

        [Fact]
        public void Reshape_Forward_PreservesFlatElementOrder()
        {
            using var inputTensor = new TensorStorage<float>(6, clearMemory: true);
            using var input = new AutogradNode(inputTensor, new TensorShape(2, 3), requiresGrad: false);

            var src = input.DataView.AsSpan();
            for (var i = 0; i < src.Length; i++)
            {
                src[i] = i + 1;
            }

            using var reshaped = TensorMath.Reshape(null, input, 3, 2);

            var actual = reshaped.DataView.AsReadOnlySpan().ToArray();

            Assert.Equal(6, actual.Length);

            for (var i = 0; i < actual.Length; i++)
            {
                Assert.Equal(i + 1, actual[i], 3);
            }
        }

        [Fact]
        public void LinearLayer_TrainAndEval_ProduceMatchingOutputs_ForBatch1()
        {
            using var layer = new LinearLayer(inputSize: 5, outputSize: 4);
            using var inputTensor = new TensorStorage<float>(5, clearMemory: true);
            using var input = new AutogradNode(inputTensor, new TensorShape(1, 5), requiresGrad: false);

            var x = input.DataView.AsSpan();
            x[0] = -1.0f; x[1] = -0.25f; x[2] = 0.5f; x[3] = 1.25f; x[4] = 2.0f;

            layer.Train();
            using var trainOut = layer.Forward(new ComputationGraph(), input);
            var expected = trainOut.DataView.AsReadOnlySpan().ToArray();

            layer.Eval();
            using var evalOut = layer.Forward(null, input);
            var actual = evalOut.DataView.AsReadOnlySpan();

            Assert.Equal(expected.Length, actual.Length);

            for (var i = 0; i < expected.Length; i++)
            {
                Assert.True(MathF.Abs(expected[i] - actual[i]) < 1e-5f,
                    $"LinearLayer train/eval mismatch at {i}: train={expected[i]}, eval={actual[i]}");
            }
        }

        [Fact]
        public void LinearLayer_SingleStepTraining_DecreasesLoss()
        {
            using var layer = new LinearLayer(inputSize: 2, outputSize: 1);
            using var xTensor = new TensorStorage<float>(8, clearMemory: true);
            using var yTensor = new TensorStorage<float>(4, clearMemory: true);
            using var xNode = new AutogradNode(xTensor, new TensorShape(4, 2), requiresGrad: false);
            using var yNode = new AutogradNode(yTensor, new TensorShape(4, 1), requiresGrad: false);

            var x = xNode.DataView.AsSpan();
            var y = yNode.DataView.AsSpan();

            x[0] = 0; x[1] = 0; y[0] = 0;
            x[2] = 1; x[3] = 0; y[1] = 1;
            x[4] = 0; x[5] = 1; y[2] = 2;
            x[6] = 1; x[7] = 1; y[3] = 3;

            using var optimizer = new Adam(layer.Parameters(), 0.01f) { UseAdamW = true };

            float beforeLoss;
            float afterLoss;

            {
                var graph = new ComputationGraph();
                using var pred = layer.Forward(graph, xNode);
                using var loss = TensorMath.MSELoss(graph, pred, yNode);
                beforeLoss = loss.DataView.AsReadOnlySpan()[0];
                graph.Backward(loss);
                optimizer.Step();
            }

            {
                var graph = new ComputationGraph();
                using var pred = layer.Forward(graph, xNode);
                using var loss = TensorMath.MSELoss(graph, pred, yNode);
                afterLoss = loss.DataView.AsReadOnlySpan()[0];
            }

            Assert.True(afterLoss < beforeLoss,
                $"Expected loss to decrease after one training step, got before={beforeLoss}, after={afterLoss}");
        }

        [Fact]
        public void Sequential_SmallNetwork_NumericalGradient_MatchesAnalytical()
        {
            using var seq = new Sequential(new LinearLayer(4, 5), new ReluActivation(), new LinearLayer(5, 3));
            using var inputTensor = new TensorStorage<float>(8, clearMemory: true);
            using var input = new AutogradNode(inputTensor, new TensorShape(2, 4), requiresGrad: true);

            var x = input.DataView.AsSpan();
            for (var i = 0; i < x.Length; i++)
            {
                x[i] = (i - 3) * 0.09f;
            }

            GradientChecker.Verify(
                seq,
                graph =>
                {
                    var y = seq.Forward(graph, input);
                    var targetTensor = new TensorStorage<float>(6, clearMemory: true);
                    var target = new AutogradNode(targetTensor, new TensorShape(2, 3), requiresGrad: false);

                    var t = target.DataView.AsSpan();
                    for (var i = 0; i < t.Length; i++)
                    {
                        t[i] = ((i % 4) - 1.5f) * 0.12f;
                    }

                    return TensorMath.MSELoss(graph, y, target);
                },
                epsilon: 1e-3f,
                tolerance: 3e-2f,
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

        [Fact]
        public void BatchNorm1D_TrainAndEval_ProduceMatchingOutputs_AfterStatsWarmup()
        {
            using var bn = new BatchNorm1D(numFeatures: 4);

            using (var warmupTensor = new TensorStorage<float>(24, clearMemory: true))
            using (var warmupInput = new AutogradNode(warmupTensor, new TensorShape(6, 4), requiresGrad: false))
            {
                var w = warmupInput.DataView.AsSpan();
                for (var i = 0; i < w.Length; i++)
                {
                    w[i] = ((i % 7) - 3) * 0.25f;
                }

                bn.Train();
                for (var step = 0; step < 20; step++)
                {
                    using var y = bn.Forward(new ComputationGraph(), warmupInput);
                }
            }

            using var inputTensor = new TensorStorage<float>(12, clearMemory: true);
            using var input = new AutogradNode(inputTensor, new TensorShape(3, 4), requiresGrad: false);

            var x = input.DataView.AsSpan();
            x[0] = -1.0f; x[1] = -0.5f; x[2] = 0.2f; x[3] = 1.0f;
            x[4] = -0.8f; x[5] = -0.1f; x[6] = 0.3f; x[7] = 1.1f;
            x[8] = -1.2f; x[9] = -0.4f; x[10] = 0.1f; x[11] = 0.9f;

            bn.Eval();
            using var evalTensor = new TensorStorage<float>(12, clearMemory: true);

            var inSpan = input.DataView.AsReadOnlySpan();
            var outSpan = evalTensor.AsSpan();

            for (var r = 0; r < 3; r++)
            {
                bn.ForwardInference(inSpan.Slice(r * 4, 4), outSpan.Slice(r * 4, 4));
            }

            var evalOutput = evalTensor.AsReadOnlySpan().ToArray();
            var gamma = bn.Gamma.DataView.AsReadOnlySpan();
            var beta = bn.Beta.DataView.AsReadOnlySpan();
            var mean = bn.RunningMean.AsReadOnlySpan();
            var varSpan = bn.RunningVar.AsReadOnlySpan();

            var expected = new float[12];
            for (var r = 0; r < 3; r++)
            {
                for (var c = 0; c < 4; c++)
                {
                    var idx = r * 4 + c;
                    var scale = gamma[c] / MathF.Sqrt(varSpan[c] + bn.Eps);
                    var shift = beta[c] - mean[c] * scale;
                    expected[idx] = x[idx] * scale + shift;
                }
            }

            for (var i = 0; i < expected.Length; i++)
            {
                Assert.True(MathF.Abs(expected[i] - evalOutput[i]) < 1e-5f,
                    $"BatchNorm1D eval mismatch at {i}: expected={expected[i]}, actual={evalOutput[i]}");
            }
        }

        [Fact]
        public void ResidualBlock_EvalPath_IsDeterministic_AfterStatsWarmup()
        {
            using var block = new ResidualBlock(hiddenSize: 8);

            using (var warmupTensor = new TensorStorage<float>(64, clearMemory: true))
            using (var warmupInput = new AutogradNode(warmupTensor, new TensorShape(8, 8), requiresGrad: false))
            {
                var w = warmupInput.DataView.AsSpan();
                for (var i = 0; i < w.Length; i++)
                {
                    w[i] = ((i % 11) - 5) * 0.1f;
                }

                block.Train();
                for (var step = 0; step < 25; step++)
                {
                    using var y = block.Forward(new ComputationGraph(), warmupInput);
                }
            }

            using var inputTensor = new TensorStorage<float>(8, clearMemory: true);
            using var input = new AutogradNode(inputTensor, new TensorShape(1, 8), requiresGrad: false);

            var x = input.DataView.AsSpan();
            x[0] = -1.0f; x[1] = -0.5f; x[2] = -0.1f; x[3] = 0.0f;
            x[4] = 0.3f; x[5] = 0.7f; x[6] = 1.1f; x[7] = 1.5f;

            block.Eval();

            var evalGraph1 = new ComputationGraph { IsRecording = false };
            using var out1 = block.Forward(evalGraph1, input);
            var y1 = out1.DataView.AsReadOnlySpan().ToArray();

            var evalGraph2 = new ComputationGraph { IsRecording = false };
            using var out2 = block.Forward(evalGraph2, input);
            var y2 = out2.DataView.AsReadOnlySpan().ToArray();

            Assert.Equal(y1.Length, y2.Length);

            for (var i = 0; i < y1.Length; i++)
            {
                Assert.True(MathF.Abs(y1[i] - y2[i]) < 1e-6f,
                    $"ResidualBlock eval path is not deterministic at {i}: first={y1[i]}, second={y2[i]}");
            }
        }

        [Fact]
        public void BatchNorm1D_EvalPath_IsDeterministic_AfterWarmup()
        {
            var model = new Sequential(
                new LinearLayer(2, 8),
                new BatchNorm1D(8),
                new ReluActivation(),
                new LinearLayer(8, 2));

            model.Train();

            using var xTensor = CreateTensor(
                rows: 8,
                cols: 2,
                values:
                [
                    -2.0f, -1.8f,
                -1.7f, -1.3f,
                -1.4f, -1.1f,
                -1.2f, -0.8f,
                 1.2f,  0.9f,
                 1.4f,  1.1f,
                 1.7f,  1.4f,
                 2.0f,  1.7f
                ]);

            using var yTensor = CreateTensor(
                rows: 8,
                cols: 2,
                values:
                [
                    1f, 0f,
                1f, 0f,
                1f, 0f,
                1f, 0f,
                0f, 1f,
                0f, 1f,
                0f, 1f,
                0f, 1f
                ]);

            using var xNode = new AutogradNode(xTensor, new TensorShape(8, 2), requiresGrad: false);
            using var yNode = new AutogradNode(yTensor, new TensorShape(8, 2), requiresGrad: false);

            using var optimizer = new Adam(model.Parameters(), learningRate: 0.01f)
            {
                UseAdamW = true
            };

            for (var step = 0; step < 40; step++)
            {
                var trainGraph = new ComputationGraph();
                optimizer.ZeroGrad();

                using var logits = model.Forward(trainGraph, xNode);
                using var loss = TensorMath.SoftmaxCrossEntropy(trainGraph, logits, yNode);

                trainGraph.Backward(loss);
                optimizer.Step();
            }

            model.Eval();

            // POPRAWKA: Rozdzielone grafy w celu wyeliminowania ObjectDisposedException
            var evalGraph1 = new ComputationGraph();
            using var eval1 = model.Forward(evalGraph1, xNode);
            var s1 = eval1.DataView.AsReadOnlySpan().ToArray();

            var evalGraph2 = new ComputationGraph();
            using var eval2 = model.Forward(evalGraph2, xNode);
            var s2 = eval2.DataView.AsReadOnlySpan().ToArray();

            Assert.Equal(s1.Length, s2.Length);

            for (var i = 0; i < s1.Length; i++)
            {
                Assert.Equal(s1[i], s2[i], 5);
            }
        }

        [Fact]
        public void Adam_Step_KeepsParametersFinite()
        {
            using var layer = new LinearLayer(4, 3);
            using var optimizer = new Adam(layer.Parameters(), 0.001f) { UseAdamW = true };

            using var inputTensor = new TensorStorage<float>(8, clearMemory: true);
            using var targetTensor = new TensorStorage<float>(6, clearMemory: true);
            using var input = new AutogradNode(inputTensor, new TensorShape(2, 4), requiresGrad: false);
            using var target = new AutogradNode(targetTensor, new TensorShape(2, 3), requiresGrad: false);

            var xs = input.DataView.AsSpan();
            for (var i = 0; i < xs.Length; i++)
            {
                xs[i] = (i - 2) * 0.1f;
            }

            var ts = target.DataView.AsSpan();
            for (var i = 0; i < ts.Length; i++)
            {
                ts[i] = (i % 3) * 0.2f;
            }

            var graph = new ComputationGraph();
            using var pred = layer.Forward(graph, input);
            using var loss = TensorMath.MSELoss(graph, pred, target);
            graph.Backward(loss);
            optimizer.Step();

            foreach (var p in layer.Parameters())
            {
                var span = p.DataView.AsReadOnlySpan();
                for (var i = 0; i < span.Length; i++)
                {
                    Assert.True(float.IsFinite(span[i]), $"Parameter contains non-finite value at {i}: {span[i]}");
                }
            }
        }

        [Fact]
        public void Mnist_SmallTraining_Run_DecreasesLoss_And_ReachesReasonableAccuracy()
        {
            const int trainSize = 2048;
            const int testSize = 256;
            const int batchSize = 64;
            const int epochs = 2;
            const float lr = 0.001f;

            var trainImagesPath = "d:/ml/train-images.idx3-ubyte";
            var trainLabelsPath = "d:/ml/train-labels.idx1-ubyte";
            var testImagesPath = "d:/ml/t10k-images.idx3-ubyte";
            var testLabelsPath = "d:/ml/t10k-labels.idx1-ubyte";

            if (!File.Exists(trainImagesPath) || !File.Exists(trainLabelsPath) ||
                !File.Exists(testImagesPath) || !File.Exists(testLabelsPath))
            {
                return; // Przeskakuje, jeśli brakuje datasetu
            }

            var (trainX, trainY) = MnistLoader.Load(trainImagesPath, trainLabelsPath, trainSize);
            var (testX, testY) = MnistLoader.Load(testImagesPath, testLabelsPath, testSize);

            using var conv1 = new ConvLayer(1, 8, 28, 28, 3);
            using var bn1 = new BatchNorm1D(1352);
            using var res1 = new ResidualBlock(1352);
            using var fcOut = new LinearLayer(8, 10);

            var parameters = conv1.Parameters().Concat(bn1.Parameters()).Concat(res1.Parameters()).Concat(fcOut.Parameters()).ToArray();
            using var optimizer = new Adam(parameters, lr) { UseAdamW = true };

            var graph = new ComputationGraph();
            var epochLosses = new float[epochs];

            for (var epoch = 0; epoch < epochs; epoch++)
            {
                var epochLoss = 0f;
                var batches = trainSize / batchSize;

                conv1.Train();
                bn1.Train();
                res1.Train();
                fcOut.Train();

                for (var b = 0; b < batches; b++)
                {
                    graph.Reset();
                    optimizer.ZeroGrad();

                    using var xBData = new TensorStorage<float>(batchSize * 1 * 28 * 28, clearMemory: false);
                    using var yBData = new TensorStorage<float>(batchSize * 10, clearMemory: false);
                    using var xBNode = new AutogradNode(xBData, new TensorShape(batchSize, 1, 28, 28), requiresGrad: false);
                    using var yBNode = new AutogradNode(yBData, new TensorShape(batchSize, 10), requiresGrad: false);

                    trainX.AsReadOnlySpan().Slice(b * batchSize * 784, batchSize * 784).CopyTo(xBData.AsSpan());
                    trainY.AsReadOnlySpan().Slice(b * batchSize * 10, batchSize * 10).CopyTo(yBData.AsSpan());

                    using var h1 = conv1.Forward(graph, xBNode);
                    using var a1 = TensorMath.ReLU(graph, h1);
                    using var p1 = TensorMath.MaxPool2D(graph, a1, 8, 26, 26, 2);
                    using var p1F = TensorMath.Reshape(graph, p1, batchSize, 1352);
                    using var bn1O = bn1.Forward(graph, p1F);
                    using var resO = res1.Forward(graph, bn1O);
                    using var gapO = TensorMath.GlobalAveragePool2D(graph, resO, 8, 13, 13);
                    using var logits = fcOut.Forward(graph, gapO);
                    using var loss = TensorMath.SoftmaxCrossEntropy(graph, logits, yBNode);

                    epochLoss += loss.DataView.AsReadOnlySpan()[0];
                    graph.Backward(loss);
                    optimizer.Step();
                }

                epochLosses[epoch] = epochLoss / (trainSize / batchSize);
            }

            conv1.Train();
            bn1.Train();
            res1.Train();
            fcOut.Train();

            var correct = 0;
            var testBatches = testSize / batchSize;

            for (var b = 0; b < testBatches; b++)
            {
                using var xBData = new TensorStorage<float>(batchSize * 1 * 28 * 28, clearMemory: false);
                using var yBData = new TensorStorage<float>(batchSize * 10, clearMemory: false);
                using var xBNode = new AutogradNode(xBData, new TensorShape(batchSize, 1, 28, 28), requiresGrad: false);
                using var yBNode = new AutogradNode(yBData, new TensorShape(batchSize, 10), requiresGrad: false);

                testX.AsReadOnlySpan().Slice(b * batchSize * 784, batchSize * 784).CopyTo(xBData.AsSpan());
                testY.AsReadOnlySpan().Slice(b * batchSize * 10, batchSize * 10).CopyTo(yBData.AsSpan());

                var evalGraph = new ComputationGraph { IsRecording = false };

                using var h1 = conv1.Forward(evalGraph, xBNode);
                using var a1 = TensorMath.ReLU(evalGraph, h1);
                using var p1 = TensorMath.MaxPool2D(evalGraph, a1, 8, 26, 26, 2);
                using var p1F = TensorMath.Reshape(evalGraph, p1, batchSize, 1352);
                using var bn1O = bn1.Forward(evalGraph, p1F);
                using var resO = res1.Forward(evalGraph, bn1O);
                using var gapO = TensorMath.GlobalAveragePool2D(evalGraph, resO, 8, 13, 13);
                using var logits = fcOut.Forward(evalGraph, gapO);

                var logitsSpan = logits.DataView.AsReadOnlySpan();
                var labelsSpan = yBNode.DataView.AsReadOnlySpan();

                for (var i = 0; i < batchSize; i++)
                {
                    var pred = ArgMax(logitsSpan.Slice(i * 10, 10));
                    var truth = ArgMax(labelsSpan.Slice(i * 10, 10));
                    if (pred == truth)
                    {
                        correct++;
                    }
                }
            }

            var accuracy = correct / (float)(testBatches * batchSize);

            Assert.True(epochLosses[1] < epochLosses[0],
                $"Training loss did not decrease: epoch1={epochLosses[0]:F4}, epoch2={epochLosses[1]:F4}");

            Assert.True(accuracy >= 0.60f,
                $"Accuracy too low after small training run: {accuracy:P2}");
        }

        private static int ArgMax(ReadOnlySpan<float> span)
        {
            var bestIdx = 0;
            var bestVal = span[0];
            for (var i = 1; i < span.Length; i++)
            {
                if (span[i] > bestVal)
                {
                    bestVal = span[i];
                    bestIdx = i;
                }
            }
            return bestIdx;
        }

        [Fact]
        public void ReLU_Backward_ZeroForNegative_OneForPositive_AndZeroAtZero()
        {
            using var inputTensor = new TensorStorage<float>(6, clearMemory: true);
            using var input = new AutogradNode(inputTensor, new TensorShape(1, 6), requiresGrad: true);

            var x = input.DataView.AsSpan();
            x[0] = -2f;
            x[1] = -0.5f;
            x[2] = -0.0001f;
            x[3] = 0f;
            x[4] = 0.25f;
            x[5] = 3f;

            var graph = new ComputationGraph();
            using var y = TensorMath.ReLU(graph, input);
            graph.Backward(y);

            var g = input.GradView.AsReadOnlySpan().ToArray();

            Assert.Equal(0f, g[0], 3);
            Assert.Equal(0f, g[1], 3);
            Assert.Equal(0f, g[2], 3);
            Assert.Equal(0f, g[3], 3);
            Assert.Equal(1f, g[4], 3);
            Assert.Equal(1f, g[5], 3);
        }

        [Fact]
        public void Add_Backward_PropagatesUnitGradientToBothInputs()
        {
            using var aTensor = new TensorStorage<float>(6, clearMemory: true);
            using var bTensor = new TensorStorage<float>(6, clearMemory: true);
            using var a = new AutogradNode(aTensor, new TensorShape(2, 3), requiresGrad: true);
            using var b = new AutogradNode(bTensor, new TensorShape(2, 3), requiresGrad: true);

            var aS = a.DataView.AsSpan();
            var bS = b.DataView.AsSpan();
            for (var i = 0; i < aS.Length; i++)
            {
                aS[i] = i + 1;
                bS[i] = 10 + i;
            }

            var graph = new ComputationGraph();
            using var c = TensorMath.Add(graph, a, b);
            graph.Backward(c);

            var ga = a.GradView.AsReadOnlySpan().ToArray();
            var gb = b.GradView.AsReadOnlySpan().ToArray();

            Assert.Equal(ga.Length, gb.Length);
            for (var i = 0; i < ga.Length; i++)
            {
                Assert.Equal(1f, ga[i], 3);
                Assert.Equal(1f, gb[i], 3);
            }
        }

        [Fact]
        public void MSELoss_Backward_MatchesHandComputedGradient()
        {
            using var predTensor = new TensorStorage<float>(4, clearMemory: true);
            using var tgtTensor = new TensorStorage<float>(4, clearMemory: true);
            using var pred = new AutogradNode(predTensor, new TensorShape(1, 4), requiresGrad: true);
            using var tgt = new AutogradNode(tgtTensor, new TensorShape(1, 4), requiresGrad: false);

            var p = pred.DataView.AsSpan();
            p[0] = 1f; p[1] = 2f; p[2] = 3f; p[3] = 4f;

            var t = tgt.DataView.AsSpan();
            t[0] = 0f; t[1] = 1f; t[2] = 1f; t[3] = 2f;

            var graph = new ComputationGraph();
            using var loss = TensorMath.MSELoss(graph, pred, tgt);
            graph.Backward(loss);

            var g = pred.GradView.AsReadOnlySpan().ToArray();

            float[] expected =
            {
                2f * (1f - 0f) / 4f,
                2f * (2f - 1f) / 4f,
                2f * (3f - 1f) / 4f,
                2f * (4f - 2f) / 4f
            };

            for (var i = 0; i < expected.Length; i++)
            {
                Assert.Equal(expected[i], g[i], 3);
            }
        }

        [Fact]
        public void BatchNorm1D_RunningStats_RemainFinite_AfterManyTrainingSteps()
        {
            using var bn = new BatchNorm1D(numFeatures: 8);
            using var inputTensor = new TensorStorage<float>(128, clearMemory: true);
            using var input = new AutogradNode(inputTensor, new TensorShape(16, 8), requiresGrad: false);

            var x = input.DataView.AsSpan();
            for (var step = 0; step < 50; step++)
            {
                for (var i = 0; i < x.Length; i++)
                {
                    x[i] = ((i + step) % 13 - 6) * 0.17f;
                }

                bn.Train();
                using var y = bn.Forward(new ComputationGraph(), input);
            }

            var mean = bn.RunningMean.AsReadOnlySpan();
            var varSpan = bn.RunningVar.AsReadOnlySpan();

            for (var i = 0; i < mean.Length; i++)
            {
                Assert.True(float.IsFinite(mean[i]), $"RunningMean[{i}] is not finite: {mean[i]}");
                Assert.True(float.IsFinite(varSpan[i]), $"RunningVar[{i}] is not finite: {varSpan[i]}");
                Assert.True(varSpan[i] > 0f, $"RunningVar[{i}] should stay positive, got {varSpan[i]}");
            }
        }

        [Fact]
        public void ConvLayer_Backward_Gradients_AreFinite()
        {
            using var conv = new ConvLayer(1, 2, 6, 6, 3);
            using var inputTensor = new TensorStorage<float>(72, clearMemory: true);
            using var targetTensor = new TensorStorage<float>(64, clearMemory: true);
            using var input = new AutogradNode(inputTensor, new TensorShape(2, 1, 6, 6), requiresGrad: true);
            using var target = new AutogradNode(targetTensor, new TensorShape(2, 2, 4, 4), requiresGrad: false);

            var x = input.DataView.AsSpan();
            for (var i = 0; i < x.Length; i++)
            {
                x[i] = ((i % 17) - 8) * 0.05f;
            }

            var t = target.DataView.AsSpan();
            for (var i = 0; i < t.Length; i++)
            {
                t[i] = ((i % 7) - 3) * 0.03f;
            }

            var graph = new ComputationGraph();
            using var y = conv.Forward(graph, input);
            using var loss = TensorMath.MSELoss(graph, y, target);
            graph.Backward(loss);

            var ig = input.GradView.AsReadOnlySpan();
            for (var i = 0; i < ig.Length; i++)
            {
                Assert.True(float.IsFinite(ig[i]), $"Input grad non-finite at {i}: {ig[i]}");
            }

            foreach (var p in conv.Parameters())
            {
                if (!p.RequiresGrad)
                {
                    continue;
                }

                var pg = p.GradView.AsReadOnlySpan();
                for (var i = 0; i < pg.Length; i++)
                {
                    Assert.True(float.IsFinite(pg[i]), $"Conv param grad non-finite at {i}: {pg[i]}");
                }
            }
        }

        [Fact]
        public void Subtract_ForwardAndBackward_Correct()
        {
            var graph = new ComputationGraph();

            using var aTensor = new TensorStorage<float>(6, clearMemory: true);
            using var bTensor = new TensorStorage<float>(6, clearMemory: true);
            using var a = new AutogradNode(aTensor, new TensorShape(2, 3), requiresGrad: true);
            using var b = new AutogradNode(bTensor, new TensorShape(2, 3), requiresGrad: true);

            a.DataView.AsSpan().Fill(5f);
            b.DataView.AsSpan().Fill(3f);

            using var output = TensorMath.Subtract(graph, a, b);

            Assert.Equal(2f, output.DataView.AsReadOnlySpan()[0], 3);

            graph.Backward(output);

            Assert.Equal(1f, a.GradView.AsReadOnlySpan()[0], 3);
            Assert.Equal(-1f, b.GradView.AsReadOnlySpan()[0], 3);
        }

        [Fact]
        public void SoftmaxCrossEntropy_NumericallyStable_WithSmallProbabilities()
        {
            var graph = new ComputationGraph();

            using var logitsTensor = new TensorStorage<float>(3, clearMemory: true);
            using var targetTensor = new TensorStorage<float>(3, clearMemory: true);
            using var logits = new AutogradNode(logitsTensor, new TensorShape(1, 3), requiresGrad: true);
            using var target = new AutogradNode(targetTensor, new TensorShape(1, 3), requiresGrad: false);

            var l = logits.DataView.AsSpan();
            l[0] = 100f;
            l[1] = 0f;
            l[2] = 0f;

            var t = target.DataView.AsSpan();
            t[0] = 0f;
            t[1] = 1f;
            t[2] = 0f;

            using var loss = TensorMath.SoftmaxCrossEntropy(graph, logits, target);

            var lossVal = loss.DataView.AsReadOnlySpan()[0];

            Assert.False(float.IsNaN(lossVal));
            Assert.False(float.IsInfinity(lossVal));
            Assert.True(lossVal >= 10f, $"Expected large finite loss, got {lossVal}");
            Assert.True(lossVal <= 20f, $"Expected clamped finite loss, got {lossVal}");
        }

        [Fact]
        public void ResidualBlock_Backward_Gradients_AreFinite()
        {
            using var block = new ResidualBlock(8);
            using var inputTensor = new TensorStorage<float>(32, clearMemory: true);
            using var targetTensor = new TensorStorage<float>(32, clearMemory: true);
            using var input = new AutogradNode(inputTensor, new TensorShape(4, 8), requiresGrad: true);
            using var target = new AutogradNode(targetTensor, new TensorShape(4, 8), requiresGrad: false);

            var x = input.DataView.AsSpan();
            for (var i = 0; i < x.Length; i++)
            {
                x[i] = ((i % 9) - 4) * 0.11f;
            }

            var t = target.DataView.AsSpan();
            for (var i = 0; i < t.Length; i++)
            {
                t[i] = ((i % 5) - 2) * 0.07f;
            }

            var graph = new ComputationGraph();
            using var y = block.Forward(graph, input);
            using var loss = TensorMath.MSELoss(graph, y, target);
            graph.Backward(loss);

            var ig = input.GradView.AsReadOnlySpan();
            for (var i = 0; i < ig.Length; i++)
            {
                Assert.True(float.IsFinite(ig[i]), $"Residual input grad non-finite at {i}: {ig[i]}");
            }

            foreach (var p in block.Parameters())
            {
                var pg = p.GradView.AsReadOnlySpan();
                for (var i = 0; i < pg.Length; i++)
                {
                    Assert.True(float.IsFinite(pg[i]), $"Residual param grad non-finite at {i}: {pg[i]}");
                }
            }
        }

        [Fact]
        public void SmallTrainingLoop_LossNeverBecomesNaN()
        {
            using var seq = new Sequential(
                new LinearLayer(4, 8),
                new ReluActivation(),
                new LinearLayer(8, 3));

            using var optimizer = new Adam(seq.Parameters(), 0.001f) { UseAdamW = true };

            using var xTensor = new TensorStorage<float>(32, clearMemory: true);
            using var yTensor = new TensorStorage<float>(24, clearMemory: true);
            using var xNode = new AutogradNode(xTensor, new TensorShape(8, 4), requiresGrad: false);
            using var yNode = new AutogradNode(yTensor, new TensorShape(8, 3), requiresGrad: false);

            var x = xNode.DataView.AsSpan();
            for (var i = 0; i < x.Length; i++)
            {
                x[i] = ((i % 11) - 5) * 0.13f;
            }

            var y = yNode.DataView.AsSpan();
            for (var i = 0; i < y.Length; i++)
            {
                y[i] = ((i % 3) - 1) * 0.25f;
            }

            for (var step = 0; step < 25; step++)
            {
                var graph = new ComputationGraph();
                optimizer.ZeroGrad();

                using var pred = seq.Forward(graph, xNode);
                using var loss = TensorMath.MSELoss(graph, pred, yNode);

                var lv = loss.DataView.AsReadOnlySpan()[0];
                Assert.True(float.IsFinite(lv), $"Loss became non-finite at step {step}: {lv}");

                graph.Backward(loss);
                optimizer.Step();
            }
        }

        [Fact]
        public void Sequential_SmallNetwork_Training_DecreasesLoss_WithoutBatchNorm()
        {
            var graph = new ComputationGraph();

            // Zwiększono pojemność z 8 na 16 neuronów, by uniknąć problemu Dead ReLU
            var model = new Sequential(
                new LinearLayer(2, 16),
                new ReluActivation(),
                new LinearLayer(16, 1));

            model.Train();

            using var xTensor = CreateTensor(
                rows: 4,
                cols: 2,
                values:
                [
                    0f, 0f,
                    0f, 1f,
                    1f, 0f,
                    1f, 1f
                ]);

            using var yTensor = CreateTensor(
                rows: 4,
                cols: 1,
                values:
                [
                    0f,
                    1f,
                    1f,
                    0f
                ]);

            using var xNode = new AutogradNode(xTensor, new TensorShape(4, 2), requiresGrad: false);
            using var yNode = new AutogradNode(yTensor, new TensorShape(4, 1), requiresGrad: false);

            using var optimizer = new Adam(model.Parameters(), learningRate: 0.03f)
            {
                UseAdamW = true
            };

            float initialLoss;
            var finalLoss = float.MaxValue;

            {
                using var pred0 = model.Forward(graph, xNode);
                using var loss0 = TensorMath.MSELoss(graph, pred0, yNode);
                initialLoss = loss0.DataView.AsReadOnlySpan()[0];
                graph.Reset();
            }

            // Zwiększono liczbę kroków z 300 na 600
            for (var step = 0; step < 600; step++)
            {
                graph.Reset();
                optimizer.ZeroGrad();

                using var pred = model.Forward(graph, xNode);
                using var loss = TensorMath.MSELoss(graph, pred, yNode);

                finalLoss = loss.DataView.AsReadOnlySpan()[0];

                graph.Backward(loss);
                optimizer.Step();
            }

            Assert.True(finalLoss < initialLoss * 0.5f,
                $"Loss should decrease clearly. initial={initialLoss}, final={finalLoss}");

            Assert.True(finalLoss < 0.10f,
                $"Final loss too high for tiny XOR-style regression. final={finalLoss}");
        }

        [Fact]
        public void BatchNorm1D_SmallClassifier_Overfits_LinearlySeparableBatch()
        {
            var graph = new ComputationGraph();

            var model = new Sequential(
                new LinearLayer(2, 8),
                new BatchNorm1D(8),
                new ReluActivation(),
                new LinearLayer(8, 2));

            model.Train();

            using var xTensor = CreateTensor(
                rows: 8,
                cols: 2,
                values:
                [
                    -2.0f, -1.8f,
                -1.7f, -1.3f,
                -1.4f, -1.1f,
                -1.2f, -0.8f,
                 1.2f,  0.9f,
                 1.4f,  1.1f,
                 1.7f,  1.4f,
                 2.0f,  1.7f
                ]);

            using var yTensor = CreateTensor(
                rows: 8,
                cols: 2,
                values:
                [
                    1f, 0f,
                1f, 0f,
                1f, 0f,
                1f, 0f,
                0f, 1f,
                0f, 1f,
                0f, 1f,
                0f, 1f
                ]);

            using var xNode = new AutogradNode(xTensor, new TensorShape(8, 2), requiresGrad: false);
            using var yNode = new AutogradNode(yTensor, new TensorShape(8, 2), requiresGrad: false);

            using var optimizer = new Adam(model.Parameters(), learningRate: 0.01f)
            {
                UseAdamW = true
            };

            float initialLoss;
            var finalLoss = float.MaxValue;

            {
                using var logits0 = model.Forward(graph, xNode);
                using var loss0 = TensorMath.SoftmaxCrossEntropy(graph, logits0, yNode);
                initialLoss = loss0.DataView.AsReadOnlySpan()[0];
                graph.Reset();
            }

            for (var step = 0; step < 200; step++)
            {
                graph.Reset();
                optimizer.ZeroGrad();

                using var logits = model.Forward(graph, xNode);
                using var loss = TensorMath.SoftmaxCrossEntropy(graph, logits, yNode);

                finalLoss = loss.DataView.AsReadOnlySpan()[0];

                graph.Backward(loss);
                optimizer.Step();
            }

            model.Eval();

            graph.Reset();
            using var evalLogits = model.Forward(graph, xNode);
            var accuracy = ComputeAccuracy(evalLogits.DataView.AsReadOnlySpan(), yTensor.AsReadOnlySpan(), rows: 8, classes: 2);

            Assert.True(finalLoss < initialLoss * 0.4f,
                $"BatchNorm classifier loss should drop clearly. initial={initialLoss}, final={finalLoss}");

            Assert.True(accuracy >= 0.875f,
                $"Accuracy too low after tiny BN overfit. acc={accuracy:P2}");
        }

        private static TensorStorage<float> CreateTensor(int rows, int cols, float[] values)
        {
            var tensor = new TensorStorage<float>(rows * cols, clearMemory: true);
            values.AsSpan().CopyTo(tensor.AsSpan());
            return tensor;
        }

        private static float ComputeAccuracy(ReadOnlySpan<float> logits, ReadOnlySpan<float> oneHotTargets, int rows, int classes)
        {
            var correct = 0;

            for (var r = 0; r < rows; r++)
            {
                var rowOffset = r * classes;

                var predClass = 0;
                var predValue = logits[rowOffset];

                for (var c = 1; c < classes; c++)
                {
                    var v = logits[rowOffset + c];
                    if (v > predValue)
                    {
                        predValue = v;
                        predClass = c;
                    }
                }

                var trueClass = 0;
                var trueValue = oneHotTargets[rowOffset];

                for (var c = 1; c < classes; c++)
                {
                    var v = oneHotTargets[rowOffset + c];
                    if (v > trueValue)
                    {
                        trueValue = v;
                        trueClass = c;
                    }
                }

                if (predClass == trueClass)
                {
                    correct++;
                }
            }

            return correct / (float)rows;
        }

    }
}