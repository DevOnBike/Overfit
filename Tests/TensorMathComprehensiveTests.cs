// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Numerics.Tensors;
using DevOnBike.Overfit.Core;

namespace DevOnBike.Overfit.Tests
{
    public class TensorMathComprehensiveTests
    {
        // Prywatna instancja grafu na potrzeby weryfikacji operacji
        private readonly ComputationGraph _graph = new();

        // ====================================================================
        // 1. TESTY LOGIKI MATEMATYCZNEJ (FORWARD & BACKWARD THEORY)
        // ====================================================================

        [Fact]
        public void AddBias_ForwardAndBackward_NCHW_Correct()
        {
            using var input = new AutogradNode(new FastTensor<float>(2, 3, 1, 1));
            using var bias = new AutogradNode(new FastTensor<float>(3));

            input.Data.AsSpan().Fill(1f);
            var bSpan = bias.Data.AsSpan();
            bSpan[0] = 10f;
            bSpan[1] = 20f;
            bSpan[2] = 30f;

            using var output = TensorMath.AddBias(_graph, input, bias);

            Assert.Equal(11f, output.Data[0, 0, 0, 0]);
            Assert.Equal(21f, output.Data[0, 1, 0, 0]);
            Assert.Equal(31f, output.Data[1, 2, 0, 0]);

            output.Grad.AsSpan().Fill(1f);
            _graph.Backward(output);

            Assert.Equal(2f, bias.Grad[0]);
            Assert.Equal(2f, bias.Grad[1]);
        }

        [Fact]
        public void MatMul_FullGradientCheck_MathematicalTruth()
        {
            using var a = new AutogradNode(new FastTensor<float>(2, 3));
            using var b = new AutogradNode(new FastTensor<float>(3, 2));

            a.Data.AsSpan().Fill(1f);
            b.Data.AsSpan().Fill(2f);

            using var output = TensorMath.MatMul(_graph, a, b);

            Assert.Equal(6f, output.Data[0, 0]);

            output.Grad.AsSpan().Fill(1f);
            _graph.Backward(output);

            Assert.Equal(4f, a.Grad[0, 0]);
            Assert.Equal(2f, b.Grad[0, 0]);
        }

        [Fact]
        public void ReLU_GradientMasking_MathematicalTruth()
        {
            using var input = new AutogradNode(new FastTensor<float>(1, 4));
            input.Data.AsSpan()[0] = -5f;
            input.Data.AsSpan()[1] = 0f;
            input.Data.AsSpan()[2] = 5f;
            input.Data.AsSpan()[3] = 10f;

            using var output = TensorMath.ReLU(_graph, input);
            Assert.Equal(0f, output.Data[0, 0]);
            Assert.Equal(0f, output.Data[0, 1]);
            Assert.Equal(5f, output.Data[0, 2]);

            _graph.Backward(output);

            Assert.Equal(0f, input.Grad[0, 0]);
            Assert.Equal(0f, input.Grad[0, 1]);
            Assert.Equal(1f, input.Grad[0, 2]);
            Assert.Equal(1f, input.Grad[0, 3]);
        }

        [Fact]
        public void MaxPool2D_GradientRouting_MathematicalTruth()
        {
            using var input = new AutogradNode(new FastTensor<float>(1, 1, 2, 2));
            var span = input.Data.AsSpan();
            span[0] = 1f;
            span[1] = 2f;
            span[2] = 3f;
            span[3] = 10f;

            using var output = TensorMath.MaxPool2D(_graph, input, 1, 2, 2, 2);
            Assert.Equal(10f, output.Data[0, 0, 0, 0]);

            _graph.Backward(output);

            Assert.Equal(0f, input.Grad[0, 0, 0, 0]);
            Assert.Equal(0f, input.Grad[0, 0, 0, 1]);
            Assert.Equal(0f, input.Grad[0, 0, 1, 0]);
            Assert.Equal(1f, input.Grad[0, 0, 1, 1]);
        }

        [Fact]
        public void GlobalAveragePool2D_NCHW_Reduction_Check()
        {
            using var input = new AutogradNode(new FastTensor<float>(1, 2, 2, 2));
            var inS = input.Data.AsSpan();
            inS.Slice(0, 4).Fill(10f);
            inS.Slice(4, 4).Fill(20f);

            using var output = TensorMath.GlobalAveragePool2D(_graph, input, 2, 2, 2);

            Assert.Equal(10f, output.Data[0, 0]);
            Assert.Equal(20f, output.Data[0, 1]);
        }

        [Fact]
        public void BatchNorm1D_NormalizationTheory()
        {
            using var input = new AutogradNode(new FastTensor<float>(3, 1));
            var inS = input.Data.AsSpan();
            inS[0] = 10f;
            inS[1] = 20f;
            inS[2] = 30f;

            using var gamma = new AutogradNode(new FastTensor<float>(1));
            using var beta = new AutogradNode(new FastTensor<float>(1));
            gamma.Data.AsSpan().Fill(1f);
            beta.Data.AsSpan().Fill(0f);

            using var rm = new FastTensor<float>(1);
            using var rv = new FastTensor<float>(1);
            rv.AsSpan().Fill(1f);

            using var output = TensorMath.BatchNorm1D(_graph, input, gamma, beta, rm, rv, 0.1f, 1e-5f, true);

            Assert.InRange(output.Data[1, 0], -0.01f, 0.01f);
            Assert.Equal(-output.Data[0, 0], output.Data[2, 0], 3);
        }

        [Fact]
        public void SoftmaxCrossEntropy_MathematicalTruth()
        {
            using var logits = new AutogradNode(new FastTensor<float>(1, 3));
            using var target = new AutogradNode(new FastTensor<float>(1, 3), false);

            var lS = logits.Data.AsSpan();
            lS[0] = 2.0f;
            lS[1] = 1.0f;
            lS[2] = 0.1f;
            target.Data[0, 0] = 1.0f;

            using var loss = TensorMath.SoftmaxCrossEntropy(_graph, logits, target);

            Assert.InRange(loss.Data[0, 0], 0.41f, 0.43f);

            _graph.Backward(loss);

            Assert.InRange(logits.Grad[0, 0], -0.35f, -0.33f);
            Assert.InRange(logits.Grad[0, 1], 0.23f, 0.25f);
        }

        [Fact]
        public void Dropout_Scaling_And_InferenceMode()
        {
            using var input = new AutogradNode(new FastTensor<float>(1, 100));
            input.Data.AsSpan().Fill(1f);

            using var outputTrain = TensorMath.Dropout(_graph, input, 0.5f, true);
            var sumTrain = TensorPrimitives.Sum(outputTrain.Data.AsSpan());

            Assert.InRange(sumTrain, 60f, 140f);

            // ZMIANA: Tryb ewaluacji przyjmuje NULL
            using var outputEval = TensorMath.Dropout(null, input, 0.5f, false);
            Assert.Equal(100f, TensorPrimitives.Sum(outputEval.Data.AsSpan()));
        }

        [Fact]
        public void MSELoss_MathematicalGradient_Check()
        {
            using var pred = new AutogradNode(new FastTensor<float>(1, 2));
            using var target = new AutogradNode(new FastTensor<float>(1, 2), false);

            pred.Data[0, 0] = 10f;
            target.Data[0, 0] = 5f;

            using var loss = TensorMath.MSELoss(_graph, pred, target);
            Assert.Equal(12.5f, loss.Data[0, 0]);

            _graph.Backward(loss);

            Assert.Equal(5f, pred.Grad[0, 0]);
        }

        [Fact]
        public void Reshape_ZeroCopy_And_Gradient_Flows()
        {
            using var tensor = new AutogradNode(new FastTensor<float>(1, 8, 13, 13));
            tensor.Data.AsSpan().Fill(7f);

            using var reshaped = TensorMath.Reshape(_graph, tensor, 1, 1352);

            Assert.Equal(1352, reshaped.Data.Shape[1]);
            Assert.Equal(7f, reshaped.Data[0, 1351]);

            _graph.Backward(reshaped);

            Assert.Equal(1f, tensor.Grad.AsSpan()[0]);
            Assert.Equal(1f, tensor.Grad.AsSpan()[1351]);
        }

        // ====================================================================
        // 2. TESTY NUMERYCZNE (CZYSTA PRAWDA RÓŻNICZKOWA)
        // ====================================================================

        [Fact]
        public void NumericalCheck_MatMul_ShouldBePrecise()
        {
            using var a = new AutogradNode(new FastTensor<float>(2, 3));
            using var b = new AutogradNode(new FastTensor<float>(3, 2));

            for (var i = 0; i < a.Data.Size; i++)
            {
                a.Data.AsSpan()[i] = (i + 1) * 0.1f;
            }
            for (var i = 0; i < b.Data.Size; i++)
            {
                b.Data.AsSpan()[i] = (i + 1) * 0.2f;
            }

            var lossFunc = () =>
            {
                var mm = TensorMath.MatMul(_graph, a, b);
                var target = new AutogradNode(new FastTensor<float>(2, 2), false);
                return TensorMath.MSELoss(_graph, mm, target);
            };

            VerifyGradients(lossFunc, a);
            VerifyGradients(lossFunc, b);
        }

        [Fact]
        public void NumericalCheck_AddBias_ShouldBePrecise()
        {
            using var input = new AutogradNode(new FastTensor<float>(3, 2));
            using var bias = new AutogradNode(new FastTensor<float>(2));

            input.Data.AsSpan().Fill(1.5f);
            bias.Data.AsSpan().Fill(0.5f);

            var lossFunc = () =>
            {
                var res = TensorMath.AddBias(_graph, input, bias);
                var target = new AutogradNode(new FastTensor<float>(3, 2), false);
                return TensorMath.MSELoss(_graph, res, target);
            };

            VerifyGradients(lossFunc, input);
            VerifyGradients(lossFunc, bias);
        }

        [Fact]
        public void NumericalCheck_Linear_ShouldBePrecise()
        {
            using var input = new AutogradNode(new FastTensor<float>(2, 3));
            using var weights = new AutogradNode(new FastTensor<float>(3, 2));
            using var bias = new AutogradNode(new FastTensor<float>(2));

            for (var i = 0; i < input.Data.Size; i++)
            {
                input.Data.AsSpan()[i] = (i + 1) * 0.1f;
            }
            for (var i = 0; i < weights.Data.Size; i++)
            {
                weights.Data.AsSpan()[i] = (i + 1) * 0.2f;
            }
            for (var i = 0; i < bias.Data.Size; i++)
            {
                bias.Data.AsSpan()[i] = (i + 1) * 0.3f;
            }

            var lossFunc = () =>
            {
                var lin = TensorMath.Linear(_graph, input, weights, bias);
                var target = new AutogradNode(new FastTensor<float>(2, 2), false);
                return TensorMath.MSELoss(_graph, lin, target);
            };

            VerifyGradients(lossFunc, input);
            VerifyGradients(lossFunc, weights);
            VerifyGradients(lossFunc, bias);
        }

        [Fact]
        public void NumericalCheck_Conv2D_ShouldBePrecise()
        {
            using var input = new AutogradNode(new FastTensor<float>(1, 1, 4, 4));
            using var weights = new AutogradNode(new FastTensor<float>(1, 9));

            input.Data.AsSpan().Fill(0.5f);
            weights.Data.AsSpan().Fill(0.1f);

            var lossFunc = () =>
            {
                var conv = TensorMath.Conv2D(_graph, input, weights, 1, 1, 4, 4, 3);
                var target = new AutogradNode(new FastTensor<float>(1, 1, 2, 2), false);
                return TensorMath.MSELoss(_graph, conv, target);
            };

            VerifyGradients(lossFunc, weights);
            VerifyGradients(lossFunc, input);
        }

        [Fact]
        public void NumericalCheck_BatchNorm1D_ShouldBePrecise()
        {
            using var input = new AutogradNode(new FastTensor<float>(4, 2));
            using var gamma = new AutogradNode(new FastTensor<float>(2));
            using var beta = new AutogradNode(new FastTensor<float>(2));

            input.Data.AsSpan().Fill(1.5f);
            gamma.Data.AsSpan().Fill(1.0f);
            beta.Data.AsSpan().Fill(0.0f);

            using var rm = new FastTensor<float>(2);
            using var rv = new FastTensor<float>(2);
            rv.AsSpan().Fill(1f);

            var lossFunc = () =>
            {
                var bn = TensorMath.BatchNorm1D(_graph, input, gamma, beta, rm, rv, 0.1f, 1e-5f, true);
                var target = new AutogradNode(new FastTensor<float>(4, 2), false);
                return TensorMath.MSELoss(_graph, bn, target);
            };

            VerifyGradients(lossFunc, gamma);
            VerifyGradients(lossFunc, beta);
            VerifyGradients(lossFunc, input);
        }

        [Fact]
        public void DirectionalLoss_MathematicalGradient_Check()
        {
            using var pred = new AutogradNode(new FastTensor<float>(1, 4));
            using var target = new AutogradNode(new FastTensor<float>(1, 4), false);

            var pData = pred.Data.AsSpan();
            var tData = target.Data.AsSpan();

            pData[0] = 1f;
            tData[0] = 2f;
            pData[1] = 1f;
            tData[1] = -2f;
            pData[2] = -1f;
            tData[2] = -2f;
            pData[3] = -1f;
            tData[3] = 2f;

            using var loss = TensorMath.DirectionalLoss(_graph, pred, target);
            Assert.Equal(15f, loss.Data[0, 0]);

            _graph.Backward(loss);

            var pGrad = pred.Grad.AsSpan();
            Assert.Equal(-0.5f, pGrad[0]);
            Assert.Equal(6.5f, pGrad[1]);
            Assert.Equal(0.5f, pGrad[2]);
            Assert.Equal(-6.5f, pGrad[3]);
        }

        [Fact]
        public void NumericalCheck_DirectionalLoss_ShouldBePrecise()
        {
            using var pred = new AutogradNode(new FastTensor<float>(2, 3));
            using var target = new AutogradNode(new FastTensor<float>(2, 3), false);

            var pData = pred.Data.AsSpan();
            var tData = target.Data.AsSpan();

            pData[0] = 0.5f;
            tData[0] = 0.8f;
            pData[1] = -0.5f;
            tData[1] = 0.8f;
            pData[2] = 1.2f;
            tData[2] = -0.3f;
            pData[3] = -1.0f;
            tData[3] = -1.5f;
            pData[4] = 0.1f;
            tData[4] = 0.5f;
            pData[5] = 2.0f;
            tData[5] = 2.5f;

            var lossFunc = () => TensorMath.DirectionalLoss(_graph, pred, target);

            VerifyGradients(lossFunc, pred);
        }

        // ====================================================================
        // HELPER: NUMERICAL GRADIENT CHECKER
        // ====================================================================

        private void VerifyGradients(Func<AutogradNode> lossFunc, AutogradNode parameter, float epsilon = 1e-3f, float tolerance = 1e-2f)
        {
            // Zerujemy gradient parametru — ten sam kontrakt co optimizer.ZeroGrad()
            // Bez tego gradient kumuluje się z poprzednich wywołań VerifyGradients
            parameter.Grad.AsSpan().Clear();

            _graph.Reset();
            using var lossNode = lossFunc();
            _graph.Backward(lossNode);

            var analyticalGrads = parameter.Grad.AsSpan().ToArray();
            var numGrads = new float[parameter.Data.Size];
            var dataSpan = parameter.Data.AsSpan();

            for (var i = 0; i < parameter.Data.Size; i++)
            {
                var originalValue = dataSpan[i];

                dataSpan[i] = originalValue + epsilon;
                _graph.Reset();
                using (var lossPlus = lossFunc())
                {
                    var fPlus = lossPlus.Data[0, 0];

                    dataSpan[i] = originalValue - epsilon;
                    _graph.Reset();
                    using (var lossMinus = lossFunc())
                    {
                        var fMinus = lossMinus.Data[0, 0];
                        numGrads[i] = (fPlus - fMinus) / (2 * epsilon);
                    }
                }
                dataSpan[i] = originalValue;
            }

            for (var i = 0; i < analyticalGrads.Length; i++)
            {
                var diff = Math.Abs(analyticalGrads[i] - numGrads[i]);
                Assert.True(diff < tolerance,
                $"Błąd gradientu w elemencie {i}! Analityczny: {analyticalGrads[i]}, Numeryczny: {numGrads[i]}, Różnica: {diff}");
            }
        }
    }
}