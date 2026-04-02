using DevOnBike.Overfit.Core;
using System.Numerics.Tensors;

namespace DevOnBike.Overfit.Tests
{
    public class TensorMathComprehensiveTests : IDisposable
    {
        public TensorMathComprehensiveTests()
        {
            ComputationGraph.Active = new ComputationGraph();
        }

        public void Dispose() => ComputationGraph.Active = null;

        // ====================================================================
        // 1. TESTY LOGIKI MATEMATYCZNEJ (FORWARD & BACKWARD THEORY)
        // ====================================================================

        [Fact]
        public void AddBias_ForwardAndBackward_NCHW_Correct()
        {
            using var input = new AutogradNode(new FastTensor<float>(2, 3, 1, 1));
            using var bias = new AutogradNode(new FastTensor<float>(3)); // Tensor 1D

            input.Data.AsSpan().Fill(1f);
            var bSpan = bias.Data.AsSpan();
            bSpan[0] = 10f; bSpan[1] = 20f; bSpan[2] = 30f;

            using var output = TensorMath.AddBias(input, bias);

            // Forward: Output jest 4D
            Assert.Equal(11f, output.Data[0, 0, 0, 0]);
            Assert.Equal(21f, output.Data[0, 1, 0, 0]);
            Assert.Equal(31f, output.Data[1, 2, 0, 0]);

            // Backward: Gradient biasu jest sumą po wymiarze batcha
            output.Grad.AsSpan().Fill(1f);
            ComputationGraph.Active.Backward(output);

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

            using var output = TensorMath.MatMul(a, b);

            // Forward: każdy element to 1*2 + 1*2 + 1*2 = 6
            Assert.Equal(6f, output.Data[0, 0]);

            // Backward: Grad A = GradOut * B^T
            output.Grad.AsSpan().Fill(1f);
            ComputationGraph.Active.Backward(output);

            Assert.Equal(4f, a.Grad[0, 0]); // 2 + 2
            Assert.Equal(2f, b.Grad[0, 0]); // 1 + 1
        }

        [Fact]
        public void ReLU_GradientMasking_MathematicalTruth()
        {
            using var input = new AutogradNode(new FastTensor<float>(1, 4));
            input.Data.AsSpan()[0] = -5f;
            input.Data.AsSpan()[1] = 0f;
            input.Data.AsSpan()[2] = 5f;
            input.Data.AsSpan()[3] = 10f;

            using var output = TensorMath.ReLU(input);
            Assert.Equal(0f, output.Data[0, 0]);
            Assert.Equal(0f, output.Data[0, 1]);
            Assert.Equal(5f, output.Data[0, 2]);

            ComputationGraph.Active.Backward(output);

            // Elementy <= 0 muszą maskować gradient na 0
            Assert.Equal(0f, input.Grad[0, 0]);
            Assert.Equal(0f, input.Grad[0, 1]);
            // Elementy > 0 przepuszczają standardowy gradient z Backward (1f)
            Assert.Equal(1f, input.Grad[0, 2]);
            Assert.Equal(1f, input.Grad[0, 3]);
        }

        [Fact]
        public void MaxPool2D_GradientRouting_MathematicalTruth()
        {
            using var input = new AutogradNode(new FastTensor<float>(1, 1, 2, 2));
            var span = input.Data.AsSpan();
            span[0] = 1f; span[1] = 2f;
            span[2] = 3f; span[3] = 10f; // Max element

            using var output = TensorMath.MaxPool2D(input, 1, 2, 2, 2);
            Assert.Equal(10f, output.Data[0, 0, 0, 0]);

            ComputationGraph.Active.Backward(output);

            // Gradient (1f) może popłynąć TYLKO do elementu, który wygrał Pooling
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
            inS.Slice(0, 4).Fill(10f); // Kanał 0
            inS.Slice(4, 4).Fill(20f); // Kanał 1

            using var output = TensorMath.GlobalAveragePool2D(input, 2, 2, 2);

            Assert.Equal(10f, output.Data[0, 0]);
            Assert.Equal(20f, output.Data[0, 1]);
        }

        [Fact]
        public void BatchNorm1D_NormalizationTheory()
        {
            using var input = new AutogradNode(new FastTensor<float>(3, 1));
            var inS = input.Data.AsSpan();
            inS[0] = 10f; inS[1] = 20f; inS[2] = 30f;

            using var gamma = new AutogradNode(new FastTensor<float>(1));
            using var beta = new AutogradNode(new FastTensor<float>(1));
            gamma.Data.AsSpan().Fill(1f);
            beta.Data.AsSpan().Fill(0f);

            using var rm = new FastTensor<float>(1);
            using var rv = new FastTensor<float>(1);
            rv.AsSpan().Fill(1f);

            using var output = TensorMath.BatchNorm1D(input, gamma, beta, rm, rv, 0.1f, 1e-5f, true);

            // Średnia to 20. Wartość 20 powinna znormalizować się do 0
            Assert.InRange(output.Data[1, 0], -0.01f, 0.01f);
            // Elementy 10 i 30 powinny być symetryczne wokół zera
            Assert.Equal(-output.Data[0, 0], output.Data[2, 0], 3);
        }

        [Fact]
        public void SoftmaxCrossEntropy_MathematicalTruth()
        {
            using var logits = new AutogradNode(new FastTensor<float>(1, 3));
            using var target = new AutogradNode(new FastTensor<float>(1, 3), false);

            var lS = logits.Data.AsSpan();
            lS[0] = 2.0f; lS[1] = 1.0f; lS[2] = 0.1f;
            target.Data[0, 0] = 1.0f;

            using var loss = TensorMath.SoftmaxCrossEntropy(logits, target);

            Assert.InRange(loss.Data[0, 0], 0.41f, 0.43f);

            ComputationGraph.Active.Backward(loss);

            // Pochodna CE to (Pred - Target)
            Assert.InRange(logits.Grad[0, 0], -0.35f, -0.33f);
            Assert.InRange(logits.Grad[0, 1], 0.23f, 0.25f);
        }

        [Fact]
        public void Dropout_Scaling_And_InferenceMode()
        {
            using var input = new AutogradNode(new FastTensor<float>(1, 100));
            input.Data.AsSpan().Fill(1f);

            using var outputTrain = TensorMath.Dropout(input, 0.5f, isTraining: true);
            var sumTrain = TensorPrimitives.Sum(outputTrain.Data.AsSpan());

            Assert.InRange(sumTrain, 60f, 140f);

            using var outputEval = TensorMath.Dropout(input, 0.5f, isTraining: false);
            Assert.Equal(100f, TensorPrimitives.Sum(outputEval.Data.AsSpan()));
        }

        [Fact]
        public void MSELoss_MathematicalGradient_Check()
        {
            using var pred = new AutogradNode(new FastTensor<float>(1, 2));
            using var target = new AutogradNode(new FastTensor<float>(1, 2), false);

            pred.Data[0, 0] = 10f;
            target.Data[0, 0] = 5f;

            using var loss = TensorMath.MSELoss(pred, target);
            Assert.Equal(12.5f, loss.Data[0, 0]);

            ComputationGraph.Active.Backward(loss);

            // Grad = (2/N) * (Pred - Target)
            Assert.Equal(5f, pred.Grad[0, 0]);
        }

        [Fact]
        public void Reshape_ZeroCopy_And_Gradient_Flows()
        {
            using var tensor = new AutogradNode(new FastTensor<float>(1, 8, 13, 13));
            tensor.Data.AsSpan().Fill(7f);

            using var reshaped = TensorMath.Reshape(tensor, 1, 1352);

            Assert.Equal(1352, reshaped.Data.Shape[1]);
            Assert.Equal(7f, reshaped.Data[0, 1351]);

            ComputationGraph.Active.Backward(reshaped);

            // Sprawdzamy czy Reshape przepuszcza domyślny gradient 1f
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

            for (var i = 0; i < a.Data.Size; i++) a.Data.AsSpan()[i] = (i + 1) * 0.1f;
            for (var i = 0; i < b.Data.Size; i++) b.Data.AsSpan()[i] = (i + 1) * 0.2f;

            var lossFunc = () => {
                var mm = TensorMath.MatMul(a, b);
                var target = new AutogradNode(new FastTensor<float>(2, 2), false);
                return TensorMath.MSELoss(mm, target);
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

            var lossFunc = () => {
                var res = TensorMath.AddBias(input, bias);
                var target = new AutogradNode(new FastTensor<float>(3, 2), false);
                return TensorMath.MSELoss(res, target);
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

            for (var i = 0; i < input.Data.Size; i++) input.Data.AsSpan()[i] = (i + 1) * 0.1f;
            for (var i = 0; i < weights.Data.Size; i++) weights.Data.AsSpan()[i] = (i + 1) * 0.2f;
            for (var i = 0; i < bias.Data.Size; i++) bias.Data.AsSpan()[i] = (i + 1) * 0.3f;

            var lossFunc = () => {
                var lin = TensorMath.Linear(input, weights, bias);
                var target = new AutogradNode(new FastTensor<float>(2, 2), false);
                return TensorMath.MSELoss(lin, target);
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

            var lossFunc = () => {
                var conv = TensorMath.Conv2D(input, weights, 1, 1, 4, 4, 3);
                var target = new AutogradNode(new FastTensor<float>(1, 1, 2, 2), false);
                return TensorMath.MSELoss(conv, target);
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

            var lossFunc = () => {
                var bn = TensorMath.BatchNorm1D(input, gamma, beta, rm, rv, 0.1f, 1e-5f, true);
                var target = new AutogradNode(new FastTensor<float>(4, 2), false);
                return TensorMath.MSELoss(bn, target);
            };

            VerifyGradients(lossFunc, gamma);
            VerifyGradients(lossFunc, beta);
            VerifyGradients(lossFunc, input);
        }

        // ====================================================================
        // HELPER: NUMERICAL GRADIENT CHECKER
        // ====================================================================

        private void VerifyGradients(Func<AutogradNode> lossFunc, AutogradNode parameter, float epsilon = 1e-4f, float tolerance = 2e-3f)
        {
            ComputationGraph.Active.Reset();
            using var lossNode = lossFunc();
            ComputationGraph.Active.Backward(lossNode);

            var analyticalGrads = parameter.Grad.AsSpan().ToArray();
            var numGrads = new float[parameter.Data.Size];
            var dataSpan = parameter.Data.AsSpan();

            for (var i = 0; i < parameter.Data.Size; i++)
            {
                var originalValue = dataSpan[i];

                dataSpan[i] = originalValue + epsilon;
                ComputationGraph.Active.Reset();
                using (var lossPlus = lossFunc())
                {
                    var fPlus = lossPlus.Data[0, 0];

                    dataSpan[i] = originalValue - epsilon;
                    ComputationGraph.Active.Reset();
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