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

            // Backward
            output.Grad.AsSpan().Fill(1f);
            ComputationGraph.Active.Backward(output);

            // Bias jest 1D - używamy indeksu [i]
            Assert.Equal(2f, bias.Grad[0]);
            Assert.Equal(2f, bias.Grad[1]);
        }

        [Fact]
        public void BatchNorm1D_NormalizationTheory()
        {
            // Batch=3, Features=1 (Tensor 2D)
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

            // Output jest 2D - używamy indeksu [i, j]
            // Średnia z 10,20,30 to 20. Wartość 20 (indeks [1,0]) powinna stać się 0.
            Assert.InRange(output.Data[1, 0], -0.01f, 0.01f);

            // Symetria: -1.22 i 1.22
            Assert.Equal(-output.Data[0, 0], output.Data[2, 0], 3);
        }

        [Fact]
        public void SoftmaxCrossEntropy_ForwardAndBackward_MathematicalTruth()
        {
            using var logits = new AutogradNode(new FastTensor<float>(1, 3));
            using var target = new AutogradNode(new FastTensor<float>(1, 3), false);

            var lS = logits.Data.AsSpan();
            lS[0] = 2.0f; lS[1] = 1.0f; lS[2] = 0.1f;
            target.Data[0, 0] = 1.0f;

            using var loss = TensorMath.SoftmaxCrossEntropy(logits, target);

            // Loss dla Softmaxa (1, 3)
            Assert.InRange(loss.Data[0, 0], 0.41f, 0.43f);

            ComputationGraph.Active.Backward(loss);

            // Gradient (Pred - Target)
            Assert.InRange(logits.Grad[0, 0], -0.35f, -0.33f);
            Assert.InRange(logits.Grad[0, 1], 0.23f, 0.25f);
        }

        [Fact]
        public void Reshape_ZeroCopy_Integrity()
        {
            using var tensor = new FastTensor<float>(1, 8, 13, 13);
            tensor.AsSpan().Fill(7f);

            // Reshape 4D -> 2D
            using var reshaped = tensor.Reshape(1, 1352);

            Assert.Equal(1352, reshaped.Shape[1]);
            // Dostęp do spłaszczonego tensora 2D
            Assert.Equal(7f, reshaped[0, 1351]);
        }

        [Fact]
        public void MatMul_FullGradientCheck_MathematicalTruth()
        {
            // Test mnożenia macierzy [2x3] * [3x2] = [2x2]
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

            // Grad A powinien wynosić sumę wierszy B (2+2=4)
            Assert.Equal(4f, a.Grad[0, 0]);
            // Grad B powinien wynosić sumę kolumn A (1+1=2)
            Assert.Equal(2f, b.Grad[0, 0]);
        }

        [Fact]
        public void Dropout_Scaling_And_InferenceMode()
        {
            using var input = new AutogradNode(new FastTensor<float>(1, 100));
            input.Data.AsSpan().Fill(1f);

            // Tryb treningowy: wartości powinny być skalowane o 1/(1-p)
            using var outputTrain = TensorMath.Dropout(input, 0.5f, isTraining: true);
            var sumTrain = TensorPrimitives.Sum(outputTrain.Data.AsSpan());

            // Statystycznie suma powinna oscylować wokół 100 (50*0 + 50*2)
            Assert.InRange(sumTrain, 60f, 140f);

            // Tryb inferencji: Dropout musi być przezroczysty
            using var outputEval = TensorMath.Dropout(input, 0.5f, isTraining: false);
            Assert.Equal(100f, TensorPrimitives.Sum(outputEval.Data.AsSpan()));
        }

        [Fact]
        public void GlobalAveragePool2D_NCHW_Reduction_Check()
        {
            // Wejście NCHW: 1 batch, 2 kanały, 2x2 przestrzeń
            using var input = new AutogradNode(new FastTensor<float>(1, 2, 2, 2));
            var inS = input.Data.AsSpan();
            inS.Slice(0, 4).Fill(10f); // Kanał 0
            inS.Slice(4, 4).Fill(20f); // Kanał 1

            using var output = TensorMath.GlobalAveragePool2D(input, 2, 2, 2);

            // Wynik powinien mieć kształt [1, 2]
            Assert.Equal(10f, output.Data[0, 0]);
            Assert.Equal(20f, output.Data[0, 1]);
        }

        [Fact]
        public void MSELoss_MathematicalGradient_Check()
        {
            using var pred = new AutogradNode(new FastTensor<float>(1, 2));
            using var target = new AutogradNode(new FastTensor<float>(1, 2), false);

            pred.Data[0, 0] = 10f;
            target.Data[0, 0] = 5f;

            // MSE = ((10-5)^2 + 0) / 2 = 12.5
            using var loss = TensorMath.MSELoss(pred, target);
            Assert.Equal(12.5f, loss.Data[0, 0]);

            ComputationGraph.Active.Backward(loss);

            // Gradient MSE: (2/n) * (pred - target) = (2/2) * (10 - 5) = 5
            Assert.Equal(5f, pred.Grad[0, 0]);
        }

        [Fact]
        public void NumericalCheck_MatMul_ShouldBePrecise()
        {
            // Małe wymiary, by test trwał krótko
            using var a = new AutogradNode(new FastTensor<float>(2, 3));
            using var b = new AutogradNode(new FastTensor<float>(3, 2));

            // Inicjalizacja małymi wartościami, by uniknąć nasycenia
            for (int i = 0; i < a.Data.Size; i++) a.Data.AsSpan()[i] = (i + 1) * 0.1f;
            for (int i = 0; i < b.Data.Size; i++) b.Data.AsSpan()[i] = (i + 1) * 0.2f;

            // Funkcja straty: suma wszystkich elementów wyniku mnożenia macierzy
            Func<AutogradNode> lossFunc = () => {
                var mm = TensorMath.MatMul(a, b);
                var ones = new AutogradNode(new FastTensor<float>(2, 2), false);
                ones.Data.AsSpan().Fill(1f);
                return TensorMath.MSELoss(mm, ones); // Używamy MSE, by dostać skalar
            };

            VerifyGradients(lossFunc, a);
            VerifyGradients(lossFunc, b);
        }

        [Fact]
        public void NumericalCheck_Conv2D_ShouldBePrecise()
        {
            // Test splotu: obraz 4x4, filtr 3x3
            using var input = new AutogradNode(new FastTensor<float>(1, 1, 4, 4));
            using var weights = new AutogradNode(new FastTensor<float>(1, 9)); // 1 outC * (3*3*1)

            input.Data.AsSpan().Fill(0.5f);
            weights.Data.AsSpan().Fill(0.1f);

            Func<AutogradNode> lossFunc = () => {
                var conv = TensorMath.Conv2D(input, weights, 1, 1, 4, 4, 3);
                var target = new AutogradNode(new FastTensor<float>(1, 1, 2, 2), false);
                return TensorMath.MSELoss(conv, target);
            };

            // Sprawdzamy gradienty dla wag (filtrów)
            VerifyGradients(lossFunc, weights);
            // Sprawdzamy gradienty dla wejścia (obrazu)
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

            Func<AutogradNode> lossFunc = () => {
                var bn = TensorMath.BatchNorm1D(input, gamma, beta, rm, rv, 0.1f, 1e-5f, true);
                var target = new AutogradNode(new FastTensor<float>(4, 2), false);
                return TensorMath.MSELoss(bn, target);
            };

            VerifyGradients(lossFunc, gamma);
            VerifyGradients(lossFunc, beta);
            VerifyGradients(lossFunc, input);
        }

        private void VerifyGradients(Func<AutogradNode> lossFunc, AutogradNode parameter, float epsilon = 1e-4f, float tolerance = 1e-3f)
        {
            // 1. Obliczamy gradient analityczny (Twój autograd)
            ComputationGraph.Active.Reset();
            using var lossNode = lossFunc();
            ComputationGraph.Active.Backward(lossNode);

            // Kopiujemy gradienty do tablicy, aby ich nie nadpisać
            var analyticalGrads = parameter.Grad.AsSpan().ToArray();

            // 2. Obliczamy gradient numeryczny metodą różnic skończonych: [f(x+e) - f(x-e)] / (2e)
            var numGrads = new float[parameter.Data.Size];
            var dataSpan = parameter.Data.AsSpan();

            for (int i = 0; i < parameter.Data.Size; i++)
            {
                float originalValue = dataSpan[i];

                // f(x + epsilon)
                dataSpan[i] = originalValue + epsilon;
                ComputationGraph.Active.Reset();
                using (var lossPlus = lossFunc())
                {
                    float fPlus = lossPlus.Data[0, 0];

                    // f(x - epsilon)
                    dataSpan[i] = originalValue - epsilon;
                    ComputationGraph.Active.Reset();
                    using (var lossMinus = lossFunc())
                    {
                        float fMinus = lossMinus.Data[0, 0];

                        // Przybliżenie pochodnej
                        numGrads[i] = (fPlus - fMinus) / (2 * epsilon);
                    }
                }

                // Przywracamy oryginalną wartość parametru
                dataSpan[i] = originalValue;
            }

            // 3. Porównujemy wyniki
            for (int i = 0; i < analyticalGrads.Length; i++)
            {
                float diff = Math.Abs(analyticalGrads[i] - numGrads[i]);
                // Jeśli różnica jest zbyt duża, test padnie
                Assert.True(diff < tolerance,
                    $"Błąd gradientu w elemencie {i}! Analityczny: {analyticalGrads[i]}, Numeryczny: {numGrads[i]}, Różnica: {diff}");
            }
        }
    }
}