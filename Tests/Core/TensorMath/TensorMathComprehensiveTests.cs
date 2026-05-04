// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Numerics.Tensors;
using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.Ops;
using DevOnBike.Overfit.Tensors;
using DevOnBike.Overfit.Tensors.Core; // Zmieniono na Tensors.Core

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
            using var inputTensor = new TensorStorage<float>(2 * 3 * 1 * 1, clearMemory: true);
            using var biasTensor = new TensorStorage<float>(3, clearMemory: true);
            using var input = new AutogradNode(inputTensor, new TensorShape(2, 3, 1, 1), requiresGrad: true);
            using var bias = new AutogradNode(biasTensor, new TensorShape(3), requiresGrad: true);

            input.DataView.AsSpan().Fill(1f);
            var bSpan = bias.DataView.AsSpan();
            bSpan[0] = 10f;
            bSpan[1] = 20f;
            bSpan[2] = 30f;

            using var output = TensorMath.AddBias(_graph, input, bias);

            var outSpan = output.DataView.AsSpan();
            Assert.Equal(11f, outSpan[0]); // Batch 0, Ch 0
            Assert.Equal(21f, outSpan[1]); // Batch 0, Ch 1
            Assert.Equal(31f, outSpan[2]); // Batch 0, Ch 2

            output.GradView.AsSpan().Fill(1f);
            _graph.Backward(output);

            var inGrad = input.GradView.AsSpan();
            var bGrad = bias.GradView.AsSpan();

            Assert.Equal(1f, inGrad[0]);
            Assert.Equal(2f, bGrad[0]); // Batch size = 2, gradient sumuje się po batchach i pikselach
            Assert.Equal(2f, bGrad[1]);
        }

        [Fact]
        public void MatMul_ForwardAndBackward_Correct()
        {
            using var inputTensor = new TensorStorage<float>(2 * 3, clearMemory: true);
            using var weightsTensor = new TensorStorage<float>(3 * 4, clearMemory: true);
            using var input = new AutogradNode(inputTensor, new TensorShape(2, 3), requiresGrad: true);
            using var weights = new AutogradNode(weightsTensor, new TensorShape(3, 4), requiresGrad: true);

            input.DataView.AsSpan().Fill(2f);
            weights.DataView.AsSpan().Fill(3f);

            using var output = TensorMath.MatMul(_graph, input, weights);

            var outSpan = output.DataView.AsSpan();
            Assert.Equal(2 * 3 * 3f, outSpan[0]); // 18

            output.GradView.AsSpan().Fill(1f);
            _graph.Backward(output);

            var inGrad = input.GradView.AsSpan();
            var wGrad = weights.GradView.AsSpan();

            Assert.Equal(4 * 3f, inGrad[0]); // 4 kolumny wag, każda waga = 3 -> grad względem wejścia = sum(grad_out * W^T)
            Assert.Equal(2 * 2f, wGrad[0]);  // 2 wiersze wejścia, każdy = 2 -> grad względem wagi = sum(X^T * grad_out)
        }

        [Fact]
        public void Conv2D_ForwardAndBackward_Correct()
        {
            int inC = 1, outC = 1, h = 3, w = 3, k = 2;
            using var inputTensor = new TensorStorage<float>(1 * inC * h * w, clearMemory: true);
            using var weightsTensor = new TensorStorage<float>(outC * inC * k * k, clearMemory: true);
            using var input = new AutogradNode(inputTensor, new TensorShape(1, inC, h, w), requiresGrad: true);
            using var weights = new AutogradNode(weightsTensor, new TensorShape(outC, inC * k * k), requiresGrad: true);

            input.DataView.AsSpan().Fill(1f);
            weights.DataView.AsSpan().Fill(2f);

            using var output = TensorMath.Conv2D(_graph, input, weights, inC, outC, h, w, k);

            var outSpan = output.DataView.AsSpan();
            Assert.Equal(4 * 2f, outSpan[0]); // 2x2 okno jedynek * 2 = 8

            output.GradView.AsSpan().Fill(1f);
            _graph.Backward(output);

            var inGrad = input.GradView.AsSpan();
            var wGrad = weights.GradView.AsSpan();

            Assert.Equal(2f, inGrad[0]); // Narożnik użyty raz
            Assert.Equal(4f, inGrad[1]); // Krawędź użyta dwa razy
            Assert.Equal(4f, wGrad[0]);  // 4 pozycje filtra
        }

        [Fact]
        public void ReLU_ForwardAndBackward_Correct()
        {
            using var inputTensor = new TensorStorage<float>(4, clearMemory: true);
            using var input = new AutogradNode(inputTensor, new TensorShape(1, 4), requiresGrad: true);
            var inSpan = input.DataView.AsSpan();
            inSpan[0] = 5.0f;
            inSpan[1] = -2.0f;
            inSpan[2] = 0.0f;
            inSpan[3] = 3.0f;

            using var output = TensorMath.ReLU(_graph, input);
            var outSpan = output.DataView.AsSpan();

            Assert.Equal(5.0f, outSpan[0]);
            Assert.Equal(0.0f, outSpan[1]);
            Assert.Equal(0.0f, outSpan[2]);
            Assert.Equal(3.0f, outSpan[3]);

            // Wywołanie Backward() automatycznie zaleje węzeł wyjściowy gradientem = 1.0f
            _graph.Backward(output);

            var gradSpan = input.GradView.AsSpan();

            // Oczekujemy teraz prądu 1.0f, bo taki generuje na starcie ComputationGraph
            Assert.Equal(1.0f, gradSpan[0]); // Przepuścił prąd 
            Assert.Equal(0.0f, gradSpan[1]); // Zablokował (ujemne)
            Assert.Equal(0.0f, gradSpan[2]); // Zablokował (zero)
            Assert.Equal(1.0f, gradSpan[3]); // Przepuścił prąd 
        }

        [Fact]
        public void SoftmaxCrossEntropy_ForwardAndBackward_Correct()
        {
            using var logitsTensor = new TensorStorage<float>(3, clearMemory: true);
            using var targetsTensor = new TensorStorage<float>(3, clearMemory: true);
            using var logits = new AutogradNode(logitsTensor, new TensorShape(1, 3), requiresGrad: true);
            using var targets = new AutogradNode(targetsTensor, new TensorShape(1, 3), requiresGrad: false);

            var lSpan = logits.DataView.AsSpan();
            lSpan[0] = 2.0f; lSpan[1] = 1.0f; lSpan[2] = 0.1f;

            var tSpan = targets.DataView.AsSpan();
            tSpan[0] = 1.0f; tSpan[1] = 0.0f; tSpan[2] = 0.0f; // Target class 0

            using var loss = TensorMath.SoftmaxCrossEntropy(_graph, logits, targets);
            Assert.True(loss.DataView.AsReadOnlySpan()[0] > 0f);

            loss.GradView.AsSpan()[0] = 1.0f;
            _graph.Backward(loss);

            var grad = logits.GradView.AsSpan();
            var sum = grad[0] + grad[1] + grad[2];

            Assert.True(Math.Abs(sum) < 1e-5f); // Suma gradientów softmax_ce wynosi ~0
            Assert.True(grad[0] < 0f); // Klasa poprawna ma gradient ujemny (chce zwiększyć wartość)
            Assert.True(grad[1] > 0f); // Błędne mają dodatni (chcą zmniejszyć)
            Assert.True(grad[2] > 0f);
        }

        [Fact]
        public void GlobalAveragePool2D_Forward_Correct()
        {
            int batch = 1, channels = 2, h = 2, w = 2;
            using var inputTensor = new TensorStorage<float>(batch * channels * h * w, clearMemory: true);
            using var input = new AutogradNode(inputTensor, new TensorShape(batch, channels, h, w), requiresGrad: false);
            var inSpan = input.DataView.AsSpan();

            // Kanał 0
            inSpan[0] = 1f; inSpan[1] = 3f; inSpan[2] = 5f; inSpan[3] = 7f; // Średnia: 4
            // Kanał 1
            inSpan[4] = 10f; inSpan[5] = 10f; inSpan[6] = 20f; inSpan[7] = 0f; // Średnia: 10

            using var output = TensorMath.GlobalAveragePool2D(_graph, input, channels, h, w);
            var outSpan = output.DataView.AsSpan();

            Assert.Equal(2, outSpan.Length); // Batch = 1, Channels = 2
            Assert.Equal(4f, outSpan[0]);
            Assert.Equal(10f, outSpan[1]);
        }

        [Fact]
        public void MSELoss_ForwardAndBackward_Correct()
        {
            using var predTensor = new TensorStorage<float>(4, clearMemory: true);
            using var targetTensor = new TensorStorage<float>(4, clearMemory: true);
            using var pred = new AutogradNode(predTensor, new TensorShape(1, 4), requiresGrad: true);
            using var target = new AutogradNode(targetTensor, new TensorShape(1, 4), requiresGrad: false);

            pred.DataView.AsSpan().Fill(2f);
            target.DataView.AsSpan().Fill(1f);

            using var loss = TensorMath.MSELoss(_graph, pred, target);

            Assert.Equal(1f, loss.DataView.AsReadOnlySpan()[0]);

            loss.GradView.AsSpan()[0] = 1f;
            _graph.Backward(loss);

            var pGrad = pred.GradView.AsSpan();
            Assert.Equal(2f * (2f - 1f) / 4f, pGrad[0]); // 0.5
        }

        [Fact]
        public void DirectionalLoss_ForwardAndBackward_Correct()
        {
            using var predTensor = new TensorStorage<float>(4, clearMemory: true);
            using var targetTensor = new TensorStorage<float>(4, clearMemory: true);
            using var pred = new AutogradNode(predTensor, new TensorShape(1, 4), requiresGrad: true);
            using var target = new AutogradNode(targetTensor, new TensorShape(1, 4), requiresGrad: false);

            pred.DataView.AsSpan().Fill(2f);
            target.DataView.AsSpan().Fill(-1f); // Zły kierunek (pred > 0, target < 0)

            using var loss = TensorMath.DirectionalLoss(_graph, pred, target, gamma: 10f);

            loss.GradView.AsSpan()[0] = 1f;
            _graph.Backward(loss);

            var pGrad = pred.GradView.AsSpan();
            Assert.True(pGrad[0] > 0);
        }

        // ====================================================================
        // 2. TESTY NUMERYCZNE (NUMERICAL GRADIENT CHECKING)
        // ====================================================================

        [Fact]
        public void AddBias_NumericalGradient_MatchesAnalytical()
        {
            using var inputTensor = new TensorStorage<float>(6, clearMemory: true);
            using var biasTensor = new TensorStorage<float>(3, clearMemory: true);
            using var input = new AutogradNode(inputTensor, new TensorShape(2, 3, 1, 1), requiresGrad: true);
            using var bias = new AutogradNode(biasTensor, new TensorShape(3), requiresGrad: true);

            input.DataView.AsSpan().Fill(0.5f);
            bias.DataView.AsSpan().Fill(0.2f);

            // Przekazujemy bezpośrednio operację. VerifyGradients zajmie się sumowaniem.
            VerifyGradients(bias, () => TensorMath.AddBias(_graph, input, bias));
        }

        [Fact]
        public void ReLU_NumericalGradient_MatchesAnalytical()
        {
            using var inputTensor = new TensorStorage<float>(4, clearMemory: true);
            using var input = new AutogradNode(inputTensor, new TensorShape(1, 4), requiresGrad: true);

            // Wartości wejściowe "z dala od zera", aby test numeryczny f(x+e)-f(x-e) działał płynnie 
            // poza punktem nieciągłości matematycznej ReLU.
            var span = input.DataView.AsSpan();
            span[0] = 1.5f; span[1] = -0.5f; span[2] = 2.0f; span[3] = -1.0f;

            VerifyGradients(input, () => TensorMath.ReLU(_graph, input));
        }

        [Fact]
        public void MatMul_NumericalGradient_MatchesAnalytical()
        {
            using var aTensor = new TensorStorage<float>(6, clearMemory: true);
            using var bTensor = new TensorStorage<float>(6, clearMemory: true);
            using var a = new AutogradNode(aTensor, new TensorShape(2, 3), requiresGrad: true);
            using var b = new AutogradNode(bTensor, new TensorShape(3, 2), requiresGrad: true);

            a.DataView.AsSpan().Fill(0.5f);
            b.DataView.AsSpan().Fill(0.2f);

            VerifyGradients(a, () => TensorMath.MatMul(_graph, a, b));
        }

        [Fact]
        public void Conv2D_NumericalGradient_MatchesAnalytical()
        {
            using var inputTensor = new TensorStorage<float>(16, clearMemory: true);
            using var weightsTensor = new TensorStorage<float>(4, clearMemory: true);
            using var input = new AutogradNode(inputTensor, new TensorShape(1, 1, 4, 4), requiresGrad: true);
            using var weights = new AutogradNode(weightsTensor, new TensorShape(1, 4), requiresGrad: true);

            input.DataView.AsSpan().Fill(0.5f);
            weights.DataView.AsSpan().Fill(0.2f);

            VerifyGradients(weights, () => TensorMath.Conv2D(_graph, input, weights, 1, 1, 4, 4, 2));
        }

        /// <summary>
        /// Uniwersalna metoda porównująca Gradient Analityczny z Numerycznym Różniczkowaniem.
        /// </summary>
        private void VerifyGradients(AutogradNode parameter, Func<AutogradNode> opFunc, float epsilon = 1e-3f, float tolerance = 1e-2f)
        {
            // 1. Resetujemy gradient z poprzednich wywołań
            parameter.ZeroGrad();
            _graph.Reset();

            // 2. Generujemy graf
            using var outNode = opFunc();

            // ---> NAPRAWA GRAFU: Inicjujemy tensor wyjściowy jedynkami! 
            // Matematycznie jest to równoznaczne ze zrobieniem Loss = Sum(outNode) i Loss.Backward().
            outNode.GradView.AsSpan().Fill(1f);

            // 3. Propagacja wsteczna (Liczy gradienty analityczne bezpośrednio w grafie)
            _graph.Backward(outNode);

            var analyticalGrads = parameter.GradView.AsSpan().ToArray();
            var numGrads = new float[parameter.DataView.Size];
            var dataSpan = parameter.DataView.AsSpan();

            // 4. Numeryczne sprawdzanie każdego elementu
            for (var i = 0; i < parameter.DataView.Size; i++)
            {
                var originalValue = dataSpan[i];

                // Krok do przodu f(x + epsilon)
                dataSpan[i] = originalValue + epsilon;
                _graph.Reset();
                using (var outPlus = opFunc())
                {
                    // Liczymy numeryczną sumę na wyjściu (TensorPrimitives wspiera przyspieszenie sprzętowe)
                    var fPlus = TensorPrimitives.Sum(outPlus.DataView.AsReadOnlySpan());

                    // Krok w tył f(x - epsilon)
                    dataSpan[i] = originalValue - epsilon;
                    _graph.Reset();
                    using (var outMinus = opFunc())
                    {
                        var fMinus = TensorPrimitives.Sum(outMinus.DataView.AsReadOnlySpan());

                        // Aproksymacja pochodnej numerycznej
                        numGrads[i] = (fPlus - fMinus) / (2 * epsilon);
                    }
                }

                // Przywrócenie stanu oryginalnego
                dataSpan[i] = originalValue;
            }

            // 5. Asercje
            for (var i = 0; i < analyticalGrads.Length; i++)
            {
                var diff = Math.Abs(analyticalGrads[i] - numGrads[i]);
                Assert.True(diff < tolerance,
                $"Błąd gradientu w elemencie {i}! Analityczny: {analyticalGrads[i]:F5}, Numeryczny: {numGrads[i]:F5}, Różnica: {diff:F5}");
            }
        }
    }
}