using DevOnBike.Overfit.Core;

namespace DevOnBike.Overfit.Tests
{
    public class AutogradTests : IDisposable
    {
        private const int Precision = 6;

        public AutogradTests()
        {
            // Każdy test w xUnit to nowa instancja klasy - inicjujemy graf
            ComputationGraph.Active = new ComputationGraph();
        }

        public void Dispose()
        {
            // Czyszczenie referencji statycznej
            ComputationGraph.Active = null;
        }

        // ====================================================================
        // 1. PODSTAWOWE OPERACJE
        // ====================================================================

        [Fact]
        public void TensorAdd_ForwardAndBackward_Correct()
        {
            using var a = new AutogradNode(new FastMatrix<double>(1, 2));
            using var b = new AutogradNode(new FastMatrix<double>(1, 2));
            a.Data.CopyFrom([1.0, 2.0]);
            b.Data.CopyFrom([3.0, 4.0]);

            using var res = TensorMath.Add(a, b);
            Assert.Equal([4.0, 6.0], res.Data.AsReadOnlySpan().ToArray()); //

            ComputationGraph.Active.Backward(res); // Automatycznie inicjuje 1.0
            Assert.Equal([1.0, 1.0], a.Grad.AsReadOnlySpan().ToArray()); //
            Assert.Equal([1.0, 1.0], b.Grad.AsReadOnlySpan().ToArray()); //
        }

        [Fact]
        public void TensorMatMul_ForwardAndBackward_Correct()
        {
            using var a = new AutogradNode(new FastMatrix<double>(1, 2));
            using var b = new AutogradNode(new FastMatrix<double>(2, 1));
            a.Data.CopyFrom([2.0, 3.0]);
            b.Data.CopyFrom([4.0, 5.0]);

            using var res = TensorMath.MatMul(a, b); //
            Assert.Equal(23.0, res.Data[0, 0]);

            ComputationGraph.Active.Backward(res);
            // dL/da = grad(res) * b^T = 1.0 * [4, 5]
            Assert.Equal([4.0, 5.0], a.Grad.AsReadOnlySpan().ToArray()); //
        }

        [Fact]
        public void AddBias_ForwardAndBackward_Correct()
        {
            using var input = new AutogradNode(new FastMatrix<double>(2, 2));
            using var bias = new AutogradNode(new FastMatrix<double>(1, 2));
            input.Data.AsSpan().Fill(1.0);
            bias.Data.CopyFrom([10.0, 20.0]);

            using var res = TensorMath.AddBias(input, bias); //
            ComputationGraph.Active.Backward(res);

            // Bias zbiera gradient z obu wierszy (1.0 + 1.0)
            Assert.Equal([2.0, 2.0], bias.Grad.AsReadOnlySpan().ToArray()); //
        }

        // ====================================================================
        // 2. AKTYWACJE I REGULARYZACJA
        // ====================================================================

        [Fact]
        public void ReLU_ForwardAndBackward_Correct()
        {
            using var input = new AutogradNode(new FastMatrix<double>(1, 3));
            input.Data.CopyFrom([-1.0, 0.0, 5.0]);

            using var res = TensorMath.ReLU(input); //
            Assert.Equal([0.0, 0.0, 5.0], res.Data.AsReadOnlySpan().ToArray());

            ComputationGraph.Active.Backward(res);
            Assert.Equal([0.0, 0.0, 1.0], input.Grad.AsReadOnlySpan().ToArray()); //
        }

        [Fact]
        public void Dropout_ForwardAndBackward_Correct()
        {
            using var input = new AutogradNode(new FastMatrix<double>(1, 10));
            input.Data.AsSpan().Fill(1.0);

            // Skalowanie 1/(1-0.5) = 2.0
            using var res = TensorMath.Dropout(input, 0.5, isTraining: true);
            ComputationGraph.Active.Backward(res);

            var grads = input.Grad.AsReadOnlySpan();
            var data = res.Data.AsReadOnlySpan();

            for (var i = 0; i < 10; i++)
            {
                if (data[i] == 0) Assert.Equal(0.0, grads[i]);
                else Assert.Equal(2.0, grads[i]); //
            }
        }

        // ====================================================================
        // 3. FUNKCJE STRAT
        // ====================================================================

        [Fact]
        public void MSELoss_ForwardAndBackward_Correct()
        {
            using var p = new AutogradNode(new FastMatrix<double>(2, 1));
            using var t = new AutogradNode(new FastMatrix<double>(2, 1));
            p.Data.CopyFrom([3.0, 5.0]);
            t.Data.CopyFrom([1.0, 9.0]);

            using var loss = TensorMath.MSELoss(p, t); //
            Assert.Equal(10.0, loss.Data[0, 0]); // ((3-1)^2 + (5-9)^2)/2 = 10

            ComputationGraph.Active.Backward(loss);
            // dL/dp = (2/N)*(p-t) = (2/2)*[2, -4] = [2.0, -4.0]
            Assert.Equal([2.0, -4.0], p.Grad.AsReadOnlySpan().ToArray());
        }

        [Fact]
        public void SoftmaxCrossEntropy_ForwardAndBackward_Correct()
        {
            using var logits = new AutogradNode(new FastMatrix<double>(1, 2));
            using var target = new AutogradNode(new FastMatrix<double>(1, 2));
            logits.Data.CopyFrom([0.0, 0.0]); // Softmax -> [0.5, 0.5]
            target.Data.CopyFrom([0.0, 1.0]);

            using var loss = TensorMath.SoftmaxCrossEntropy(logits, target); //
            ComputationGraph.Active.Backward(loss);

            // dL/dx = (probs - target) = [0.5, 0.5] - [0.0, 1.0] = [0.5, -0.5]
            Assert.Equal(0.5, logits.Grad[0, 0], Precision);
            Assert.Equal(-0.5, logits.Grad[0, 1], Precision);
        }

        // ====================================================================
        // 4. CNN I OPERACJE PRZESTRZENNE
        // ====================================================================

        [Fact]
        public void Conv2D_ForwardAndBackward_Flows()
        {
            using var input = new AutogradNode(new FastMatrix<double>(1, 9)); // 3x3
            using var weights = new AutogradNode(new FastMatrix<double>(1, 4)); // 2x2
            input.Data.AsSpan().Fill(1.0);
            weights.Data.AsSpan().Fill(1.0);

            using var res = TensorMath.Conv2D(input, weights, 1, 1, 3, 3, 2); //
            Assert.Equal(4, res.Data.Cols); // 2x2 wyjście
            Assert.Equal(4.0, res.Data[0, 0]); // Suma wag (1*1*4)

            ComputationGraph.Active.Backward(res);
            Assert.NotNull(weights.Grad);
            Assert.NotNull(input.Grad);
        }

        [Fact]
        public void MaxPool2D_ForwardAndBackward_Correct()
        {
            using var input = new AutogradNode(new FastMatrix<double>(1, 4)); // 2x2
            input.Data.CopyFrom([1, 2, 3, 10]);

            using var res = TensorMath.MaxPool2D(input, 1, 2, 2, 2); //
            Assert.Equal(10.0, res.Data[0, 0]);

            ComputationGraph.Active.Backward(res);
            // Gradient trafia tylko do indeksu z wartością maksymalną (10.0)
            Assert.Equal([0.0, 0.0, 0.0, 1.0], input.Grad.AsReadOnlySpan().ToArray());
        }

        [Fact]
        public void GlobalAveragePool2D_ForwardAndBackward_Correct()
        {
            using var input = new AutogradNode(new FastMatrix<double>(1, 4)); // 2x2
            input.Data.CopyFrom([1, 2, 3, 4]); // Suma=10, Avg=2.5

            using var res = TensorMath.GlobalAveragePool2D(input, 1, 2, 2); //
            Assert.Equal(2.5, res.Data[0, 0]);

            ComputationGraph.Active.Backward(res);
            // dL/dx = 1/N = 0.25
            Assert.All(input.Grad.AsReadOnlySpan().ToArray(), x => Assert.Equal(0.25, x));
        }

        // ====================================================================
        // 5. WARSTWY ZŁOŻONE (BatchNorm1D)
        // ====================================================================

        [Fact]
        public void BatchNorm1D_ForwardAndBackward_Flows()
        {
            using var input = new AutogradNode(new FastMatrix<double>(2, 1));
            input.Data.CopyFrom([10.0, 20.0]); // Mean=15, Var=25, Std=5

            using var gamma = new AutogradNode(new FastMatrix<double>(1, 1));
            using var beta = new AutogradNode(new FastMatrix<double>(1, 1));
            gamma.Data[0, 0] = 1.0; beta.Data[0, 0] = 0.0;

            using var rm = new FastMatrix<double>(1, 1);
            using var rv = new FastMatrix<double>(1, 1);
            rv[0, 0] = 1.0;

            using var res = TensorMath.BatchNorm1D(input, gamma, beta, rm, rv, 0.1, 1e-5, true); //
            // (20 - 15) / 5 = 1.0
            Assert.Equal(1.0, res.Data[1, 0], 1e-3);

            ComputationGraph.Active.Backward(res);
            Assert.NotNull(input.Grad);
            Assert.NotNull(gamma.Grad);
        }

        // ====================================================================
        // 6. OBSŁUGA FLAG (RequiresGrad)
        // ====================================================================

        [Fact]
        public void Tensor_RequiresGradFalse_DoesNotCalculateGradient()
        {
            using var matA = new FastMatrix<double>(1, 1);
            using var matB = new FastMatrix<double>(1, 1);

            using var a = new AutogradNode(matA, requiresGrad: false); //
            using var b = new AutogradNode(matB, requiresGrad: true);

            using var c = TensorMath.Add(a, b);
            ComputationGraph.Active.Backward(c);

            Assert.Null(a.Grad); //
            Assert.NotNull(b.Grad);
            Assert.Equal(1.0, b.Grad[0, 0]);
        }
    }
}