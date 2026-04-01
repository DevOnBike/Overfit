using DevOnBike.Overfit.Core;

namespace DevOnBike.Overfit.Tests
{
    public class TensorMathComprehensiveTests : IDisposable
    {
        private const double Delta = 1e-7;
        private const int Precision = 6;

        public TensorMathComprehensiveTests()
        {
            // Inicjalizacja grafu dla każdego testu
            ComputationGraph.Active = new ComputationGraph();
        }

        public void Dispose()
        {
            // Sprzątanie po teście
            ComputationGraph.Active = null;
        }

        // ====================================================================
        // 1. BROADCASTING & WIDOKI (VIEW OPS)
        // ====================================================================

        [Fact]
        public void BroadcastRowVector_Layout_Correct()
        {
            double[] row = [1.0, 2.0, 3.0];
            var view = TensorMath.BroadcastRowVector(row.AsSpan(), 2);

            Assert.Equal(2, view.Rows);
            Assert.Equal(3, view.Cols);
            Assert.Equal(0, view.RowStride); // Klucz do broadcastingu bez kopiowania
            Assert.Equal(1.0, view[1, 0]);
        }

        // ====================================================================
        // 2. PODSTAWOWA ALGEBRA (ADD & MATMUL)
        // ====================================================================

        [Fact]
        public void Add_ForwardAndBackward_Correct()
        {
            using var a = new AutogradNode(new FastMatrix<double>(1, 2));
            using var b = new AutogradNode(new FastMatrix<double>(1, 2));
            a.Data.CopyFrom([1.0, 2.0]);
            b.Data.CopyFrom([3.0, 4.0]);

            using var res = TensorMath.Add(a, b);
            Assert.Equal([4.0, 6.0], res.Data.AsReadOnlySpan().ToArray());

            ComputationGraph.Active.Backward(res);
            Assert.Equal([1.0, 1.0], a.Grad.AsReadOnlySpan().ToArray());
            Assert.Equal([1.0, 1.0], b.Grad.AsReadOnlySpan().ToArray());
        }

        [Fact]
        public void MatMul_FullCheck()
        {
            using var a = new AutogradNode(new FastMatrix<double>(1, 2));
            using var b = new AutogradNode(new FastMatrix<double>(2, 1));
            a.Data.CopyFrom([2, 3]);
            b.Data.CopyFrom([4, 5]);

            // Forward
            using var res = TensorMath.MatMul(a, b);
            Assert.Equal(23.0, res.Data[0, 0]); // 2*4 + 3*5

            // Backward
            ComputationGraph.Active.Backward(res);
            Assert.Equal(4.0, a.Grad[0, 0]); // dL/da = grad_out * b^T
            Assert.Equal(2.0, b.Grad[0, 0]); // dL/db = a^T * grad_out
        }

        [Fact]
        public void MatMulRaw_And_Add_Views_Correct()
        {
            using var a = new FastMatrix<double>(2, 2);
            using var b = new FastMatrix<double>(2, 1);
            using var c = new FastMatrix<double>(2, 1);
            a.CopyFrom([1, 2, 3, 4]);
            b.CopyFrom([5, 6]);

            // MatMul view
            TensorMath.MatMul(a.AsView(), b.AsView(), c.AsView());
            Assert.Equal(17.0, c[0, 0]);

            // MatMulAdd view
            TensorMath.MatMulAdd(a.AsView(), b.AsView(), c.AsView());
            Assert.Equal(34.0, c[0, 0]); // 17 + 17

            // MatMulRaw
            using var raw = TensorMath.MatMulRaw(a.AsView(), b.AsView());
            Assert.Equal(17.0, raw[0, 0]);
        }

        // ====================================================================
        // 3. WARSTWY NEURONOWE (BIAS & LINEAR)
        // ====================================================================

        [Fact]
        public void AddBias_And_Linear_Correct()
        {
            using var x = new AutogradNode(new FastMatrix<double>(1, 2));
            using var w = new AutogradNode(new FastMatrix<double>(2, 1));
            using var b = new AutogradNode(new FastMatrix<double>(1, 1));
            x.Data.CopyFrom([1, 2]); w.Data.CopyFrom([3, 4]); b.Data[0, 0] = 5;

            // Test Linear (MatMul + AddBias)
            using var res = TensorMath.Linear(x, w, b);
            Assert.Equal(16.0, res.Data[0, 0]); // (1*3 + 2*4) + 5

            ComputationGraph.Active.Backward(res);
            Assert.Equal(1.0, b.Grad[0, 0]);
        }

        // ====================================================================
        // 4. AKTYWACJE I REGULARYZACJA
        // ====================================================================

        [Fact]
        public void ReLU_Backward_MasksCorrectly()
        {
            using var input = new AutogradNode(new FastMatrix<double>(1, 2));
            input.Data.CopyFrom([-5.0, 5.0]);

            using var res = TensorMath.ReLU(input);
            ComputationGraph.Active.Backward(res);

            Assert.Equal(0.0, input.Grad[0, 0]); // Dla -5.0
            Assert.Equal(1.0, input.Grad[0, 1]); // Dla 5.0
        }

        [Fact]
        public void Dropout_Train_ScalesGradients()
        {
            using var input = new AutogradNode(new FastMatrix<double>(1, 100));
            input.Data.AsSpan().Fill(1.0);
            var p = 0.5;

            using var res = TensorMath.Dropout(input, p, isTraining: true);
            ComputationGraph.Active.Backward(res);

            var grad = input.Grad.AsReadOnlySpan();
            var data = res.Data.AsReadOnlySpan();
            var scale = 1.0 / (1.0 - p);

            for (var i = 0; i < 100; i++)
            {
                if (data[i] == 0) Assert.Equal(0.0, grad[i]);
                else Assert.Equal(scale, grad[i]); // dL/dx = scale
            }
        }

        // ====================================================================
        // 5. FUNKCJE STRAT (LOSS)
        // ====================================================================

        [Fact]
        public void MSELoss_ForwardAndBackward_Correct()
        {
            using var p = new AutogradNode(new FastMatrix<double>(2, 1));
            using var t = new AutogradNode(new FastMatrix<double>(2, 1));
            p.Data.CopyFrom([3.0, 5.0]);
            t.Data.CopyFrom([1.0, 9.0]);

            using var loss = TensorMath.MSELoss(p, t);
            Assert.Equal(10.0, loss.Data[0, 0]); // ((3-1)^2 + (5-9)^2)/2 = 10

            ComputationGraph.Active.Backward(loss);
            // dL/dp = (2/N)*(p-t) = (2/2)*[2, -4]
            Assert.Equal(2.0, p.Grad[0, 0]);
            Assert.Equal(-4.0, p.Grad[1, 0]);
        }

        [Fact]
        public void SoftmaxCrossEntropy_Correct()
        {
            using var logits = new AutogradNode(new FastMatrix<double>(1, 2));
            using var target = new AutogradNode(new FastMatrix<double>(1, 2));
            logits.Data.CopyFrom([0.0, 0.0]); // p = [0.5, 0.5]
            target.Data.CopyFrom([0.0, 1.0]);

            using var loss = TensorMath.SoftmaxCrossEntropy(logits, target);
            Assert.Equal(Math.Log(2), loss.Data[0, 0], Precision);

            ComputationGraph.Active.Backward(loss);
            // dL/dx = p - t = [0.5, -0.5]
            Assert.Equal(0.5, logits.Grad[0, 0], Precision);
            Assert.Equal(-0.5, logits.Grad[0, 1], Precision);
        }

        // ====================================================================
        // 6. CNN I OPERACJE PRZESTRZENNE
        // ====================================================================

        [Fact]
        public void Conv2D_ForwardAndBackward_Check()
        {
            using var input = new AutogradNode(new FastMatrix<double>(1, 9)); // 3x3
            using var weights = new AutogradNode(new FastMatrix<double>(1, 4)); // 2x2
            input.Data.AsSpan().Fill(1.0);
            weights.Data.CopyFrom([1, 1, 1, 1]);

            using var res = TensorMath.Conv2D(input, weights, 1, 1, 3, 3, 2);
            Assert.Equal(4.0, res.Data[0, 0]); // Suma wag

            ComputationGraph.Active.Backward(res);
            Assert.NotNull(weights.Grad);
            Assert.NotNull(input.Grad);
        }

        [Fact]
        public void MaxPool2D_And_GlobalAvgPool_Correct()
        {
            using var input = new AutogradNode(new FastMatrix<double>(1, 4)); // 2x2
            input.Data.CopyFrom([1, 2, 3, 4]);

            // MaxPool
            using var maxP = TensorMath.MaxPool2D(input, 1, 2, 2, 2);
            Assert.Equal(4.0, maxP.Data[0, 0]);
            ComputationGraph.Active.Backward(maxP);
            Assert.Equal(1.0, input.Grad[0, 3]);

            input.Grad.Clear();

            // GlobalAvgPool
            using var avgP = TensorMath.GlobalAveragePool2D(input, 1, 2, 2);
            Assert.Equal(2.5, avgP.Data[0, 0]);
            ComputationGraph.Active.Backward(avgP);
            Assert.Equal(0.25, input.Grad[0, 0]); // 1/4
        }

        [Fact]
        public void Im2Col_Col2Im_Roundtrip_Consistency()
        {
            double[] input = [1, 2, 3, 4, 5, 6, 7, 8, 9];
            var col = new double[16];
            var output = new double[9];

            TensorMath.Im2Col(input, 1, 3, 3, 2, 1, 0, col);
            TensorMath.Col2Im(col, 1, 3, 3, 2, 1, 0, output);

            // Wartość 5.0 (środek) przy oknie 2x2 nakłada się 4 razy
            Assert.Equal(20.0, output[4]);
        }

        // ====================================================================
        // 7. NORMALIZACJA (BATCHNORM)
        // ====================================================================

        [Fact]
        public void BatchNorm1D_FullFlow_WithNonZeroGradients()
        {
            using var input = new AutogradNode(new FastMatrix<double>(2, 1));
            input.Data.CopyFrom([10.0, 20.0]); // Mean=15, Std=5 -> xHat=[-1, 1]

            using var gamma = new AutogradNode(new FastMatrix<double>(1, 1));
            using var beta = new AutogradNode(new FastMatrix<double>(1, 1));
            gamma.Data[0, 0] = 1.0;
            beta.Data[0, 0] = 0.0;

            using var rm = new FastMatrix<double>(1, 1);
            using var rv = new FastMatrix<double>(1, 1);
            rv[0, 0] = 1.0;

            // Forward
            using var bnOut = TensorMath.BatchNorm1D(input, gamma, beta, rm, rv, 0.1, 1e-5, isTraining: true);

            // Używamy MSELoss, aby uniknąć zerowania gradientu Gamma
            using var target = new AutogradNode(new FastMatrix<double>(2, 1), false);
            target.Data.Clear();
            using var loss = TensorMath.MSELoss(bnOut, target);

            ComputationGraph.Active.Backward(loss);

            // Gamma grad = sum(grad_out * xHat) = sum(xHat * xHat) = (-1*-1) + (1*1) = 2.0
            Assert.Equal(2.0, gamma.Grad[0, 0], 1e-3);
            Assert.NotNull(input.Grad);
        }
    }
}