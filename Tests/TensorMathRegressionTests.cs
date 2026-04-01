using DevOnBike.Overfit.Core;

namespace DevOnBike.Overfit.Tests
{
    public class TensorMathRegressionTests
    {
        private const double Delta = 1e-7;

        // Pomocnik do czyszczenia grafu po teście (Transients only)
        private void CleanupGraph(List<AutogradNode> graph, params AutogradNode[] keep)
        {
            var keepSet = keep.ToHashSet();
            foreach (var node in graph)
            {
                if (!keepSet.Contains(node)) node.Dispose();
            }
        }

        // ====================================================================
        // 1. BROADCASTING & VIEW OPS
        // ====================================================================

        [Fact]
        public void Test_BroadcastRowVector()
        {
            double[] data = [1, 2, 3];
            var view = TensorMath.BroadcastRowVector(data, 2);

            Assert.Equal(2, view.Rows);
            Assert.Equal(3, view.Cols);
            Assert.Equal(0, view.RowStride); // Klucz do broadcastingu
            Assert.Equal(1.0, view[0, 0]);
            Assert.Equal(1.0, view[1, 0]); // Ten sam element fizyczny
        }

        // ====================================================================
        // 2. CORE MATH (VIEW VERSIONS)
        // ====================================================================

        [Fact]
        public void Test_Raw_Add()
        {
            using var a = new FastMatrix<double>(1, 3);
            using var b = new FastMatrix<double>(1, 3);
            using var res = new FastMatrix<double>(1, 3);
            a.CopyFrom([1, 2, 3]);
            b.CopyFrom([4, 5, 6]);

            TensorMath.Add(a.AsView(), b.AsView(), res.AsView());
            Assert.Equal(5.0, res[0, 0]);
            Assert.Equal(9.0, res[0, 2]);
        }

        [Fact]
        public void Test_Raw_MatMul_And_Add()
        {
            using var a = new FastMatrix<double>(2, 2);
            using var b = new FastMatrix<double>(2, 1);
            using var c = new FastMatrix<double>(2, 1);
            a.CopyFrom([1, 2, 3, 4]);
            b.CopyFrom([5, 6]);

            TensorMath.MatMul(a.AsView(), b.AsView(), c.AsView());
            // [1*5 + 2*6] = [17]
            // [3*5 + 4*6] = [39]
            Assert.Equal(17.0, c[0, 0]);
            Assert.Equal(39.0, c[1, 0]);

            TensorMath.MatMulAdd(a.AsView(), b.AsView(), c.AsView());
            Assert.Equal(34.0, c[0, 0]); // 17 + 17
        }

        [Fact]
        public void Test_MatMulRaw_Overloads()
        {
            using var a = new FastMatrix<double>(2, 2);
            using var b = new FastMatrix<double>(2, 2);
            a.CopyFrom([1, 0, 0, 1]);
            b.CopyFrom([5, 6, 7, 8]);

            using var res1 = TensorMath.MatMulRaw(a.AsView(), b.AsView());
            using var res2 = TensorMath.MatMulRaw(a, b); // Parallel version

            Assert.Equal(res1.AsReadOnlySpan().ToArray(), res2.AsReadOnlySpan().ToArray());
            Assert.Equal(5.0, res2[0, 0]);
        }

        // ====================================================================
        // 3. AUTOGRAD NODE OPS
        // ====================================================================

        [Fact]
        public void Test_Node_Add_And_Bias()
        {
            using var input = new AutogradNode(new FastMatrix<double>(2, 2));
            using var bias = new AutogradNode(new FastMatrix<double>(1, 2));
            input.Data.CopyFrom([1, 1, 1, 1]);
            bias.Data.CopyFrom([10, 20]);

            using var res = TensorMath.AddBias(input, bias);
            Assert.Equal(11.0, res.Data[0, 0]);
            Assert.Equal(21.0, res.Data[0, 1]);

            var g = res.Backward();
            Assert.Equal(2.0, bias.Grad[0, 0]); // 1.0 + 1.0 z obu wierszy
            CleanupGraph(g, input, bias);
        }

        [Fact]
        public void Test_Node_MatMul()
        {
            using var a = new AutogradNode(new FastMatrix<double>(1, 2));
            using var b = new AutogradNode(new FastMatrix<double>(2, 1));
            a.Data.CopyFrom([2, 3]);
            b.Data.CopyFrom([4, 5]);

            using var res = TensorMath.MatMul(a, b);
            Assert.Equal(23.0, res.Data[0, 0]);

            var g = res.Backward();
            Assert.Equal(4.0, a.Grad[0, 0]); // d(ab)/da = b
            Assert.Equal(2.0, b.Grad[0, 0]); // d(ab)/db = a
            CleanupGraph(g, a, b);
        }

        // ====================================================================
        // 4. ACTIVATIONS & DROPOUT
        // ====================================================================

        [Fact]
        public void Test_ReLU()
        {
            using var input = new AutogradNode(new FastMatrix<double>(1, 2));
            input.Data.CopyFrom([-5.0, 5.0]);

            using var res = TensorMath.ReLU(input);
            Assert.Equal(0.0, res.Data[0, 0]);
            Assert.Equal(5.0, res.Data[0, 1]);

            res.Backward();
            Assert.Equal(0.0, input.Grad[0, 0]);
            Assert.Equal(1.0, input.Grad[0, 1]);
        }

        [Fact]
        public void Test_Dropout_Eval_And_Train()
        {
            using var input = new AutogradNode(new FastMatrix<double>(1, 100));
            input.Data.AsSpan().Fill(1.0);

            // Eval mode: Identity
            using var resEval = TensorMath.Dropout(input, 0.5, false);
            Assert.Equal(1.0, resEval.Data[0, 0]);

            // Train mode: Scaling
            using var resTrain = TensorMath.Dropout(input, 0.5, true);
            // Średnia powinna być bliska 1.0 przez skalowanie 1/(1-p)
            double sum = 0;
            foreach (var v in resTrain.Data.AsReadOnlySpan()) sum += v;
            Assert.InRange(sum / 100.0, 0.5, 1.5);
        }

        // ====================================================================
        // 5. LOSS FUNCTIONS
        // ====================================================================

        [Fact]
        public void Test_MSE_Both_Implementations()
        {
            using var p = new AutogradNode(new FastMatrix<double>(1, 1));
            using var t = new AutogradNode(new FastMatrix<double>(1, 1));
            p.Data[0, 0] = 10; t.Data[0, 0] = 5;

            using var loss1 = TensorMath.MSE(p, t);
            using var loss2 = TensorMath.MSELoss(p, t);

            Assert.Equal(25.0, loss1.Data[0, 0]);
            Assert.Equal(25.0, loss2.Data[0, 0]);

            loss2.Backward();
            Assert.Equal(10.0, p.Grad[0, 0]); // 2 * (10-5)
        }

        [Fact]
        public void Test_SoftmaxCrossEntropy()
        {
            using var logits = new AutogradNode(new FastMatrix<double>(1, 2));
            using var target = new AutogradNode(new FastMatrix<double>(1, 2));
            logits.Data.CopyFrom([0.0, 0.0]); // Softmax -> [0.5, 0.5]
            target.Data.CopyFrom([0.0, 1.0]);

            using var loss = TensorMath.SoftmaxCrossEntropy(logits, target);
            // -ln(0.5) = 0.693...
            Assert.Equal(Math.Log(2), loss.Data[0, 0], 1e-5);

            loss.Backward();
            Assert.Equal(-0.5, logits.Grad[0, 1], Delta); // p - t = 0.5 - 1.0
        }

        // ====================================================================
        // 6. CNN & POOLING
        // ====================================================================

        [Fact]
        public void Test_Im2Col_Col2Im_Roundtrip()
        {
            int C = 1, H = 3, W = 3, K = 2;
            double[] input = [1, 2, 3, 4, 5, 6, 7, 8, 9];
            var col = new double[1 * 2 * 2 * 2 * 2]; // C*K*K * outH*outW
            var output = new double[9];

            TensorMath.Im2Col(input, C, H, W, K, 1, 0, col);
            TensorMath.Col2Im(col, C, H, W, K, 1, 0, output);

            // To nie jest czysty roundtrip (sumowanie overlapów!), ale dla K=1 byłby.
            // Sprawdzamy czy środek (5) jest zsumowany poprawnie (występuje w 4 oknach)
            Assert.Equal(5.0 * 4, output[4]);
        }

        [Fact]
        public void Test_Conv2D_Forward()
        {
            using var input = new AutogradNode(new FastMatrix<double>(1, 9)); // 1x3x3
            using var weights = new AutogradNode(new FastMatrix<double>(1, 4)); // 1 filtr 2x2
            input.Data.AsSpan().Fill(1.0);
            weights.Data.AsSpan().Fill(1.0);

            using var res = TensorMath.Conv2D(input, weights, 1, 1, 3, 3, 2);
            // Każde okno 2x2 sumuje cztery jedynki -> wynik 4.0
            Assert.Equal(4.0, res.Data[0, 0]);
            Assert.Equal(4, res.Data.Cols); // 2x2 output
        }

        [Fact]
        public void Test_MaxPool2D()
        {
            using var input = new AutogradNode(new FastMatrix<double>(1, 4)); // 2x2
            input.Data.CopyFrom([1, 2, 3, 4]);
            using var res = TensorMath.MaxPool2D(input, 1, 2, 2, 2);

            Assert.Equal(4.0, res.Data[0, 0]);
            res.Backward();
            Assert.Equal(1.0, input.Grad[0, 3]); // Gradient tylko na maxie
        }

        // ====================================================================
        // 7. NORMALIZATION & GLOBAL OPS
        // ====================================================================

        [Fact]
        public void Test_BatchNorm1D_Training_Flow()
        {
            using var input = new AutogradNode(new FastMatrix<double>(2, 1));
            input.Data.CopyFrom([10.0, 20.0]); // Mean=15, Var=25

            using var gamma = new AutogradNode(new FastMatrix<double>(1, 1));
            using var beta = new AutogradNode(new FastMatrix<double>(1, 1));
            gamma.Data[0, 0] = 1.0; beta.Data[0, 0] = 0.0;

            using var rm = new FastMatrix<double>(1, 1);
            using var rv = new FastMatrix<double>(1, 1);
            rv[0, 0] = 1.0;

            using var res = TensorMath.BatchNorm1D(input, gamma, beta, rm, rv, 0.1, 1e-5, true);

            // Xhat dla 20: (20-15)/sqrt(25) = 1.0
            Assert.Equal(1.0, res.Data[1, 0], 1e-3);
            Assert.Equal(15.0 * 0.1, rm[0, 0]); // Momentum update
        }

        [Fact]
        public void Test_GlobalAveragePool2D()
        {
            using var input = new AutogradNode(new FastMatrix<double>(1, 4)); // 1 channel 2x2
            input.Data.CopyFrom([1, 2, 3, 4]);
            using var res = TensorMath.GlobalAveragePool2D(input, 1, 2, 2);

            Assert.Equal(2.5, res.Data[0, 0]);
            res.Backward();
            Assert.Equal(0.25, input.Grad[0, 0]); // 1 / (2*2)
        }

        [Fact]
        public void Test_Linear_Full_Combo()
        {
            using var x = new AutogradNode(new FastMatrix<double>(1, 2));
            using var w = new AutogradNode(new FastMatrix<double>(2, 1));
            using var b = new AutogradNode(new FastMatrix<double>(1, 1));
            x.Data.CopyFrom([1, 2]); w.Data.CopyFrom([3, 4]); b.Data[0, 0] = 5;

            using var res = TensorMath.Linear(x, w, b);
            // (1*3 + 2*4) + 5 = 11 + 5 = 16
            Assert.Equal(16.0, res.Data[0, 0]);
        }
    }
}