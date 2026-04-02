using DevOnBike.Overfit.Core;

namespace DevOnBike.Overfit.Tests
{
    public class TensorMathComprehensiveTests : IDisposable
    {
        private const float Delta = 1e-7f;
        private const int Precision = 6;

        public TensorMathComprehensiveTests()
        {
            ComputationGraph.Active = new ComputationGraph();
        }

        public void Dispose()
        {
            ComputationGraph.Active = null;
        }

        [Fact]
        public void BroadcastRowVector_Layout_Correct()
        {
            float[] row = [1.0f, 2.0f, 3.0f];
            var view = TensorMath.BroadcastRowVector(row.AsSpan(), 2);

            Assert.Equal(2, view.Rows);
            Assert.Equal(3, view.Cols);
            Assert.Equal(0, view.RowStride);
            Assert.Equal(1.0f, view[1, 0]);
        }

        [Fact]
        public void Add_ForwardAndBackward_Correct()
        {
            using var a = new AutogradNode(new FastMatrix<float>(1, 2));
            using var b = new AutogradNode(new FastMatrix<float>(1, 2));
            a.Data.CopyFrom([1.0f, 2.0f]);
            b.Data.CopyFrom([3.0f, 4.0f]);

            using var res = TensorMath.Add(a, b);
            Assert.Equal([4.0f, 6.0f], res.Data.AsReadOnlySpan().ToArray());

            ComputationGraph.Active.Backward(res);
            Assert.Equal([1.0f, 1.0f], a.Grad.AsReadOnlySpan().ToArray());
            Assert.Equal([1.0f, 1.0f], b.Grad.AsReadOnlySpan().ToArray());
        }

        [Fact]
        public void MatMul_FullCheck()
        {
            using var a = new AutogradNode(new FastMatrix<float>(1, 2));
            using var b = new AutogradNode(new FastMatrix<float>(2, 1));
            a.Data.CopyFrom([2f, 3f]);
            b.Data.CopyFrom([4f, 5f]);

            using var res = TensorMath.MatMul(a, b);
            Assert.Equal(23.0f, res.Data[0, 0]);

            ComputationGraph.Active.Backward(res);
            Assert.Equal(4.0f, a.Grad[0, 0]);
            Assert.Equal(2.0f, b.Grad[0, 0]);
        }

        [Fact]
        public void MatMulRaw_And_Add_Views_Correct()
        {
            using var a = new FastMatrix<float>(2, 2);
            using var b = new FastMatrix<float>(2, 1);
            using var c = new FastMatrix<float>(2, 1);
            a.CopyFrom([1f, 2f, 3f, 4f]);
            b.CopyFrom([5f, 6f]);

            TensorMath.MatMul(a.AsView(), b.AsView(), c.AsView());
            Assert.Equal(17.0f, c[0, 0]);

            TensorMath.MatMulAdd(a.AsView(), b.AsView(), c.AsView());
            Assert.Equal(34.0f, c[0, 0]);

            using var raw = TensorMath.MatMulRaw(a.AsView(), b.AsView());
            Assert.Equal(17.0f, raw[0, 0]);
        }

        [Fact]
        public void AddBias_And_Linear_Correct()
        {
            using var x = new AutogradNode(new FastMatrix<float>(1, 2));
            using var w = new AutogradNode(new FastMatrix<float>(2, 1));
            using var b = new AutogradNode(new FastMatrix<float>(1, 1));
            x.Data.CopyFrom([1f, 2f]); w.Data.CopyFrom([3f, 4f]); b.Data[0, 0] = 5f;

            using var res = TensorMath.Linear(x, w, b);
            Assert.Equal(16.0f, res.Data[0, 0]);

            ComputationGraph.Active.Backward(res);
            Assert.Equal(1.0f, b.Grad[0, 0]);
        }

        [Fact]
        public void ReLU_Backward_MasksCorrectly()
        {
            using var input = new AutogradNode(new FastMatrix<float>(1, 2));
            input.Data.CopyFrom([-5.0f, 5.0f]);

            using var res = TensorMath.ReLU(input);
            ComputationGraph.Active.Backward(res);

            Assert.Equal(0.0f, input.Grad[0, 0]);
            Assert.Equal(1.0f, input.Grad[0, 1]);
        }

        [Fact]
        public void Dropout_Train_ScalesGradients()
        {
            using var input = new AutogradNode(new FastMatrix<float>(1, 100));
            input.Data.AsSpan().Fill(1.0f);
            var p = 0.5f;

            using var res = TensorMath.Dropout(input, p, isTraining: true);
            ComputationGraph.Active.Backward(res);

            var grad = input.Grad.AsReadOnlySpan();
            var data = res.Data.AsReadOnlySpan();
            var scale = 1.0f / (1.0f - p);

            for (var i = 0; i < 100; i++)
            {
                if (data[i] == 0f) Assert.Equal(0.0f, grad[i]);
                else Assert.Equal(scale, grad[i]);
            }
        }

        [Fact]
        public void MSELoss_ForwardAndBackward_Correct()
        {
            using var p = new AutogradNode(new FastMatrix<float>(2, 1));
            using var t = new AutogradNode(new FastMatrix<float>(2, 1));
            p.Data.CopyFrom([3.0f, 5.0f]);
            t.Data.CopyFrom([1.0f, 9.0f]);

            using var loss = TensorMath.MSELoss(p, t);
            Assert.Equal(10.0f, loss.Data[0, 0]);

            ComputationGraph.Active.Backward(loss);
            Assert.Equal(2.0f, p.Grad[0, 0]);
            Assert.Equal(-4.0f, p.Grad[1, 0]);
        }

        [Fact]
        public void SoftmaxCrossEntropy_Correct()
        {
            using var logits = new AutogradNode(new FastMatrix<float>(1, 2));
            using var target = new AutogradNode(new FastMatrix<float>(1, 2));
            logits.Data.CopyFrom([0.0f, 0.0f]);
            target.Data.CopyFrom([0.0f, 1.0f]);

            using var loss = TensorMath.SoftmaxCrossEntropy(logits, target);
            Assert.Equal(MathF.Log(2f), loss.Data[0, 0], Precision);

            ComputationGraph.Active.Backward(loss);
            Assert.Equal(0.5f, logits.Grad[0, 0], Precision);
            Assert.Equal(-0.5f, logits.Grad[0, 1], Precision);
        }

        [Fact]
        public void Conv2D_ForwardAndBackward_Check()
        {
            using var input = new AutogradNode(new FastMatrix<float>(1, 9));
            using var weights = new AutogradNode(new FastMatrix<float>(1, 4));
            input.Data.AsSpan().Fill(1.0f);
            weights.Data.CopyFrom([1f, 1f, 1f, 1f]);

            using var res = TensorMath.Conv2D(input, weights, 1, 1, 3, 3, 2);
            Assert.Equal(4.0f, res.Data[0, 0]);

            ComputationGraph.Active.Backward(res);
            Assert.NotNull(weights.Grad);
            Assert.NotNull(input.Grad);
        }

        [Fact]
        public void MaxPool2D_And_GlobalAvgPool_Correct()
        {
            using var input = new AutogradNode(new FastMatrix<float>(1, 4));
            input.Data.CopyFrom([1f, 2f, 3f, 4f]);

            using var maxP = TensorMath.MaxPool2D(input, 1, 2, 2, 2);
            Assert.Equal(4.0f, maxP.Data[0, 0]);
            ComputationGraph.Active.Backward(maxP);
            Assert.Equal(1.0f, input.Grad[0, 3]);

            input.Grad.Clear();

            using var avgP = TensorMath.GlobalAveragePool2D(input, 1, 2, 2);
            Assert.Equal(2.5f, avgP.Data[0, 0]);
            ComputationGraph.Active.Backward(avgP);
            Assert.Equal(0.25f, input.Grad[0, 0]);
        }

        [Fact]
        public void Im2Col_Col2Im_Roundtrip_Consistency()
        {
            float[] input = [1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f];
            var col = new float[16];
            var output = new float[9];

            TensorMath.Im2Col(input, 1, 3, 3, 2, 1, 0, col);
            TensorMath.Col2Im(col, 1, 3, 3, 2, 1, 0, output);

            Assert.Equal(20.0f, output[4]);
        }

        [Fact]
        public void BatchNorm1D_FullFlow_WithNonZeroGradients()
        {
            using var input = new AutogradNode(new FastMatrix<float>(2, 1));
            input.Data.CopyFrom([10.0f, 20.0f]);

            using var gamma = new AutogradNode(new FastMatrix<float>(1, 1));
            using var beta = new AutogradNode(new FastMatrix<float>(1, 1));
            gamma.Data[0, 0] = 1.0f;
            beta.Data[0, 0] = 0.0f;

            using var rm = new FastMatrix<float>(1, 1);
            using var rv = new FastMatrix<float>(1, 1);
            rv[0, 0] = 1.0f;

            using var bnOut = TensorMath.BatchNorm1D(input, gamma, beta, rm, rv, 0.1f, 1e-5f, isTraining: true);

            using var target = new AutogradNode(new FastMatrix<float>(2, 1), false);
            target.Data.Clear();
            using var loss = TensorMath.MSELoss(bnOut, target);

            ComputationGraph.Active.Backward(loss);

            Assert.Equal(2.0f, gamma.Grad[0, 0], 1e-3f);
            Assert.NotNull(input.Grad);
        }
    }
}