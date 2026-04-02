using DevOnBike.Overfit.Core;

namespace DevOnBike.Overfit.Tests
{
    public class AutogradTests : IDisposable
    {
        private const int Precision = 6;

        public AutogradTests()
        {
            ComputationGraph.Active = new ComputationGraph();
        }

        public void Dispose()
        {
            ComputationGraph.Active = null;
        }

        [Fact]
        public void TensorAdd_ForwardAndBackward_Correct()
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
        public void TensorMatMul_ForwardAndBackward_Correct()
        {
            using var a = new AutogradNode(new FastMatrix<float>(1, 2));
            using var b = new AutogradNode(new FastMatrix<float>(2, 1));
            a.Data.CopyFrom([2.0f, 3.0f]);
            b.Data.CopyFrom([4.0f, 5.0f]);

            using var res = TensorMath.MatMul(a, b);
            Assert.Equal(23.0f, res.Data[0, 0]);

            ComputationGraph.Active.Backward(res);
            Assert.Equal([4.0f, 5.0f], a.Grad.AsReadOnlySpan().ToArray());
        }

        [Fact]
        public void AddBias_ForwardAndBackward_Correct()
        {
            using var input = new AutogradNode(new FastMatrix<float>(2, 2));
            using var bias = new AutogradNode(new FastMatrix<float>(1, 2));
            input.Data.AsSpan().Fill(1.0f);
            bias.Data.CopyFrom([10.0f, 20.0f]);

            using var res = TensorMath.AddBias(input, bias);
            ComputationGraph.Active.Backward(res);

            Assert.Equal([2.0f, 2.0f], bias.Grad.AsReadOnlySpan().ToArray());
        }

        [Fact]
        public void ReLU_ForwardAndBackward_Correct()
        {
            using var input = new AutogradNode(new FastMatrix<float>(1, 3));
            input.Data.CopyFrom([-1.0f, 0.0f, 5.0f]);

            using var res = TensorMath.ReLU(input);
            Assert.Equal([0.0f, 0.0f, 5.0f], res.Data.AsReadOnlySpan().ToArray());

            ComputationGraph.Active.Backward(res);
            Assert.Equal([0.0f, 0.0f, 1.0f], input.Grad.AsReadOnlySpan().ToArray());
        }

        [Fact]
        public void Dropout_ForwardAndBackward_Correct()
        {
            using var input = new AutogradNode(new FastMatrix<float>(1, 10));
            input.Data.AsSpan().Fill(1.0f);

            using var res = TensorMath.Dropout(input, 0.5f, isTraining: true);
            ComputationGraph.Active.Backward(res);

            var grads = input.Grad.AsReadOnlySpan();
            var data = res.Data.AsReadOnlySpan();

            for (var i = 0; i < 10; i++)
            {
                if (data[i] == 0f) Assert.Equal(0.0f, grads[i]);
                else Assert.Equal(2.0f, grads[i]);
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
            Assert.Equal([2.0f, -4.0f], p.Grad.AsReadOnlySpan().ToArray());
        }

        [Fact]
        public void SoftmaxCrossEntropy_ForwardAndBackward_Correct()
        {
            using var logits = new AutogradNode(new FastMatrix<float>(1, 2));
            using var target = new AutogradNode(new FastMatrix<float>(1, 2));
            logits.Data.CopyFrom([0.0f, 0.0f]);
            target.Data.CopyFrom([0.0f, 1.0f]);

            using var loss = TensorMath.SoftmaxCrossEntropy(logits, target);
            ComputationGraph.Active.Backward(loss);

            Assert.Equal(0.5f, logits.Grad[0, 0], Precision);
            Assert.Equal(-0.5f, logits.Grad[0, 1], Precision);
        }

        [Fact]
        public void Conv2D_ForwardAndBackward_Flows()
        {
            using var input = new AutogradNode(new FastMatrix<float>(1, 9));
            using var weights = new AutogradNode(new FastMatrix<float>(1, 4));
            input.Data.AsSpan().Fill(1.0f);
            weights.Data.AsSpan().Fill(1.0f);

            using var res = TensorMath.Conv2D(input, weights, 1, 1, 3, 3, 2);
            Assert.Equal(4, res.Data.Cols);
            Assert.Equal(4.0f, res.Data[0, 0]);

            ComputationGraph.Active.Backward(res);
            Assert.NotNull(weights.Grad);
            Assert.NotNull(input.Grad);
        }

        [Fact]
        public void MaxPool2D_ForwardAndBackward_Correct()
        {
            using var input = new AutogradNode(new FastMatrix<float>(1, 4));
            input.Data.CopyFrom([1f, 2f, 3f, 10f]);

            using var res = TensorMath.MaxPool2D(input, 1, 2, 2, 2);
            Assert.Equal(10.0f, res.Data[0, 0]);

            ComputationGraph.Active.Backward(res);
            Assert.Equal([0.0f, 0.0f, 0.0f, 1.0f], input.Grad.AsReadOnlySpan().ToArray());
        }

        [Fact]
        public void GlobalAveragePool2D_ForwardAndBackward_Correct()
        {
            using var input = new AutogradNode(new FastMatrix<float>(1, 4));
            input.Data.CopyFrom([1f, 2f, 3f, 4f]);

            using var res = TensorMath.GlobalAveragePool2D(input, 1, 2, 2);
            Assert.Equal(2.5f, res.Data[0, 0]);

            ComputationGraph.Active.Backward(res);
            Assert.All(input.Grad.AsReadOnlySpan().ToArray(), x => Assert.Equal(0.25f, x));
        }

        [Fact]
        public void BatchNorm1D_ForwardAndBackward_Flows()
        {
            using var input = new AutogradNode(new FastMatrix<float>(2, 1));
            input.Data.CopyFrom([10.0f, 20.0f]);

            using var gamma = new AutogradNode(new FastMatrix<float>(1, 1));
            using var beta = new AutogradNode(new FastMatrix<float>(1, 1));
            gamma.Data[0, 0] = 1.0f; beta.Data[0, 0] = 0.0f;

            using var rm = new FastMatrix<float>(1, 1);
            using var rv = new FastMatrix<float>(1, 1);
            rv[0, 0] = 1.0f;

            using var res = TensorMath.BatchNorm1D(input, gamma, beta, rm, rv, 0.1f, 1e-5f, true);
            Assert.Equal(1.0f, res.Data[1, 0], 1e-3f);

            ComputationGraph.Active.Backward(res);
            Assert.NotNull(input.Grad);
            Assert.NotNull(gamma.Grad);
        }

        [Fact]
        public void Tensor_RequiresGradFalse_DoesNotCalculateGradient()
        {
            using var matA = new FastMatrix<float>(1, 1);
            using var matB = new FastMatrix<float>(1, 1);

            using var a = new AutogradNode(matA, requiresGrad: false);
            using var b = new AutogradNode(matB, requiresGrad: true);

            using var c = TensorMath.Add(a, b);
            ComputationGraph.Active.Backward(c);

            Assert.Null(a.Grad);
            Assert.NotNull(b.Grad);
            Assert.Equal(1.0f, b.Grad[0, 0]);
        }
    }
}