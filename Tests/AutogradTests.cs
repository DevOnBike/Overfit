// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Core;

namespace DevOnBike.Overfit.Tests
{
    public class AutogradTests
    {
        private const int Precision = 6;

        [Fact]
        public void TensorAdd_ForwardAndBackward_Correct()
        {
            var graph = new ComputationGraph();
            using var a = new AutogradNode(new FastTensor<float>(1, 2));
            using var b = new AutogradNode(new FastTensor<float>(1, 2));
            ((Span<float>)[1.0f, 2.0f]).CopyTo(a.Data.AsSpan());
            ((Span<float>)[3.0f, 4.0f]).CopyTo(b.Data.AsSpan());

            using var res = TensorMath.Add(graph, a, b);
            Assert.Equal([4.0f, 6.0f], res.Data.AsSpan().ToArray());

            graph.Backward(res);
            Assert.Equal([1.0f, 1.0f], a.Grad.AsSpan().ToArray());
            Assert.Equal([1.0f, 1.0f], b.Grad.AsSpan().ToArray());
        }

        [Fact]
        public void TensorMatMul_ForwardAndBackward_Correct()
        {
            var graph = new ComputationGraph();
            using var a = new AutogradNode(new FastTensor<float>(1, 2));
            using var b = new AutogradNode(new FastTensor<float>(2, 1));
            ((Span<float>)[2.0f, 3.0f]).CopyTo(a.Data.AsSpan());
            ((Span<float>)[4.0f, 5.0f]).CopyTo(b.Data.AsSpan());

            using var res = TensorMath.MatMul(graph, a, b);
            Assert.Equal(23.0f, res.Data[0, 0]);

            graph.Backward(res);
            Assert.Equal([4.0f, 5.0f], a.Grad.AsSpan().ToArray());
        }

        [Fact]
        public void AddBias_ForwardAndBackward_Correct()
        {
            var graph = new ComputationGraph();
            using var input = new AutogradNode(new FastTensor<float>(2, 2));
            using var bias = new AutogradNode(new FastTensor<float>(2));
            input.Data.AsSpan().Fill(1.0f);
            ((Span<float>)[10.0f, 20.0f]).CopyTo(bias.Data.AsSpan());

            using var res = TensorMath.AddBias(graph, input, bias);
            graph.Backward(res);

            Assert.Equal([2.0f, 2.0f], bias.Grad.AsSpan().ToArray());
        }

        [Fact]
        public void ReLU_ForwardAndBackward_Correct()
        {
            var graph = new ComputationGraph();
            using var input = new AutogradNode(new FastTensor<float>(1, 3));
            ((Span<float>)[-1.0f, 0.0f, 5.0f]).CopyTo(input.Data.AsSpan());

            using var res = TensorMath.ReLU(graph, input);
            Assert.Equal([0.0f, 0.0f, 5.0f], res.Data.AsSpan().ToArray());

            graph.Backward(res);
            Assert.Equal([0.0f, 0.0f, 1.0f], input.Grad.AsSpan().ToArray());
        }

        [Fact]
        public void Dropout_ForwardAndBackward_Correct()
        {
            var graph = new ComputationGraph();
            using var input = new AutogradNode(new FastTensor<float>(1, 10));
            input.Data.AsSpan().Fill(1.0f);

            using var res = TensorMath.Dropout(graph, input, 0.5f, true);
            graph.Backward(res);

            var grads = input.Grad.AsSpan();
            var data = res.Data.AsSpan();

            for (var i = 0; i < 10; i++)
            {
                if (data[i] == 0f) Assert.Equal(0.0f, grads[i]);
                else Assert.Equal(2.0f, grads[i]);
            }
        }

        [Fact]
        public void MSELoss_ForwardAndBackward_Correct()
        {
            var graph = new ComputationGraph();
            using var p = new AutogradNode(new FastTensor<float>(2, 1));
            using var t = new AutogradNode(new FastTensor<float>(2, 1));
            ((Span<float>)[3.0f, 5.0f]).CopyTo(p.Data.AsSpan());
            ((Span<float>)[1.0f, 9.0f]).CopyTo(t.Data.AsSpan());

            using var loss = TensorMath.MSELoss(graph, p, t);
            Assert.Equal(10.0f, loss.Data[0, 0]);

            graph.Backward(loss);
            Assert.Equal([2.0f, -4.0f], p.Grad.AsSpan().ToArray());
        }

        [Fact]
        public void SoftmaxCrossEntropy_ForwardAndBackward_Correct()
        {
            var graph = new ComputationGraph();
            using var logits = new AutogradNode(new FastTensor<float>(1, 2));
            using var target = new AutogradNode(new FastTensor<float>(1, 2));
            logits.Data.AsSpan().Fill(0.0f);
            ((Span<float>)[0.0f, 1.0f]).CopyTo(target.Data.AsSpan());

            using var loss = TensorMath.SoftmaxCrossEntropy(graph, logits, target);
            graph.Backward(loss);

            Assert.Equal(0.5f, logits.Grad[0, 0], Precision);
            Assert.Equal(-0.5f, logits.Grad[0, 1], Precision);
        }

        [Fact]
        public void Conv2D_ForwardAndBackward_Flows()
        {
            var graph = new ComputationGraph();
            using var input = new AutogradNode(new FastTensor<float>(1, 1, 3, 3));
            using var weights = new AutogradNode(new FastTensor<float>(1, 4));
            input.Data.AsSpan().Fill(1.0f);
            weights.Data.AsSpan().Fill(1.0f);

            using var res = TensorMath.Conv2D(graph, input, weights, 1, 1, 3, 3, 2);
            Assert.Equal(4.0f, res.Data[0, 0, 0, 0]);

            graph.Backward(res);
            Assert.NotNull(weights.Grad);
            Assert.NotNull(input.Grad);
        }

        [Fact]
        public void MaxPool2D_ForwardAndBackward_Correct()
        {
            var graph = new ComputationGraph();
            using var input = new AutogradNode(new FastTensor<float>(1, 1, 2, 2));
            ((Span<float>)[1f, 2f, 3f, 10f]).CopyTo(input.Data.AsSpan());

            using var res = TensorMath.MaxPool2D(graph, input, 1, 2, 2, 2);
            Assert.Equal(10.0f, res.Data[0, 0, 0, 0]);

            graph.Backward(res);
            Assert.Equal([0.0f, 0.0f, 0.0f, 1.0f], input.Grad.AsSpan().ToArray());
        }

        [Fact]
        public void GlobalAveragePool2D_ForwardAndBackward_Correct()
        {
            var graph = new ComputationGraph();
            using var input = new AutogradNode(new FastTensor<float>(1, 1, 2, 2));
            ((Span<float>)[1f, 2f, 3f, 4f]).CopyTo(input.Data.AsSpan());

            using var res = TensorMath.GlobalAveragePool2D(graph, input, 1, 2, 2);
            Assert.Equal(2.5f, res.Data[0, 0]);

            graph.Backward(res);
            Assert.All(input.Grad.AsSpan().ToArray(), action: x => Assert.Equal(0.25f, x));
        }

        [Fact]
        public void BatchNorm1D_ForwardAndBackward_Flows()
        {
            var graph = new ComputationGraph();
            using var input = new AutogradNode(new FastTensor<float>(2, 1));
            ((Span<float>)[10.0f, 20.0f]).CopyTo(input.Data.AsSpan());

            using var gamma = new AutogradNode(new FastTensor<float>(1));
            using var beta = new AutogradNode(new FastTensor<float>(1));
            gamma.Data[0] = 1.0f;
            beta.Data[0] = 0.0f;

            using var rm = new FastTensor<float>(1);
            using var rv = new FastTensor<float>(1);
            rv[0] = 1.0f;

            using var res = TensorMath.BatchNorm1D(graph, input, gamma, beta, rm, rv, 0.1f, 1e-5f, true);
            Assert.Equal(1.0f, res.Data[1, 0], 1e-3f);

            graph.Backward(res);
            Assert.NotNull(input.Grad);
            Assert.NotNull(gamma.Grad);
        }

        [Fact]
        public void Tensor_RequiresGradFalse_DoesNotCalculateGradient()
        {
            var graph = new ComputationGraph();
            using var matA = new FastTensor<float>(1, 1);
            using var matB = new FastTensor<float>(1, 1);

            using var a = new AutogradNode(matA, false);
            using var b = new AutogradNode(matB, true);

            using var c = TensorMath.Add(graph, a, b);
            graph.Backward(c);

            Assert.Null(a.Grad);
            Assert.NotNull(b.Grad);
            Assert.Equal(1.0f, b.Grad[0, 0]);
        }
    }
}