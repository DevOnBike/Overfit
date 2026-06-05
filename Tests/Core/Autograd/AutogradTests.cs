// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.Tensors;
using DevOnBike.Overfit.Tensors.Core;

namespace DevOnBike.Overfit.Tests.Core.Autograd
{
    public class AutogradTests
    {
        private const int Precision = 6;

        [Fact]
        public void TensorAdd_ForwardAndBackward_Correct()
        {
            var graph = new ComputationGraph();
            using var a = new AutogradNode(new TensorStorage<float>(2, clearMemory: true), new TensorShape(1, 2), requiresGrad: true);
            using var b = new AutogradNode(new TensorStorage<float>(2, clearMemory: true), new TensorShape(1, 2), requiresGrad: true);

            ((Span<float>)[1.0f, 2.0f]).CopyTo(a.DataView.AsSpan());
            ((Span<float>)[3.0f, 4.0f]).CopyTo(b.DataView.AsSpan());

            using var res = Ops.TensorMath.Add(graph, a, b);
            Assert.Equal([4.0f, 6.0f], res.DataView.AsSpan().ToArray());

            graph.Backward(res);
            Assert.Equal([1.0f, 1.0f], a.GradView.AsSpan().ToArray());
            Assert.Equal([1.0f, 1.0f], b.GradView.AsSpan().ToArray());
        }

        [Fact]
        public void TensorBatchNorm_ForwardAndBackward_Correct()
        {
            var graph = new ComputationGraph();
            using var input = new AutogradNode(new TensorStorage<float>(2, clearMemory: true), new TensorShape(2), requiresGrad: true);
            ((Span<float>)[2.0f, 4.0f]).CopyTo(input.DataView.AsSpan());

            using var gamma = new AutogradNode(new TensorStorage<float>(1, clearMemory: true), new TensorShape(1), requiresGrad: true);
            gamma.DataView.AsSpan()[0] = 1.0f;

            using var beta = new AutogradNode(new TensorStorage<float>(1, clearMemory: true), new TensorShape(1), requiresGrad: true);
            beta.DataView.AsSpan()[0] = 0.0f;

            using var rm = new TensorStorage<float>(1, clearMemory: true);
            using var rv = new TensorStorage<float>(1, clearMemory: true);
            rv.AsSpan()[0] = 1.0f;

            using var res = Ops.TensorMath.BatchNorm1D(graph, input, gamma, beta, rm, rv, 0.1f, 1e-5f, true);
            Assert.Equal(1.0f, res.DataView[1, 0], 1e-3f);

            graph.Backward(res);

            // If RequiresGrad = true, GradView will expose a Span with gradients without error
            Assert.False(input.GradView.AsSpan().IsEmpty);
            Assert.False(gamma.GradView.AsSpan().IsEmpty);
        }

        [Fact]
        public void Tensor_RequiresGradFalse_DoesNotCalculateGradient()
        {
            var graph = new ComputationGraph();
            using var matA = new TensorStorage<float>(1, clearMemory: true);
            using var matB = new TensorStorage<float>(1, clearMemory: true);

            using var a = new AutogradNode(matA, new TensorShape(1), requiresGrad: false);
            using var b = new AutogradNode(matB, new TensorShape(1), requiresGrad: false);

            using var res = Ops.TensorMath.Add(graph, a, b);
            graph.Backward(res);

            // Accessing GradView for a node with RequiresGrad = false throws an exception.
            Assert.Throws<OverfitRuntimeException>(() => _ = a.GradView);
            Assert.Throws<OverfitRuntimeException>(() => _ = b.GradView);
        }
    }
}