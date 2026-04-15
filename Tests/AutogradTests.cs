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
            using var a = new AutogradNode(new FastTensor<float>(1, 2, clearMemory: true), requiresGrad: true);
            using var b = new AutogradNode(new FastTensor<float>(1, 2, clearMemory: true), requiresGrad: true);

            ((Span<float>)[1.0f, 2.0f]).CopyTo(a.DataView.AsSpan());
            ((Span<float>)[3.0f, 4.0f]).CopyTo(b.DataView.AsSpan());

            using var res = TensorMath.Add(graph, a, b);
            Assert.Equal([4.0f, 6.0f], res.DataView.AsSpan().ToArray());

            graph.Backward(res);
            Assert.Equal([1.0f, 1.0f], a.GradView.AsSpan().ToArray());
            Assert.Equal([1.0f, 1.0f], b.GradView.AsSpan().ToArray());
        }

        [Fact]
        public void BatchNorm1D_ForwardAndBackward_Flows()
        {
            var graph = new ComputationGraph();
            using var input = new AutogradNode(new FastTensor<float>(2, 1, clearMemory: true), requiresGrad: true);
            ((Span<float>)[10.0f, 20.0f]).CopyTo(input.DataView.AsSpan());

            using var gamma = new AutogradNode(new FastTensor<float>(1, clearMemory: true), requiresGrad: true);
            using var beta = new AutogradNode(new FastTensor<float>(1, clearMemory: true), requiresGrad: true);
            gamma.DataView.AsSpan()[0] = 1.0f;
            beta.DataView.AsSpan()[0] = 0.0f;

            using var rm = new FastTensor<float>(1, clearMemory: true);
            using var rv = new FastTensor<float>(1, clearMemory: true);
            rv.GetView().AsSpan()[0] = 1.0f;

            using var res = TensorMath.BatchNorm1D(graph, input, gamma, beta, rm, rv, 0.1f, 1e-5f, true);
            Assert.Equal(0.70710677f, res.DataView[1, 0], 1e-3f);

            graph.Backward(res);

            // Jeżeli RequiresGrad = true, GradView bez błędu udostępni Span z gradientami
            Assert.False(input.GradView.AsSpan().IsEmpty);
            Assert.False(gamma.GradView.AsSpan().IsEmpty);
        }

        [Fact]
        public void Tensor_RequiresGradFalse_DoesNotCalculateGradient()
        {
            var graph = new ComputationGraph();
            using var matA = new FastTensor<float>(1, 1, clearMemory: true);
            using var matB = new FastTensor<float>(1, 1, clearMemory: true);

            using var a = new AutogradNode(matA, requiresGrad: false);
            using var b = new AutogradNode(matB, requiresGrad: false);

            using var res = TensorMath.Add(graph, a, b);
            graph.Backward(res);

            // Pobieranie GradView dla węzła z RequiresGrad = false wyrzuca wyjątek.
            // Używamy odrzucenia (discard `_ = `), aby stworzyć poprawną instrukcję void
            // i uniknąć błędu pakowania (boxing) super-szybkiej struktury `ref struct`.
            Assert.Throws<InvalidOperationException>(() => { _ = a.GradView; });
            Assert.Throws<InvalidOperationException>(() => { _ = b.GradView; });
        }
    }
}