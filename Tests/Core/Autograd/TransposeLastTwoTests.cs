// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.Tensors;
using DevOnBike.Overfit.Tensors.Core;

namespace DevOnBike.Overfit.Tests.Core.Autograd
{
    /// <summary>
    /// <see cref="ComputationGraph.TransposeLastTwo"/> — forward reorders the last two axes; its
    /// backward is itself a transpose of the upstream gradient (exact, no tolerance needed).
    /// </summary>
    public sealed class TransposeLastTwoTests
    {
        [Fact]
        public void Forward_Transposes_2D()
        {
            using var graph = new ComputationGraph(1 << 16);
            var store = new TensorStorage<float>(6, clearMemory: false);
            // [2,3] = [[1,2,3],[4,5,6]]
            for (var i = 0; i < 6; i++)
            {
                store.AsSpan()[i] = i + 1;
            }
            using var x = new AutogradNode(store, new TensorShape(2, 3), requiresGrad: true);

            var y = graph.TransposeLastTwo(x);   // [3,2] = [[1,4],[2,5],[3,6]]

            Assert.Equal(new[] { 1f, 4f, 2f, 5f, 3f, 6f }, y.DataView.AsReadOnlySpan().ToArray());
        }

        [Fact]
        public void Backward_IsTransposeOfUpstreamGradient()
        {
            using var graph = new ComputationGraph(1 << 16);
            var store = new TensorStorage<float>(6, clearMemory: true);
            using var x = new AutogradNode(store, new TensorShape(2, 3), requiresGrad: true);

            var y = graph.TransposeLastTwo(x);   // [3,2]
            // Seed dL/dy = [[10,20],[30,40],[50,60]] (shape [3,2]).
            var g = y.GradView.AsSpan();
            for (var i = 0; i < 6; i++)
            {
                g[i] = (i + 1) * 10f;
            }

            graph.BackwardFromGrad(y);

            // dL/dx is the transpose back to [2,3]: [[10,30,50],[20,40,60]].
            Assert.Equal(new[] { 10f, 30f, 50f, 20f, 40f, 60f }, x.GradView.AsReadOnlySpan().ToArray());
        }

        [Fact]
        public void Forward_Transposes_3D_Batched()
        {
            using var graph = new ComputationGraph(1 << 16);
            var store = new TensorStorage<float>(12, clearMemory: false);
            // [2,2,3]: batch0 = [[1,2,3],[4,5,6]], batch1 = [[7,8,9],[10,11,12]]
            for (var i = 0; i < 12; i++)
            {
                store.AsSpan()[i] = i + 1;
            }
            using var x = new AutogradNode(store, new TensorShape(2, 2, 3), requiresGrad: true);

            var y = graph.TransposeLastTwo(x);   // [2,3,2]

            // batch0 transposed = [[1,4],[2,5],[3,6]]; batch1 = [[7,10],[8,11],[9,12]]
            Assert.Equal(
                new[] { 1f, 4f, 2f, 5f, 3f, 6f, 7f, 10f, 8f, 11f, 9f, 12f },
                y.DataView.AsReadOnlySpan().ToArray());
        }
    }
}
