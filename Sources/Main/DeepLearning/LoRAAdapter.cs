// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.Tensors;
using DevOnBike.Overfit.Tensors.Core;

namespace DevOnBike.Overfit.DeepLearning
{
    /// <summary>
    /// A low-rank adaptation (LoRA) adapter for one linear projection: a trainable rank-<c>r</c>
    /// residual <c>ΔW = A·B</c> applied as <c>Δ(x) = (x·A)·B</c>, added on top of a frozen base
    /// projection. <c>A</c> is <c>[inDim, r]</c>, <c>B</c> is <c>[r, outDim]</c>. <b>B starts at zero</b>
    /// (standard LoRA init) so the adapter contributes nothing until trained — the model initially equals
    /// the frozen quantized base. Only A and B carry gradients; the base never does (QLoRA).
    /// </summary>
    public sealed class LoRAAdapter : IDisposable
    {
        private readonly TensorStorage<float> _aStore;
        private readonly TensorStorage<float> _bStore;

        public LoRAAdapter(int inDim, int outDim, int rank, Random rng)
        {
            Rank = rank;
            // A ~ small random (Kaiming-ish), B = 0.
            _aStore = new TensorStorage<float>(inDim * rank, clearMemory: false);
            var a = _aStore.AsSpan();
            var scale = (float)(1.0 / Math.Sqrt(inDim));
            for (var i = 0; i < a.Length; i++) { a[i] = (float)(rng.NextDouble() * 2 - 1) * scale; }

            _bStore = new TensorStorage<float>(rank * outDim, clearMemory: true); // zeros
            A = new AutogradNode(_aStore, new TensorShape(inDim, rank), requiresGrad: true);
            B = new AutogradNode(_bStore, new TensorShape(rank, outDim), requiresGrad: true);
        }

        public int Rank { get; }
        public AutogradNode A { get; }
        public AutogradNode B { get; }

        /// <summary>The LoRA residual for input <paramref name="x"/> (<c>[T, inDim]</c>): <c>(x·A)·B</c>.</summary>
        public AutogradNode Apply(ComputationGraph graph, AutogradNode x)
            => graph.MatMul(graph.MatMul(x, A), B);

        public void Dispose()
        {
            A.Dispose();
            B.Dispose();
            _aStore.Dispose();
            _bStore.Dispose();
        }
    }
}
