// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.DeepLearning.Abstractions;
using DevOnBike.Overfit.Tensors;

namespace DevOnBike.Overfit.DeepLearning
{
    /// <summary>
    /// Reshapes a tensor of shape [batch, C, H, W] to [batch, C*H*W].
    /// Implemented as a zero-copy view — data layout in memory is unchanged,
    /// only the TensorShape metadata is updated via <see cref="AutogradNode.ViewOf"/>.
    /// </summary>
    public sealed class FlattenLayer : IModule
    {
        public bool IsTraining { get; private set; } = true;

        public void Train() => IsTraining = true;

        public void Eval() => IsTraining = false;

        public AutogradNode Forward(ComputationGraph graph, AutogradNode input)
        {
            // TensorShape.Flatten2D() computes new TensorShape(D0, D1*D2*D3).
            // ViewOf shares the underlying TensorStorage — zero allocation, zero copy.
            var flatShape = input.Shape.Flatten2D();
            return AutogradNode.ViewOf(input, flatShape, input.RequiresGrad);
        }

        public void ForwardInference(ReadOnlySpan<float> input, Span<float> output)
        {
            // Flatten is element-identity — data layout does not change.
            input.CopyTo(output);
        }

        public void InvalidateParameterCaches() { }

        public IEnumerable<AutogradNode> Parameters() => [];

        public void Save(BinaryWriter bw) { }

        public void Load(BinaryReader br) { }

        public void Dispose() { }
    }
}
