// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Numerics.Tensors;
using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.DeepLearning.Abstractions;
using DevOnBike.Overfit.Ops;

namespace DevOnBike.Overfit.DeepLearning
{
    /// <summary>
    /// Sigmoid activation: f(x) = 1 / (1 + exp(-x)), output range (0, 1).
    /// Commonly used in binary classification output layers and gating mechanisms.
    /// </summary>
    public sealed class SigmoidActivation : IModule
    {
        public bool IsTraining { get; private set; } = true;

        public void Train() => IsTraining = true;

        public void Eval() => IsTraining = false;

        public AutogradNode Forward(ComputationGraph graph, AutogradNode input)
            => TensorMath.Sigmoid(graph, input);

        public void ForwardInference(ReadOnlySpan<float> input, Span<float> output)
            => TensorPrimitives.Sigmoid(input, output);

        public IEnumerable<AutogradNode> Parameters() => [];

        public void InvalidateParameterCaches() { }

        public void Save(BinaryWriter bw) { }

        public void Load(BinaryReader br) { }

        public void Dispose() { }
    }
}
