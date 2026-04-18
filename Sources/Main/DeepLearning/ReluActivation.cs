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
    ///     Implements the Rectified Linear Unit (ReLU) activation function as a standalone module.
    ///     Formula: f(x) = max(0, x).
    /// </summary>
    public sealed class ReluActivation : IModule
    {
        public bool IsTraining { get; private set; } = true;

        public void Train()
        {
            IsTraining = true;
        }

        public void Eval()
        {
            IsTraining = false;
        }

        public AutogradNode Forward(ComputationGraph graph, AutogradNode input)
        {
            return TensorMath.ReLU(graph, input);
        }

        public IEnumerable<AutogradNode> Parameters()
        {
            return [];
        }

        public void Save(BinaryWriter bw)
        {

        }

        public void Load(BinaryReader br)
        {

        }

        public void ForwardInference(ReadOnlySpan<float> input, Span<float> output)
        {
            // TensorPrimitives natywnie używa AVX-512 do operacji Max
            TensorPrimitives.Max(input, 0f, output);
        }

        public void Dispose()
        {
        }

        public void InvalidateParameterCaches()
        {
        }
    }
}