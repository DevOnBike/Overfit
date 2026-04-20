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
    ///     Implements the hyperbolic tangent (tanh) activation function as a standalone module.
    ///     Formula: f(x) = tanh(x), with output range (-1, 1).
    /// </summary>
    /// <remarks>
    ///     Useful as the output activation for policies that produce bounded continuous actions
    ///     (e.g. control signals in [-1, 1]) or as an intermediate activation where a
    ///     zero-centred nonlinearity is preferred over ReLU.
    /// </remarks>
    public sealed class TanhActivation : IModule
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
            return TensorMath.Tanh(graph, input);
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
            // TensorPrimitives.Tanh uses SIMD where available.
            TensorPrimitives.Tanh(input, output);
        }

        public void InvalidateParameterCaches()
        {
            // Activation modules hold no parameters, so there are no derived caches to
            // invalidate. Implemented explicitly (rather than relying on the default
            // interface method) to keep the class buildable against any version of IModule
            // the surrounding repo may have.
        }

        public void Dispose()
        {
        }
    }
}