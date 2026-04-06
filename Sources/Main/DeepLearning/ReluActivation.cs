// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Core;

namespace DevOnBike.Overfit.DeepLearning
{
    /// <summary>
    /// Implements the Rectified Linear Unit (ReLU) activation function as a standalone module.
    /// Formula: f(x) = max(0, x).
    /// </summary>
    public sealed class ReluActivation : IModule
    {
        public bool IsTraining { get; private set; } = true;

        public void Train() => IsTraining = true;
        public void Eval() => IsTraining = false;

        /// <summary>
        /// Applies the ReLU operation via the global math engine.
        /// </summary>
        public AutogradNode Forward(ComputationGraph graph, AutogradNode input)
        {
            return TensorMath.ReLU(graph, input);
        }

        /// <summary>
        /// ReLU is a non-parametric function and contains no learnable weights.
        /// </summary>
        public IEnumerable<AutogradNode> Parameters()
        {
            return [];
        }

        public void Save(BinaryWriter bw) { }
        public void Load(BinaryReader br) { }

        public void Dispose() { }
    }
}