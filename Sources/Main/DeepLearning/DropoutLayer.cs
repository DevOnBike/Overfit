// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.DeepLearning.Abstractions;

namespace DevOnBike.Overfit.DeepLearning
{
    /// <summary>
    /// Standard element-wise dropout layer (inverted dropout, identity at inference). No trainable
    /// parameters — drop into an MLP/Sequential after an activation to regularise.
    /// </summary>
    public sealed class DropoutLayer : IModule
    {
        private readonly float _probability;

        public DropoutLayer(float probability = 0.5f)
        {
            if (probability is < 0f or >= 1f)
            {
                throw new ArgumentOutOfRangeException(nameof(probability), "probability must be in [0,1).");
            }
            _probability = probability;
        }

        public bool IsTraining { get; private set; } = true;

        public void Train() => IsTraining = true;
        public void Eval() => IsTraining = false;

        public AutogradNode Forward(ComputationGraph graph, AutogradNode input)
            => graph.Dropout(input, _probability, IsTraining);

        public void ForwardInference(ReadOnlySpan<float> input, Span<float> output)
            => input.CopyTo(output);

        public IEnumerable<AutogradNode> Parameters() => [];

        public void InvalidateParameterCaches()
        {
        }

        public void Save(BinaryWriter bw)
        {
        }

        public void Load(BinaryReader br)
        {
        }

        public void Dispose()
        {
        }
    }
}
