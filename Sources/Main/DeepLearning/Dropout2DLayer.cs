// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.DeepLearning.Abstractions;

namespace DevOnBike.Overfit.DeepLearning
{
    /// <summary>
    /// Spatial-dropout layer for conv feature maps <c>[N, C, H, W]</c> — drops whole channels during
    /// training (each channel's H·W block kept or zeroed together, inverted-scaled), identity at
    /// inference. No trainable parameters. Place after a conv/activation (or pool) to regularise.
    /// </summary>
    public sealed class Dropout2DLayer : IModule
    {
        private readonly float _probability;

        public Dropout2DLayer(float probability = 0.25f)
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
            => graph.Dropout2D(input, _probability, IsTraining);

        public void ForwardInference(ReadOnlySpan<float> input, Span<float> output)
            => input.CopyTo(output);   // dropout is identity at inference

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
