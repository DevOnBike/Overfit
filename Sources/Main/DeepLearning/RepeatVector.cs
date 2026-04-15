// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.

using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.DeepLearning.Abstractions;
using DevOnBike.Overfit.Ops;

namespace DevOnBike.Overfit.DeepLearning
{
    /// <summary>
    ///     Repeats a latent vector n times along a new sequence dimension.
    ///     Input:  [batch, hiddenSize]
    ///     Output: [batch, seqLen, hiddenSize]
    /// </summary>
    public sealed class RepeatVector : IModule
    {
        private readonly int _seqLen;

        public RepeatVector(int seqLen)
        {
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(seqLen);
            _seqLen = seqLen;
        }

        public bool IsTraining { get; private set; } = true;

        public void Train() => IsTraining = true;
        public void Eval() => IsTraining = false;

        public void ForwardInference(ReadOnlySpan<float> input, Span<float> output)
        {
            for (var t = 0; t < _seqLen; t++)
            {
                input.CopyTo(output.Slice(t * input.Length, input.Length));
            }
        }

        public AutogradNode Forward(ComputationGraph graph, AutogradNode input)
        {
            return TensorMath.RepeatVector(graph, input, _seqLen);
        }

        public IEnumerable<AutogradNode> Parameters() => [];

        public void Save(BinaryWriter bw) { }
        public void Load(BinaryReader br) { }
        public void Dispose() { }
    }
}