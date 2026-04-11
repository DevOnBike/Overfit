// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Core;

namespace DevOnBike.Overfit.DeepLearning
{
    /// <summary>
    /// Repeats a latent vector n times along a new sequence dimension.
    ///
    /// Input:  [batch, hiddenSize]
    /// Output: [batch, seqLen, hiddenSize]
    ///
    /// Used in LSTM decoder — expands the encoder's final hidden state
    /// into a full sequence so the decoder can reconstruct each timestep.
    /// </summary>
    public sealed class RepeatVector : IModule
    {
        private readonly int _seqLen;

        public bool IsTraining { get; private set; } = true;

        public RepeatVector(int seqLen)
        {
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(seqLen);
            _seqLen = seqLen;
        }

        public void Train() => IsTraining = true;
        public void Eval() => IsTraining = false;

        // ---------------------------------------------------------------------------
        // Forward
        // ---------------------------------------------------------------------------

        /// <summary>
        /// Repeats input [batch, hiddenSize] → [batch, seqLen, hiddenSize].
        /// Each timestep in output is an identical copy of the input row.
        /// </summary>
        public AutogradNode Forward(ComputationGraph graph, AutogradNode input)
        {
            return TensorMath.RepeatVector(graph, input, _seqLen);
        }

        // ---------------------------------------------------------------------------
        // IModule
        // ---------------------------------------------------------------------------

        public IEnumerable<AutogradNode> Parameters() => [];

        public void Save(BinaryWriter bw) {}
        public void Load(BinaryReader br) {}
        public void Dispose() {}
    }
}