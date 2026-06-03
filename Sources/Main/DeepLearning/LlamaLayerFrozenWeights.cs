// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Autograd;

namespace DevOnBike.Overfit.DeepLearning
{
    /// <summary>
    /// The frozen quantized weights of one decoder layer for <see cref="TrainableLlamaModel"/> — the
    /// combined-tensor projections plus the two RMSNorm gain initializers (copied into trainable γ).
    /// </summary>
    public sealed class LlamaLayerFrozenWeights
    {
        public required IDequantRowSource Wq { get; init; }
        public required IDequantRowSource Wk { get; init; }
        public required IDequantRowSource Wv { get; init; }
        public required IDequantRowSource Wo { get; init; }
        public required IDequantRowSource Gate { get; init; }
        public required IDequantRowSource Up { get; init; }
        public required IDequantRowSource Down { get; init; }
        public required float[] Ln1GammaInit { get; init; }
        public required float[] Ln2GammaInit { get; init; }
    }
}
