// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Autograd
{
    /// <summary>
    ///     Opcodes for the computation graph tape. Packed as byte for memory efficiency.
    /// </summary>
    public enum OpCode : byte
    {
        None,
        Add,
        Subtract,
        MatMul,
        AddBias,
        ReLU,
        Dropout,
        Conv2D,
        MaxPool2D,
        GlobalAveragePool2D,
        BatchNorm1D,
        BatchNorm2D,
        MseLoss,
        SoftmaxCrossEntropy,
        Reshape,
        DirectionalLoss,
        Sigmoid,
        Tanh,
        Multiply,
        GateSlice,
        TimestepSlice,
        StackTimesteps,
        RepeatVector,
        FusedLSTMStep,
        Linear,
        AddInPlace,
        LayerNorm,
        Embedding,
        ScaledDotProductAttention,
        Gelu,
        Checkpoint,
        TransposeLastTwo,
        DepthwiseConv2D,

        /// <summary>
        /// Linear projection through a FROZEN quantized weight (Q4_K / Q6_K), dequantized on the
        /// fly. The base weight is not an <see cref="AutogradNode"/> and never receives a gradient
        /// — only the INPUT gradient flows back (the QLoRA base path: frozen 4-bit weights in RAM,
        /// trainable adapters elsewhere). The weight is held in a graph-side list, index in I0.
        /// </summary>
        FrozenQuantizedLinear,

        /// <summary>RMS normalization (Llama-style): <c>y = x / sqrt(mean(x²)+eps) · gamma</c>.
        /// No mean subtraction, no beta. Saves per-row inv-RMS for backward.</summary>
        RmsNorm,

        /// <summary>SiLU / swish activation: <c>y = x · sigmoid(x)</c> (the SwiGLU gate).</summary>
        SiLU,

        /// <summary>Rotary Position Embedding (adjacent-pair / GGUF layout). Per-pair 2-D rotation
        /// by a precomputed angle; backward is the inverse rotation. cos/sin are constants.</summary>
        Rope,

        /// <summary>GQA KV-head broadcast (HF <c>repeat_kv</c>): expand <c>[kvHeads,…]</c> →
        /// <c>[kvHeads·groupSize,…]</c>; backward sums each group's gradient into the shared KV head.
        /// kvHeads in I0, groupSize in I1.</summary>
        ExpandKvHeads,

        /// <summary>Swap the first two axes of a rank-3 tensor (<c>[A,B,C]→[B,A,C]</c>) — the
        /// token-major ↔ head-major reshuffle for attention. Backward transposes back.</summary>
        Transpose01
    }
}