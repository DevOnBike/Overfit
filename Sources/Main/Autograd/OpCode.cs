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
        FrozenQuantizedLinear
    }
}