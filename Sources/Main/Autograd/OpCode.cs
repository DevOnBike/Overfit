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
        ScaledDotProductAttention
    }
}