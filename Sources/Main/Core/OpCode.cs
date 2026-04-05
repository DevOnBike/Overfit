// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Core
{
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
        MSELoss,
        SoftmaxCrossEntropy,
        Reshape,
        DirectionalLoss
    }
}