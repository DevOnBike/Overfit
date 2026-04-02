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
        Reshape
    }
}