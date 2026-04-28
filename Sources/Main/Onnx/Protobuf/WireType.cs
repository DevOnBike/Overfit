namespace DevOnBike.Overfit.Onnx.Protobuf
{
    internal enum WireType
    {
        Varint = 0,
        Fixed64 = 1,
        LengthDelimited = 2,
        StartGroup = 3,    // Deprecated
        EndGroup = 4,      // Deprecated
        Fixed32 = 5,
    }
}