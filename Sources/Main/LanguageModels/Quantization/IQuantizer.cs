namespace DevOnBike.Overfit.LanguageModels.Quantization
{
    public interface IQuantizer
    {
        QuantizationOptions Options { get; }

        int QuantizeWeights(
            ReadOnlySpan<float> source,
            Span<sbyte> destination,
            Span<float> scales,
            Span<int> zeroPoints);

        int DequantizeWeights(
            ReadOnlySpan<sbyte> source,
            ReadOnlySpan<float> scales,
            ReadOnlySpan<int> zeroPoints,
            Span<float> destination);
    }
}
