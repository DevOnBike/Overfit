using DevOnBike.Overfit.LanguageModels.Contracts;

namespace DevOnBike.Overfit.LanguageModels.Quantization
{
    public interface IQuantizedModel : ISlmModel
    {
        QuantizationOptions Quantization { get; }

        long QuantizedParameterBytes { get; }

        long OriginalParameterBytes { get; }

        double CompressionRatio { get; }
    }
}
