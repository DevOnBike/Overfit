namespace DevOnBike.Overfit.LanguageModels.Quantization
{
    public enum QuantizationScaleKind
    {
        PerTensor = 0,
        PerChannel = 1,
        PerGroup = 2
    }
}