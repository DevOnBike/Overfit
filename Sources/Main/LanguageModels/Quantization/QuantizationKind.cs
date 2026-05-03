namespace DevOnBike.Overfit.LanguageModels.Quantization
{
    public enum QuantizationKind
    {
        None = 0,
        Int8WeightOnly = 1,
        Int8WeightAndActivation = 2
    }
}