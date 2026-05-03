namespace DevOnBike.Overfit.LanguageModels.Contracts
{
    public enum SamplingStrategy
    {
        Greedy = 0,
        Temperature = 1,
        TopK = 2,
        TopP = 3,
        TopKTopP = 4
    }
}