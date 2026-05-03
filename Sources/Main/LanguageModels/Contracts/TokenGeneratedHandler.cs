namespace DevOnBike.Overfit.LanguageModels.Contracts
{
    public delegate bool TokenGeneratedHandler(
        int tokenId,
        int position,
        ReadOnlySpan<float> logits);
}