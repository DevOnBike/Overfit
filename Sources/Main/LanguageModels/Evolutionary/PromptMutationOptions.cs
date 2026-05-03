namespace DevOnBike.Overfit.LanguageModels.Evolutionary
{
    public readonly struct PromptMutationOptions
    {
        public PromptMutationOptions(
            int maxPromptTokens,
            int maxGeneratedTokens,
            float temperature,
            int topK,
            int seed)
        {
            MaxPromptTokens = maxPromptTokens;
            MaxGeneratedTokens = maxGeneratedTokens;
            Temperature = temperature;
            TopK = topK;
            Seed = seed;
        }

        public int MaxPromptTokens { get; }

        public int MaxGeneratedTokens { get; }

        public float Temperature { get; }

        public int TopK { get; }

        public int Seed { get; }
    }
}
