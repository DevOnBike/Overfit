namespace DevOnBike.Overfit.LanguageModels.Contracts
{
    public readonly struct GenerationStats
    {
        public GenerationStats(
            int promptTokens,
            int generatedTokens,
            long elapsedNanoseconds,
            long allocatedBytes,
            bool usedKeyValueCache)
        {
            PromptTokens = promptTokens;
            GeneratedTokens = generatedTokens;
            ElapsedNanoseconds = elapsedNanoseconds;
            AllocatedBytes = allocatedBytes;
            UsedKeyValueCache = usedKeyValueCache;
        }

        public int PromptTokens { get; }

        public int GeneratedTokens { get; }

        public long ElapsedNanoseconds { get; }

        public long AllocatedBytes { get; }

        public bool UsedKeyValueCache { get; }

        public double NanosecondsPerToken =>
            GeneratedTokens == 0 ? 0.0 : (double)ElapsedNanoseconds / GeneratedTokens;

        public double TokensPerSecond =>
            ElapsedNanoseconds == 0 ? 0.0 : GeneratedTokens / (ElapsedNanoseconds / 1_000_000_000.0);
    }
}
