namespace DevOnBike.Overfit.Monitoring.Contracts
{
    /// <summary>
    /// Pair of features whose Pearson correlation exceeds
    /// <see cref="TrainingDataAnalyzerConfig.HighCorrelationThreshold"/>.
    /// </summary>
    public sealed record CorrelationPair
    {
        public int FeatureIndexA { get; init; }
        public int FeatureIndexB { get; init; }
        public string FeatureNameA { get; init; } = string.Empty;
        public string FeatureNameB { get; init; } = string.Empty;

        /// <summary>Pearson r ∈ [-1, 1].</summary>
        public float Correlation { get; init; }
    }
}