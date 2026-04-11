namespace DevOnBike.Overfit.Data.Contracts
{
    /// <summary>Full report returned by <see cref="FeatureImportanceAnalyzer.Analyze"/>.</summary>
    public sealed record FeatureImportanceReport
    {
        /// <summary>Per-dimension results ordered by descending MeanImportance.</summary>
        public IReadOnlyList<FeatureImportanceResult> Results { get; init; } = [];

        /// <summary>Features classified as <see cref="FeatureImportanceVerdict.Confirmed"/>.</summary>
        public IReadOnlyList<FeatureImportanceResult> Confirmed { get; init; } = [];

        /// <summary>Features classified as <see cref="FeatureImportanceVerdict.Tentative"/>.</summary>
        public IReadOnlyList<FeatureImportanceResult> Tentative { get; init; } = [];

        /// <summary>Features classified as <see cref="FeatureImportanceVerdict.Rejected"/>.</summary>
        public IReadOnlyList<FeatureImportanceResult> Rejected { get; init; } = [];

        /// <summary>Number of training samples used for the analysis.</summary>
        public int SampleCount { get; init; }

        /// <summary>Number of permutation iterations run per feature.</summary>
        public int Iterations { get; init; }
    }
}