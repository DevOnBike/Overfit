using DevOnBike.Overfit.Data.Abstractions;
namespace DevOnBike.Overfit.Data
{
    /// <summary>
    /// Generic fallback provider that names features "feature_0", "feature_1" etc.
    /// Use when domain-specific names are not available or not needed.
    /// </summary>
    public sealed class IndexedFeatureNameProvider : IFeatureNameProvider
    {
        public int ExpectedDimension { get; }
        
        private readonly string _prefix;

        /// <param name="expectedDimension">
        ///   Expected vector length. Pass 0 to skip validation.
        /// </param>
        /// <param name="prefix">Name prefix. Default: "feature".</param>
        public IndexedFeatureNameProvider(int expectedDimension = 0, string prefix = "feature")
        {
            ArgumentException.ThrowIfNullOrEmpty(prefix);
            
            ExpectedDimension = expectedDimension;
            _prefix = prefix;
        }
        
        public string GetName(int featureIndex)
        {
            return $"{_prefix}_{featureIndex}";
        }
    }
}