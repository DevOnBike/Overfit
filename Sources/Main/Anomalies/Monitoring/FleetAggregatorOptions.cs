namespace DevOnBike.Overfit.Anomalies.Monitoring
{
    public sealed class FleetAggregatorOptions
    {
        /// <summary>Must match AlignerOptions.WindowSize.</summary>
        public int WindowSize { get; init; } = 60;

        /// <summary>Must match AlignerOptions.MetricCount.</summary>
        public int MetricCount { get; init; } = 12;
    }
}