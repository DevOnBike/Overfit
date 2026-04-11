namespace DevOnBike.Overfit.Anomalies.Monitoring.Contracts
{
    public sealed class RawMetricSeries
    {
        public PodKey Pod { get; init; }
        
        public byte MetricTypeId { get; init; }
        
        public List<RawSample> Samples { get; } = [];

        public int Length => Samples.Count;
    }
}