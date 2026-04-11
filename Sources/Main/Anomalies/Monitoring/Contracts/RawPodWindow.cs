namespace DevOnBike.Overfit.Anomalies.Monitoring.Contracts
{
    /// <summary>
    /// Wyrównane okno jednego poda [WindowSize × MetricCount] row-major.
    /// Dane nienormalizowane. DC zachowane do agregacji fleet.
    /// PodName trzymany zewnętrznie w PodIndex[].
    /// </summary>
    public sealed class RawPodWindow
    {
        public DataCenter DC { get; init; }
        public float[] Data { get; init; } = [];
    }

}