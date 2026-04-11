namespace DevOnBike.Overfit.Anomalies.Monitoring.Contracts
{
    public sealed class AlignerOptions
    {
        /// <summary>
        /// Number of time steps in a single window (e.g. 60 steps × 15s = 15 minutes of history).
        /// Must match the LSTM input sequence length — changing this requires retraining the model.
        /// </summary>
        public int WindowSize { get; init; } = 60;

        /// <summary>
        /// Prometheus scrape interval in seconds. Defines the spacing of the alignment grid.
        /// Must match the step parameter used in /api/v1/query_range calls.
        /// </summary>
        public int StepSeconds { get; init; } = 15;

        /// <summary>
        /// Number of metrics per time step. Determines the feature dimension of the output tensor.
        /// Must match the number of distinct MetricTypeId values ingested from Prometheus.
        /// Changing this requires retraining the model.
        /// </summary>
        public int MetricCount { get; init; } = 12;

        /// <summary>
        /// Maximum number of consecutive missing samples that will be filled by forward-fill.
        /// Gaps exceeding this limit are left as NaN. A value of 2 tolerates up to 30s of data loss
        /// at the default 15s step without discarding the window.
        /// </summary>
        public int MaxGapSteps { get; init; } = 2;

        /// <summary>Alignment grid step in milliseconds.</summary>
        public int StepMs => StepSeconds * 1000;

        /// <summary>
        /// Nearest-neighbour search tolerance: half the grid step.
        /// A sample is accepted for a grid point if it falls within this distance.
        /// </summary>
        public int ToleranceMs => StepMs / 2;
    }
}