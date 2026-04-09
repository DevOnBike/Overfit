// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Monitoring
{
    /// <summary>
    /// A single metric sample for one pod.
    /// Readonly struct — zero allocations when passing through the pipeline.
    /// </summary>
    public readonly struct MetricSnapshot
    {
        public DateTime Timestamp { get; init; }
        public string PodName { get; init; }

        // --- resources ---
        public float CpuUsage { get; init; }   // 0-1 (normalized to pod limit)
        public float MemoryBytes { get; init; }   // RSS bytes

        // --- HTTP/gRPC traffic ---
        public float RequestLatencyP95 { get; init; }   // ms
        public float RequestsPerSecond { get; init; }
        public float ErrorRate { get; init; }   // 0-1

        // --- .NET runtime ---
        public float GcPauseMs { get; init; }   // total GC time in the sample window
        public float ThreadPoolQueue { get; init; }   // ThreadPool queue length
        public float HeapBytes { get; init; }   // managed heap bytes

        /// <summary>
        /// Feature vector size — pipeline and model contract.
        /// Changing this requires rebuilding the autoencoder weights.
        /// </summary>
        public const int FeatureCount = 8;

        /// <summary>
        /// Writes the feature vector to the specified <paramref name="destination"/>.
        /// Zero allocation — the only allowed path in the production pipeline.
        /// The order is fixed and must match the encoder's expectations.
        /// </summary>
        public void WriteFeatureVector(Span<float> destination)
        {
            if (destination.Length < FeatureCount)
            {
                throw new ArgumentException($"Destination za krótki: potrzeba {FeatureCount}, dostępne {destination.Length}.", nameof(destination));
            }

            destination[0] = CpuUsage;
            destination[1] = MemoryBytes;
            destination[2] = RequestLatencyP95;
            destination[3] = RequestsPerSecond;
            destination[4] = ErrorRate;
            destination[5] = GcPauseMs;
            destination[6] = ThreadPoolQueue;
            destination[7] = HeapBytes;
        }
    }
}