// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Monitoring
{
    /// <summary>
    /// Pojedyncza próbka metryk jednego poda.
    /// Readonly struct — zero alokacji przy przesyłaniu przez pipeline.
    /// </summary>
    public readonly struct MetricSnapshot
    {
        public DateTime Timestamp { get; init; }
        public string PodName { get; init; }

        // --- zasoby ---
        public float CpuUsage { get; init; }   // 0–1 (znormalizowane do limitu poda)
        public float MemoryBytes { get; init; }   // bajty RSS

        // --- ruch HTTP/gRPC ---
        public float RequestLatencyP95 { get; init; }   // ms
        public float RequestsPerSecond { get; init; }
        public float ErrorRate { get; init; }   // 0–1

        // --- runtime .NET ---
        public float GcPauseMs { get; init; }   // sumaryczny czas GC w oknie próbki
        public float ThreadPoolQueue { get; init; }   // długość kolejki ThreadPool
        public float HeapBytes { get; init; }   // bajty managed heap

        /// <summary>
        /// Rozmiar wektora cech — kontrakt pipeline'u i modelu.
        /// Zmiana wymaga przebudowy wag autoenkodera.
        /// </summary>
        public const int FeatureCount = 8;

        /// <summary>
        /// Zapisuje wektor cech do podanego <paramref name="destination"/>.
        /// Zero alokacji — jedyna dozwolona ścieżka w pipeline produkcyjnym.
        /// Kolejność jest stała i musi odpowiadać oczekiwaniom encodera.
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