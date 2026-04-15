// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Diagnostics;
using System.Diagnostics.Metrics;

namespace DevOnBike.Overfit.Diagnostics
{
    namespace DevOnBike.Overfit.Diagnostics
    {
        public static class OverfitTelemetry
        {
            public const string MeterName = "DevOnBike.Overfit";
            public const string Version = "1.0.0";

            // 1. Zegary i Liczniki
            public static readonly Meter Meter = new(MeterName, Version);

            // Histogram pozwala Grafanie rysować percentyle (p95, p99) opóźnień predykcji
            public static readonly Histogram<double> InferenceDuration =
                Meter.CreateHistogram<double>("overfit.inference.duration_ms", "ms", "Czas wykonania predykcji");

            // Licznik predykcji
            public static readonly Counter<long> InferenceCount =
                Meter.CreateCounter<long>("overfit.inference.total", "{inferences}", "Całkowita liczba predykcji");

            // Śledzenie natywnego RAMu (którego GC nie widzi)
            public static readonly UpDownCounter<long> NativeMemoryBytes =
                Meter.CreateUpDownCounter<long>("overfit.memory.native_bytes", "By", "Ilość zaalokowanej natywnej pamięci");

            // 2. Rozproszone śledzenie (Traces dla np. Jaegera)
            public static readonly ActivitySource Tracer = new(MeterName, Version);
        }

    }
}