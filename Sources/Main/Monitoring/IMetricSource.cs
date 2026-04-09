// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Monitoring
{
    /// <summary>
    /// Kontrakt źródła metryk.
    /// ValueTask — synchroniczne ścieżki (testy, buforowane odczyty) nie alokują Taska.
    /// </summary>
    public interface IMetricSource : IDisposable
    {
        string PodName { get; }

        /// <summary>
        /// Odczytuje aktualną próbkę metryk.
        /// Implementacja może czekać do końca okna scrapingu przed zwrotem.
        /// </summary>
        ValueTask<MetricSnapshot> ReadAsync(CancellationToken ct = default);
    }
}