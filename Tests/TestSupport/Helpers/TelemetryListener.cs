// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Diagnostics.Metrics;
using DevOnBike.Overfit.Diagnostics;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests.Helpers
{
    public class TelemetryListener : IDisposable
    {
        private readonly ITestOutputHelper _output;
        private readonly MeterListener _listener;

        public TelemetryListener(ITestOutputHelper output)
        {
            _output = output;
            _listener = new MeterListener();
        }

        public void Subscribe()
        {
            Subscribe(OverfitTelemetry.MeterName);
        }

        public void Subscribe(string targetMeterName)
        {
            _listener.InstrumentPublished = (instrument, listener) =>
            {
                if (string.Equals(instrument.Meter.Name, targetMeterName, StringComparison.OrdinalIgnoreCase))
                {
                    listener.EnableMeasurementEvents(instrument);
                }
            };

            _listener.SetMeasurementEventCallback<double>(OnMeasurementDouble);
            _listener.SetMeasurementEventCallback<long>(OnMeasurementLong);
            _listener.MeasurementsCompleted += OnMeasurementCompleted;

            _listener.Start();
        }

        private void OnMeasurementCompleted(Instrument instrument, object? state)
        {
            _output.WriteLine($"[Telemetry] Zakończono pomiary dla: {instrument.Name}");
        }

        private void OnMeasurementDouble(
            Instrument instrument,
            double measurement,
            ReadOnlySpan<KeyValuePair<string, object?>> tags,
            object? state)
        {
            PrintMeasurement(instrument.Name, measurement.ToString("F4"), tags);
        }

        private void OnMeasurementLong(
            Instrument instrument,
            long measurement,
            ReadOnlySpan<KeyValuePair<string, object?>> tags,
            object? state)
        {
            PrintMeasurement(instrument.Name, measurement.ToString(), tags);
        }

        private void PrintMeasurement(string instrumentName, string value, ReadOnlySpan<KeyValuePair<string, object?>> tags)
        {
            var tagsString = string.Join(", ", tags.ToArray().Select(t => $"{t.Key}={t.Value}"));
            var formattedTags = string.IsNullOrEmpty(tagsString) ? "Brak tagów" : tagsString;

            _output.WriteLine($"[METRYKA] {instrumentName} = {value} | Tagi: [{formattedTags}]");
        }

        public void Dispose()
        {
            _listener.Dispose();
        }
    }
}