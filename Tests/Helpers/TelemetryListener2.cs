// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Buffers;
using System.Diagnostics.Metrics;
using System.Globalization;
using System.Text;
using DevOnBike.Overfit.Diagnostics;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests.Helpers
{
    public sealed class TelemetryListener2 : IDisposable
    {
        private readonly ITestOutputHelper _output;
        private readonly MeterListener _listener;
        private readonly object _gate = new();
        private readonly Dictionary<MetricKey, MetricAggregate> _metrics = new();

        private readonly bool _includeTags;
        private readonly int _maxRows;
        private readonly string? _metricNamePrefix;

        private string _targetMeterName = OverfitTelemetry.MeterName;
        private bool _started;
        private bool _disposed;

        public TelemetryListener2(ITestOutputHelper output)
            : this(output, includeTags: false, maxRows: 128, metricNamePrefix: null)
        {
        }

        public TelemetryListener2(
            ITestOutputHelper output,
            bool includeTags,
            int maxRows = 128,
            string? metricNamePrefix = null)
        {
            _output = output ?? throw new ArgumentNullException(nameof(output));
            _includeTags = includeTags;
            _maxRows = maxRows <= 0 ? 128 : maxRows;
            _metricNamePrefix = string.IsNullOrWhiteSpace(metricNamePrefix) ? null : metricNamePrefix;

            _listener = new MeterListener();
        }

        public int SeriesCount
        {
            get
            {
                lock (_gate)
                {
                    return _metrics.Count;
                }
            }
        }

        public void Subscribe() => Subscribe(OverfitTelemetry.MeterName);

        public void Subscribe(string targetMeterName)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);

            if (_started)
            {
                throw new InvalidOperationException("TelemetryListener is already started.");
            }

            _targetMeterName = string.IsNullOrWhiteSpace(targetMeterName)
                ? OverfitTelemetry.MeterName
                : targetMeterName;

            _listener.InstrumentPublished = OnInstrumentPublished;

            // ZMIANA ZERO-ALLOC: Silnie typowane rejestracje!
            // Żadnych generyków i ukrytego Boxingu struktur do "object".
            _listener.SetMeasurementEventCallback<double>(OnMeasurementDouble);
            _listener.SetMeasurementEventCallback<float>(OnMeasurementFloat);
            _listener.SetMeasurementEventCallback<long>(OnMeasurementLong);
            _listener.SetMeasurementEventCallback<int>(OnMeasurementInt);
            _listener.SetMeasurementEventCallback<short>(OnMeasurementShort);
            _listener.SetMeasurementEventCallback<byte>(OnMeasurementByte);

            _listener.Start();
            _started = true;
        }

        private void OnInstrumentPublished(Instrument instrument, MeterListener listener)
        {
            if (!string.Equals(instrument.Meter.Name, _targetMeterName, StringComparison.OrdinalIgnoreCase)) return;
            if (_metricNamePrefix is not null && !instrument.Name.StartsWith(_metricNamePrefix, StringComparison.Ordinal)) return;

            listener.EnableMeasurementEvents(instrument);
        }

        // ====================================================================
        // SILNIE TYPOWANE HANDLERY (Omijają Convert.ToDouble(object))
        // ====================================================================

        private void OnMeasurementDouble(Instrument inst, double measurement, ReadOnlySpan<KeyValuePair<string, object?>> tags, object? state)
            => RecordMeasurement(inst, measurement, tags);

        private void OnMeasurementFloat(Instrument inst, float measurement, ReadOnlySpan<KeyValuePair<string, object?>> tags, object? state)
            => RecordMeasurement(inst, measurement, tags);

        private void OnMeasurementLong(Instrument inst, long measurement, ReadOnlySpan<KeyValuePair<string, object?>> tags, object? state)
            => RecordMeasurement(inst, measurement, tags);

        private void OnMeasurementInt(Instrument inst, int measurement, ReadOnlySpan<KeyValuePair<string, object?>> tags, object? state)
            => RecordMeasurement(inst, measurement, tags);

        private void OnMeasurementShort(Instrument inst, short measurement, ReadOnlySpan<KeyValuePair<string, object?>> tags, object? state)
            => RecordMeasurement(inst, measurement, tags);

        private void OnMeasurementByte(Instrument inst, byte measurement, ReadOnlySpan<KeyValuePair<string, object?>> tags, object? state)
            => RecordMeasurement(inst, measurement, tags);

        // ====================================================================

        private void RecordMeasurement(Instrument instrument, double value, ReadOnlySpan<KeyValuePair<string, object?>> tags)
        {
            if (_disposed) return;

            // Jeśli tagi są wyłączone, ten string jest internowany w .NET (zero alokacji)
            string formattedTags = string.Empty;

            if (_includeTags && tags.Length > 0)
            {
                formattedTags = FormatTags(tags);
            }

            var key = new MetricKey(instrument.Name, instrument.Unit ?? string.Empty, formattedTags);

            lock (_gate)
            {
                if (_metrics.TryGetValue(key, out var aggregate))
                {
                    aggregate.Add(value);
                    // Struktura (value type), więc po prostu nadpisujemy
                    _metrics[key] = aggregate;
                }
                else
                {
                    // To wywoła się tylko RAZ dla każdej unikalnej serii metryk.
                    _metrics.Add(key, new MetricAggregate(value));
                }
            }
        }

        private static string FormatTags(ReadOnlySpan<KeyValuePair<string, object?>> tags)
        {
            if (tags.Length == 0) return string.Empty;

            // ZMIANA ZERO-ALLOC: Używamy ArrayPool zamiast stackalloc, 
            // ponieważ Tagi zawierają typy referencyjne (string i object).
            var rentedArray = ArrayPool<KeyValuePair<string, object?>>.Shared.Rent(tags.Length);

            try
            {
                // Ograniczamy widok tylko do faktycznej liczby tagów (ArrayPool może zwrócić dłuższą tablicę)
                var span = rentedArray.AsSpan(0, tags.Length);
                tags.CopyTo(span);

                // Sortowanie bez alokacji na spanie
                span.Sort((a, b) => string.CompareOrdinal(a.Key, b.Key));

                var sb = new StringBuilder(128);
                for (var i = 0; i < span.Length; i++)
                {
                    if (i != 0) sb.Append(", ");
                    sb.Append(span[i].Key);
                    sb.Append('=');
                    sb.Append(span[i].Value?.ToString() ?? "null");
                }

                return sb.ToString();
            }
            finally
            {
                // WAŻNE: Zwracamy tablicę do puli. 
                // clearArray: true jest tu krytyczne, bo tablica trzyma referencje do obiektów, 
                // a nie chcemy blokować GC przed ich posprzątaniem (wyciek pamięci).
                ArrayPool<KeyValuePair<string, object?>>.Shared.Return(rentedArray, clearArray: true);
            }
        }

        public void WriteSummary(string? title = null)
        {
            // ... (Zawartość tej metody pozostaje bez zmian, bo jest wywoływana tylko na końcu testu)
            MetricRow[] rows;
            lock (_gate)
            {
                rows = _metrics.Select(static pair => new MetricRow(pair.Key, pair.Value))
                    .OrderBy(static row => row.Key.Name, StringComparer.Ordinal)
                    .ThenBy(static row => row.Key.Tags, StringComparer.Ordinal).ToArray();
            }

            var sb = new StringBuilder(4096);
            sb.AppendLine(title is null ? "=== OVERFIT TELEMETRY SUMMARY ===" : $"=== OVERFIT TELEMETRY SUMMARY: {title} ===");
            sb.Append("meter: ").AppendLine(_targetMeterName);

            if (_metricNamePrefix is not null) sb.Append("metric prefix: ").AppendLine(_metricNamePrefix);

            sb.Append("include tags: ").AppendLine(_includeTags ? "true" : "false");

            if (rows.Length == 0)
            {
                sb.AppendLine("no metrics captured");
                _output.WriteLine(sb.ToString());
                return;
            }

            sb.Append("series: ").AppendLine(rows.Length.ToString(CultureInfo.InvariantCulture));
            sb.AppendLine("metric".PadRight(64) + " | samples".PadLeft(10) + " | sum".PadLeft(16) + " | min".PadLeft(14) + " | max".PadLeft(14) + " | last".PadLeft(14) + " | unit");
            sb.AppendLine(new string('-', 64 + 10 + 16 + 14 + 14 + 14 + 16));

            var limit = Math.Min(rows.Length, _maxRows);
            for (var i = 0; i < limit; i++)
            {
                var row = rows[i];
                var aggregate = row.Aggregate;
                sb.Append(row.Key.Name.PadRight(64)).Append(" | ")
                  .Append(aggregate.SampleCount.ToString(CultureInfo.InvariantCulture).PadLeft(8)).Append(" | ")
                  .Append(FormatDouble(aggregate.Sum).PadLeft(14)).Append(" | ")
                  .Append(FormatDouble(aggregate.Min).PadLeft(12)).Append(" | ")
                  .Append(FormatDouble(aggregate.Max).PadLeft(12)).Append(" | ")
                  .Append(FormatDouble(aggregate.Last).PadLeft(12)).Append(" | ")
                  .AppendLine(row.Key.Unit);

                if (_includeTags && row.Key.Tags.Length != 0)
                {
                    sb.Append("  tags: ").AppendLine(row.Key.Tags);
                }
            }

            if (rows.Length > limit) sb.Append("... truncated ").Append(rows.Length - limit).AppendLine(" metric series");
            _output.WriteLine(sb.ToString());
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;

            try
            {
                if (_started) WriteSummary();
            }
            finally
            {
                _listener.Dispose();
            }
        }

        private static string FormatDouble(double value)
        {
            if (double.IsNaN(value)) return "NaN";
            if (double.IsPositiveInfinity(value)) return "+Inf";
            if (double.IsNegativeInfinity(value)) return "-Inf";
            return value.ToString("0.###", CultureInfo.InvariantCulture);
        }

        private readonly struct MetricKey : IEquatable<MetricKey>
        {
            public MetricKey(string name, string unit, string tags) { Name = name; Unit = unit; Tags = tags; }
            public string Name { get; }
            public string Unit { get; }
            public string Tags { get; }
            public bool Equals(MetricKey other) => Name == other.Name && Unit == other.Unit && Tags == other.Tags;
            public override bool Equals(object? obj) => obj is MetricKey other && Equals(other);
            public override int GetHashCode() => HashCode.Combine(Name, Unit, Tags);
        }

        private struct MetricAggregate
        {
            public MetricAggregate(double value) { SampleCount = 1; Sum = value; Min = value; Max = value; Last = value; }
            public long SampleCount { get; private set; }
            public double Sum { get; private set; }
            public double Min { get; private set; }
            public double Max { get; private set; }
            public double Last { get; private set; }
            public void Add(double value)
            {
                SampleCount++; Sum += value;
                if (value < Min) Min = value;
                if (value > Max) Max = value;
                Last = value;
            }
        }

        private readonly struct MetricRow
        {
            public MetricRow(MetricKey key, MetricAggregate aggregate) { Key = key; Aggregate = aggregate; }
            public MetricKey Key { get; }
            public MetricAggregate Aggregate { get; }
        }
    }
}