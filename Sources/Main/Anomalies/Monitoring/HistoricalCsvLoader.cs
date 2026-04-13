// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Globalization;
using System.Linq;
using System.Text;
using DevOnBike.Overfit.Anomalies.Monitoring.Contracts;

namespace DevOnBike.Overfit.Anomalies.Monitoring
{
    /// <summary>
    ///     Loads historical <see cref="MetricSnapshot" /> data from a CSV file.
    ///     Expected CSV format (header row required):
    ///     <code>
    /// timestamp,pod_name,cpu_usage_ratio,cpu_throttle_ratio,memory_working_set_bytes,
    /// oom_events_rate,latency_p50_ms,latency_p95_ms,latency_p99_ms,requests_per_second,
    /// error_rate,gc_gen2_heap_bytes,gc_pause_ratio,thread_pool_queue_length
    /// 2026-04-03T10:00:00Z,payment-7f9b4c,0.32,0.01,285000000,...
    /// </code>
    ///     The CSV can be produced by:
    ///     - Grafana "Export CSV" on a dashboard panel
    ///     - A custom script querying /api/v1/query_range for each metric
    ///     - <see cref="HistoricalCsvExporter" /> (saves output of PrometheusHistoricalSource)
    ///     Missing or unparseable values default to 0 — the same conservative fallback
    ///     used throughout the pipeline.
    /// </summary>
    public static class HistoricalCsvLoader
    {
        private static readonly string[] ExpectedHeaders =
        [
            "timestamp", "pod_name",
            "cpu_usage_ratio", "cpu_throttle_ratio", "memory_working_set_bytes", "oom_events_rate",
            "latency_p50_ms", "latency_p95_ms", "latency_p99_ms", "requests_per_second",
            "error_rate", "gc_gen2_heap_bytes", "gc_pause_ratio", "thread_pool_queue_length"
        ];

        // -------------------------------------------------------------------------
        // Load
        // -------------------------------------------------------------------------

        /// <summary>
        ///     Loads snapshots from a CSV file.
        ///     Rows with parse errors are skipped and counted in the returned diagnostics.
        /// </summary>
        /// <param name="path">Path to the CSV file.</param>
        /// <param name="skippedRows">Number of data rows that could not be parsed.</param>
        /// <returns>Chronologically ordered list of snapshots.</returns>
        /// <exception cref="FileNotFoundException">When the file does not exist.</exception>
        /// <exception cref="InvalidDataException">When the header row is missing or malformed.</exception>
        public static IReadOnlyList<MetricSnapshot> Load(string path, out int skippedRows)
        {
            if (!File.Exists(path))
            {
                throw new FileNotFoundException($"CSV file not found: {path}");
            }

            using var reader = new StreamReader(path, Encoding.UTF8);
            return Load(reader, out skippedRows);
        }

        /// <summary>Overload that reads from a pre-opened <see cref="TextReader" />.</summary>
        public static IReadOnlyList<MetricSnapshot> Load(TextReader reader, out int skippedRows)
        {
            ArgumentNullException.ThrowIfNull(reader);

            var header = reader.ReadLine();
            if (header is null)
            {
                throw new InvalidDataException("CSV file is empty — no header row found.");
            }

            var columnIndex = ParseHeader(header);
            var snapshots = new List<MetricSnapshot>();
            skippedRows = 0;

            string line;
            while ((line = reader.ReadLine()) is not null)
            {
                if (string.IsNullOrWhiteSpace(line)) { continue; }

                if (TryParseRow(line, columnIndex, out var snapshot))
                {
                    snapshots.Add(snapshot);
                }
                else
                {
                    skippedRows++;
                }
            }

            snapshots.Sort((a, b) => a.Timestamp.CompareTo(b.Timestamp));
            return snapshots;
        }

        // -------------------------------------------------------------------------
        // Save (export)
        // -------------------------------------------------------------------------

        /// <summary>
        ///     Saves a list of snapshots to a CSV file with the canonical header.
        ///     Useful for caching the output of <see cref="PrometheusHistoricalSource" />
        ///     so you can inspect the data before training.
        /// </summary>
        public static void Save(string path, IReadOnlyList<MetricSnapshot> snapshots)
        {
            ArgumentNullException.ThrowIfNull(path);
            ArgumentNullException.ThrowIfNull(snapshots);

            using var writer = new StreamWriter(path, false, Encoding.UTF8);
            Save(writer, snapshots);
        }

        /// <summary>Overload that writes to a pre-opened <see cref="TextWriter" />.</summary>
        public static void Save(TextWriter writer, IReadOnlyList<MetricSnapshot> snapshots)
        {
            ArgumentNullException.ThrowIfNull(writer);
            ArgumentNullException.ThrowIfNull(snapshots);

            writer.WriteLine(string.Join(",", ExpectedHeaders));

            foreach (var s in snapshots)
            {
                writer.WriteLine(FormatRow(s));
            }
        }

        // -------------------------------------------------------------------------
        // Private
        // -------------------------------------------------------------------------

        private static Dictionary<string, int> ParseHeader(string header)
        {
            var cols = header.Split(',');
            var index = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);

            for (var i = 0; i < cols.Length; i++)
            {
                index[cols[i].Trim()] = i;
            }

            // Validate all required columns are present
            var missing = ExpectedHeaders.Where(h => !index.ContainsKey(h)).ToList();
            if (missing.Count > 0)
            {
                throw new InvalidDataException(
                $"CSV header is missing required columns: {string.Join(", ", missing)}. " +
                $"Expected: {string.Join(", ", ExpectedHeaders)}");
            }

            return index;
        }

        private static bool TryParseRow(
            string line,
            Dictionary<string, int> col,
            out MetricSnapshot snapshot)
        {
            snapshot = default;
            var parts = line.Split(',');

            if (!DateTime.TryParse(
                Get(parts, col, "timestamp"),
                CultureInfo.InvariantCulture,
                DateTimeStyles.RoundtripKind,
                out var ts))
            {
                return false;
            }

            snapshot = new MetricSnapshot
            {
                Timestamp = ts.ToUniversalTime(),
                PodName = Get(parts, col, "pod_name"),
                CpuUsageRatio = ParseFloat(parts, col, "cpu_usage_ratio"),
                CpuThrottleRatio = ParseFloat(parts, col, "cpu_throttle_ratio"),
                MemoryWorkingSetBytes = ParseFloat(parts, col, "memory_working_set_bytes"),
                OomEventsRate = ParseFloat(parts, col, "oom_events_rate"),
                LatencyP50Ms = ParseFloat(parts, col, "latency_p50_ms"),
                LatencyP95Ms = ParseFloat(parts, col, "latency_p95_ms"),
                LatencyP99Ms = ParseFloat(parts, col, "latency_p99_ms"),
                RequestsPerSecond = ParseFloat(parts, col, "requests_per_second"),
                ErrorRate = ParseFloat(parts, col, "error_rate"),
                GcGen2HeapBytes = ParseFloat(parts, col, "gc_gen2_heap_bytes"),
                GcPauseRatio = ParseFloat(parts, col, "gc_pause_ratio"),
                ThreadPoolQueueLength = ParseFloat(parts, col, "thread_pool_queue_length")
            };

            return true;
        }

        private static string Get(string[] parts, Dictionary<string, int> col, string name)
        {
            if (!col.TryGetValue(name, out var idx) || idx >= parts.Length) { return string.Empty; }
            return parts[idx].Trim();
        }

        private static float ParseFloat(string[] parts, Dictionary<string, int> col, string name)
        {
            var s = Get(parts, col, name);
            return float.TryParse(s, NumberStyles.Float, CultureInfo.InvariantCulture, out var v)
                ? v : 0f;
        }

        private static string FormatRow(MetricSnapshot s)
        {
            var sb = new StringBuilder();
            sb.Append(s.Timestamp.ToString("O", CultureInfo.InvariantCulture)).Append(',');
            sb.Append(s.PodName).Append(',');
            sb.Append(s.CpuUsageRatio.ToString(CultureInfo.InvariantCulture)).Append(',');
            sb.Append(s.CpuThrottleRatio.ToString(CultureInfo.InvariantCulture)).Append(',');
            sb.Append(s.MemoryWorkingSetBytes.ToString(CultureInfo.InvariantCulture)).Append(',');
            sb.Append(s.OomEventsRate.ToString(CultureInfo.InvariantCulture)).Append(',');
            sb.Append(s.LatencyP50Ms.ToString(CultureInfo.InvariantCulture)).Append(',');
            sb.Append(s.LatencyP95Ms.ToString(CultureInfo.InvariantCulture)).Append(',');
            sb.Append(s.LatencyP99Ms.ToString(CultureInfo.InvariantCulture)).Append(',');
            sb.Append(s.RequestsPerSecond.ToString(CultureInfo.InvariantCulture)).Append(',');
            sb.Append(s.ErrorRate.ToString(CultureInfo.InvariantCulture)).Append(',');
            sb.Append(s.GcGen2HeapBytes.ToString(CultureInfo.InvariantCulture)).Append(',');
            sb.Append(s.GcPauseRatio.ToString(CultureInfo.InvariantCulture)).Append(',');
            sb.Append(s.ThreadPoolQueueLength.ToString(CultureInfo.InvariantCulture));
            return sb.ToString();
        }
    }
}