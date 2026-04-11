using DevOnBike.Overfit.Anomalies.Monitoring;
using DevOnBike.Overfit.Anomalies.Monitoring.Contracts;

namespace DevOnBike.Overfit.Tests.Monitoring
{
    public sealed class HistoricalCsvLoaderTests
    {
        private static string MakeCsv(IEnumerable<string> rows)
        {
            var sb = new System.Text.StringBuilder();
            sb.AppendLine(
            "timestamp,pod_name,cpu_usage_ratio,cpu_throttle_ratio,memory_working_set_bytes," +
            "oom_events_rate,latency_p50_ms,latency_p95_ms,latency_p99_ms,requests_per_second," +
            "error_rate,gc_gen2_heap_bytes,gc_pause_ratio,thread_pool_queue_length");
            foreach (var row in rows) { sb.AppendLine(row); }
            return sb.ToString();
        }

        private static string ValidRow(string ts = "2026-04-03T10:00:00Z")
            => $"{ts},pod-1,0.32,0.01,285000000,0,60,110,180,100,0.005,50000000,0.02,10";

        // -------------------------------------------------------------------------
        // Load — argument validation
        // -------------------------------------------------------------------------

        [Fact]
        public void Load_WhenFileDoesNotExist_ThenThrowsFileNotFoundException()
            => Assert.Throws<FileNotFoundException>(
            () => HistoricalCsvLoader.Load("/tmp/no-such-file.csv", out _));

        [Fact]
        public void Load_WhenEmptyFile_ThenThrowsInvalidDataException()
        {
            using var reader = new StringReader("");
            Assert.Throws<InvalidDataException>(
            () => HistoricalCsvLoader.Load(reader, out _));
        }

        [Fact]
        public void Load_WhenHeaderMissingColumns_ThenThrowsInvalidDataException()
        {
            using var reader = new StringReader("timestamp,pod_name\n2026-04-03T10:00:00Z,pod-1");
            Assert.Throws<InvalidDataException>(
            () => HistoricalCsvLoader.Load(reader, out _));
        }

        // -------------------------------------------------------------------------
        // Load — correctness
        // -------------------------------------------------------------------------

        [Fact]
        public void Load_WhenValidCsv_ThenReturnsCorrectCount()
        {
            var csv = MakeCsv([ValidRow("2026-04-03T10:00:00Z"), ValidRow("2026-04-03T10:00:10Z")]);
            using var reader = new StringReader(csv);

            var result = HistoricalCsvLoader.Load(reader, out var skipped);

            Assert.Equal(2, result.Count);
            Assert.Equal(0, skipped);
        }

        [Fact]
        public void Load_WhenValidRow_ThenFieldsAreParsedCorrectly()
        {
            var csv = MakeCsv([ValidRow("2026-04-03T10:00:00Z")]);
            using var reader = new StringReader(csv);

            var result = HistoricalCsvLoader.Load(reader, out _);
            var snap = result[0];

            Assert.Equal("pod-1", snap.PodName);
            Assert.Equal(0.32f, snap.CpuUsageRatio, 0.001f);
            Assert.Equal(0.01f, snap.CpuThrottleRatio, 0.001f);
            Assert.Equal(285000000f, snap.MemoryWorkingSetBytes, 1f);
            Assert.Equal(60f, snap.LatencyP50Ms, 0.001f);
            Assert.Equal(110f, snap.LatencyP95Ms, 0.001f);
            Assert.Equal(180f, snap.LatencyP99Ms, 0.001f);
            Assert.Equal(100f, snap.RequestsPerSecond, 0.001f);
            Assert.Equal(0.005f, snap.ErrorRate, 0.0001f);
            Assert.Equal(50000000f, snap.GcGen2HeapBytes, 1f);
            Assert.Equal(0.02f, snap.GcPauseRatio, 0.001f);
            Assert.Equal(10f, snap.ThreadPoolQueueLength, 0.001f);
        }

        [Fact]
        public void Load_WhenRowHasBadTimestamp_ThenRowIsSkipped()
        {
            var csv = MakeCsv(["not-a-date,pod-1,0.32,0.01,285000000,0,60,110,180,100,0.005,50000000,0.02,10"]);
            using var reader = new StringReader(csv);

            var result = HistoricalCsvLoader.Load(reader, out var skipped);

            Assert.Empty(result);
            Assert.Equal(1, skipped);
        }

        [Fact]
        public void Load_WhenRowHasBadFloat_ThenFieldDefaultsToZero()
        {
            var csv = MakeCsv(["2026-04-03T10:00:00Z,pod-1,NOT_A_FLOAT,0.01,285000000,0,60,110,180,100,0.005,50000000,0.02,10"]);
            using var reader = new StringReader(csv);

            var result = HistoricalCsvLoader.Load(reader, out _);

            Assert.Single(result);
            Assert.Equal(0f, result[0].CpuUsageRatio); // defaulted to 0
        }

        [Fact]
        public void Load_WhenMultipleRows_ThenSortedByTimestamp()
        {
            var csv = MakeCsv([
                ValidRow("2026-04-03T10:00:20Z"),
                ValidRow("2026-04-03T10:00:00Z"),
                ValidRow("2026-04-03T10:00:10Z")
            ]);
            using var reader = new StringReader(csv);

            var result = HistoricalCsvLoader.Load(reader, out _);

            Assert.True(result[0].Timestamp < result[1].Timestamp);
            Assert.True(result[1].Timestamp < result[2].Timestamp);
        }

        [Fact]
        public void Load_WhenEmptyLines_ThenSkipped()
        {
            var csv = "timestamp,pod_name,cpu_usage_ratio,cpu_throttle_ratio,memory_working_set_bytes," +
                      "oom_events_rate,latency_p50_ms,latency_p95_ms,latency_p99_ms,requests_per_second," +
                      "error_rate,gc_gen2_heap_bytes,gc_pause_ratio,thread_pool_queue_length\n" +
                      "\n" +
                      ValidRow() + "\n" +
                      "   \n";
            using var reader = new StringReader(csv);

            var result = HistoricalCsvLoader.Load(reader, out var skipped);

            Assert.Single(result);
            Assert.Equal(0, skipped);
        }

        // -------------------------------------------------------------------------
        // Save → Load round-trip
        // -------------------------------------------------------------------------

        [Fact]
        public void SaveLoad_WhenRoundtripped_ThenDataIsPreserved()
        {
            var snapshots = new List<MetricSnapshot>
            {
                new()
                {
                    Timestamp             = new DateTime(2026, 4, 3, 10, 0, 0, DateTimeKind.Utc),
                    PodName               = "my-pod",
                    CpuUsageRatio         = 0.42f,
                    CpuThrottleRatio      = 0.03f,
                    MemoryWorkingSetBytes = 300_000_000f,
                    OomEventsRate         = 0f,
                    LatencyP50Ms          = 65f,
                    LatencyP95Ms          = 115f,
                    LatencyP99Ms          = 195f,
                    RequestsPerSecond     = 88f,
                    ErrorRate             = 0.007f,
                    GcGen2HeapBytes       = 45_000_000f,
                    GcPauseRatio          = 0.018f,
                    ThreadPoolQueueLength = 12f
                }
            };

            using var ms = new MemoryStream();
            using var writer = new StreamWriter(ms, System.Text.Encoding.UTF8, leaveOpen: true);
            HistoricalCsvLoader.Save(writer, snapshots);
            writer.Flush();

            ms.Position = 0;
            using var reader = new StreamReader(ms);
            var loaded = HistoricalCsvLoader.Load(reader, out var skipped);

            Assert.Equal(0, skipped);
            Assert.Single(loaded);
            Assert.Equal(snapshots[0].CpuUsageRatio, loaded[0].CpuUsageRatio, 0.001f);
            Assert.Equal(snapshots[0].MemoryWorkingSetBytes, loaded[0].MemoryWorkingSetBytes, 1f);
            Assert.Equal(snapshots[0].LatencyP95Ms, loaded[0].LatencyP95Ms, 0.001f);
            Assert.Equal(snapshots[0].GcPauseRatio, loaded[0].GcPauseRatio, 0.001f);
            Assert.Equal(snapshots[0].PodName, loaded[0].PodName);
        }
    }
}