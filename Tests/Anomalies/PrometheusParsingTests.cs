// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Anomalies.Monitoring;
using DevOnBike.Overfit.Anomalies.Monitoring.Contracts;

namespace DevOnBike.Overfit.Tests.Anomalies
{
    /// <summary>
    /// Golden-JSON tests for the Prometheus response parsers. They cover the
    /// sample-extraction logic in <see cref="PrometheusMetricSource"/> and
    /// <see cref="PrometheusHistoricalSource"/> — previously commented out, so
    /// both sources silently returned empty lists. No live Prometheus required:
    /// the fixtures under test_fixtures/prometheus/ are captured /api/v1/query
    /// and /api/v1/query_range responses.
    /// </summary>
    public sealed class PrometheusParsingTests
    {
        private static string LoadFixture(string name)
        {
            var path = Path.Combine(AppContext.BaseDirectory, "test_fixtures", "prometheus", name);
            return File.ReadAllText(path);
        }

        private static RawMetricSeries Get(List<RawMetricSeries> series, string podName)
        {
            foreach (var s in series)
            {
                if (s.Pod.PodName == podName)
                {
                    return s;
                }
            }

            throw new Xunit.Sdk.XunitException($"No series found for pod '{podName}'.");
        }

        [Fact]
        public void ParseInstantResponse_ExtractsPods_SkipsPodless_KeepsNonFinite()
        {
            var json = LoadFixture("instant_query.json");

            var series = PrometheusMetricSource.ParseInstantResponse(
                json,
                metricTypeId: (byte)MetricIndex.CpuUsageRatio,
                dc: DataCenter.West);

            // Fixture has 4 results; the one with no "pod" label is skipped.
            Assert.Equal(3, series.Count);

            foreach (var s in series)
            {
                Assert.Equal(DataCenter.West, s.Pod.DC);
                Assert.Equal((byte)MetricIndex.CpuUsageRatio, s.MetricTypeId);
                Assert.Single(s.Samples);
            }

            var apiGw = Get(series, "api-gateway-7d9f");
            Assert.Equal(1736500000500L, apiGw.Samples[0].Timestamp); // 1736500000.5 s -> ms
            Assert.Equal(0.4231f, apiGw.Samples[0].Value, 4);

            // Prometheus "NaN" sample is preserved as NaN, not silently dropped.
            var dbProxy = Get(series, "db-proxy-xyz");
            Assert.True(float.IsNaN(dbProxy.Samples[0].Value));
        }

        [Fact]
        public void ParseRangeResponse_ExtractsSamples_SkipsPodlessAndEmpty()
        {
            var json = LoadFixture("range_query.json");

            var series = PrometheusHistoricalSource.ParseRangeResponse(
                json,
                metricTypeId: (byte)MetricIndex.CpuUsageRatio,
                dc: DataCenter.East);

            // Fixture has 4 results: podless skipped, empty-values skipped -> 2 remain.
            Assert.Equal(2, series.Count);

            var apiGw = Get(series, "api-gateway-7d9f");
            Assert.Equal(DataCenter.East, apiGw.Pod.DC);
            Assert.Equal(3, apiGw.Samples.Count);
            Assert.Equal(1736500000000L, apiGw.Samples[0].Timestamp);
            Assert.Equal(1736500015000L, apiGw.Samples[1].Timestamp);
            Assert.Equal(1736500030000L, apiGw.Samples[2].Timestamp);
            Assert.Equal(0.40f, apiGw.Samples[0].Value, 4);
            Assert.Equal(0.45f, apiGw.Samples[2].Value, 4);

            var scheduler = Get(series, "scheduler-111");
            Assert.Equal(2, scheduler.Samples.Count);
        }

        [Fact]
        public void ParseInstantResponse_EmptyResult_ReturnsEmpty()
        {
            var series = PrometheusMetricSource.ParseInstantResponse(
                "{\"status\":\"success\",\"data\":{\"resultType\":\"vector\",\"result\":[]}}",
                metricTypeId: 0,
                dc: DataCenter.West);

            Assert.Empty(series);
        }

        [Fact]
        public void ParseRangeResponse_MissingData_ReturnsEmpty()
        {
            var series = PrometheusHistoricalSource.ParseRangeResponse(
                "{\"status\":\"error\",\"errorType\":\"bad_data\"}",
                metricTypeId: 0,
                dc: DataCenter.West);

            Assert.Empty(series);
        }
    }
}
