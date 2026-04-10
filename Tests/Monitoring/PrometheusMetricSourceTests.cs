// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com
using System.Net;
using System.Text;
using DevOnBike.Overfit.Monitoring;
using DevOnBike.Overfit.Monitoring.Contracts;

namespace DevOnBike.Overfit.Tests.Monitoring
{
    public sealed class PrometheusMetricSourceTests
    {
        // -------------------------------------------------------------------------
        // Test doubles
        // -------------------------------------------------------------------------

        /// <summary>
        /// Fake HttpMessageHandler — returns a fixed Prometheus response for every request.
        /// Avoids any real network traffic.
        /// </summary>
        private sealed class FakePrometheusHandler(string responseBody, HttpStatusCode code = HttpStatusCode.OK)
            : HttpMessageHandler
        {
            public int CallCount { get; private set; }
            public List<string> RequestedUrls { get; } = [];

            protected override Task<HttpResponseMessage> SendAsync(
                HttpRequestMessage request, CancellationToken ct)
            {
                CallCount++;
                RequestedUrls.Add(request.RequestUri?.ToString() ?? "");

                return Task.FromResult(new HttpResponseMessage(code)
                {
                    Content = new StringContent(responseBody, Encoding.UTF8, "application/json")
                });
            }
        }

        private static string PrometheusResponse(double value) => $$"""
                                                                    {
                                                                      "status": "success",
                                                                      "data": {
                                                                        "resultType": "vector",
                                                                        "result": [
                                                                          { "metric": {}, "value": [1712345678.123, "{{value.ToString(System.Globalization.CultureInfo.InvariantCulture)}}"] }
                                                                        ]
                                                                      }
                                                                    }
                                                                    """;

        private static string EmptyPrometheusResponse() => """
                                                           { "status": "success", "data": { "resultType": "vector", "result": [] } }
                                                           """;

        private static PrometheusMetricSource MakeSource(
            HttpMessageHandler handler,
            string podName = "test-pod",
            TimeSpan? scrapeInterval = null)
        {
            var config = new PrometheusMetricSourceConfig
            {
                PrometheusBaseUrl = "http://prometheus:9090",
                PodName = podName,
                ScrapeInterval = scrapeInterval ?? TimeSpan.Zero // no delay in tests
            };
            var http = new HttpClient(handler)
            {
                BaseAddress = new Uri(config.PrometheusBaseUrl)
            };
            return new PrometheusMetricSource(config, http);
        }

        // -------------------------------------------------------------------------
        // Constructor validation
        // -------------------------------------------------------------------------

        [Fact]
        public void Constructor_WhenConfigIsNull_ThenThrowsArgumentNullException()
            => Assert.Throws<ArgumentNullException>(
            () => new PrometheusMetricSource(null!));

        [Fact]
        public void Constructor_WhenBaseUrlIsEmpty_ThenThrowsArgumentException()
        {
            var config = new PrometheusMetricSourceConfig
            {
                PrometheusBaseUrl = "",
                PodName = "pod-1"
            };
            Assert.Throws<ArgumentException>(() => new PrometheusMetricSource(config));
        }

        [Fact]
        public void Constructor_WhenPodNameIsEmpty_ThenThrowsArgumentException()
        {
            var config = new PrometheusMetricSourceConfig
            {
                PrometheusBaseUrl = "http://prometheus:9090",
                PodName = ""
            };
            Assert.Throws<ArgumentException>(() => new PrometheusMetricSource(config));
        }

        // -------------------------------------------------------------------------
        // PodName
        // -------------------------------------------------------------------------

        [Fact]
        public void PodName_WhenConfigured_ThenMatchesConfig()
        {
            using var source = MakeSource(new FakePrometheusHandler("{}"), podName: "my-pod");
            Assert.Equal("my-pod", source.PodName);
        }

        // -------------------------------------------------------------------------
        // ParseFirstScalar — unit tested directly (internal static)
        // -------------------------------------------------------------------------

        [Fact]
        public void ParseFirstScalar_WhenValidResponse_ThenReturnsValue()
        {
            var json = PrometheusResponse(1.234);
            var value = PrometheusMetricSource.ParseFirstScalar(json);
            Assert.True(MathF.Abs(value - 1.234f) < 0.001f, $"got {value}");
        }

        [Fact]
        public void ParseFirstScalar_WhenEmptyResult_ThenReturnsZero()
        {
            var value = PrometheusMetricSource.ParseFirstScalar(EmptyPrometheusResponse());
            Assert.Equal(0f, value);
        }

        [Fact]
        public void ParseFirstScalar_WhenNaN_ThenReturnsZero()
        {
            var json = """{"data":{"result":[{"value":[1234,"NaN"]}]}}""";
            Assert.Equal(0f, PrometheusMetricSource.ParseFirstScalar(json));
        }

        [Fact]
        public void ParseFirstScalar_WhenPosInf_ThenReturnsZero()
        {
            var json = """{"data":{"result":[{"value":[1234,"+Inf"]}]}}""";
            Assert.Equal(0f, PrometheusMetricSource.ParseFirstScalar(json));
        }

        [Fact]
        public void ParseFirstScalar_WhenNegInf_ThenReturnsZero()
        {
            var json = """{"data":{"result":[{"value":[1234,"-Inf"]}]}}""";
            Assert.Equal(0f, PrometheusMetricSource.ParseFirstScalar(json));
        }

        [Fact]
        public void ParseFirstScalar_WhenMalformedJson_ThenReturnsZero()
            => Assert.Equal(0f, PrometheusMetricSource.ParseFirstScalar("not json"));

        [Fact]
        public void ParseFirstScalar_WhenIntegerValue_ThenReturnsAsFloat()
        {
            var json = PrometheusResponse(42);
            var value = PrometheusMetricSource.ParseFirstScalar(json);
            Assert.True(MathF.Abs(value - 42f) < 0.001f);
        }

        [Fact]
        public void ParseFirstScalar_WhenValueUsesInvariantLocale_ThenParsesCorrectly()
        {
            // Prometheus always uses "." as decimal separator — must parse regardless of OS locale
            var json = """{"data":{"result":[{"value":[1234,"0.075"]}]}}""";
            var value = PrometheusMetricSource.ParseFirstScalar(json);
            Assert.True(MathF.Abs(value - 0.075f) < 0.001f, $"got {value}");
        }

        // -------------------------------------------------------------------------
        // ReadAsync — integration with fake handler
        // -------------------------------------------------------------------------

        [Fact]
        public async Task ReadAsync_WhenCalled_ThenIssues12HttpRequests()
        {
            var handler = new FakePrometheusHandler(PrometheusResponse(0.5));
            using var source = MakeSource(handler);

            await source.ReadAsync();

            Assert.Equal(12, handler.CallCount);
        }

        [Fact]
        public async Task ReadAsync_WhenCalled_ThenSnapshotHasCorrectPodName()
        {
            var handler = new FakePrometheusHandler(PrometheusResponse(0.1));
            using var source = MakeSource(handler, podName: "prod-pod");

            var snap = await source.ReadAsync();

            Assert.Equal("prod-pod", snap.PodName);
        }

        [Fact]
        public async Task ReadAsync_WhenPrometheusReturnsValue_ThenCpuUsageIsSet()
        {
            var handler = new FakePrometheusHandler(PrometheusResponse(0.42));
            using var source = MakeSource(handler);

            var snap = await source.ReadAsync();

            Assert.True(MathF.Abs(snap.CpuUsageRatio - 0.42f) < 0.001f, $"got {snap.CpuUsageRatio}");
        }

        [Fact]
        public async Task ReadAsync_WhenPrometheusReturnsEmpty_ThenAllValuesAreZero()
        {
            var handler = new FakePrometheusHandler(EmptyPrometheusResponse());
            using var source = MakeSource(handler);

            var snap = await source.ReadAsync();

            Assert.Equal(0f, snap.CpuUsageRatio);
            Assert.Equal(0f, snap.MemoryWorkingSetBytes);
            Assert.Equal(0f, snap.LatencyP95Ms);
            Assert.Equal(0f, snap.RequestsPerSecond);
            Assert.Equal(0f, snap.ErrorRate);
            Assert.Equal(0f, snap.GcPauseRatio);
            Assert.Equal(0f, snap.ThreadPoolQueueLength);
            Assert.Equal(0f, snap.GcGen2HeapBytes);
            Assert.Equal(0f, snap.CpuThrottleRatio);
            Assert.Equal(0f, snap.OomEventsRate);
            Assert.Equal(0f, snap.LatencyP50Ms);
            Assert.Equal(0f, snap.LatencyP99Ms);
        }

        [Fact]
        public async Task ReadAsync_WhenCpuExceeds1_ThenClampedTo1()
        {
            var handler = new FakePrometheusHandler(PrometheusResponse(999.0));
            using var source = MakeSource(handler);

            var snap = await source.ReadAsync();

            Assert.Equal(1f, snap.CpuUsageRatio); // CPU clamped to [0,1]
        }

        [Fact]
        public async Task ReadAsync_WhenCancelled_ThenThrowsOperationCanceledException()
        {
            var handler = new FakePrometheusHandler(PrometheusResponse(0.1));
            var config = new PrometheusMetricSourceConfig
            {
                PrometheusBaseUrl = "http://prometheus:9090",
                PodName = "pod-1",
                ScrapeInterval = TimeSpan.FromSeconds(10) // long delay
            };
            var http = new HttpClient(handler);
            using var source = new PrometheusMetricSource(config, http);

            using var cts = new CancellationTokenSource(TimeSpan.FromMilliseconds(50));
            await Assert.ThrowsAnyAsync<OperationCanceledException>(
            () => source.ReadAsync(cts.Token).AsTask());
        }

        [Fact]
        public async Task ReadAsync_WhenHttpFails_ThenThrowsHttpRequestException()
        {
            var handler = new FakePrometheusHandler("{}", HttpStatusCode.InternalServerError);
            using var source = MakeSource(handler);

            await Assert.ThrowsAsync<HttpRequestException>(
            () => source.ReadAsync().AsTask());
        }

        [Fact]
        public async Task ReadAsync_WhenTimestampSet_ThenIsRecent()
        {
            var before = DateTime.UtcNow;
            var handler = new FakePrometheusHandler(PrometheusResponse(0.1));
            using var source = MakeSource(handler);

            var snap = await source.ReadAsync();
            var after = DateTime.UtcNow;

            Assert.True(snap.Timestamp >= before && snap.Timestamp <= after);
        }

        // -------------------------------------------------------------------------
        // Dispose
        // -------------------------------------------------------------------------

        [Fact]
        public void Dispose_WhenCalledTwice_ThenDoesNotThrow()
        {
            var handler = new FakePrometheusHandler(PrometheusResponse(0.1));
            var source = MakeSource(handler);
            source.Dispose();
            source.Dispose(); // idempotent
        }

        [Fact]
        public async Task ReadAsync_WhenDisposed_ThenThrowsObjectDisposedException()
        {
            var handler = new FakePrometheusHandler(PrometheusResponse(0.1));
            var source = MakeSource(handler);
            source.Dispose();

            await Assert.ThrowsAsync<ObjectDisposedException>(
            () => source.ReadAsync().AsTask());
        }
    }
}