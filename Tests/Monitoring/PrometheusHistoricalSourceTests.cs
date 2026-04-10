using DevOnBike.Overfit.Monitoring;
using DevOnBike.Overfit.Monitoring.Contracts;

namespace DevOnBike.Overfit.Tests.Monitoring
{
    public sealed class PrometheusHistoricalSourceTests
    {
        // -------------------------------------------------------------------------
        // ParseRangeSeries — unit tested directly
        // -------------------------------------------------------------------------

        private static string RangeResponse(params (double ts, double val)[] points)
        {
            var values = string.Join(",",
            points.Select(p =>
                $"[{p.ts.ToString(System.Globalization.CultureInfo.InvariantCulture)}," +
                $"\"{p.val.ToString(System.Globalization.CultureInfo.InvariantCulture)}\"]"));

            return $$$"""{"data":{"result":[{"values":[{{{values}}}]}]}}""";
        }

        [Fact]
        public void ParseRangeSeries_WhenValidResponse_ThenReturnsCorrectCount()
        {
            var json = RangeResponse((1712345678.0, 0.42), (1712345688.0, 0.45));
            var result = PrometheusHistoricalSource.ParseRangeSeries(json);
            Assert.Equal(2, result.Count);
        }

        [Fact]
        public void ParseRangeSeries_WhenValidResponse_ThenValuesParsedCorrectly()
        {
            var json = RangeResponse((1712345678.0, 0.42));
            var result = PrometheusHistoricalSource.ParseRangeSeries(json);

            Assert.Single(result);
            var (_, value) = result.First();
            Assert.True(MathF.Abs(value - 0.42f) < 0.001f);
        }

        [Fact]
        public void ParseRangeSeries_WhenEmptyResult_ThenReturnsEmptyDictionary()
        {
            var json = """{"data":{"result":[]}}""";
            var result = PrometheusHistoricalSource.ParseRangeSeries(json);
            Assert.Empty(result);
        }

        [Fact]
        public void ParseRangeSeries_WhenMalformedJson_ThenReturnsEmptyDictionary()
        {
            var result = PrometheusHistoricalSource.ParseRangeSeries("not json");
            Assert.Empty(result);
        }

        [Fact]
        public void ParseRangeSeries_WhenTimestampIsFloat_ThenParsedCorrectly()
        {
            // Prometheus timestamps include sub-second precision
            var json = RangeResponse((1712345678.123, 0.5));
            var result = PrometheusHistoricalSource.ParseRangeSeries(json);
            Assert.Single(result);
        }

        // -------------------------------------------------------------------------
        // Constructor validation
        // -------------------------------------------------------------------------

        [Fact]
        public void Constructor_WhenConfigIsNull_ThenThrowsArgumentNullException()
            => Assert.Throws<ArgumentNullException>(
            () => new PrometheusHistoricalSource(null!));

        [Fact]
        public void Constructor_WhenRangeEndBeforeStart_ThenThrowsArgumentException()
        {
            var config = new PrometheusHistoricalSourceConfig
            {
                PrometheusBaseUrl = "http://prometheus:9090",
                PodName = "pod-1",
                RangeStart = DateTime.UtcNow,
                RangeEnd = DateTime.UtcNow.AddHours(-1) // end before start
            };
            Assert.Throws<ArgumentException>(() => new PrometheusHistoricalSource(config));
        }
    }
}