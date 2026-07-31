// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Anomalies.Monitoring;
using DevOnBike.Overfit.Anomalies.Contracts;

namespace DevOnBike.Overfit.Tests.Anomalies
{
    /// <summary>
    /// Turning "our metric is called X" into the query the sources take.
    ///
    /// <para>The point of the map is that the client answers a question they know the answer to, instead of
    /// writing PromQL — which is where onboarding goes wrong, and where a mistake returns an empty result
    /// that Prometheus reports as success.</para>
    /// </summary>
    public sealed class MetricMapTests
    {
        [Fact]
        public void AGaugeIsSummedByPod()
        {
            var query = QueryFor(new MetricBinding(
                MetricIndex.MemoryWorkingSetBytes, "container_memory_working_set_bytes",
                MetricSourceKind.Gauge));

            Assert.Equal(
                "sum by (pod) (container_memory_working_set_bytes{%selector%})",
                query);
        }

        /// <summary>
        /// A counter's absolute value is an artefact of uptime, so comparing it across pods compares how long
        /// each has been running.
        /// </summary>
        [Fact]
        public void ACounterBecomesARate()
        {
            var query = QueryFor(new MetricBinding(
                MetricIndex.RequestsPerSecond, "http_requests_total", MetricSourceKind.Counter));

            Assert.Contains("rate(http_requests_total{%selector%}[2m])", query, StringComparison.Ordinal);
            Assert.Contains("sum by (pod)", query, StringComparison.Ordinal);
        }

        /// <summary>Two restarts is a fact; two restarts per second is noise.</summary>
        [Fact]
        public void AnEventCountBecomesAnIncrease()
        {
            var query = QueryFor(new MetricBinding(
                MetricIndex.ContainerRestarts, "kube_pod_container_status_restarts_total",
                MetricSourceKind.EventCount));

            Assert.Contains("increase(", query, StringComparison.Ordinal);
            Assert.DoesNotContain("rate(", query, StringComparison.Ordinal);
        }

        /// <summary>
        /// The <c>le</c> in the grouping is load-bearing: without the pod label surviving the quantile every
        /// sample is discarded during parsing and the feature is lost silently.
        /// </summary>
        [Fact]
        public void AHistogramKeepsThePodLabelThroughTheQuantile()
        {
            var query = QueryFor(new MetricBinding(
                MetricIndex.LatencyP95Ms, "http_server_duration_seconds",
                MetricSourceKind.HistogramSeconds));

            Assert.Contains("sum by (pod, le)", query, StringComparison.Ordinal);
            Assert.Contains("http_server_duration_seconds_bucket", query, StringComparison.Ordinal);
            Assert.Contains("histogram_quantile(0.95", query, StringComparison.Ordinal);
            Assert.Contains("* 1000", query, StringComparison.Ordinal);
        }

        /// <summary>The quantile follows the feature, so the caller names the histogram once.</summary>
        [Theory]
        [InlineData(MetricIndex.LatencyP50Ms, "0.5")]
        [InlineData(MetricIndex.LatencyP95Ms, "0.95")]
        [InlineData(MetricIndex.LatencyP99Ms, "0.99")]
        public void TheQuantileIsTakenFromTheTargetFeature(MetricIndex metric, string expected)
        {
            var query = QueryFor(new MetricBinding(metric, "d", MetricSourceKind.HistogramSeconds));

            Assert.Contains($"histogram_quantile({expected},", query, StringComparison.Ordinal);
        }

        /// <summary>A ratio is already normalised; rating or summing it produces a meaningless number.</summary>
        [Fact]
        public void ARatioIsPassedThrough()
        {
            var query = QueryFor(new MetricBinding(
                MetricIndex.CpuThrottleRatio, "container_cpu_cfs_throttled_ratio", MetricSourceKind.Ratio));

            Assert.Equal("container_cpu_cfs_throttled_ratio{%selector%}", query);
        }

        /// <summary>
        /// An unmapped feature gets an <b>empty</b> template, which is how the sources are told not to issue a
        /// query at all. Issuing one built from a metric this cluster does not have returns an empty result
        /// that cannot be told apart from a real one.
        /// </summary>
        [Fact]
        public void AnUnmappedFeatureProducesNoQuery()
        {
            var map = new MetricMap([
                new MetricBinding(MetricIndex.RequestsPerSecond, "reqs", MetricSourceKind.Counter),
            ]);

            var overrides = map.ToQueryOverrides();

            Assert.NotEqual(string.Empty, overrides[MetricIndex.RequestsPerSecond]);
            Assert.Equal(string.Empty, overrides[MetricIndex.CpuThrottleRatio]);
            Assert.False(map.IsMapped(MetricIndex.CpuThrottleRatio));
        }

        /// <summary>
        /// The absent list is the deployment's honest blind spots, available before anything runs rather than
        /// inferred from a cycle that saw nothing.
        /// </summary>
        [Fact]
        public void UnmappedListsEveryFeatureWithoutABinding()
        {
            var map = new MetricMap([
                new MetricBinding(MetricIndex.RequestsPerSecond, "reqs", MetricSourceKind.Counter),
                new MetricBinding(MetricIndex.LatencyP95Ms, "dur", MetricSourceKind.HistogramSeconds),
            ]);

            Assert.Equal(2, map.MappedCount);
            Assert.Equal((int)MetricIndex.Count - 2, map.Unmapped.Count);
            Assert.Contains(MetricIndex.CpuThrottleRatio, map.Unmapped);
            Assert.DoesNotContain(MetricIndex.RequestsPerSecond, map.Unmapped);
        }

        [Fact]
        public void ALaterBindingOverridesAnEarlierOne()
        {
            var map = new MetricMap([
                new MetricBinding(MetricIndex.RequestsPerSecond, "old", MetricSourceKind.Counter),
                new MetricBinding(MetricIndex.RequestsPerSecond, "new", MetricSourceKind.Counter),
            ]);

            Assert.Contains("new", map.ToQueryOverrides()[MetricIndex.RequestsPerSecond],
                StringComparison.Ordinal);
        }

        [Fact]
        public void ABlankSourceNameIsNotABinding()
        {
            var map = new MetricMap([
                new MetricBinding(MetricIndex.RequestsPerSecond, "   ", MetricSourceKind.Counter),
            ]);

            Assert.False(map.IsMapped(MetricIndex.RequestsPerSecond));
            Assert.Equal(0, map.MappedCount);
        }

        [Fact]
        public void DescribeNamesBothWhatIsMappedAndWhatIsNot()
        {
            var map = new MetricMap([
                new MetricBinding(MetricIndex.RequestsPerSecond, "reqs", MetricSourceKind.Counter),
            ]);

            var report = map.Describe();

            Assert.Contains("RequestsPerSecond", report, StringComparison.Ordinal);
            Assert.Contains("reqs", report, StringComparison.Ordinal);
            Assert.Contains("not available", report, StringComparison.Ordinal);
        }

        private static string QueryFor(MetricBinding binding)
        {
            return new MetricMap([binding]).ToQueryOverrides()[binding.Target];
        }
    }
}
