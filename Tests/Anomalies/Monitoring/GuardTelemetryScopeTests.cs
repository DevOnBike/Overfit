// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Anomalies.Contracts;
using DevOnBike.Overfit.Anomalies.Monitoring;

namespace DevOnBike.Overfit.Tests.Anomalies.Monitoring
{
    /// <summary>
    /// The <c>scope</c> label, and the exposition rule that makes several scopes renderable at all.
    ///
    /// <para><b>The label exists for one series.</b> Without it,
    /// <c>overfit_guard_last_cycle_timestamp_seconds</c> across several scopes is the most recent of them, so
    /// one healthy scope keeps it fresh while the others are stalled and "the guard has stopped" never fires
    /// — the pathology this whole subsystem removes, reintroduced by something that looks like
    /// configuration.</para>
    /// </summary>
    public sealed class GuardTelemetryScopeTests
    {
        private static GuardTelemetry Cycled(string scope, int opened, DateTimeOffset at)
        {
            var telemetry = new GuardTelemetry(scope);

            telemetry.Cycle(
                new GuardCycleResult(
                    Findings: 0, Incidents: 0, Opened: opened, Ongoing: 0, Resolved: 0,
                    BlindMetrics: 0, PartialMetrics: 0, UnevaluableMetrics: 0),
                pods: 3,
                at,
                suppressed: false);

            return telemetry;
        }

        /// <summary>An upgrade must not retarget somebody's alerts: no scope, no label, same output as before.</summary>
        [Fact]
        public void ASingleScopelessInstrumentRendersBareSeriesNames()
        {
            var text = Cycled(string.Empty, 1, DateTimeOffset.UnixEpoch.AddSeconds(1000)).ToPrometheusText();

            Assert.Contains("\noverfit_guard_cycles_total 1", text, StringComparison.Ordinal);
            Assert.DoesNotContain("scope=", text, StringComparison.Ordinal);
        }

        [Fact]
        public void AScopedInstrumentLabelsEverySeries()
        {
            var text = Cycled("payments/api-.*", 1, DateTimeOffset.UnixEpoch.AddSeconds(1000))
                .ToPrometheusText();

            Assert.Contains("overfit_guard_cycles_total{scope=\"payments/api-.*\"} 1", text,
                StringComparison.Ordinal);
            Assert.Contains("overfit_guard_last_cycle_timestamp_seconds{scope=\"payments/api-.*\"}", text,
                StringComparison.Ordinal);
        }

        /// <summary>
        /// The rule that makes multi-scope renderable: the text format allows one <c># HELP</c> and one
        /// <c># TYPE</c> per metric name per document, and a second one makes Prometheus reject the ENTIRE
        /// scrape — so a naive concatenation of per-scope renderings takes the whole endpoint down rather
        /// than one scope.
        /// </summary>
        [Fact]
        public void SeveralScopesShareOneHeaderPerSeries()
        {
            var text = GuardTelemetry.Render(
            [
                Cycled("payments/api-.*", 1, DateTimeOffset.UnixEpoch.AddSeconds(1000)),
                Cycled("payments/worker-.*", 2, DateTimeOffset.UnixEpoch.AddSeconds(2000)),
                Cycled("search/index-.*", 3, DateTimeOffset.UnixEpoch.AddSeconds(3000)),
            ]);

            var helps = 0;
            var samples = 0;

            foreach (var line in text.Split('\n'))
            {
                if (line.StartsWith("# HELP overfit_guard_cycles_total", StringComparison.Ordinal))
                {
                    helps++;
                }

                if (line.StartsWith("overfit_guard_cycles_total{", StringComparison.Ordinal))
                {
                    samples++;
                }
            }

            Assert.Equal(1, helps);
            Assert.Equal(3, samples);
        }

        /// <summary>
        /// The whole point, stated as an assertion: each scope keeps its own last-cycle timestamp, so a
        /// stalled scope stays stale no matter how healthy its neighbours are.
        /// </summary>
        [Fact]
        public void AStalledScopeKeepsItsOwnStaleTimestamp()
        {
            var text = GuardTelemetry.Render(
            [
                Cycled("busy", 1, DateTimeOffset.UnixEpoch.AddSeconds(9000)),
                Cycled("stalled", 1, DateTimeOffset.UnixEpoch.AddSeconds(10)),
            ]);

            Assert.Contains("overfit_guard_last_cycle_timestamp_seconds{scope=\"busy\"} 9000", text,
                StringComparison.Ordinal);
            Assert.Contains("overfit_guard_last_cycle_timestamp_seconds{scope=\"stalled\"} 10", text,
                StringComparison.Ordinal);
        }

        /// <summary>
        /// A pod regex may legitimately contain a backslash — <c>api-\d+</c> is an ordinary selector — and
        /// unescaped it would end the label value early and corrupt every series after it in the document.
        /// </summary>
        [Fact]
        public void ABackslashInAScopeNameIsEscaped()
        {
            var text = GuardTelemetry.Render(
                [Cycled("payments/api-\\d+", 1, DateTimeOffset.UnixEpoch.AddSeconds(10))]);

            Assert.Contains("scope=\"payments/api-\\\\d+\"", text, StringComparison.Ordinal);
        }
    }
}
