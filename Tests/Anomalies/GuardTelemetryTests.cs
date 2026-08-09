// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Anomalies.Contracts;
using DevOnBike.Overfit.Anomalies.Incidents;
using DevOnBike.Overfit.Anomalies.Incidents.Abstractions;
using DevOnBike.Overfit.Anomalies.Monitoring;

namespace DevOnBike.Overfit.Tests.Anomalies
{
    /// <summary>
    /// The guard watching itself.
    ///
    /// <para>Until this existed it only logged — so a loop that had stopped, or whose queries had begun
    /// failing, produced an absence of incidents that is indistinguishable from a healthy cluster. Every
    /// document here argues that silence and health must be told apart; the component making the argument
    /// could not be told apart from a healthy cluster itself.</para>
    /// </summary>
    public sealed class GuardTelemetryTests
    {
        private static readonly DateTimeOffset T0 = new(2026, 8, 1, 12, 0, 0, TimeSpan.Zero);

        [Fact]
        public void ACycleIsCountedAndTimestamped()
        {
            var guard = Guard(out _);

            guard.RunCycle(Window(), T0.AddMinutes(20));

            var text = guard.Telemetry.ToPrometheusText();

            Assert.Contains("overfit_guard_cycles_total 1", text, StringComparison.Ordinal);
            Assert.Contains("overfit_guard_pods 4", text, StringComparison.Ordinal);

            // The series the one necessary alert is written against.
            Assert.Contains(
                $"overfit_guard_last_cycle_timestamp_seconds {T0.AddMinutes(20).ToUnixTimeSeconds()}",
                text, StringComparison.Ordinal);
        }

        /// <summary>
        /// A failed cycle updates no incident counter, so without its own counter the only evidence is a log
        /// line nobody is watching. A loop failing every five minutes must be visible as a number.
        /// </summary>
        [Fact]
        public void AFailedCycleIsCountedSeparately()
        {
            var telemetry = new GuardTelemetry();

            telemetry.Failed();
            telemetry.Failed();

            var text = telemetry.ToPrometheusText();

            Assert.Contains("overfit_guard_cycle_failures_total 2", text, StringComparison.Ordinal);
            Assert.Contains("overfit_guard_cycles_total 0", text, StringComparison.Ordinal);
        }

        [Fact]
        public void BlindChannelsAreExportedAsANumber()
        {
            var guard = Guard(out _);

            guard.RunCycle(Window(), T0.AddMinutes(20));

            var text = guard.Telemetry.ToPrometheusText();

            // The window fills two channels of thirteen, so eleven are blind. Graphable over a month rather
            // than reconstructed from log lines.
            Assert.Contains("overfit_guard_blind_metrics 11", text, StringComparison.Ordinal);
        }

        [Fact]
        public void SuppressedCyclesAreCounted()
        {
            var guard = new AnomalyGuard(
                new AnomalyGuardOptions
                {
                    Namespace = "lab",
                    Workload = "svc",
                    MinimumHistoryDays = 0,
                    MaintenanceWindows = [new MaintenanceWindow(T0, T0.AddHours(1), "svc", "deploy")],
                    Grouping = IncidentGroupingOptions.Balanced with { Topology = TopologyWeights.SingleNode },
                },
                new NullSink(),
                IncidentTrackingOptions.Balanced);

            guard.RunCycle(Window(), T0.AddMinutes(20));

            Assert.Contains(
                "overfit_guard_suppressed_cycles_total 1",
                guard.Telemetry.ToPrometheusText(),
                StringComparison.Ordinal);
        }

        /// <summary>
        /// Prometheus rejects a body without HELP and TYPE for every series, and a scrape that fails to parse
        /// is another way of being invisible.
        /// </summary>
        [Fact]
        public void EverySeriesCarriesHelpAndType()
        {
            var text = new GuardTelemetry().ToPrometheusText();
            var lines = text.Split('\n', StringSplitOptions.RemoveEmptyEntries);
            var series = 0;

            for (var i = 0; i < lines.Length; i++)
            {
                if (lines[i].StartsWith("overfit_guard_", StringComparison.Ordinal))
                {
                    series++;

                    Assert.StartsWith("# TYPE ", lines[i - 1], StringComparison.Ordinal);
                    Assert.StartsWith("# HELP ", lines[i - 2], StringComparison.Ordinal);
                }
            }

            // Sixteen. Eleven once overfit_guard_state_failures_total joined them — a durable-state write
            // that fails is otherwise silent until the next restart reopens everything at once — and four
            // more for operator feedback. Those four exist because every response an operator can give makes
            // the guard quieter, so the amount of silence they have bought has to be visible from outside.
            //
            // The sixteenth is overfit_guard_peer_findings_standing_total (AN-D1). Same argument as the
            // feedback four, applied to a gate rather than to an operator: the novelty gate holds peer
            // findings back, and a channel that has gone quiet is indistinguishable from a healthy cluster
            // unless the amount being held is exported.
            Assert.Equal(16, series);
        }

        private static AnomalyGuard Guard(out NullSink sink)
        {
            sink = new NullSink();

            return new AnomalyGuard(
                new AnomalyGuardOptions
                {
                    Namespace = "lab",
                    Workload = "svc",
                    MinimumHistoryDays = 0,
                    Grouping = IncidentGroupingOptions.Balanced with { Topology = TopologyWeights.SingleNode },
                },
                sink,
                IncidentTrackingOptions.Balanced);
        }

        private static MetricWindow Window()
        {
            var names = new List<string> { "pod-0", "pod-1", "pod-2", "pod-3" };
            var window = new MetricWindow(names, 80, T0, TimeSpan.FromSeconds(15));

            for (var pod = 0; pod < names.Count; pod++)
            {
                var cpu = window.Series(pod, MetricIndex.CpuUsageRatio);
                var rps = window.Series(pod, MetricIndex.RequestsPerSecond);

                for (var i = 0; i < window.Length; i++)
                {
                    cpu[i] = 0.5;
                    rps[i] = 10.0;
                }
            }

            return window;
        }

        private sealed class NullSink : IIncidentSink
        {
            public void Report(ReadOnlySpan<IncidentLogRecord> rows)
            {
            }
        }
    }
}
