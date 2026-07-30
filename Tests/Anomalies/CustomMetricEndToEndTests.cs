// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Anomalies.Incidents;
using DevOnBike.Overfit.Anomalies.Incidents.Abstractions;
using DevOnBike.Overfit.Anomalies.Incidents.Contracts;
using DevOnBike.Overfit.Anomalies.Monitoring.Contracts;
using DevOnBike.Overfit.Anomalies.Rules.Contracts;
using DevOnBike.Overfit.Statistics;

namespace DevOnBike.Overfit.Tests.Anomalies
{
    /// <summary>
    /// A metric this project does not model, carried all the way to a reported incident.
    ///
    /// <para><b>What makes this possible is that only the top of the stack knows the enum.</b>
    /// <c>MetricSnapshot.FeatureCount</c> is a trained model's input contract and cannot move per customer —
    /// adding a thirteenth channel once broke two committed checkpoints and a 201 000-row fixture. But
    /// <c>SignalFinding.Signal</c> is a string, so the grouper, tracker and reporter never needed the enum,
    /// and a custom channel reaches them unchanged.</para>
    /// </summary>
    public sealed class CustomMetricEndToEndTests
    {
        private const string Lag = "kafka_consumer_lag";

        private static readonly DateTimeOffset T0 = new(2026, 7, 30, 12, 0, 0, TimeSpan.Zero);

        /// <summary>One pod's lag far above its siblings must be named, by the client's own metric name.</summary>
        [Fact]
        public void ACustomMetricProducesAPeerFinding()
        {
            var sink = new CapturingSink();
            var guard = Guard(sink, Binding());

            guard.RunCycle(Window(pods: 8, outlierLag: 90_000.0), T0);

            var signals = new SortedSet<string>(StringComparer.Ordinal);

            foreach (var row in sink.Rows)
            {
                signals.Add(row.Signal);
            }

            Assert.Contains(Lag, signals);
        }

        /// <summary>The absolute rule reaches a custom metric too — the case relative methods cannot cover.</summary>
        [Fact]
        public void ACustomMetricCanCarryAnAbsoluteRule()
        {
            var sink = new CapturingSink();
            var binding = Binding() with
            {
                Rule = new SustainedThresholdOptions(
                    Threshold: 50_000.0, MinBreachFraction: 0.25, MinimumSamples: 20),
            };

            // Every pod above the line: nothing to compare against, so only a rule can see it.
            guardRun(sink, binding, Window(pods: 8, outlierLag: 90_000.0, baseLag: 80_000.0));

            var reasons = new List<string>();

            foreach (var row in sink.Rows)
            {
                if (string.Equals(row.Signal, Lag, StringComparison.Ordinal))
                {
                    reasons.Add(row.Message);
                }
            }

            Assert.Contains(reasons, r => r.Contains("of the window", StringComparison.Ordinal));
        }

        /// <summary>Its class follows the configuration, because the grouper orders an incident cause-first.</summary>
        [Fact]
        public void TheConfiguredSignalClassIsCarried()
        {
            var sink = new CapturingSink();
            var guard = Guard(sink, Binding() with { Class = SignalClass.Symptom });

            guard.RunCycle(Window(pods: 8, outlierLag: 90_000.0), T0);

            foreach (var row in sink.Rows)
            {
                if (string.Equals(row.Signal, Lag, StringComparison.Ordinal)
                    && row.Kind == IncidentLogRecordKind.Finding)
                {
                    Assert.Equal(SignalClass.Symptom, row.Class);
                }
            }
        }

        /// <summary>A custom channel nobody reports is blindness, exactly as for a modelled one.</summary>
        [Fact]
        public void AnUnreportedCustomChannelCountsAsBlind()
        {
            var sink = new CapturingSink();
            var guard = Guard(sink, Binding());

            // The window carries the channel but no pod fills it.
            var window = new MetricWindow(
                ["pod-0", "pod-1", "pod-2"], 60, T0, TimeSpan.FromSeconds(15), [Lag]);

            var before = guard.RunCycle(window, T0);

            Assert.True(before.BlindMetrics >= 1);
        }

        private static void guardRun(CapturingSink sink, CustomMetricBinding binding, MetricWindow window)
        {
            Guard(sink, binding).RunCycle(window, T0);
        }

        private static CustomMetricBinding Binding()
        {
            return new CustomMetricBinding(
                Name: Lag,
                Source: "kafka_consumergroup_lag",
                Kind: MetricSourceKind.Gauge,
                SignalKind: PeerSignalKind.LoadIndependent,
                Class: SignalClass.Resource,
                MinAbsoluteGap: 1000.0);
        }

        private static AnomalyGuard Guard(IIncidentSink sink, CustomMetricBinding binding)
        {
            return new AnomalyGuard(
                new AnomalyGuardOptions
                {
                    Namespace = "overfit",
                    Workload = "consumer",
                    CustomMetrics = [binding],
                    Grouping = IncidentGroupingOptions.Balanced with
                    {
                        Topology = TopologyWeights.SingleNode
                    },
                },
                sink,
                IncidentTrackingOptions.Balanced);
        }

        /// <summary>
        /// Eight pods, because one outlier in three is half of each other member's leave-one-out baseline and
        /// the group comes back Inconclusive — the masking bound, measured.
        /// </summary>
        private static MetricWindow Window(int pods, double outlierLag, double baseLag = 500.0)
        {
            var names = new List<string>(pods);

            for (var p = 0; p < pods; p++)
            {
                names.Add($"consumer-111-pod{p:d2}");
            }

            var window = new MetricWindow(names, 60, T0, TimeSpan.FromSeconds(15), [Lag]);
            var rng = new Random(20260730);

            for (var p = 0; p < pods; p++)
            {
                var lag = window.Series(p, Lag);
                var rps = window.Series(p, MetricIndex.RequestsPerSecond);

                for (var i = 0; i < window.Length; i++)
                {
                    lag[i] = (p == pods - 1 ? outlierLag : baseLag)
                             * (1.0 + ((rng.NextDouble() - 0.5) * 0.1));
                    rps[i] = 5.0;
                }
            }

            return window;
        }

        private sealed class CapturingSink : IIncidentSink
        {
            public List<IncidentLogRecord> Rows { get; } = [];

            public void Report(ReadOnlySpan<IncidentLogRecord> rows)
            {
                for (var i = 0; i < rows.Length; i++)
                {
                    Rows.Add(rows[i]);
                }
            }
        }
    }
}
