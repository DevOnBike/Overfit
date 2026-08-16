// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Anomalies.Contracts;
using DevOnBike.Overfit.Anomalies.Incidents;
using DevOnBike.Overfit.Anomalies.Incidents.Abstractions;
using DevOnBike.Overfit.Statistics;

namespace DevOnBike.Overfit.Tests.Anomalies
{
    /// <summary>
    /// The flattening from incident tree to log rows.
    ///
    /// <para>These assertions are about a <b>schema</b>, not an implementation detail: once a dashboard query
    /// or a log filter is saved against these field names, they are an interface. A change here should have
    /// to be argued for.</para>
    /// </summary>
    public sealed class IncidentReporterTests
    {
        private static readonly DateTimeOffset Start = new(2026, 7, 30, 12, 0, 0, TimeSpan.Zero);

        [Fact]
        public void EmitsOneIncidentRowPlusOneRowPerFinding_JoinedByKey()
        {
            var sink = new CapturingSink();
            var incidents = Group(
                Finding("pod-a", "cpu_usage", SignalClass.Resource, 0.9),
                Finding("pod-a", "latency_p95", SignalClass.Symptom, 0.6));

            var written = IncidentReporter.Report(incidents, sink);

            Assert.Equal(3, written);
            Assert.Equal(3, sink.Rows.Count);

            Assert.Equal(IncidentLogRecordKind.Incident, sink.Rows[0].Kind);
            Assert.Equal(IncidentLogRecordKind.Finding, sink.Rows[1].Kind);
            Assert.Equal(IncidentLogRecordKind.Finding, sink.Rows[2].Kind);

            Assert.All(sink.Rows, r => Assert.Equal(0, r.IncidentKey));
        }

        [Fact]
        public void TheIncidentRowCarriesTheGroupsShape_NotTheFirstFindings()
        {
            var sink = new CapturingSink();
            var incidents = Group(
                Finding("pod-a", "cpu_usage", SignalClass.Resource, 0.4),
                Finding("pod-b", "latency_p95", SignalClass.Symptom, 0.95));

            IncidentReporter.Report(incidents, sink);
            var row = sink.Rows[0];

            // Peak, not the primary finding's own severity — the incident is as bad as its worst signal.
            Assert.Equal(0.95, row.Severity, 6);
            Assert.Equal(2, row.Subjects);
            Assert.Equal(2, row.Signals);
        }

        /// <summary>
        /// A common-mode row carries no pod by design. Consumers must be able to tell that apart from a
        /// missing field, because "about the workload" and "unknown pod" route completely differently.
        /// </summary>
        [Fact]
        public void ACommonModeFinding_NamesNoPod_AndSaysSo()
        {
            var sink = new CapturingSink();
            var incidents = Group(
                new SignalFinding(
                    new IncidentSubject("overfit", "overfit-server", string.Empty, string.Empty, string.Empty),
                    "latency_p50", SignalClass.Symptom, Start, Start.AddMinutes(12), 0.5,
                    "the deployment as a whole"));

            IncidentReporter.Report(incidents, sink);

            var finding = sink.Rows[1];

            Assert.False(finding.NamesAPod);
            Assert.Equal(string.Empty, finding.Pod);
            Assert.Equal("overfit-server", finding.Workload);
        }

        [Fact]
        public void APodFinding_NamesItsPod()
        {
            var sink = new CapturingSink();
            IncidentReporter.Report(Group(Finding("pod-a", "cpu_usage", SignalClass.Resource, 0.9)), sink);

            Assert.True(sink.Rows[1].NamesAPod);
            Assert.Equal("pod-a", sink.Rows[1].Pod);
        }

        [Fact]
        public void NothingToReport_TouchesTheSinkNotAtAll()
        {
            var sink = new CapturingSink();

            Assert.Equal(0, IncidentReporter.Report([], sink));
            Assert.Equal(0, sink.Calls);
        }

        [Fact]
        public void RejectsNulls()
        {
            Assert.Throws<ArgumentNullException>(() => IncidentReporter.Report(null!, new CapturingSink()));
            Assert.Throws<ArgumentNullException>(() => IncidentReporter.Report([], null!));
        }

        /// <summary>The window, not the moment of detection — detection latency differs per detector.</summary>
        [Fact]
        public void RowsCarryTheEvaluatedWindow()
        {
            var sink = new CapturingSink();
            IncidentReporter.Report(Group(Finding("pod-a", "cpu_usage", SignalClass.Resource, 0.9)), sink);

            Assert.Equal(Start, sink.Rows[0].Start);
            Assert.Equal(TimeSpan.FromMinutes(12), sink.Rows[0].Duration);
        }

        private static SignalFinding Finding(string pod, string signal, SignalClass cls, double severity)
        {
            return new SignalFinding(
                new IncidentSubject("overfit", "overfit-server", string.Empty, pod, "node-1"),
                signal, cls, Start, Start.AddMinutes(12), severity, $"{signal} on {pod}");
        }

        private static IReadOnlyList<TrackedIncident> Group(params SignalFinding[] findings)
        {
            var pipeline = new IncidentPipeline();

            foreach (var finding in findings)
            {
                // Severity on a rule finding is the breach fraction, so that is the field to set.
                pipeline.ObserveRule(
                    finding.Subject, finding.Signal,
                    new SustainedThresholdResult(
                        Status: DetectionStatus.Anomalous,
                        Reason: finding.Reason,
                        BreachFraction: finding.Severity,
                        BreachedSamples: 30,
                        UsableSamples: 40,
                        PeakValue: 1.0,
                        MedianValue: 0.5),
                    finding.Start, finding.End, default, finding.Class);
            }

            var grouped = pipeline.Group(IncidentGroupingOptions.Balanced with
            {
                Topology = TopologyWeights.SingleNode
            });

            return new IncidentTracker(IncidentTrackingOptions.Balanced).Observe(grouped, Start);
        }

        private sealed class CapturingSink : IIncidentSink
        {
            public List<IncidentLogRecord> Rows { get; } = [];

            public int Calls
            {
                get; private set;
            }

            public void Report(ReadOnlySpan<IncidentLogRecord> rows)
            {
                Calls++;

                for (var i = 0; i < rows.Length; i++)
                {
                    Rows.Add(rows[i]);
                }
            }
        }
    }
}
