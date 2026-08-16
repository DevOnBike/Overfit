// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Anomalies.Contracts;
using DevOnBike.Overfit.Anomalies.Hosting;
using Microsoft.Extensions.Logging;

namespace DevOnBike.Overfit.Tests.Anomalies
{
    /// <summary>
    /// <see cref="LoggerIncidentSink"/> — <c>XC-16</c>, and the first tests this class has ever had.
    ///
    /// <para><b>What it cost to not have them.</b> The sink dropped every finding belonging to an incident
    /// that was already open. The reasoning was sound for an operator's alert stream and catastrophic for
    /// anyone diagnosing the guard: during the <c>AN-D3</c> injection it reported a fleet-wide CPU step
    /// correctly for half an hour while emitting <c>cycle: findings=3</c> and not one word about what those
    /// findings were. From outside, a detector working perfectly and a detector reporting nothing produced
    /// identical output — which is this subsystem's entire failure mode. The verdict had to be recovered by
    /// refetching the window from Prometheus and re-running the detector offline.</para>
    ///
    /// <para>Both halves are pinned here: ongoing evidence must be emitted (at <c>Debug</c>, so it does not
    /// return to the operator's stream), and a deployment-wide finding must carry the detector's own
    /// sentence — it did not, while the per-pod finding beside it did.</para>
    /// </summary>
    public sealed class LoggerIncidentSinkTests
    {
        /// <summary>
        /// <b>The defect, stated as a test.</b> An ongoing finding used to produce nothing at all.
        /// </summary>
        [Fact]
        public void AnOngoingFindingIsStillRecorded()
        {
            var logger = new CapturingLogger();
            var sink = new LoggerIncidentSink(logger, IncidentLogOptions.Shadow);

            sink.Report([Finding(IncidentState.Ongoing, pod: "lab-workload-abc", message: "still climbing")]);

            var entry = Assert.Single(logger.Entries);

            Assert.Contains("still climbing", entry.Message, StringComparison.Ordinal);
            Assert.Contains("lab-workload-abc", entry.Message, StringComparison.Ordinal);
        }

        /// <summary>
        /// And it goes to <c>Debug</c>, not to the operator's level. The suppression existed for a real
        /// reason — repeating every finding of every open incident on every cycle is the
        /// notifications-per-hour behaviour the incident tracker exists to remove — so the fix must not
        /// hand that back.
        /// </summary>
        [Fact]
        public void OngoingEvidenceDoesNotReachTheOperatorsLevel()
        {
            var logger = new CapturingLogger();
            var sink = new LoggerIncidentSink(logger, IncidentLogOptions.Shadow);

            sink.Report([Finding(IncidentState.Ongoing, pod: "lab-workload-abc")]);

            Assert.Equal(LogLevel.Debug, Assert.Single(logger.Entries).Level);

            // Shadow puts new findings at Information, so the two are genuinely different levels and this
            // test cannot pass by both being Debug.
            Assert.Equal(LogLevel.Information, IncidentLogOptions.Shadow.FindingLevel);
        }

        /// <summary>A newly opened finding is unaffected — it still reaches the configured level.</summary>
        [Fact]
        public void AnOpenedFindingStillReachesTheConfiguredLevel()
        {
            var logger = new CapturingLogger();
            var sink = new LoggerIncidentSink(logger, IncidentLogOptions.Shadow);

            sink.Report([Finding(IncidentState.Opened, pod: "lab-workload-abc", message: "rose by 210%")]);

            var entry = Assert.Single(logger.Entries);

            Assert.Equal(LogLevel.Information, entry.Level);
            Assert.Contains("rose by 210%", entry.Message, StringComparison.Ordinal);
        }

        /// <summary>
        /// <b>The second half of XC-16.</b> A deployment-wide finding names no pod, and used to carry no
        /// reason either — so it said THAT the workload moved and never what the detector concluded. The
        /// families that report here disagree with each other: a trend and a level shift both land on the
        /// workload subject and call for opposite responses.
        /// </summary>
        [Fact]
        public void ADeploymentWideFindingCarriesTheDetectorsReason()
        {
            var logger = new CapturingLogger();
            var sink = new LoggerIncidentSink(logger, IncidentLogOptions.Shadow);

            sink.Report([Finding(
                IncidentState.Opened,
                pod: string.Empty,
                message: "This is a step, not a drift: it affects the workload as a whole.")]);

            var entry = Assert.Single(logger.Entries);

            Assert.Contains("no individual replica is implicated", entry.Message, StringComparison.Ordinal);
            Assert.Contains("This is a step, not a drift", entry.Message, StringComparison.Ordinal);
        }

        [Fact]
        public void AnOngoingDeploymentWideFindingCarriesItsReasonToo()
        {
            var logger = new CapturingLogger();
            var sink = new LoggerIncidentSink(logger, IncidentLogOptions.Shadow);

            sink.Report([Finding(IncidentState.Ongoing, pod: string.Empty, message: "still stepped")]);

            var entry = Assert.Single(logger.Entries);

            Assert.Equal(LogLevel.Debug, entry.Level);
            Assert.Contains("still stepped", entry.Message, StringComparison.Ordinal);
        }

        /// <summary>
        /// Turning evidence off turns ALL of it off, ongoing included — otherwise the quiet profile would
        /// have quietly acquired a new channel.
        /// </summary>
        [Fact]
        public void EvidenceOffSuppressesOngoingEvidenceAsWell()
        {
            var logger = new CapturingLogger();
            var sink = new LoggerIncidentSink(logger, IncidentLogOptions.Quiet);

            sink.Report([
                Finding(IncidentState.Ongoing, pod: "lab-workload-abc"),
                Finding(IncidentState.Opened, pod: "lab-workload-abc"),
            ]);

            Assert.Empty(logger.Entries);
        }

        /// <summary>
        /// An ongoing INCIDENT still says nothing, and that is deliberate: the incident was announced with
        /// its summary when it opened, and repeating it is the notification storm. Only the evidence moved.
        /// </summary>
        [Fact]
        public void AnOngoingIncidentIsStillSilent()
        {
            var logger = new CapturingLogger();
            var sink = new LoggerIncidentSink(logger, IncidentLogOptions.Shadow);

            sink.Report([Record(IncidentLogRecordKind.Incident, IncidentState.Ongoing, "pod", "msg")]);

            Assert.Empty(logger.Entries);
        }

        private static IncidentLogRecord Finding(
            IncidentState state, string pod, string message = "reason text")
            => Record(IncidentLogRecordKind.Finding, state, pod, message);

        private static IncidentLogRecord Record(
            IncidentLogRecordKind kind, IncidentState state, string pod, string message)
            => new(
                IncidentKey: 1,
                IncidentId: 7,
                State: state,
                Kind: kind,
                Namespace: "lab",
                Workload: "lab-workload",
                Pod: pod,
                Node: "node-0",
                Signal: "CpuUsageRatio",
                Class: SignalClass.Resource,
                Severity: 0.5,
                Start: new DateTimeOffset(2026, 8, 11, 9, 0, 0, TimeSpan.Zero),
                End: new DateTimeOffset(2026, 8, 11, 9, 20, 0, TimeSpan.Zero),
                Subjects: 1,
                Signals: 1,
                Message: message);

        /// <summary>Captures level and rendered text; the sink's product is the fields, the text is the check.</summary>
        private sealed class CapturingLogger : ILogger<LoggerIncidentSink>
        {
            public List<(LogLevel Level, string Message)> Entries { get; } = [];

            public IDisposable? BeginScope<TState>(TState state)
                where TState : notnull => null;

            // Everything enabled, including Debug — otherwise this test class could not tell "not emitted"
            // from "emitted below the threshold", which is the exact distinction it is here to make.
            public bool IsEnabled(LogLevel logLevel) => true;

            public void Log<TState>(
                LogLevel logLevel, EventId eventId, TState state, Exception? exception,
                Func<TState, Exception?, string> formatter)
                => Entries.Add((logLevel, formatter(state, exception)));
        }
    }
}
