// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Anomalies.Contracts;
using DevOnBike.Overfit.Anomalies.Hosting;
using DevOnBike.Overfit.Anomalies.Incidents;
using DevOnBike.Overfit.Anomalies.Incidents.Abstractions;
using DevOnBike.Overfit.Anomalies.Monitoring.Abstractions;
using DevOnBike.Overfit.Statistics;
using Microsoft.Extensions.Logging;

namespace DevOnBike.Overfit.Tests.Anomalies
{
    /// <summary>
    /// The calibrator's floor proposals, as they reach the log — <c>AN-D4b</c>.
    ///
    /// <para><b>What was wrong.</b> <c>FloorCalibrator</c> computes three proposals per metric and the
    /// service printed two. The step proposal — the number that tells an operator what to write into the
    /// new <c>minStepChange</c> — was computed on every interval and read by nothing. Measured on the
    /// running lab guard: 60 minutes of log, zero mentions.</para>
    ///
    /// <para>A second, older defect of the same shape sat beside it: all three gates were skipped unless
    /// the PEER proposal happened to be positive, so a channel with a good step or trend proposal and an
    /// empty peer one was never mentioned at all.</para>
    /// </summary>
    public sealed class FloorProposalLoggingTests
    {
        private static readonly DateTimeOffset T0 = new(2026, 8, 11, 8, 0, 0, TimeSpan.Zero);
        private static readonly TimeSpan Cadence = TimeSpan.FromMinutes(5);

        /// <summary><b>The defect, as a test.</b> The step proposal reaches the log.</summary>
        [Fact]
        public async Task TheStepProposalIsReported()
        {
            var lines = await RunUntilProposal();

            Assert.Contains(lines, l => l.Contains("step in the workload's level", StringComparison.Ordinal));
        }

        /// <summary>All three gates are named, so none of them can go unexamined for months again.</summary>
        [Fact]
        public async Task AllThreeGatesAreNamed()
        {
            var lines = await RunUntilProposal();

            foreach (var gate in new[] { "peer gap", "trend change across a window", "step in the workload's level" })
            {
                Assert.Contains(lines, l => l.Contains(gate, StringComparison.Ordinal));
            }
        }

        /// <summary>
        /// The step proposal is compared against the EFFECTIVE floor. With no <c>minStepChange</c>
        /// configured the gate really runs on the trend floor, so a trend floor that already covers the
        /// proposal must silence it — otherwise the guard advises changing a number that is already high
        /// enough, every interval, for ever.
        /// </summary>
        [Fact]
        public async Task ATrendFloorThatAlreadyCoversTheStepSilencesTheProposal()
        {
            var trend = new double[(int)MetricIndex.Count];

            // Far above anything the synthetic window can produce.
            for (var m = 0; m < trend.Length; m++)
            {
                trend[m] = 1e9;
            }

            var lines = await RunUntilProposal(trend);

            Assert.DoesNotContain(lines, l => l.Contains("step in the workload's level", StringComparison.Ordinal));
        }

        /// <summary>
        /// And a configured step floor overrides the trend one for that comparison — the log must follow the
        /// same fallback the detector uses, or it proposes against a number nothing reads.
        /// </summary>
        [Fact]
        public async Task AConfiguredStepFloorIsWhatTheProposalIsComparedAgainst()
        {
            var trend = new double[(int)MetricIndex.Count];
            var step = new double[(int)MetricIndex.Count];

            for (var m = 0; m < trend.Length; m++)
            {
                trend[m] = 1e9;      // would silence it
                step[m] = 1e-9;      // but the step table is what counts, and it is tiny
            }

            var lines = await RunUntilProposal(trend, step);

            Assert.Contains(lines, l => l.Contains("step in the workload's level", StringComparison.Ordinal));
        }

        /// <summary>
        /// <b>The older defect beside it.</b> All three gates used to be skipped unless the PEER proposal
        /// was positive, so a channel whose replicas sit on top of each other — no peer gap to propose —
        /// was never mentioned at all, however much its level moved. A flat fleet that drifts is exactly
        /// that shape, and it is not exotic: replicas of one deployment under an even load balancer are
        /// supposed to look alike.
        ///
        /// <para>This arm exists because a mutation restoring the old gate left every other test in this
        /// class green — they all use a fleet whose pods differ, so the peer proposal is always positive
        /// and the extra condition never fires.</para>
        /// </summary>
        [Fact]
        public async Task AChannelWithNoPeerSpreadStillGetsItsOtherGatesProposed()
        {
            var lines = await RunUntilProposal(flatFleet: true);

            Assert.Contains(lines, l => l.Contains("step in the workload's level", StringComparison.Ordinal)
                || l.Contains("trend change across a window", StringComparison.Ordinal));
        }

        private static async Task<List<string>> RunUntilProposal(
            double[]? trend = null, double[]? step = null, bool flatFleet = false)
        {
            var logger = new CapturingLogger();
            var source = new SyntheticSource { Flat = flatFleet };

            using var service = new AnomalyGuardService(
                new AnomalyGuardServiceOptions
                {
                    Guard = new AnomalyGuardOptions
                    {
                        Namespace = "lab",
                        Workload = "lab-workload",
                        MinimumHistoryDays = 0,
                        ApplyCalibratedFloors = false,
                        MinAbsoluteTrendChange = trend,
                        MinAbsoluteStepChange = step,
                    },
                    Tracking = IncidentTrackingOptions.Balanced,

                    // Short enough that a proposal lands inside the loop below. The first interval is
                    // consumed by the service arming itself, so the loop must outlast two of them.
                    FloorProposalInterval = TimeSpan.FromMinutes(10),
                },
                source,
                new NullSink(),
                logger);

            // FloorProposal.MinimumWindows is 24, so a proposal cannot be usable before that many cycles
            // however good the data is — the run has to outlast it.
            for (var cycle = 0; cycle < 40; cycle++)
            {
                await service.RunCycleAsync(T0 + (cycle * Cadence), CancellationToken.None);
            }

            return logger.Lines;
        }

        /// <summary>
        /// Twelve pods whose levels differ a little and drift a little, so every gate's distribution is
        /// non-degenerate — a flat population proposes nothing and the test would pass for the wrong reason.
        /// </summary>
        private sealed class SyntheticSource : IMetricWindowSource
        {
            private int _cycle;

            /// <summary>Every pod identical, so the peer-gap distribution is degenerate and proposes nothing.</summary>
            public bool Flat { get; init; }

            public IReadOnlyList<string> StalePodsExcluded => [];

            public Task<MetricWindow?> ReadAsync(
                DateTimeOffset end, TimeSpan window, CancellationToken ct)
            {
                var names = new List<string>(12);

                for (var p = 0; p < 12; p++)
                {
                    names.Add($"lab-workload-7765564ff6-pod{p:d2}");
                }

                var result = new MetricWindow(names, 40, end - window, TimeSpan.FromSeconds(30));
                var rng = new Random(1000 + _cycle);

                for (var pod = 0; pod < names.Count; pod++)
                {
                    var series = result.Series(pod, MetricIndex.CpuUsageRatio);
                    var spread = Flat ? 0.0 : pod * 0.00002;
                    var level = 0.0015 + spread + (_cycle % 7 * 0.00001);

                    // A flat fleet still has to MOVE within the window, or trend and step propose nothing
                    // either and the arm would pass because everything is silent rather than because the
                    // right thing was said.
                    var draw = Flat ? new Random(2000 + _cycle) : rng;

                    for (var i = 0; i < result.Length; i++)
                    {
                        series[i] = level + ((draw.NextDouble() - 0.5) * 0.0002) + (i * 0.000002);
                    }
                }

                _cycle++;

                return Task.FromResult<MetricWindow?>(result);
            }

            public void Dispose()
            {
            }
        }

        private sealed class NullSink : IIncidentSink
        {
            public void Report(ReadOnlySpan<IncidentLogRecord> rows)
            {
            }
        }

        private sealed class CapturingLogger : ILogger<AnomalyGuardService>
        {
            public List<string> Lines { get; } = [];

            public IDisposable? BeginScope<TState>(TState state)
                where TState : notnull => null;

            public bool IsEnabled(LogLevel logLevel) => true;

            public void Log<TState>(
                LogLevel logLevel, EventId eventId, TState state, Exception? exception,
                Func<TState, Exception?, string> formatter)
                => Lines.Add(formatter(state, exception));
        }
    }
}
