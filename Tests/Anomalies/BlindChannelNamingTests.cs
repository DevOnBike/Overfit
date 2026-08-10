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
    /// `blind=1` on the cycle line tells an operator the guard is partly blind and nothing about which query
    /// to go and fix. The names are the actionable part, and until 2026-08-10 (`XC-11`) only the built-in
    /// channels got them.
    ///
    /// <para><b>The loop was keyed on `MetricIndex`</b>, so a channel a deployment added itself was counted
    /// and never named — on the lab that is five channels, and it is the half of the system a client is most
    /// likely to extend. The built-in half was already right, and its own comment argues the case: *"the
    /// names are the whole actionable part, and they are cheap because the window already knows."*</para>
    /// </summary>
    public sealed class BlindChannelNamingTests
    {
        private static readonly DateTimeOffset T0 = new(2026, 8, 10, 9, 0, 0, TimeSpan.Zero);

        [Fact]
        public async Task ACustomChannelNobodyReportedIsNamedAndNotJustCounted()
        {
            var log = new CapturingLogger();
            using var source = new OneWindowSource(WindowWithout("GcCommittedBytes"));
            using var service = Service(source, log);

            await service.RunCycleAsync(T0, CancellationToken.None);

            Assert.Contains(
                log.Messages,
                m => m.Contains("Blind on", StringComparison.Ordinal)
                     && m.Contains("GcCommittedBytes", StringComparison.Ordinal));
        }

        /// <summary>
        /// The control, and without it the test above would pass on a guard that named every channel every
        /// cycle. A custom channel every pod reported must NOT be announced as blind.
        /// </summary>
        [Fact]
        public async Task ACustomChannelEveryPodReportedIsNotNamed()
        {
            var log = new CapturingLogger();
            using var source = new OneWindowSource(WindowWith("GcCommittedBytes"));
            using var service = Service(source, log);

            await service.RunCycleAsync(T0, CancellationToken.None);

            Assert.DoesNotContain(
                log.Messages,
                m => m.Contains("Blind on", StringComparison.Ordinal)
                     && m.Contains("GcCommittedBytes", StringComparison.Ordinal));
        }

        private static AnomalyGuardService Service(IMetricWindowSource source, ILogger<AnomalyGuardService> log)
        {
            return new AnomalyGuardService(
                new AnomalyGuardServiceOptions
                {
                    Guard = new AnomalyGuardOptions
                    {
                        Namespace = "lab",
                        Workload = "svc",
                        CustomMetrics =
                        [
                            new CustomMetricBinding(
                                Name: "GcCommittedBytes",
                                Source: "dotnet_gc_committed_bytes",
                                Kind: MetricSourceKind.Gauge,
                                SignalKind: PeerSignalKind.LoadIndependent,
                                Class: SignalClass.Resource),
                        ],
                        Grouping = IncidentGroupingOptions.Balanced with
                        {
                            Topology = TopologyWeights.SingleNode,
                        },
                    },
                    Tracking = IncidentTrackingOptions.Balanced,
                },
                source,
                new DiscardingSink(),
                log);
        }

        /// <summary>A window that declares the channel and carries no samples for it — the blind case.</summary>
        private static MetricWindow WindowWithout(string channel)
        {
            return Build(channel, report: false);
        }

        /// <summary>The same window with every pod reporting it — the control.</summary>
        private static MetricWindow WindowWith(string channel)
        {
            return Build(channel, report: true);
        }

        private static MetricWindow Build(string channel, bool report)
        {
            var names = new List<string> { "pod-0", "pod-1", "pod-2", "pod-3" };
            var window = new MetricWindow(names, 60, T0, TimeSpan.FromSeconds(15), [channel]);

            for (var pod = 0; pod < names.Count; pod++)
            {
                // A built-in channel is always populated, so `blind` is never zero for the wrong reason and
                // the cycle has something to evaluate either way.
                var rps = window.Series(pod, MetricIndex.RequestsPerSecond);
                var custom = window.Series(pod, channel);

                for (var i = 0; i < window.Length; i++)
                {
                    rps[i] = 5.0;
                    custom[i] = report ? 40e6 : double.NaN;
                }
            }

            return window;
        }

        /// <summary>
        /// Serves one prepared window and then nothing.
        ///
        /// <para>A local copy rather than a shared helper: the equivalents in
        /// <c>AnomalyGuardServiceCycleTests</c> and <c>AnomalyGuardLoopTests</c> are private nested types, and
        /// promoting them would edit two unrelated files to save eighteen lines here.</para>
        /// </summary>
        private sealed class OneWindowSource : IMetricWindowSource
        {
            private MetricWindow? _window;

            public OneWindowSource(MetricWindow window)
            {
                _window = window;
            }

            public IReadOnlyList<string> StalePodsExcluded => [];

            public Task<MetricWindow?> ReadAsync(
                DateTimeOffset end, TimeSpan window, CancellationToken ct = default)
            {
                var served = _window;
                _window = null;

                return Task.FromResult(served);
            }

            // `IMetricWindowSource` extends `IDisposable` for the Prometheus implementation's `HttpClient`.
            // A window is not disposable and this source owns nothing else, so there is nothing to release.
            public void Dispose()
            {
            }
        }

        private sealed class DiscardingSink : IIncidentSink
        {
            public void Report(ReadOnlySpan<IncidentLogRecord> rows)
            {
            }
        }

        private sealed class CapturingLogger : ILogger<AnomalyGuardService>
        {
            public List<string> Messages { get; } = [];

            public IDisposable? BeginScope<TState>(TState state)
                where TState : notnull
            {
                return null;
            }

            public bool IsEnabled(LogLevel logLevel)
            {
                return true;
            }

            public void Log<TState>(
                LogLevel logLevel,
                EventId eventId,
                TState state,
                Exception? exception,
                Func<TState, Exception?, string> formatter)
            {
                ArgumentNullException.ThrowIfNull(formatter);

                Messages.Add(formatter(state, exception));
            }
        }
    }
}
