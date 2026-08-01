// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Anomalies.Contracts;
using DevOnBike.Overfit.Anomalies.Incidents;
using DevOnBike.Overfit.Anomalies.Incidents.Abstractions;
using DevOnBike.Overfit.Anomalies.Monitoring.Abstractions;

namespace DevOnBike.Overfit.Tests.Anomalies
{
    /// <summary>
    /// A pod the cluster lists and that reports nothing.
    ///
    /// <para>The blind spot these close is structural rather than a matter of thresholds: every other family
    /// judges a time series, and this pod has none. Eleven healthy replicas and twelve replicas one of which
    /// never started produce the same window, and before this the guard said nothing about either — which the
    /// operator reads as health.</para>
    /// </summary>
    public sealed class SilentPodTests
    {
        private static readonly DateTimeOffset T0 = new(2026, 8, 1, 12, 0, 0, TimeSpan.Zero);

        [Fact]
        public void OneQuietCycleIsNotAFinding()
        {
            var sink = new CapturingSink();
            var guard = Guard(sink, roster: ["pod-0", "pod-1", "pod-2", "pod-missing"]);

            guard.RunCycle(Window(3), T0);

            // A pod created moments ago has no samples yet and is not a problem.
            Assert.DoesNotContain(sink.Messages, m => m.Contains("reported no metrics", StringComparison.Ordinal));
        }

        [Fact]
        public void PersistentSilenceIsReported()
        {
            var sink = new CapturingSink();
            var guard = Guard(sink, roster: ["pod-0", "pod-1", "pod-2", "pod-missing"]);

            guard.RunCycle(Window(3), T0);
            guard.RunCycle(Window(3), T0.AddMinutes(5));

            Assert.Contains(sink.Messages, m => m.Contains("reported no metrics", StringComparison.Ordinal));
            Assert.Contains(sink.Pods, p => p == "pod-missing");
        }

        /// <summary>
        /// A pod that comes back must clear its counter, or a slow starter is reported for ever afterwards on
        /// the strength of cycles it has already recovered from.
        /// </summary>
        [Fact]
        public void ReportingAgainResetsTheCount()
        {
            var sink = new CapturingSink();
            var guard = Guard(sink, roster: ["pod-0", "pod-1", "pod-2", "pod-3"]);

            guard.RunCycle(Window(3), T0);
            guard.RunCycle(Window(4), T0.AddMinutes(5));
            guard.RunCycle(Window(3), T0.AddMinutes(10));

            Assert.DoesNotContain(sink.Messages, m => m.Contains("reported no metrics", StringComparison.Ordinal));
        }

        /// <summary>
        /// Scale-down is not a fault. When the cluster stops listing a pod, it stops being expected — and the
        /// guard must not go on reporting a replica that was deliberately removed.
        /// </summary>
        [Fact]
        public void APodTheClusterHasForgottenIsNotReported()
        {
            var sink = new CapturingSink();
            var roster = new List<string> { "pod-0", "pod-1", "pod-2", "pod-3" };
            var guard = Guard(sink, roster);

            guard.RunCycle(Window(3), T0);

            roster.Remove("pod-3");

            guard.RunCycle(Window(3), T0.AddMinutes(5));

            Assert.DoesNotContain(sink.Messages, m => m.Contains("reported no metrics", StringComparison.Ordinal));
        }

        [Fact]
        public void WithoutARosterTheCheckIsSkipped()
        {
            var sink = new CapturingSink();

            var guard = new AnomalyGuard(
                new AnomalyGuardOptions
                {
                    Namespace = "lab",
                    Workload = "svc",
                    Grouping = IncidentGroupingOptions.Balanced with { Topology = TopologyWeights.SingleNode },
                },
                sink,
                IncidentTrackingOptions.Balanced);

            guard.RunCycle(Window(3), T0);
            guard.RunCycle(Window(3), T0.AddMinutes(5));

            Assert.DoesNotContain(sink.Messages, m => m.Contains("reported no metrics", StringComparison.Ordinal));
        }

        /// <summary>
        /// An empty roster means "nothing known", not "no pods exist". Read the other way, every reporting pod
        /// would become unexpected — the inverse of this check's job.
        /// </summary>
        [Fact]
        public void AnEmptyRosterReportsNothing()
        {
            var sink = new CapturingSink();
            var guard = Guard(sink, roster: []);

            guard.RunCycle(Window(3), T0);
            guard.RunCycle(Window(3), T0.AddMinutes(5));

            Assert.DoesNotContain(sink.Messages, m => m.Contains("reported no metrics", StringComparison.Ordinal));
        }

        private static AnomalyGuard Guard(CapturingSink sink, List<string> roster)
        {
            return new AnomalyGuard(
                new AnomalyGuardOptions
                {
                    Namespace = "lab",
                    Workload = "svc",
                    PodTopology = new FakeTopology(roster),
                    Grouping = IncidentGroupingOptions.Balanced with { Topology = TopologyWeights.SingleNode },
                },
                sink,
                IncidentTrackingOptions.Balanced);
        }

        /// <summary>A perfectly healthy window of <paramref name="pods"/> replicas.</summary>
        private static MetricWindow Window(int pods)
        {
            var names = new List<string>(pods);

            for (var p = 0; p < pods; p++)
            {
                names.Add($"pod-{p}");
            }

            var window = new MetricWindow(names, 80, T0, TimeSpan.FromSeconds(15));
            var rng = new Random(20260801);

            for (var pod = 0; pod < pods; pod++)
            {
                var cpu = window.Series(pod, MetricIndex.CpuUsageRatio);
                var rps = window.Series(pod, MetricIndex.RequestsPerSecond);

                for (var i = 0; i < window.Length; i++)
                {
                    cpu[i] = 0.5 * (1.0 + ((rng.NextDouble() - 0.5) * 0.02));
                    rps[i] = 10.0 * (1.0 + ((rng.NextDouble() - 0.5) * 0.02));
                }
            }

            return window;
        }

        private sealed class FakeTopology : IPodTopology, IPodRoster
        {
            private readonly List<string> _pods;

            public FakeTopology(List<string> pods)
            {
                _pods = pods;
            }

            public IReadOnlyList<string> KnownPods => _pods;

            public bool TryResolve(string pod, out PodPlacement placement)
            {
                placement = new PodPlacement("lab", "svc", string.Empty, "node-0");

                return _pods.Contains(pod);
            }
        }

        private sealed class CapturingSink : IIncidentSink
        {
            public List<string> Messages { get; } = [];

            public List<string> Pods { get; } = [];

            public void Report(ReadOnlySpan<IncidentLogRecord> rows)
            {
                for (var i = 0; i < rows.Length; i++)
                {
                    Messages.Add(rows[i].Message);
                    Pods.Add(rows[i].Pod);
                }
            }
        }
    }
}
