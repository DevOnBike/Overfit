// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Anomalies.Contracts;
using DevOnBike.Overfit.Anomalies.Incidents;
using DevOnBike.Overfit.Anomalies.Incidents.Abstractions;

namespace DevOnBike.Overfit.Tests.Anomalies
{
    /// <summary>
    /// A durable store that cannot read or write must be visible while it is happening.
    ///
    /// <para><b>This is the quietest failure in the subsystem.</b> The stores swallow their exceptions
    /// deliberately — detection that stops because a volume filled up has replaced the problem it was bought
    /// to detect — so cycles keep completing and incidents keep being reported with nothing to suggest
    /// anything is wrong. The bill arrives at the next restart: every open incident reopens at once, and a
    /// week of calibration is gone.</para>
    ///
    /// <para><b>Written after the fix was claimed and had not landed.</b> A changelog entry said this path
    /// shipped; the reality was that <c>FileIncidentStore</c> recorded <c>LastError</c>, the guard held the
    /// interface, and the interface had no such member — so nobody could read it, and
    /// <c>GuardTelemetry.StateWriteFailed()</c> was called from nowhere in the tree. Every piece existed and
    /// none of them was connected. These tests exist so the same claim cannot be made again without the
    /// wiring.</para>
    /// </summary>
    public sealed class GuardStateFailureTests
    {
        private static readonly DateTimeOffset T0 = new(2026, 8, 3, 9, 0, 0, TimeSpan.Zero);

        [Fact]
        public void AFailedSaveIsReportedAndCounted()
        {
            // Healthy at construction, so the load succeeds and the assertion below is about the save
            // rather than about a store that was broken before the guard ever touched it.
            var store = new BrokenStore("disk is full", healthy: true);
            var guard = Guard(store);

            Assert.Null(guard.StateError);

            store.Healthy = false;

            guard.RunCycle(Window(), T0.AddMinutes(20));

            Assert.NotNull(guard.StateError);
            Assert.Contains("disk is full", guard.StateError, StringComparison.Ordinal);
            Assert.Contains(
                "overfit_guard_state_failures_total 1",
                guard.Telemetry.ToPrometheusText(),
                StringComparison.Ordinal);
        }

        /// <summary>
        /// A load that failed is not the same as a store that was empty, and both produce a cold start. The
        /// distinction only exists if it is reported at construction — by the time the first cycle runs, the
        /// guard is already running without its history and looks exactly like a healthy first run.
        /// </summary>
        [Fact]
        public void AFailedLoadIsReportedBeforeTheFirstCycle()
        {
            var guard = Guard(new BrokenStore("state file is not readable"));

            Assert.NotNull(guard.StateError);
            Assert.Equal(0, guard.RestoredIncidents);
            Assert.Contains(
                "overfit_guard_state_failures_total 1",
                guard.Telemetry.ToPrometheusText(),
                StringComparison.Ordinal);
        }

        /// <summary>
        /// A store that recovers clears the report. A latch would turn one bad minute into a permanent
        /// alarm, and an operator who has seen an alert that never clears stops reading it.
        /// </summary>
        [Fact]
        public void RecoveryClearsTheReport()
        {
            var store = new BrokenStore("temporarily unavailable");
            var guard = Guard(store);

            Assert.NotNull(guard.StateError);

            store.Healthy = true;
            guard.RunCycle(Window(), T0.AddMinutes(20));

            Assert.Null(guard.StateError);

            // The counter does not go back down — it is a total, and the minute that failed still happened.
            Assert.Contains(
                "overfit_guard_state_failures_total 1",
                guard.Telemetry.ToPrometheusText(),
                StringComparison.Ordinal);
        }

        /// <summary>
        /// Both stores failing is one cycle that could not persist, not two. The counter answers "how many
        /// cycles", so a host alerting on its rate must not see a step change from a configuration that
        /// added a second store.
        /// </summary>
        [Fact]
        public void TwoBrokenStoresCountAsOneFailedCycle()
        {
            var guard = Guard(new BrokenStore("incidents unavailable"), new BrokenStore("history unavailable"));

            Assert.Contains(
                "overfit_guard_state_failures_total 1",
                guard.Telemetry.ToPrometheusText(),
                StringComparison.Ordinal);
        }

        private static AnomalyGuard Guard(IIncidentStore store, IIncidentStore? history = null)
        {
            return new AnomalyGuard(
                new AnomalyGuardOptions
                {
                    Namespace = "lab",
                    Workload = "svc",
                    MinimumHistoryDays = 0,
                    Grouping = IncidentGroupingOptions.Balanced with { Topology = TopologyWeights.SingleNode },
                },
                new NullSink(),
                IncidentTrackingOptions.Balanced,
                store,
                T0,
                history);
        }

        private static MetricWindow Window()
        {
            var pods = new List<string> { "svc-a", "svc-b", "svc-c" };
            var window = new MetricWindow(pods, 80, T0, TimeSpan.FromSeconds(15));

            for (var pod = 0; pod < pods.Count; pod++)
            {
                var cpu = window.Series(pod, MetricIndex.CpuUsageRatio);
                var rps = window.Series(pod, MetricIndex.RequestsPerSecond);

                for (var i = 0; i < cpu.Length; i++)
                {
                    cpu[i] = 0.4;
                    rps[i] = 5.0;
                }
            }

            return window;
        }

        /// <summary>
        /// A store that behaves the way the file one does when the volume is unwritable: it fails, it says
        /// so through <see cref="LastError"/>, and it never throws.
        /// </summary>
        private sealed class BrokenStore : IIncidentStore
        {
            private readonly string _reason;

            public BrokenStore(string reason, bool healthy = false)
            {
                _reason = reason;
                Healthy = healthy;
                LastError = healthy ? null : reason;
            }

            /// <summary>Flip either way, so a failure and a recovery are both reachable.</summary>
            public bool Healthy
            {
                get; set;
            }

            public string? LastError
            {
                get; private set;
            }

            public string? Load()
            {
                LastError = Healthy ? null : _reason;

                return null;
            }

            public void Save(string state)
            {
                LastError = Healthy ? null : _reason;
            }
        }

        private sealed class NullSink : IIncidentSink
        {
            public void Report(ReadOnlySpan<IncidentLogRecord> rows)
            {
            }
        }
    }
}
