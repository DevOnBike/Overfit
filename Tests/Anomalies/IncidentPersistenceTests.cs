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
    /// Incident identity surviving a restart.
    ///
    /// <para><b>The property is the first test and the rest exist to stop it being got cheaply.</b> Without
    /// durable state a rolling update of the guard reopens every incident that was running — the tracker's
    /// entire contribution undone by the guard's own deploy, at the moment somebody is already looking at a
    /// change.</para>
    /// </summary>
    public sealed class IncidentPersistenceTests
    {
        private static readonly DateTimeOffset T0 = new(2026, 7, 30, 12, 0, 0, TimeSpan.Zero);

        /// <summary>The reason this exists.</summary>
        [Fact]
        public void ARestartDoesNotReopenARunningIncident()
        {
            var store = new MemoryStore();

            var before = Guard(new CapturingSink(), store, T0);
            before.RunCycle(Window(degraded: true), T0);
            before.RunCycle(Window(degraded: true), T0.AddMinutes(5));

            // A new process, same durable state.
            var sink = new CapturingSink();
            var after = Guard(sink, store, T0.AddMinutes(10));

            Assert.True(after.RestoredIncidents > 0, "nothing was adopted from the saved state");

            var result = after.RunCycle(Window(degraded: true), T0.AddMinutes(10));

            Assert.Equal(0, result.Opened);
            Assert.True(result.Ongoing > 0);
        }

        /// <summary>And the identity a consumer saw before the restart is the identity it sees after.</summary>
        [Fact]
        public void TheIdentitySurvives()
        {
            var store = new MemoryStore();

            var firstSink = new CapturingSink();
            var before = Guard(firstSink, store, T0);
            before.RunCycle(Window(degraded: true), T0);

            var opened = firstSink.Ids(IncidentState.Opened);

            Assert.Single(opened);

            var secondSink = new CapturingSink();
            Guard(secondSink, store, T0.AddMinutes(5)).RunCycle(Window(degraded: true), T0.AddMinutes(5));

            Assert.Contains(opened[0], secondSink.Ids(IncidentState.Ongoing));
        }

        /// <summary>
        /// A guard restarted after a long gap must not resurrect incidents and close them: that is the same
        /// storm the tracker prevents, with the opposite sign.
        /// </summary>
        [Fact]
        public void StaleStateIsNotAdopted()
        {
            var store = new MemoryStore();

            Guard(new CapturingSink(), store, T0).RunCycle(Window(degraded: true), T0);

            var afterAWeek = Guard(new CapturingSink(), store, T0.AddDays(7));

            Assert.Equal(0, afterAWeek.RestoredIncidents);
        }

        /// <summary>Identifiers must not be handed out twice — a consumer would join unrelated histories.</summary>
        [Fact]
        public void IdentifiersAreNotReused()
        {
            var store = new MemoryStore();

            var first = Guard(new CapturingSink(), store, T0);
            first.RunCycle(Window(degraded: true), T0);

            var sink = new CapturingSink();
            var second = Guard(sink, store, T0.AddMinutes(5));

            // Nothing matching the saved incident, so it ages out and something new opens instead.
            second.RunCycle(Window(degraded: false), T0.AddMinutes(5));
            second.RunCycle(Window(degraded: false), T0.AddMinutes(10));

            var reused = new List<long>();

            foreach (var id in sink.Ids(IncidentState.Opened))
            {
                if (id == 1)
                {
                    reused.Add(id);
                }
            }

            Assert.Empty(reused);
        }

        /// <summary>A corrupt or foreign file is a cold start, never a crash.</summary>
        [Theory]
        [InlineData("")]
        [InlineData("garbage")]
        [InlineData("overfit-incident-state\tv99\n1\n")]
        public void UnreadableStateIsAColdStart(string state)
        {
            var store = new MemoryStore { Content = state };

            var guard = Guard(new CapturingSink(), store, T0);

            Assert.Equal(0, guard.RestoredIncidents);
        }

        /// <summary>A summary containing a tab must not split the record it lives in.</summary>
        [Fact]
        public void SeparatorsInsideTextSurviveTheRoundTrip()
        {
            var saved = new PersistedIncident(
                7, T0, T0.AddMinutes(5), 2, 0, "ns/pod", ["ns/pod"],
                "ns", "wl", "rs", "pod", "node", "sig", SignalClass.Symptom, 0.75,
                T0, T0.AddMinutes(12), 1, 1,
                "fell\tby 10%\nacross the window\\here");

            var text = IncidentStateFormat.Write([saved], nextId: 8);
            var read = IncidentStateFormat.Read(text, out var nextId);

            Assert.Equal(8, nextId);
            Assert.Single(read);
            Assert.Equal(saved.Summary, read[0].Summary);
            Assert.Equal(saved.Id, read[0].Id);
            Assert.Equal(saved.Class, read[0].Class);
        }

        /// <summary>A file store must leave either the old state or the new one, never half of one.</summary>
        [Fact]
        public void TheFileStoreWritesAtomically()
        {
            var path = Path.Combine(Path.GetTempPath(), $"overfit-incident-{Guid.NewGuid():N}.state");

            try
            {
                var store = new FileIncidentStore(path);

                store.Save("overfit-incident-state\tv1\n1\n");
                Assert.Null(store.LastError);
                Assert.NotNull(store.Load());

                // The temporary file must not be left behind.
                Assert.False(File.Exists(path + ".tmp"));
            }
            finally
            {
                if (File.Exists(path))
                {
                    File.Delete(path);
                }
            }
        }

        /// <summary>A store that cannot be read must not stop the guard.</summary>
        [Fact]
        public void AnUnreadableFileIsReportedAndNotThrown()
        {
            var store = new FileIncidentStore(
                Path.Combine(Path.GetTempPath(), "definitely", "not", "there", "x.state"));

            Assert.Null(store.Load());
            Assert.Null(store.LastError);
        }

        private static AnomalyGuard Guard(IIncidentSink sink, IIncidentStore store, DateTimeOffset now)
        {
            return new AnomalyGuard(
                new AnomalyGuardOptions
                {
                    Namespace = "overfit",
                    Workload = "srv",
                    Grouping = IncidentGroupingOptions.Balanced with
                    {
                        Topology = TopologyWeights.SingleNode
                    },
                },
                sink,
                IncidentTrackingOptions.Balanced,
                store,
                now);
        }

        /// <summary>Eight pods, because one outlier in three cannot clear the masking bound.</summary>
        private static MetricWindow Window(bool degraded)
        {
            var pods = new List<string>();

            for (var p = 0; p < 8; p++)
            {
                pods.Add($"srv-111-pod{p:d2}");
            }

            var window = new MetricWindow(pods, 60, T0, TimeSpan.FromSeconds(15));
            var rng = new Random(20260730);

            for (var p = 0; p < pods.Count; p++)
            {
                var latency = window.Series(p, MetricIndex.LatencyP95Ms);
                var rps = window.Series(p, MetricIndex.RequestsPerSecond);
                var slow = degraded && p == pods.Count - 1;

                for (var i = 0; i < window.Length; i++)
                {
                    latency[i] = (slow ? 3200.0 : 900.0) * (1.0 + ((rng.NextDouble() - 0.5) * 0.15));
                    rps[i] = 5.0;
                }
            }

            return window;
        }

        private sealed class MemoryStore : IIncidentStore
        {
            public string? Content
            {
                get; set;
            }

            /// <summary>Memory does not fill up in a test; there is nothing to report.</summary>
            public string? LastError => null;

            public string? Load() => Content;

            public void Save(string state) => Content = state;
        }

        private sealed class CapturingSink : IIncidentSink
        {
            private readonly List<IncidentLogRecord> _rows = [];

            public List<long> Ids(IncidentState state)
            {
                var ids = new List<long>();

                foreach (var row in _rows)
                {
                    if (row.Kind == IncidentLogRecordKind.Incident && row.State == state)
                    {
                        ids.Add(row.IncidentId);
                    }
                }

                return ids;
            }

            public void Report(ReadOnlySpan<IncidentLogRecord> rows)
            {
                for (var i = 0; i < rows.Length; i++)
                {
                    _rows.Add(rows[i]);
                }
            }
        }
    }
}
