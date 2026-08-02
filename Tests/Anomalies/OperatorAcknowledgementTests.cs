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
    /// One operator, one button, and the two effects it has to have.
    ///
    /// <para>The suppression is the relief somebody needs at three in the morning; the label is what makes
    /// the threshold right by the end of the week. Either one alone is a different, worse product — mutes
    /// that accumulate for a year, or a button that appears to do nothing — so these tests check both halves
    /// and, more importantly, the two things the button must <b>refuse</b> to do.</para>
    /// </summary>
    public sealed class OperatorAcknowledgementTests
    {
        private static readonly DateTimeOffset T0 = new(2026, 8, 2, 12, 0, 0, TimeSpan.Zero);

        [Fact]
        public void AcknowledgingAsNoiseMutesTheSignalOnThatPod()
        {
            var sink = new CapturingSink();
            var guard = Guard(sink);

            guard.RunCycle(OneReplicaHigh(), T0.AddMinutes(20));

            var id = FirstIncidentId(sink);

            Assert.Contains(sink.Messages, m => m.Contains("peers", StringComparison.Ordinal));

            var echo = guard.Acknowledge(
                id, OperatorLabelKind.Noise, TimeSpan.FromHours(1), "known sawtooth", T0.AddMinutes(21));

            Assert.Contains("muted", echo, StringComparison.Ordinal);
            Assert.Equal(1, guard.Suppressions.ActiveCount(T0.AddMinutes(21)));

            sink.Messages.Clear();
            guard.RunCycle(OneReplicaHigh(), T0.AddMinutes(25));

            Assert.DoesNotContain(sink.Messages, m => m.Contains("peers", StringComparison.Ordinal));
        }

        /// <summary>
        /// The label carries the finding's size in the signal's own units — which is the only reason a
        /// <c>--real</c> acknowledgement can constrain anything later.
        /// </summary>
        [Fact]
        public void TheLabelCarriesTheFindingsOwnMagnitude()
        {
            var sink = new CapturingSink();
            var guard = Guard(sink);

            guard.RunCycle(OneReplicaHigh(), T0.AddMinutes(20));
            guard.Acknowledge(
                FirstIncidentId(sink), OperatorLabelKind.Real, null, "this was the leak", T0.AddMinutes(21));

            Assert.Equal(1, guard.Labels.Count);

            var label = guard.Labels.Labels[0];

            Assert.Equal(OperatorLabelKind.Real, label.Kind);
            Assert.True(label.ConstrainsFloors, $"magnitude {label.Magnitude} cannot constrain a floor");

            // The high replica sits at 90 MB against 40 MB peers, so the gap is tens of megabytes.
            Assert.InRange(label.Magnitude, 10e6, 100e6);
        }

        /// <summary>
        /// Confirming a finding and silencing it in the same breath is a contradiction. Accepting it quietly
        /// would leave an operator believing they had escalated something they had in fact hidden.
        /// </summary>
        [Fact]
        public void ConfirmingSomethingAsRealNeverOpensASuppression()
        {
            var sink = new CapturingSink();
            var guard = Guard(sink);

            guard.RunCycle(OneReplicaHigh(), T0.AddMinutes(20));

            var echo = guard.Acknowledge(
                FirstIncidentId(sink), OperatorLabelKind.Real, TimeSpan.FromDays(7), "real", T0.AddMinutes(21));

            Assert.Contains("no suppression", echo, StringComparison.Ordinal);
            Assert.Equal(0, guard.Suppressions.ActiveCount(T0.AddMinutes(21)));
        }

        [Fact]
        public void TheMuteExpiresAndTheSignalComesBack()
        {
            var sink = new CapturingSink();
            var guard = Guard(sink);

            guard.RunCycle(OneReplicaHigh(), T0.AddMinutes(20));
            guard.Acknowledge(
                FirstIncidentId(sink), OperatorLabelKind.Noise, TimeSpan.FromMinutes(10), "for now",
                T0.AddMinutes(21));

            sink.Messages.Clear();
            guard.RunCycle(OneReplicaHigh(), T0.AddMinutes(25));

            Assert.DoesNotContain(sink.Messages, m => m.Contains("peers", StringComparison.Ordinal));

            sink.Messages.Clear();
            guard.RunCycle(OneReplicaHigh(), T0.AddMinutes(40));

            Assert.Contains(sink.Messages, m => m.Contains("peers", StringComparison.Ordinal));
            Assert.Equal(0, guard.Suppressions.ActiveCount(T0.AddMinutes(40)));
        }

        /// <summary>
        /// An identifier this guard has never reported is refused. The alternative is a label about a
        /// magnitude nobody can vouch for, which would then cap a real threshold.
        /// </summary>
        [Fact]
        public void AnUnknownIncidentIsRefused()
        {
            var guard = Guard(new CapturingSink());

            var thrown = Assert.Throws<ArgumentException>(() => guard.Acknowledge(
                9999, OperatorLabelKind.Noise, TimeSpan.FromHours(1), "who?", T0));

            Assert.Contains("not one this guard has reported", thrown.Message, StringComparison.Ordinal);
        }

        private static long FirstIncidentId(CapturingSink sink)
        {
            Assert.NotEmpty(sink.Ids);

            return sink.Ids[0];
        }

        private static AnomalyGuard Guard(CapturingSink sink)
        {
            return new AnomalyGuard(
                new AnomalyGuardOptions
                {
                    Namespace = "lab",
                    Workload = "svc",
                    PodTopology = new FakeTopology(),
                    MinimumHistoryDays = 0,
                    DecomposeCommonMode = false,
                    MinAbsoluteGap = Floors(),
                    MinAbsoluteTrendChange = Floors(),
                    Grouping = IncidentGroupingOptions.Balanced with { Topology = TopologyWeights.SingleNode },
                },
                sink,
                IncidentTrackingOptions.Balanced);
        }

        /// <summary>Floors sized to this fixture rather than to the shipped 256 MiB default.</summary>
        private static double[] Floors()
        {
            var floors = new double[(int)MetricIndex.Count];
            floors[(int)MetricIndex.MemoryWorkingSetBytes] = 1.089e6;

            return floors;
        }

        /// <summary>Flat memory, one replica far above the rest — a peer question with an absolute gap.</summary>
        private static MetricWindow OneReplicaHigh()
        {
            var names = new List<string> { "pod-0", "pod-1", "pod-2", "pod-3" };
            var window = new MetricWindow(names, 80, T0, TimeSpan.FromSeconds(15));
            var rng = new Random(20260802);

            for (var pod = 0; pod < names.Count; pod++)
            {
                var memory = window.Series(pod, MetricIndex.MemoryWorkingSetBytes);
                var cpu = window.Series(pod, MetricIndex.CpuUsageRatio);
                var level = pod == 0 ? 90e6 : 40e6;

                for (var i = 0; i < window.Length; i++)
                {
                    memory[i] = level * (1.0 + ((rng.NextDouble() - 0.5) * 0.01));
                    cpu[i] = 0.2;
                }
            }

            return window;
        }

        private sealed class FakeTopology : IPodTopology
        {
            public bool TryResolve(string pod, out PodPlacement placement)
            {
                placement = new PodPlacement("svc", "rs-1", "node-0", string.Empty, T0.AddDays(-1));

                return true;
            }
        }

        private sealed class CapturingSink : IIncidentSink
        {
            public List<string> Messages { get; } = [];

            public List<long> Ids { get; } = [];

            public void Report(ReadOnlySpan<IncidentLogRecord> rows)
            {
                for (var i = 0; i < rows.Length; i++)
                {
                    Messages.Add(rows[i].Message);
                    Ids.Add(rows[i].IncidentId);
                }
            }
        }
    }
}
