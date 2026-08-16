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
    /// The novelty gate as the guard actually runs it — <b>on both peer call sites</b>.
    ///
    /// <para><b>Every assertion here is an A/B against the same window sequence with the gate off.</b> A bare
    /// "fewer findings with the gate on" count would also be satisfied by the calibrated floor climbing above
    /// the gap, by the incident tracker's grace period, or by a suppression — three mechanisms that already
    /// quieten this path. The control arm is what makes the number attributable to the gate.</para>
    ///
    /// <para><b><c>RunCustomPeer</c> is tested separately and deliberately.</b> A gate present only on
    /// <c>RunPeer</c> leaves every channel a customer adds re-reporting a standing outlier for ever — on the
    /// deployed lab configuration that is five channels, and it is the half of the system a client is most
    /// likely to extend.</para>
    /// </summary>
    public sealed class PeerNoveltyGateTests
    {
        private const string Queue = "myapp_queue_depth";

        private static readonly DateTimeOffset T0 = new(2026, 8, 6, 8, 0, 0, TimeSpan.Zero);

        private const int Cycles = 24;

        private static PeerNoveltyOptions Novelty => PeerNoveltyOptions.Daily with
        {
            MinimumCycles = 12,
            RetainedCyclesPerSeries = 24,
        };

        /// <summary>
        /// A fixed offset on a built-in channel stops being reported; without the gate it never does.
        ///
        /// <para><b>Twelve, exactly, rather than "fewer".</b> Twelve is
        /// <see cref="PeerNoveltyOptions.MinimumCycles"/> — the warm-up during which the gate correctly
        /// forwards everything — and the thirteenth cycle is the first that can be judged. Pinning the number
        /// is what separates this from a gate that suppresses from cycle one, which a loose bound would also
        /// accept and which would be the defect this mechanism replaces wearing the fix's clothes.</para>
        /// </summary>
        [Fact]
        public void AStandingOffsetOnABuiltInChannelStopsRepeating()
        {
            var withGate = RunBuiltIn(Novelty, growthPerCycle: 0.0);
            var without = RunBuiltIn(null, growthPerCycle: 0.0);

            Assert.Equal(Cycles, without);
            Assert.Equal(Novelty.MinimumCycles, withGate);
        }

        /// <summary>
        /// A gap that keeps growing is reported every cycle, gate or no gate. Without this arm the test above
        /// is satisfied by a gate that suppresses everything.
        /// </summary>
        [Fact]
        public void AGrowingOffsetOnABuiltInChannelKeepsReporting()
        {
            var withGate = RunBuiltIn(Novelty, growthPerCycle: 900_000.0);
            var without = RunBuiltIn(null, growthPerCycle: 900_000.0);

            Assert.Equal(without, withGate);
            Assert.Equal(Cycles, withGate);
        }

        /// <summary>The same, on the call site a customer's own metric goes through, and the same count.</summary>
        [Fact]
        public void AStandingOffsetOnACustomChannelStopsRepeating()
        {
            var withGate = RunCustom(Novelty, growthPerCycle: 0.0);
            var without = RunCustom(null, growthPerCycle: 0.0);

            Assert.Equal(Cycles, without);
            Assert.Equal(Novelty.MinimumCycles, withGate);
        }

        /// <summary>And its loud arm.</summary>
        [Fact]
        public void AGrowingOffsetOnACustomChannelKeepsReporting()
        {
            var withGate = RunCustom(Novelty, growthPerCycle: 900.0);
            var without = RunCustom(null, growthPerCycle: 900.0);

            Assert.Equal(without, withGate);
            Assert.Equal(Cycles, withGate);
        }

        /// <summary>
        /// A held finding is counted, so an operator can tell the gate working from a detector that stopped.
        /// </summary>
        [Fact]
        public void HeldFindingsAreExported()
        {
            var sink = new CapturingSink();
            var guard = Guard(sink, Novelty, custom: false);

            for (var c = 0; c < Cycles; c++)
            {
                guard.RunCycle(BuiltInWindow(c, 0.0), T0.AddMinutes(5 * c));
            }

            Assert.Contains(
                "overfit_guard_peer_findings_standing_total",
                guard.Telemetry.ToPrometheusText(),
                StringComparison.Ordinal);

            var held = ReadSeries(
                guard.Telemetry.ToPrometheusText(), "overfit_guard_peer_findings_standing_total");

            Assert.True(held > 0.0, $"expected held findings, saw {held}");
        }

        /// <summary>
        /// A surviving standing finding carries the classification and half the severity, so a consumer can
        /// route it as an update rather than a page.
        /// </summary>
        [Fact]
        public void AReassertedFindingIsTaggedAndDeprioritised()
        {
            var sink = new CapturingSink();

            // A ten-minute cadence, so a reassertion lands inside the run rather than a day later.
            var guard = Guard(
                sink,
                Novelty with
                {
                    StandingReassertionInterval = TimeSpan.FromMinutes(10)
                },
                custom: false);

            for (var c = 0; c < Cycles; c++)
            {
                guard.RunCycle(BuiltInWindow(c, 0.0), T0.AddMinutes(5 * c));
            }

            var standing = new List<IncidentLogRecord>();

            for (var i = 0; i < sink.Rows.Count; i++)
            {
                if (IsPeerFinding(sink.Rows[i]) && sink.Rows[i].Novelty == NoveltyKind.Standing)
                {
                    standing.Add(sink.Rows[i]);
                }
            }

            Assert.NotEmpty(standing);

            for (var i = 0; i < standing.Count; i++)
            {
                Assert.InRange(standing[i].Severity, 0.0, 0.5);
            }
        }

        /// <summary>
        /// Enabling the gate without its change floor is refused at construction rather than run with the
        /// floor off. See <c>AnomalyGuardOptions.MinAbsoluteGapChange</c> — the value is unmeasured, and a
        /// suppression gate running against a threshold nobody chose is worse than a noisy one.
        /// </summary>
        [Fact]
        public void TheGateWithoutItsChangeFloorIsRefused()
        {
            var error = Assert.Throws<ArgumentException>(() => new AnomalyGuard(
                new AnomalyGuardOptions
                {
                    Namespace = "lab",
                    Workload = "lab-workload",
                    PeerNovelty = Novelty,
                },
                new CapturingSink(),
                IncidentTrackingOptions.Balanced));

            Assert.Contains(nameof(AnomalyGuardOptions.MinAbsoluteGapChange), error.Message,
                StringComparison.Ordinal);
        }

        /// <summary>The same refusal for a custom channel, which has no per-metric table to fall back on.</summary>
        [Fact]
        public void ACustomChannelWithoutItsChangeFloorIsRefused()
        {
            var error = Assert.Throws<ArgumentException>(() => new AnomalyGuard(
                new AnomalyGuardOptions
                {
                    Namespace = "lab",
                    Workload = "lab-workload",
                    PeerNovelty = Novelty,
                    MinAbsoluteGapChange = ChangeFloors(),
                    CustomMetrics = [Binding(withChangeFloor: false)],
                },
                new CapturingSink(),
                IncidentTrackingOptions.Balanced));

            Assert.Contains(Queue, error.Message, StringComparison.Ordinal);
        }

        private static int RunBuiltIn(PeerNoveltyOptions? novelty, double growthPerCycle)
        {
            var sink = new CapturingSink();
            var guard = Guard(sink, novelty, custom: false);

            for (var c = 0; c < Cycles; c++)
            {
                guard.RunCycle(BuiltInWindow(c, growthPerCycle), T0.AddMinutes(5 * c));
            }

            return CountPeerFindings(sink, nameof(MetricIndex.MemoryWorkingSetBytes));
        }

        private static int RunCustom(PeerNoveltyOptions? novelty, double growthPerCycle)
        {
            var sink = new CapturingSink();
            var guard = Guard(sink, novelty, custom: true);

            for (var c = 0; c < Cycles; c++)
            {
                guard.RunCycle(CustomWindow(c, growthPerCycle), T0.AddMinutes(5 * c));
            }

            return CountPeerFindings(sink, Queue);
        }

        private static int CountPeerFindings(CapturingSink sink, string signal)
        {
            var count = 0;

            for (var i = 0; i < sink.Rows.Count; i++)
            {
                if (IsPeerFinding(sink.Rows[i])
                    && string.Equals(sink.Rows[i].Signal, signal, StringComparison.Ordinal))
                {
                    count++;
                }
            }

            return count;
        }

        /// <summary>
        /// A finding row from the peer family. Matched on the detector's own wording rather than on a kind
        /// flag, because the trend family reports the same signal on the same pod and would otherwise be
        /// counted as if the gate had let it through.
        /// </summary>
        private static bool IsPeerFinding(in IncidentLogRecord row)
        {
            return row.Kind == IncidentLogRecordKind.Finding
                   && row.Message.Contains("peers", StringComparison.Ordinal);
        }

        private static double ReadSeries(string exposition, string name)
        {
            var lines = exposition.Split('\n');

            for (var i = 0; i < lines.Length; i++)
            {
                if (!lines[i].StartsWith(name + " ", StringComparison.Ordinal))
                {
                    continue;
                }

                return double.Parse(
                    lines[i][(name.Length + 1)..],
                    System.Globalization.CultureInfo.InvariantCulture);
            }

            return double.NaN;
        }

        private static AnomalyGuard Guard(IIncidentSink sink, PeerNoveltyOptions? novelty, bool custom)
        {
            var gaps = new double[(int)MetricIndex.Count];
            gaps[(int)MetricIndex.MemoryWorkingSetBytes] = 5_000_000.0;

            return new AnomalyGuard(
                new AnomalyGuardOptions
                {
                    Namespace = "lab",
                    Workload = "lab-workload",
                    MinAbsoluteGap = gaps,

                    // Off, so the only thing that can quieten the peer family across these cycles is the gate
                    // under test. Left on, the calibrator's own floor climbs above a persistent gap and the
                    // findings stop for a reason this test is not about.
                    ApplyCalibratedFloors = false,
                    MinimumHistoryDays = 0,
                    DecomposeCommonMode = false,
                    PeerNovelty = novelty,
                    MinAbsoluteGapChange = novelty is null ? null : ChangeFloors(),
                    CustomMetrics = custom ? [Binding(withChangeFloor: true)] : [],
                    Grouping = IncidentGroupingOptions.Balanced with
                    {
                        Topology = TopologyWeights.SingleNode
                    },
                },
                sink,
                IncidentTrackingOptions.Balanced);
        }

        private static double[] ChangeFloors()
        {
            var floors = new double[(int)MetricIndex.Count];

            floors[(int)MetricIndex.MemoryWorkingSetBytes] = 2_000_000.0;

            return floors;
        }

        private static CustomMetricBinding Binding(bool withChangeFloor)
        {
            return new CustomMetricBinding(
                Name: Queue,
                Source: "myapp_queue_depth",
                Kind: MetricSourceKind.Gauge,
                SignalKind: PeerSignalKind.LoadIndependent,
                Class: SignalClass.Resource,
                MinAbsoluteGap: 5.0,
                MinAbsoluteGapChange: withChangeFloor ? 2.0 : 0.0);
        }

        /// <summary>Eight pods, one of which sits 10 MB above the rest for the whole run.</summary>
        private static MetricWindow BuiltInWindow(int cycle, double growthPerCycle)
        {
            var window = NewWindow(cycle, custom: false);
            var rng = new Random(20260806 + cycle);

            for (var pod = 0; pod < window.Pods.Count; pod++)
            {
                var memory = window.Series(pod, MetricIndex.MemoryWorkingSetBytes);
                var offset = pod == 0 ? 10_000_000.0 + (growthPerCycle * cycle) : 0.0;

                for (var i = 0; i < window.Length; i++)
                {
                    memory[i] = 44_000_000.0 + offset + ((rng.NextDouble() - 0.5) * 400_000.0);
                }
            }

            return window;
        }

        /// <summary>The same shape on a channel the enum knows nothing about.</summary>
        private static MetricWindow CustomWindow(int cycle, double growthPerCycle)
        {
            var window = NewWindow(cycle, custom: true);
            var rng = new Random(20260806 + cycle);

            for (var pod = 0; pod < window.Pods.Count; pod++)
            {
                var depth = window.Series(pod, Queue);
                var offset = pod == 0 ? 40.0 + (growthPerCycle * cycle) : 0.0;

                for (var i = 0; i < window.Length; i++)
                {
                    depth[i] = 20.0 + offset + ((rng.NextDouble() - 0.5) * 1.0);
                }
            }

            return window;
        }

        private static MetricWindow NewWindow(int cycle, bool custom)
        {
            var names = new List<string>(8);

            for (var p = 0; p < 8; p++)
            {
                names.Add($"lab-workload-7765564ff6-pod{p:d2}");
            }

            var start = T0.AddMinutes(5 * cycle);

            return custom
                ? new MetricWindow(names, 60, start, TimeSpan.FromSeconds(15), [Queue])
                : new MetricWindow(names, 60, start, TimeSpan.FromSeconds(15));
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
