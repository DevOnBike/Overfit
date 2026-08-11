// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Anomalies.Contracts;
using DevOnBike.Overfit.Anomalies.Incidents;
using DevOnBike.Overfit.Anomalies.Incidents.Abstractions;
using DevOnBike.Overfit.Anomalies.Monitoring;
using DevOnBike.Overfit.Statistics;

namespace DevOnBike.Overfit.Tests.Anomalies
{
    /// <summary>
    /// <c>minStepChange</c> on a CUSTOM channel — the symmetric half of <c>AN-D4b</c>, and the asymmetry
    /// that was the only real argument for promoting <c>CpuPressure</c> to a <c>MetricIndex</c> member
    /// (<c>PS-1</c>).
    ///
    /// <para>Until this existed a custom channel could configure a peer floor, a trend floor and a gap-change
    /// floor, but its step gate read the TREND floor with no way to override it — the same defect the
    /// built-in channels had, measured at 40% of the level on `MemoryWorkingSetBytes` and 123% at the low
    /// decile of `GcGen2HeapBytes`.</para>
    ///
    /// <para>The precedence is the compatibility contract and is what most of these pin: step floor, then
    /// trend floor, then the calibrator. Falling back to the calibrator instead of the trend floor would
    /// move every deployed custom channel on the day it upgraded.</para>
    /// </summary>
    public sealed class CustomChannelStepFloorTests
    {
        [Fact]
        public void TheConfigFieldIsRead()
        {
            var bindings = Read(minTrendChange: "3.0MB", minStepChange: "0.9MB");
            var binding = Assert.Single(bindings);

            Assert.Equal(3.0e6, binding.MinAbsoluteTrendChange);
            Assert.Equal(0.9e6, binding.MinAbsoluteStepChange);
        }

        /// <summary><b>The compatibility contract.</b> Omitted means "fall back", not "no floor".</summary>
        [Fact]
        public void AnOmittedFieldLeavesTheBindingOnItsTrendFloor()
        {
            var binding = Assert.Single(Read(minTrendChange: "3.0MB", minStepChange: null));

            Assert.Equal(3.0e6, binding.MinAbsoluteTrendChange);
            Assert.Equal(0.0, binding.MinAbsoluteStepChange);
        }

        /// <summary>
        /// The guard's own precedence, exercised through a real cycle rather than asserted about a private
        /// method: a channel whose step floor is far above anything the window can produce must report no
        /// step, and the same channel with a tiny one must.
        /// </summary>
        [Theory]
        [InlineData(1e9, false)]     // step floor unreachable -> silent
        [InlineData(1e-9, true)]     // step floor negligible  -> reported
        public void TheBindingsStepFloorGovernsTheStepGate(double stepFloor, bool expectFinding)
        {
            var rows = RunCycle(trendFloor: 0.0, stepFloor: stepFloor);

            Assert.Equal(expectFinding, rows.Exists(r => r.Contains("step, not a drift", StringComparison.Ordinal)));
        }

        /// <summary>
        /// With no step floor the TREND floor governs — the behaviour every deployed custom channel had
        /// before this field existed, and the reason adding it moves nothing on its own.
        /// </summary>
        [Fact]
        public void WithNoStepFloorTheTrendFloorStillGoverns()
        {
            var blocked = RunCycle(trendFloor: 1e9, stepFloor: 0.0);
            var open = RunCycle(trendFloor: 1e-9, stepFloor: 0.0);

            Assert.DoesNotContain(blocked, r => r.Contains("step, not a drift", StringComparison.Ordinal));
            Assert.Contains(open, r => r.Contains("step, not a drift", StringComparison.Ordinal));
        }

        /// <summary>And the step floor beats the trend floor when both are set.</summary>
        [Fact]
        public void TheStepFloorWinsOverTheTrendFloor()
        {
            var rows = RunCycle(trendFloor: 1e9, stepFloor: 1e-9);

            Assert.Contains(rows, r => r.Contains("step, not a drift", StringComparison.Ordinal));
        }

        private const string Channel = "myapp_queue_depth";

        private static IReadOnlyList<CustomMetricBinding> Read(string minTrendChange, string? minStepChange)
        {
            var file = new AnomalyGuardConfigFile
            {
                Prometheus = "http://localhost:9090",
                CustomMetrics =
                {
                    [Channel] = new AnomalyGuardConfigFile.CustomEntry
                    {
                        Source = Channel,
                        Kind = nameof(MetricSourceKind.Gauge),
                        MinTrendChange = minTrendChange,
                        MinStepChange = minStepChange ?? string.Empty,
                    },
                },
            };

            // Custom bindings are built inside ReadMap, not by a reader of their own.
            var map = AnomalyGuardConfigReader.ReadMap(file, out var problems);

            Assert.Empty(problems);

            return map.Custom;
        }

        /// <summary>Runs one real cycle over a fleet that steps mid-window, and returns the reported messages.</summary>
        private static List<string> RunCycle(double trendFloor, double stepFloor)
        {
            var sink = new CapturingSink();

            var guard = new AnomalyGuard(
                new AnomalyGuardOptions
                {
                    Namespace = "lab",
                    Workload = "lab-workload",
                    DecomposeCommonMode = true,
                    MinimumHistoryDays = 0,
                    ApplyCalibratedFloors = false,
                    WarmUpGrace = TimeSpan.Zero,
                    CustomMetrics =
                    [
                        new CustomMetricBinding(
                            Channel, Channel, MetricSourceKind.Gauge, PeerSignalKind.LoadIndependent,
                            Class: SignalClass.Resource,
                            MinAbsoluteGap: 1e9,
                            MinAbsoluteTrendChange: trendFloor,
                            MinAbsoluteStepChange: stepFloor),
                    ],
                    Grouping = IncidentGroupingOptions.Balanced with
                    {
                        Topology = TopologyWeights.SingleNode
                    },
                },
                sink,
                IncidentTrackingOptions.Balanced);

            guard.RunCycle(SteppedWindow(), Start.AddMinutes(20));

            return sink.Messages;
        }

        private static readonly DateTimeOffset Start = new(2026, 8, 11, 9, 0, 0, TimeSpan.Zero);

        /// <summary>Twelve pods, all stepping from 20 to 60 half-way through — a fleet-wide step.</summary>
        private static MetricWindow SteppedWindow()
        {
            var names = new List<string>(12);

            for (var p = 0; p < 12; p++)
            {
                names.Add($"lab-workload-7765564ff6-pod{p:d2}");
            }

            var window = new MetricWindow(names, 80, Start, TimeSpan.FromSeconds(15), [Channel]);
            var rng = new Random(20260811);

            for (var pod = 0; pod < names.Count; pod++)
            {
                var series = window.Series(pod, Channel);

                for (var i = 0; i < window.Length; i++)
                {
                    series[i] = (i < window.Length / 2 ? 20.0 : 60.0) + ((rng.NextDouble() - 0.5) * 0.5);
                }
            }

            return window;
        }

        private sealed class CapturingSink : IIncidentSink
        {
            public List<string> Messages { get; } = [];

            public void Report(ReadOnlySpan<IncidentLogRecord> rows)
            {
                foreach (var row in rows)
                {
                    Messages.Add(row.Message);
                }
            }
        }
    }
}
