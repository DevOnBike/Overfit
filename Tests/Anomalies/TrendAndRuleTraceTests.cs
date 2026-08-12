// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Anomalies.Contracts;
using DevOnBike.Overfit.Anomalies.Incidents;
using DevOnBike.Overfit.Anomalies.Incidents.Abstractions;
using DevOnBike.Overfit.Anomalies.Monitoring.Abstractions;
using DevOnBike.Overfit.Statistics;

namespace DevOnBike.Overfit.Tests.Anomalies
{
    /// <summary>
    /// The decision trace on the TREND and RULE families — <c>XC-15</c>.
    ///
    /// <para><b>Why it exists.</b> The peer path could say why it stayed quiet and the rest of the detector
    /// could not, so half the guard's silence was diagnosable and half was not. Measured cost on 2026-08-10:
    /// two hours spent on a channel that looked gated and was merely slower, and a twenty-minute injected
    /// fault that produced zero rows because the custom path emitted nothing.</para>
    ///
    /// <para><b>The row worth having is the warm-up one.</b> A pod inside the grace is skipped with a bare
    /// <c>continue</c>, so before this "tested and healthy" and "never tested" were the same silence — and
    /// during a rollout that is every pod at once. Anything else the trace shows can be inferred from a
    /// finding; that one cannot be inferred from anything.</para>
    ///
    /// <para><b>The rule half is <c>AN-D14</c>, and it is the same shape one level down.</b> The rule trace
    /// covered only the built-in profiles, so a custom binding's rule — same detector, same pipeline —
    /// produced no row, and on a deployment whose rules are all custom the trace was empty every cycle while
    /// reading as "no rule fired". The property these tests pin is what an EMPTY trace means: after the
    /// change, exactly that no rule is configured on any channel.</para>
    /// </summary>
    public sealed class TrendAndRuleTraceTests
    {
        private const string Queue = "myapp_queue_depth";

        private static readonly DateTimeOffset T0 = new(2026, 8, 6, 8, 0, 0, TimeSpan.Zero);

        [Fact]
        public void TheTrendPathEmitsARowPerPodPerChannel()
        {
            var rows = new List<TrendDecisionTrace>();
            var guard = Guard(new NullSink(), custom: false);

            guard.RunCycle(Window(custom: false), T0.AddMinutes(5), null, null, rows.Add, null);

            Assert.NotEmpty(rows);
            Assert.All(rows, r => Assert.False(string.IsNullOrEmpty(r.Pod)));
            Assert.All(rows, r => Assert.False(string.IsNullOrEmpty(r.Signal)));
        }

        /// <summary>The custom half, which is where the peer trace's equivalent gap was found.</summary>
        [Fact]
        public void TheCustomTrendPathEmitsRowsToo()
        {
            var rows = new List<TrendDecisionTrace>();
            var guard = Guard(new NullSink(), custom: true);

            guard.RunCycle(Window(custom: true), T0.AddMinutes(5), null, null, rows.Add, null);

            Assert.Contains(rows, r => string.Equals(r.Signal, Queue, StringComparison.Ordinal));
        }

        /// <summary>
        /// <b>The point of the whole task.</b> A pod skipped for warm-up leaves a row saying so, instead of
        /// leaving nothing and being indistinguishable from a pod that was tested and found healthy.
        /// </summary>
        [Fact]
        public void APodSkippedForWarmUpIsTracedRatherThanSilentlyDropped()
        {
            var rows = new List<TrendDecisionTrace>();
            var guard = Guard(new NullSink(), custom: false, warmUp: TimeSpan.FromHours(6));

            // The window starts minutes after T0, so every pod is inside a six-hour grace.
            guard.RunCycle(Window(custom: false), T0.AddMinutes(5), null, null, rows.Add, null);

            var skipped = rows.Where(r => r.WarmingUp).ToList();

            Assert.NotEmpty(skipped);
            Assert.All(skipped, r => Assert.Equal(DetectionStatus.InsufficientData, r.Status));
            Assert.All(skipped, r => Assert.Contains("warm-up", r.Reason, StringComparison.OrdinalIgnoreCase));
        }

        /// <summary>Without the grace the same pods are judged, so the flag is not simply always set.</summary>
        [Fact]
        public void WithoutTheGraceNoRowIsMarkedWarmingUp()
        {
            var rows = new List<TrendDecisionTrace>();
            var guard = Guard(new NullSink(), custom: false, warmUp: TimeSpan.Zero);

            guard.RunCycle(Window(custom: false), T0.AddMinutes(5), null, null, rows.Add, null);

            Assert.DoesNotContain(rows, r => r.WarmingUp);
        }

        /// <summary>
        /// The trend row carries the detector's own sentence. That sentence is what an operator reads first,
        /// and the numbers beside it are what they check it against.
        /// </summary>
        [Fact]
        public void ATrendRowCarriesTheDetectorsReasonAndTheFloorItWasJudgedAgainst()
        {
            var rows = new List<TrendDecisionTrace>();
            var guard = Guard(new NullSink(), custom: false, warmUp: TimeSpan.Zero);

            guard.RunCycle(Window(custom: false), T0.AddMinutes(5), null, null, rows.Add, null);

            var judged = rows.First(r => !r.WarmingUp);

            Assert.False(string.IsNullOrWhiteSpace(judged.Reason));
            Assert.True(judged.SampleCount > 0);
        }

        /// <summary>
        /// Nothing is emitted when no trace sink is passed — the flag is what turns this on, not the code.
        ///
        /// <para><b>This test could not fail until 2026-08-11.</b> It asserted <c>Assert.NotNull(result)</c>
        /// on <see cref="GuardCycleResult"/>, which is a readonly record STRUCT and therefore never null;
        /// xunit v3's <c>xUnit2002</c> analyzer flagged it during the migration. It now asserts the thing it
        /// was always meant to: that the same cycle which produces rows WITH a sink produces none without
        /// one, so the overload genuinely gates the work rather than merely accepting a null.</para>
        /// </summary>
        [Fact]
        public void NoSinkMeansNoWork()
        {
            var withSink = new List<TrendDecisionTrace>();

            Guard(new NullSink(), custom: false)
                .RunCycle(Window(custom: false), T0.AddMinutes(5), null, null, withSink.Add, null);

            Assert.NotEmpty(withSink);

            var result = Guard(new NullSink(), custom: false)
                .RunCycle(Window(custom: false), T0.AddMinutes(5));

            // The cycle still did its job — it is the tracing that is off, not the detection.
            Assert.True(result.Findings >= 0);
        }

        /// <summary>
        /// <b>The custom half of the rule family, which emitted nothing at all until <c>AN-D14</c>.</b> The
        /// binding is evaluated by the same detector into the same pipeline as a built-in profile, so a
        /// firing rule that leaves no row is a finding an operator cannot explain from the trace they were
        /// given.
        /// </summary>
        [Fact]
        public void ACustomChannelsRuleThatFiresIsTraced()
        {
            var rows = new List<RuleDecisionTrace>();

            Guard(new NullSink(), custom: true, customRule: Fires, builtInRules: false)
                .RunCycle(Window(custom: true), T0.AddMinutes(5), null, null, null, rows.Add);

            var queue = rows.Where(r => string.Equals(r.Signal, Queue, StringComparison.Ordinal)).ToList();

            Assert.NotEmpty(queue);
            Assert.All(queue, r => Assert.Equal(DetectionStatus.Anomalous, r.Status));

            // The gates, separately: a row that only carried a verdict would leave the reader with the same
            // question the trace exists to answer.
            Assert.All(queue, r => Assert.Equal(Fires.Threshold, r.Threshold));
            Assert.All(queue, r => Assert.Equal(Fires.MinBreachFraction, r.MinBreachFraction));
            Assert.All(queue, r => Assert.True(r.UsableSamples >= Fires.MinimumSamples));
        }

        /// <summary>
        /// <b>The fixture that matters most.</b> A rule that did not fire has to leave a row saying so —
        /// otherwise the silence keeps two meanings and nothing about the task has changed for the operator
        /// reading the trace on a healthy cluster, which is almost every cycle.
        /// </summary>
        [Fact]
        public void ACustomChannelsRuleThatDoesNotFireStillEmitsARow()
        {
            var rows = new List<RuleDecisionTrace>();

            Guard(new NullSink(), custom: true, customRule: Silent, builtInRules: false)
                .RunCycle(Window(custom: true), T0.AddMinutes(5), null, null, null, rows.Add);

            var queue = rows.Where(r => string.Equals(r.Signal, Queue, StringComparison.Ordinal)).ToList();

            Assert.Equal(8, queue.Count);
            Assert.All(queue, r => Assert.Equal(DetectionStatus.Healthy, r.Status));
            Assert.All(queue, r => Assert.False(string.IsNullOrWhiteSpace(r.Reason)));
        }

        /// <summary>
        /// The missing-data arm, and the branch the obvious fix misses: a binding whose channel nobody
        /// reported never reaches the evaluation at all, so threading the callback into the rule loop alone
        /// would have left a configured rule silent on exactly the cycles where a broken exporter is the
        /// thing to find.
        ///
        /// <para>Never <c>Healthy</c>: no observation is silence about the channel, not compliance with the
        /// threshold.</para>
        /// </summary>
        [Fact]
        public void ACustomRuleOnAChannelNobodyReportedIsTracedAsInsufficientData()
        {
            var rows = new List<RuleDecisionTrace>();

            Guard(new NullSink(), custom: true, customRule: Fires, builtInRules: false)
                .RunCycle(
                    Window(custom: true, fillCustom: false), T0.AddMinutes(5), null, null, null, rows.Add);

            var queue = rows.Where(r => string.Equals(r.Signal, Queue, StringComparison.Ordinal)).ToList();

            Assert.Equal(8, queue.Count);
            Assert.All(queue, r => Assert.Equal(DetectionStatus.InsufficientData, r.Status));
            Assert.All(queue, r => Assert.Equal(0, r.UsableSamples));

            // The gate is still carried — what the rule WOULD have tested against is the next thing asked.
            Assert.All(queue, r => Assert.Equal(Fires.Threshold, r.Threshold));
        }

        /// <summary>
        /// A binding pointed at a channel this window does not carry at all — a query that returns nothing on
        /// this cluster, which is the shape a misconfigured custom metric arrives in.
        ///
        /// <para><c>MetricWindow.Series(pod, name)</c> throws for an unknown channel, so a version of the fix
        /// that evaluated the rule "anyway" for symmetry with the built-in path would turn a silent channel
        /// into a failed cycle. The row is emitted without reading the series.</para>
        /// </summary>
        [Fact]
        public void ACustomRuleOnAChannelTheWindowDoesNotCarryIsTracedRatherThanThrowing()
        {
            var rows = new List<RuleDecisionTrace>();

            var result = Guard(new NullSink(), custom: true, customRule: Fires, builtInRules: false)
                .RunCycle(Window(custom: false), T0.AddMinutes(5), null, null, null, rows.Add);

            Assert.True(result.BlindMetrics > 0);
            Assert.Equal(8, rows.Count(r => string.Equals(r.Signal, Queue, StringComparison.Ordinal)));
        }

        /// <summary>
        /// <b>The property the task exists to establish.</b> With no rule configured anywhere the trace is
        /// empty — and after the change that is the ONLY thing an empty trace can mean, where before it also
        /// covered "every rule here is a custom one".
        ///
        /// <para>The custom channel is present and reporting; it simply carries no <c>Rule</c>. So this is
        /// the configuration statement, not an absence of data.</para>
        /// </summary>
        [Fact]
        public void AnEmptyRuleTraceMeansNoRuleIsConfiguredAnywhere()
        {
            var rows = new List<RuleDecisionTrace>();

            Guard(new NullSink(), custom: true, customRule: null, builtInRules: false)
                .RunCycle(Window(custom: true), T0.AddMinutes(5), null, null, null, rows.Add);

            Assert.Empty(rows);
        }

        /// <summary>
        /// The trace stays a statement about RULES, not about coverage: an unreported channel that carries no
        /// rule produces nothing here. A blind channel is reported by the guard's own blind-metric count and
        /// the service's per-metric warning, and duplicating it into this trace would put a row in front of
        /// the reader for a family that was never configured.
        /// </summary>
        [Fact]
        public void AnUnreportedChannelWithNoRuleEmitsNoRow()
        {
            var rows = new List<RuleDecisionTrace>();

            var result = Guard(new NullSink(), custom: true, customRule: null, builtInRules: false)
                .RunCycle(
                    Window(custom: true, fillCustom: false), T0.AddMinutes(5), null, null, null, rows.Add);

            Assert.True(result.BlindMetrics > 0);
            Assert.Empty(rows);
        }

        /// <summary>
        /// The control: the built-in family is untouched, and both kinds of row arrive in the same cycle. A
        /// change that MOVED rows from one path to the other would satisfy every test above and fail this one.
        /// </summary>
        [Fact]
        public void TheBuiltInRowsSurviveAlongsideTheCustomOnes()
        {
            var rows = new List<RuleDecisionTrace>();

            Guard(new NullSink(), custom: true, customRule: Fires, builtInRules: true)
                .RunCycle(
                    Window(custom: true, throttle: 0.9), T0.AddMinutes(5), null, null, null, rows.Add);

            var builtIn = rows
                .Where(r => string.Equals(
                    r.Signal, nameof(MetricIndex.CpuThrottleRatio), StringComparison.Ordinal))
                .ToList();

            Assert.Equal(8, builtIn.Count);
            Assert.All(builtIn, r => Assert.Equal(DetectionStatus.Anomalous, r.Status));
            Assert.Equal(8, rows.Count(r => string.Equals(r.Signal, Queue, StringComparison.Ordinal)));
        }

        /// <summary>Every sample of the queue depth is over this line, so the rule fires on every pod.</summary>
        private static SustainedThresholdOptions Fires => new(
            Threshold: 10.0,
            MinBreachFraction: 0.5,
            MinimumSamples: 20);

        /// <summary>Far above anything the window holds, so the rule is evaluated and finds nothing.</summary>
        private static SustainedThresholdOptions Silent => new(
            Threshold: 1000.0,
            MinBreachFraction: 0.5,
            MinimumSamples: 20);

        private static AnomalyGuard Guard(
            IIncidentSink sink,
            bool custom,
            TimeSpan? warmUp = null,
            SustainedThresholdOptions? customRule = null,
            bool builtInRules = true)
        {
            var gaps = new double[(int)MetricIndex.Count];
            gaps[(int)MetricIndex.MemoryWorkingSetBytes] = 5_000_000.0;

            return new AnomalyGuard(
                new AnomalyGuardOptions
                {
                    Namespace = "lab",
                    Workload = "lab-workload",
                    MinAbsoluteGap = gaps,
                    ApplyCalibratedFloors = false,
                    MinimumHistoryDays = 0,
                    DecomposeCommonMode = false,
                    WarmUpGrace = warmUp ?? TimeSpan.Zero,

                    // The grace FAILS CLOSED: no topology, or an unknown creation time, means no pod is
                    // ever spared. So the warm-up row cannot be produced without one, and a test that
                    // omitted it saw zero skips and looked like a missing trace.
                    PodTopology = warmUp is { } g && g > TimeSpan.Zero ? new FreshTopology(T0) : null,

                    // Off by default for the rule tests: with the three shipped profiles left on, every
                    // built-in metric contributes rows and "the trace is empty" could not be asserted.
                    Rules = builtInRules ? AnomalyGuardOptions.DefaultRules : [],
                    CustomMetrics = custom
                        ? [new CustomMetricBinding(
                            Queue, Queue, MetricSourceKind.Gauge, PeerSignalKind.LoadIndependent,
                            Class: SignalClass.Resource, MinAbsoluteGap: 5.0, Rule: customRule)]
                        : [],
                    Grouping = IncidentGroupingOptions.Balanced with
                    {
                        Topology = TopologyWeights.SingleNode
                    },
                },
                sink,
                IncidentTrackingOptions.Balanced);
        }

        private static MetricWindow Window(bool custom, bool fillCustom = true, double throttle = double.NaN)
        {
            var names = new List<string>(8);

            for (var p = 0; p < 8; p++)
            {
                names.Add($"lab-workload-7765564ff6-pod{p:d2}");
            }

            var window = custom
                ? new MetricWindow(names, 60, T0, TimeSpan.FromSeconds(15), [Queue])
                : new MetricWindow(names, 60, T0, TimeSpan.FromSeconds(15));
            var rng = new Random(20260806);

            for (var pod = 0; pod < window.Pods.Count; pod++)
            {
                if (double.IsFinite(throttle))
                {
                    window.Series(pod, MetricIndex.CpuThrottleRatio).Fill(throttle);
                }

                // Declared and left at NaN: what a cycle whose query returned nothing looks like, and the
                // reason PodsReporting is 0 without the channel being unknown to the window.
                if (custom && !fillCustom)
                {
                    continue;
                }

                var series = custom
                    ? window.Series(pod, Queue)
                    : window.Series(pod, MetricIndex.MemoryWorkingSetBytes);
                var baseline = custom ? 20.0 : 44_000_000.0;
                var scale = custom ? 1.0 : 400_000.0;

                for (var i = 0; i < window.Length; i++)
                {
                    series[i] = baseline + ((rng.NextDouble() - 0.5) * scale);
                }
            }

            return window;
        }

        /// <summary>Every pod created at <c>T0</c>, so a grace wider than the window covers all of them.</summary>
        private sealed class FreshTopology : IPodTopology
        {
            private readonly DateTimeOffset _created;

            public FreshTopology(DateTimeOffset created) => _created = created;

            public bool TryResolve(string pod, out PodPlacement placement)
            {
                placement = new PodPlacement("lab-workload", "rs-1", "node-0", string.Empty, _created);

                return true;
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
