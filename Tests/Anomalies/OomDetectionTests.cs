// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Anomalies.Incidents;
using DevOnBike.Overfit.Anomalies.Contracts;
using DevOnBike.Overfit.Anomalies.Rules;
using DevOnBike.Overfit.Statistics;

namespace DevOnBike.Overfit.Tests.Anomalies
{
    /// <summary>
    /// Does the guard catch an OOM kill?
    ///
    /// <para>The fault-injection fixture in <c>k8s/overfit/fault-cpu-throttle.yaml</c> justifies choosing CPU
    /// throttling over a memory limit by asserting that "OOMKill shows up as a restart, and restarts are a
    /// load-independent signal that <b>any detector catches trivially</b>". That was an assumption and was never
    /// measured. These tests measure it.</para>
    ///
    /// <para>The shape is the one Prometheus actually produces: the counter increments once, and
    /// <c>rate(container_oom_events_total[2m])</c> at a 15-second step therefore reads
    /// <c>1/120 = 0.00833</c> for the eight samples whose window contains the event and zero everywhere else.
    /// A brief non-zero blip in an otherwise flat series — not a step, not a trend.</para>
    /// </summary>
    public sealed class OomDetectionTests
    {
        private const int WindowSamples = 80;          // 20 min at 15 s
        private const int EventSamples = 8;            // a 2-minute rate window at a 15-second step
        private const double RatePerSecond = 1.0 / 120.0;

        private static readonly DateTimeOffset From = new(2026, 7, 29, 12, 0, 0, TimeSpan.Zero);
        private static readonly DateTimeOffset To = From.AddMinutes(20);

        [Fact]
        public void PeerComparison_MissesASingleOomKill()
        {
            // Measured, not assumed. Cliff's delta between "eight non-zero samples among eighty" and "all zero"
            // is about the fraction of non-zero samples — roughly 0.10 — which is under the 0.33 materiality
            // gate. The peer family is structurally blind to one OOM kill, and only sees a sustained series of
            // them. That is why the hard rule exists and why leaving it unwired was a real gap.
            var peers = new List<PeerSeries>
            {
                new("pod-a", Flat()),
                new("pod-b", Flat()),
                new("pod-c", Flat()),
                new("pod-oom", SingleOom())
            };

            var findings = new PeerOutlierFinding[peers.Count];
            var result = new PeerGroupOutlierDetector().Detect(
                peers, PeerSignalKind.LoadIndependent, PeerOutlierOptions.Balanced, findings);

            Assert.NotEqual(DetectionStatus.Anomalous, result.Status);
            Assert.Equal(0, result.HighCount);
        }

        [Fact]
        public void TheHardRule_CatchesIt()
        {
            // ForRareEvent asks for any non-zero reading across a twentieth of the window. Eight samples of
            // eighty is 10%, so a single kill clears it — with no comparison and no history.
            var result = new SustainedThresholdRule().Evaluate(
                SingleOom(), SustainedThresholdOptions.ForRareEvent);

            Assert.Equal(DetectionStatus.Anomalous, result.Status);
            Assert.Equal(EventSamples, result.BreachedSamples);
            Assert.Equal(EventSamples / (double)WindowSamples, result.BreachFraction, 6);
        }

        [Fact]
        public void TheHardRule_StaysSilentOnAHealthyPod()
        {
            var result = new SustainedThresholdRule().Evaluate(
                Flat(), SustainedThresholdOptions.ForRareEvent);

            Assert.Equal(DetectionStatus.Healthy, result.Status);
            Assert.Equal(0, result.BreachedSamples);
        }

        [Fact]
        public void TheRuleVerdictBecomesAnIncidentNamingThePod()
        {
            var pipeline = new IncidentPipeline();
            var subject = new IncidentSubject("overfit", "overfit-server", "overfit-server-abc", "pod-oom", "node-1");

            var result = new SustainedThresholdRule().Evaluate(
                SingleOom(), SustainedThresholdOptions.ForRareEvent);

            Assert.True(pipeline.ObserveRule(subject, "container_oom_events_total", result, From, To));

            var incident = Assert.Single(pipeline.Group(IncidentGroupingOptions.Balanced));

            Assert.Equal("pod-oom", incident.Primary.Subject.Pod);

            // The catalog classifies "oom" as infrastructure, so it leads any symptom grouped with it.
            Assert.Equal(SignalClass.Infrastructure, incident.Primary.Class);
        }

        [Fact]
        public void ARestartIsCaughtByTheSameProfile()
        {
            // The strongest OOM evidence is the restart it causes. kube_pod_container_status_restarts_total is a
            // monotonic counter, so the feature carries its increase over the window: zero on a healthy pod, one
            // step on a restarted one.
            var restarted = new double[WindowSamples];
            for (var i = WindowSamples / 2; i < WindowSamples; i++)
            {
                restarted[i] = 1.0;
            }

            var result = new SustainedThresholdRule().Evaluate(
                restarted, SustainedThresholdOptions.ForRareEvent);

            Assert.Equal(DetectionStatus.Anomalous, result.Status);
        }

        /// <summary>A healthy pod: the counter never moves, so the rate is flat zero.</summary>
        private static double[] Flat() => new double[WindowSamples];

        /// <summary>One kill: a brief non-zero rate as the event passes through the 2-minute window.</summary>
        private static double[] SingleOom()
        {
            var values = new double[WindowSamples];

            for (var i = 40; i < 40 + EventSamples; i++)
            {
                values[i] = RatePerSecond;
            }

            return values;
        }
    }
}
