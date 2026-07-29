// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System;
using DevOnBike.Overfit.Anomalies.Incidents;
using DevOnBike.Overfit.Anomalies.Incidents.Contracts;
using DevOnBike.Overfit.Anomalies.Rules;
using DevOnBike.Overfit.Anomalies.Rules.Contracts;
using DevOnBike.Overfit.Statistics;

namespace DevOnBike.Overfit.Tests.Anomalies.Rules
{
    /// <summary>
    /// The hard rule for CPU throttling, pinned against the distribution the cluster lab actually produced.
    /// </summary>
    public sealed class SustainedThresholdRuleTests
    {
        private static readonly DateTimeOffset WindowStart = new(2026, 7, 29, 12, 0, 0, TimeSpan.Zero);
        private static readonly DateTimeOffset WindowEnd = WindowStart.AddMinutes(12);

        private static readonly SustainedThresholdRule Rule = new();

        [Fact]
        public void TheLoadedLabWindow_Fires()
        {
            // Measured on the degraded replica across a 12-minute loaded window: median 1.4%, p90 11.8%,
            // peak 19.8%, and 33% of samples at or above 5%.
            var result = Rule.Evaluate(LabWindow(breachFraction: 0.33), SustainedThresholdOptions.ForCpuThrottling);

            Assert.Equal(DetectionStatus.Anomalous, result.Status);
            Assert.True(result.BreachFraction >= 0.25, $"breach fraction {result.BreachFraction}");
            Assert.Contains("Held at or above", result.Reason, StringComparison.Ordinal);
        }

        [Fact]
        public void TheIdleLabWindow_StaysSilent()
        {
            // The same pod over a mostly-idle half hour: only 13% of samples reached 5%. Nothing was wrong, and
            // the rule has to agree — this is the separation the persistence requirement exists to make.
            var result = Rule.Evaluate(LabWindow(breachFraction: 0.13), SustainedThresholdOptions.ForCpuThrottling);

            Assert.Equal(DetectionStatus.Healthy, result.Status);
            Assert.True(result.BreachFraction < 0.25);
        }

        [Fact]
        public void TheLiteratureThreshold_WouldHaveMissedIt()
        {
            // Guidance commonly names 25% throttling as the point of concern. On this fault the signal peaked at
            // 19.8% and never reached it — which is why the threshold was measured rather than quoted, and why
            // this test exists to stop anyone "correcting" it back.
            var window = LabWindow(breachFraction: 0.33);

            var measured = Rule.Evaluate(window, SustainedThresholdOptions.ForCpuThrottling);
            var literature = Rule.Evaluate(
                window, SustainedThresholdOptions.ForCpuThrottling with { Threshold = 0.25 });

            Assert.Equal(DetectionStatus.Anomalous, measured.Status);
            Assert.Equal(DetectionStatus.Healthy, literature.Status);
            Assert.True(measured.PeakValue < 0.25, $"peak {measured.PeakValue} — the premise of this test is gone");
        }

        [Fact]
        public void ASingleSpike_IsNotAFault()
        {
            // An absolute threshold without persistence is a noise generator on a bursty signal.
            var window = new double[40];
            window[17] = 0.9;

            var result = Rule.Evaluate(window, SustainedThresholdOptions.ForCpuThrottling);

            Assert.Equal(DetectionStatus.Healthy, result.Status);
            Assert.Equal(1, result.BreachedSamples);
        }

        [Fact]
        public void MissingSamplesAreSilence_NotCompliance()
        {
            // A scrape that returned nothing says nothing about whether the container was throttled. Counting
            // NaN as "under the threshold" would let a broken query read as a clean bill of health.
            var window = new double[40];
            Array.Fill(window, double.NaN);

            var result = Rule.Evaluate(window, SustainedThresholdOptions.ForCpuThrottling);

            Assert.Equal(DetectionStatus.InsufficientData, result.Status);
            Assert.False(result.IsHealthy);
            Assert.Equal(0, result.UsableSamples);
        }

        [Fact]
        public void GapsDoNotDiluteTheBreach()
        {
            // Ten breaching samples among ten gaps is a fully-breached window, not a half-breached one.
            var window = new double[20];
            for (var i = 0; i < window.Length; i++)
            {
                window[i] = i % 2 == 0 ? 0.4 : double.NaN;
            }

            var result = Rule.Evaluate(
                window, SustainedThresholdOptions.ForCpuThrottling with { MinimumSamples = 10 });

            Assert.Equal(DetectionStatus.Anomalous, result.Status);
            Assert.Equal(10, result.UsableSamples);
            Assert.Equal(1.0, result.BreachFraction, 6);
        }

        [Fact]
        public void AShortWindow_IsWarmingUp_NotHealthy()
        {
            var window = new double[5];
            Array.Fill(window, 0.9);

            var result = Rule.Evaluate(window, SustainedThresholdOptions.ForCpuThrottling);

            Assert.Equal(DetectionStatus.WarmingUp, result.Status);
            Assert.False(result.IsDecided);
        }

        [Fact]
        public void ARareEventCounter_FiresOnASingleOccurrence()
        {
            // OOM kills and restarts do not happen by accident, so the profile for them needs almost no
            // persistence at all.
            var window = new double[20];
            window[3] = 1.0;

            var result = Rule.Evaluate(window, SustainedThresholdOptions.ForRareEvent);

            Assert.Equal(DetectionStatus.Anomalous, result.Status);
            Assert.Equal(1, result.BreachedSamples);
        }

        [Fact]
        public void DefaultOptions_AreRejected()
        {
            Assert.Throws<ArgumentException>(() => Rule.Evaluate(new double[40], default));
        }

        [Fact]
        public void TheVerdictReachesTheGrouperAsAFinding()
        {
            // The whole point: a signal a peer comparison cannot reach still becomes an incident. Classified as
            // Infrastructure by the catalog, so it leads any latency symptom it is grouped with.
            var pipeline = new IncidentPipeline();
            var subject = new IncidentSubject("overfit", "overfit-server-degraded", "pod-degraded", "node-1");

            var result = Rule.Evaluate(LabWindow(breachFraction: 0.33), SustainedThresholdOptions.ForCpuThrottling);

            Assert.True(pipeline.ObserveRule(
                subject, "container_cpu_cfs_throttled_periods_total", result, WindowStart, WindowEnd));

            var incident = Assert.Single(pipeline.Group(IncidentGroupingOptions.Balanced));

            Assert.Equal(SignalClass.Infrastructure, incident.Primary.Class);
            Assert.Equal("pod-degraded", incident.Primary.Subject.Pod);
            Assert.Equal(result.BreachFraction, incident.Primary.Severity, 6);
        }

        [Fact]
        public void AHealthyVerdictProducesNothing()
        {
            var pipeline = new IncidentPipeline();
            var subject = new IncidentSubject("overfit", "overfit-server", "pod-a", "node-1");

            var result = Rule.Evaluate(LabWindow(breachFraction: 0.13), SustainedThresholdOptions.ForCpuThrottling);

            Assert.False(pipeline.ObserveRule(
                subject, "container_cpu_cfs_throttled_periods_total", result, WindowStart, WindowEnd));
            Assert.Equal(0, pipeline.Count);
        }

        /// <summary>
        /// A window shaped like the lab's: bursty, capped near the measured 19.8% peak, with the requested
        /// share of samples at or above 5%. Deterministic so the breach count is exact.
        /// </summary>
        private static double[] LabWindow(double breachFraction)
        {
            const int Samples = 49;

            var values = new double[Samples];
            var breaching = (int)Math.Round(breachFraction * Samples);

            for (var i = 0; i < Samples; i++)
            {
                // Breaching samples ramp between the threshold and the measured ceiling; the rest sit in the
                // sub-threshold noise the same pod showed for most of the window.
                values[i] = i < breaching
                    ? 0.05 + (0.15 * i / Math.Max(1, breaching - 1))
                    : 0.014 * (i % 3) / 2.0;
            }

            return values;
        }
    }
}
