// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Anomalies.Monitoring.Contracts;
using DevOnBike.Overfit.Tests.TestSupport;

namespace DevOnBike.Overfit.Tests.Anomalies
{
    /// <summary>
    /// The gate on the calibration reference.
    ///
    /// <para><b>Every threshold in the guard is tuned against a generator, and the generator is tuned against
    /// the checked-in lab window.</b> That makes the fixture the root of the whole chain, and until now
    /// nothing checked it: three broken recordings in a row were installed and only caught afterwards, by
    /// reading numbers that looked odd. A contaminated reference does not announce itself — it quietly moves
    /// every constant calibrated against it.</para>
    ///
    /// <para>The negative cases matter as much as the positive one. A validator that passes everything would
    /// satisfy the first test and protect nothing, so each defect that actually occurred is reproduced here
    /// and the validator has to reject it.</para>
    /// </summary>
    public sealed class LabWindowFixtureTests
    {
        private static readonly DateTimeOffset T0 = new(2026, 7, 31, 12, 0, 0, TimeSpan.Zero);

        [Fact]
        public void TheCheckedInFixtureIsUsableAsAReference()
        {
            Assert.True(LabWindowFixture.Exists, $"fixture missing: {LabWindowFixture.Path}");

            var (window, faulted) = LabWindowFixture.Load();
            var verdict = LabWindowValidator.Validate(window, faulted);

            Assert.True(
                verdict.IsUsable,
                $"the checked-in lab window is not fit to calibrate against — {verdict.Describe()}");
        }

        /// <summary>The baseline must pass, or every negative case below proves nothing.</summary>
        [Fact]
        public void AWellFormedWindowPasses()
        {
            var verdict = LabWindowValidator.Validate(Recording(), ["pod-faulted"]);

            Assert.True(verdict.IsUsable, verdict.Describe());
        }

        /// <summary>A replica deleted before the window, still held by Prometheus.</summary>
        [Fact]
        public void APodWithNoDataAtAll_IsRejected()
        {
            var window = Recording(extraPods: ["pod-phantom"]);
            var verdict = LabWindowValidator.Validate(window, ["pod-faulted"]);

            Assert.False(verdict.IsUsable);
            Assert.Contains(verdict.Problems, p => p.Contains("pod-phantom", StringComparison.Ordinal));
        }

        /// <summary>
        /// The <c>xj98m</c> case: scraped throughout, driven never, because it failed one <c>/health</c> probe
        /// at the moment the load generator built its endpoint list.
        /// </summary>
        [Fact]
        public void APodThatReceivedNoTraffic_IsRejected()
        {
            var window = Recording();
            var idle = 1;

            window.Series(idle, MetricIndex.RequestsPerSecond).Clear();
            window.Series(idle, MetricIndex.LatencyP95Ms).Fill(double.NaN);

            var verdict = LabWindowValidator.Validate(window, ["pod-faulted"]);

            Assert.False(verdict.IsUsable);
            Assert.Contains(verdict.Problems, p => p.Contains("zero requests", StringComparison.Ordinal));
        }

        /// <summary>Settle shorter than the window: the opening scrapes predate the load.</summary>
        [Fact]
        public void AWindowThatStartsBeforeTheLoad_IsRejected()
        {
            var window = Recording();

            for (var pod = 0; pod < window.Pods.Count; pod++)
            {
                window.Series(pod, MetricIndex.LatencyP95Ms)[..12].Fill(double.NaN);
            }

            var verdict = LabWindowValidator.Validate(window, ["pod-faulted"]);

            Assert.False(verdict.IsUsable);
            Assert.Contains(verdict.Problems, p => p.Contains("before the load started", StringComparison.Ordinal));
        }

        /// <summary>
        /// A saturated node. The healthy replicas slow down until the throttled one is ordinary — the fault is
        /// still injected and no longer visible, which is the most dangerous of these because every pod still
        /// reports full, plausible data.
        /// </summary>
        [Fact]
        public void AWindowWhereTheFaultHasNoContrast_IsRejected()
        {
            var window = Recording(healthyP95: 3500.0, faultedP95: 4700.0);
            var verdict = LabWindowValidator.Validate(window, ["pod-faulted"]);

            Assert.False(verdict.IsUsable);
            Assert.Contains(verdict.Problems, p => p.Contains("healthy siblings", StringComparison.Ordinal));
        }

        /// <summary>Too little traffic per pod: the quantiles pin to histogram bucket edges.</summary>
        [Fact]
        public void AWindowRecordedOnAnUnderloadedCluster_IsRejected()
        {
            var window = Recording(healthyP95: 120.0, faultedP95: 340.0);
            var verdict = LabWindowValidator.Validate(window, ["pod-faulted"]);

            Assert.False(verdict.IsUsable);
            Assert.Contains(verdict.Problems, p => p.Contains("outside", StringComparison.Ordinal));
        }

        [Fact]
        public void ARestartInsideTheWindow_IsRejected()
        {
            var window = Recording();
            window.Series(2, MetricIndex.ContainerRestarts)[20] = 1.0;

            var verdict = LabWindowValidator.Validate(window, ["pod-faulted"]);

            Assert.False(verdict.IsUsable);
            Assert.Contains(verdict.Problems, p => p.Contains("restarted inside", StringComparison.Ordinal));
        }

        /// <summary>
        /// Three healthy replicas plus a throttled one, at the operating point the project calibrated
        /// against: healthy p95 near 860 ms, throttled near 2441 ms.
        /// </summary>
        private static MetricWindow Recording(
            double healthyP95 = 860.0,
            double faultedP95 = 2441.0,
            IReadOnlyList<string>? extraPods = null)
        {
            var names = new List<string> { "pod-0", "pod-1", "pod-2", "pod-faulted" };

            if (extraPods is not null)
            {
                names.AddRange(extraPods);
            }

            var window = new MetricWindow(names, 48, T0, TimeSpan.FromSeconds(15));
            var rng = new Random(20260731);

            for (var pod = 0; pod < names.Count; pod++)
            {
                // Anything in extraPods is a phantom: present in the pod list, absent from every series.
                if (extraPods is not null && extraPods.Contains(names[pod]))
                {
                    for (var m = 0; m < (int)MetricIndex.Count; m++)
                    {
                        window.Series(pod, (MetricIndex)m).Fill(double.NaN);
                    }

                    continue;
                }

                var level = names[pod] == "pod-faulted" ? faultedP95 : healthyP95;
                var latency = window.Series(pod, MetricIndex.LatencyP95Ms);
                var rps = window.Series(pod, MetricIndex.RequestsPerSecond);
                var restarts = window.Series(pod, MetricIndex.ContainerRestarts);

                for (var t = 0; t < window.Length; t++)
                {
                    latency[t] = level * (1.0 + ((rng.NextDouble() - 0.5) * 0.3));
                    rps[t] = 4.0 * (1.0 + ((rng.NextDouble() - 0.5) * 0.2));
                    restarts[t] = 0.0;
                }
            }

            return window;
        }
    }
}
