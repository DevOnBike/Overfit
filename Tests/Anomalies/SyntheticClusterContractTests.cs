// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Anomalies.Contracts;
using DevOnBike.Overfit.Tests.TestSupport;

namespace DevOnBike.Overfit.Tests.Anomalies
{
    /// <summary>
    /// The three properties of <see cref="SyntheticCluster"/> that the anomaly harnesses build on.
    ///
    /// <para>Written after <c>DetectionMatrixDiagnostics</c> derived a fault's affected channels by diffing
    /// an injected clone against a clean one and got "all thirteen channels, for every fault" — including
    /// faults that write to a single array. The generator was innocent: it is reproducible, and the diff was
    /// comparing deliberate <see cref="double.NaN"/> gaps with <c>!=</c>, under which NaN is unequal to
    /// itself.</para>
    ///
    /// <para>These are cheap and they are asserted rather than printed, because that derivation now depends
    /// on them. A generator that quietly stopped being reproducible, or stopped punching gaps, would turn it
    /// back into something that reports a confident wrong answer.</para>
    /// </summary>
    public sealed class SyntheticClusterContractTests
    {
        private const int Pods = 12;

        /// <summary>The documented contract: "the same seed reproduces the same cluster exactly".</summary>
        [Fact]
        public void SameSeedReproducesTheSameCluster()
        {
            var a = new SyntheticCluster(Pods, 6, 15.0, 20260801, restartsPerPodPerDay: 0.0);
            var b = new SyntheticCluster(Pods, 6, 15.0, 20260801, restartsPerPodPerDay: 0.0);

            Assert.Equal(a.Samples, b.Samples);

            for (var m = 0; m < (int)MetricIndex.Count; m++)
            {
                var metric = (MetricIndex)m;

                for (var pod = 0; pod < Pods; pod++)
                {
                    var x = a.Series(pod, metric);
                    var y = b.Series(pod, metric);

                    for (var i = 0; i < x.Length; i++)
                    {
                        // Equals, not ==: the gaps below are NaN, and NaN != NaN would make every channel
                        // look non-reproducible. This is the exact trap the derivation fell into.
                        Assert.True(
                            x[i].Equals(y[i]),
                            $"{metric} pod {pod} sample {i}: {x[i]:R} vs {y[i]:R}");
                    }
                }
            }
        }

        /// <summary>
        /// Scrape gaps exist and are rare. Missing is not zero anywhere in this pipeline, and a generator
        /// producing complete series would let a detector pass here and fail on a real Prometheus.
        /// </summary>
        [Fact]
        public void PunchesRareScrapeGaps()
        {
            var cluster = new SyntheticCluster(Pods, 6, 15.0, 20260801, restartsPerPodPerDay: 0.0);
            var gaps = 0;
            var total = 0;

            for (var pod = 0; pod < Pods; pod++)
            {
                // Any channel except the throttle ratio, which is absent by design and would swamp the rate.
                var series = cluster.Series(pod, MetricIndex.CpuUsageRatio);

                for (var i = 0; i < series.Length; i++)
                {
                    total++;

                    if (double.IsNaN(series[i]))
                    {
                        gaps++;
                    }
                }
            }

            // Generated at 0.005 per sample; the band is wide enough that the binomial scatter on ~17000
            // draws cannot reach either edge, and narrow enough to catch the rate being changed or removed.
            Assert.InRange(gaps / (double)total, 0.002, 0.010);
        }

        /// <summary>
        /// The throttle ratio is absent throughout, because no pod here carries a CPU limit and CFS
        /// accounting does not exist without one. A detector reading it must produce nothing rather than
        /// treat absence as zero.
        /// </summary>
        [Fact]
        public void ThrottleRatioIsAbsentEverywhere()
        {
            var cluster = new SyntheticCluster(Pods, 6, 15.0, 20260801, restartsPerPodPerDay: 0.0);

            for (var pod = 0; pod < Pods; pod++)
            {
                var series = cluster.Series(pod, MetricIndex.CpuThrottleRatio);

                for (var i = 0; i < series.Length; i++)
                {
                    Assert.True(double.IsNaN(series[i]), $"pod {pod} sample {i} is {series[i]:R}");
                }
            }
        }
    }
}
