// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Anomalies.Contracts;
using DevOnBike.Overfit.Anomalies.Monitoring;

namespace DevOnBike.Overfit.Tests.Anomalies
{
    /// <summary>
    /// A channel bound to a series that never varies reports a number, passes every coverage check, and can
    /// never produce a finding. Two were found on 2026-08-08 and both had been live for months, so the cases
    /// here are those two shapes rather than invented ones.
    /// </summary>
    public sealed class InertChannelTests
    {
        private const int Pods = 4;
        private const int Samples = 40;

        [Fact]
        public void AConstantNonZeroChannelIsReportedAndIsConclusive()
        {
            // The latency shape: histogram_quantile over buckets that all requests fall into returns
            // 25 + q * 25, so p95 was 48.75 on every pod on every scrape.
            var calibrator = Observe(MetricIndex.LatencyP95Ms, _ => 48.75);

            var inert = calibrator.InertChannels(minimumObservations: 100);
            var found = Single(inert, MetricIndex.LatencyP95Ms);

            Assert.Equal(48.75, found.Value);
            Assert.True(found.IsConclusive, "a constant non-zero reading cannot come from a healthy signal");
        }

        [Fact]
        public void AConstantZeroChannelIsReportedButNotConclusive()
        {
            // The OOM shape: container_oom_events_total exists, binds, and is zero for ever. Indistinguishable
            // from a correctly bound channel on a cluster that had no OOM kills, which is why the type says so
            // rather than pretending to know.
            var calibrator = Observe(MetricIndex.OomEventsRate, _ => 0.0);

            var found = Single(calibrator.InertChannels(minimumObservations: 100), MetricIndex.OomEventsRate);

            Assert.Equal(0.0, found.Value);
            Assert.False(found.IsConclusive);
        }

        [Fact]
        public void AChannelThatMovesEvenOnceIsNotReported()
        {
            // The control that makes the other two mean something, and it caught a real defect: the first
            // implementation read FloorCalibrator's magnitude accumulator, which is a per-window MEDIAN. One
            // spike among 1200 observations left the median constant at zero, so the channel was reported
            // dead despite having moved. That is a false accusation aimed at exactly the channel this check
            // exists for — a working OomEventsRate reports non-zero over roughly a tenth of a window, and a
            // median cannot see that either. The check now reads the exact range of raw samples.
            var moved = 0;
            var calibrator = Observe(MetricIndex.MemoryWorkingSetBytes, _ => moved++ == 700 ? 1.0 : 0.0);

            Assert.DoesNotContain(
                calibrator.InertChannels(minimumObservations: 100),
                c => c.Metric == MetricIndex.MemoryWorkingSetBytes);
        }

        [Fact]
        public void AChannelThatFiresLikeARealOomKillIsNotReported()
        {
            // The shape MetricIndex.OomEventsRate documents: one kill yields a non-zero rate across about a
            // tenth of the window. Four samples in forty, once in the whole history — still not inert.
            var tick = 0;
            var calibrator = Observe(
                MetricIndex.OomEventsRate,
                _ =>
                {
                    var index = tick++;
                    var insideTheKill = index >= 4000 && index < 4004;

                    return insideTheKill ? 0.066 : 0.0;
                });

            Assert.DoesNotContain(
                calibrator.InertChannels(minimumObservations: 100),
                c => c.Metric == MetricIndex.OomEventsRate);
        }

        [Fact]
        public void AChannelWithTooLittleHistoryIsNotJudgedAtAll()
        {
            // Not judged leniently — not judged. Twenty minutes of zeros is what a healthy OOM channel looks
            // like, and calling that dead would be worse than saying nothing.
            var calibrator = Observe(MetricIndex.OomEventsRate, _ => 0.0, cycles: 5);

            Assert.Empty(calibrator.InertChannels(minimumObservations: 100));
        }

        [Fact]
        public void TheDefaultHorizonIsTwentyHoursOfFiveMinuteCycles()
        {
            // Pins the number in the doc comment to the number in the code: 240 cycles.
            var calibrator = Observe(MetricIndex.LatencyP95Ms, _ => 48.75, cycles: 20);

            Assert.Empty(calibrator.InertChannels());
            Assert.NotEmpty(calibrator.InertChannels(minimumObservations: 20 * Pods));
        }

        private static InertChannel Single(IReadOnlyList<InertChannel> inert, MetricIndex metric)
        {
            var matches = inert.Where(c => c.Metric == metric).ToList();

            Assert.True(matches.Count == 1, $"expected {metric} exactly once, got {matches.Count}");

            return matches[0];
        }

        /// <summary>
        /// Feeds windows in which <paramref name="metric"/> follows <paramref name="value"/> and every other
        /// channel varies, so a reported channel is the one under test rather than the whole set.
        /// </summary>
        private static FloorCalibrator Observe(
            MetricIndex metric, Func<int, double> value, int cycles = 300)
        {
            var calibrator = new FloorCalibrator();
            var pods = new List<string>(Pods);

            for (var p = 0; p < Pods; p++)
            {
                pods.Add($"pod-{p}");
            }

            var tick = 0;

            for (var cycle = 0; cycle < cycles; cycle++)
            {
                var window = new MetricWindow(
                    pods, Samples, DateTime.UnixEpoch.AddMinutes(cycle * 5), TimeSpan.FromSeconds(30));

                for (var p = 0; p < Pods; p++)
                {
                    for (var m = 0; m < (int)MetricIndex.Count; m++)
                    {
                        var series = window.Series(p, (MetricIndex)m);

                        for (var i = 0; i < Samples; i++)
                        {
                            series[i] = (MetricIndex)m == metric
                                ? value(tick++)
                                : 1.0 + (0.01 * ((cycle * Samples) + i + p));
                        }
                    }
                }

                calibrator.Observe(window);
            }

            return calibrator;
        }
    }
}
