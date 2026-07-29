// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System;
using DevOnBike.Overfit.Anomalies.Monitoring.Contracts;

namespace DevOnBike.Overfit.Tests.TestSupport
{
    /// <summary>
    /// A healthy Kubernetes deployment, generated: <c>pods × 12 features × hours</c> of metric series with
    /// <b>no fault of any kind injected</b>.
    ///
    /// <para>That absence is the whole design. Every threshold in the guard was calibrated against
    /// <i>detectability</i> — can it find the replica we broke — and not one against <i>false positives</i>.
    /// On a population where nothing is wrong, any incident the guard produces is a false positive by
    /// construction, and counting them is the only way to get that number. It is also the number the
    /// blueprint's M0 gate turns on.</para>
    ///
    /// <para><b>The realism is not decoration — it is the validity of the measurement.</b> A generator emitting
    /// flat lines with Gaussian noise would report a flattering false-positive rate that says nothing, because
    /// every shape the detectors actually struggle with would be missing. Each element below is here because it
    /// was measured on the cluster lab or is the documented reason a detector exists:</para>
    /// <list type="bullet">
    /// <item><b>Diurnal seasonality.</b> Traffic follows a daily curve and latency, CPU and GC follow traffic.
    /// The rising limb of that curve is the hardest case a trend detector ever sees — it looks exactly like a
    /// leak over any window shorter than the period.</item>
    /// <item><b>Per-pod baseline offsets.</b> Replicas legitimately differ. Measured on the lab: p95 medians of
    /// 860 / 880 / 902 ms, a 3% spread — the exact shape that made Cliff's delta report 0.52 and 0.68 and call
    /// a healthy group split.</item>
    /// <item><b>Raw CPU spread of ~33%.</b> Measured under <i>even</i> load: 1.44 / 1.78 / 1.92 cores. Four
    /// times the peer detector's 8% size gate, which is why raw CPU is expected to be the noisiest input here.</item>
    /// <item><b>Affine cost.</b> CPU is <c>fixed + marginal × traffic</c>, not proportional. That is the
    /// measured model behind the finding that dividing by a work metric leaves a residue which grows with load
    /// imbalance and never vanishes.</item>
    /// <item><b>Scrape gaps as NaN.</b> Missing is not zero anywhere in this pipeline, and the generator has to
    /// produce the holes that make that distinction matter.</item>
    /// <item><b>Restarts.</b> A restart drops working set to near zero and ramps it back. Measured
    /// consequence: within-pod spread of 99–104% for any window containing the ramp, which is why memory was
    /// unusable in the live run.</item>
    /// <item><b>Scrape spikes.</b> Rare, large, and the entire reason the estimators are rank-based.</item>
    /// <item><b>Uneven load balancing.</b> The lab measured 22–26% shares across four replicas under a
    /// generator asking for equal load. Perfectly equal traffic is not a thing.</item>
    /// <item><b>Identically-zero counters.</b> OOM events and the 5xx ratio sit at zero on a healthy cluster —
    /// the case that exposed the zero-scale defect in the departure count.</item>
    /// <item><b>Absent CFS series.</b> No pod here carries a CPU limit, so the throttle ratio is NaN
    /// throughout, exactly as it was for three of the lab's four pods.</item>
    /// </list>
    ///
    /// <para>Deterministic for a given seed, so a false-positive count is reproducible and a regression in it
    /// is attributable.</para>
    /// </summary>
    public sealed class SyntheticCluster
    {
        /// <summary>Features per pod — the <see cref="MetricSnapshot"/> contract.</summary>
        public const int MetricCount = (int)MetricIndex.Count;

        private readonly double[][] _series;

        /// <param name="pods">Replicas in the deployment.</param>
        /// <param name="hours">Wall-clock hours to generate.</param>
        /// <param name="scrapeSeconds">Scrape interval; 15 s matches the lab's Prometheus.</param>
        /// <param name="seed">Any value; the same seed reproduces the same cluster exactly.</param>
        public SyntheticCluster(int pods, double hours, double scrapeSeconds = 15.0, int seed = 20260729)
        {
            ArgumentOutOfRangeException.ThrowIfLessThan(pods, 3);
            ArgumentOutOfRangeException.ThrowIfLessThanOrEqual(hours, 0.0);
            ArgumentOutOfRangeException.ThrowIfLessThanOrEqual(scrapeSeconds, 0.0);

            Pods = pods;
            ScrapeSeconds = scrapeSeconds;
            Samples = (int)(hours * 3600.0 / scrapeSeconds);

            _series = new double[pods * MetricCount][];

            var rng = new Random(seed);

            for (var pod = 0; pod < pods; pod++)
            {
                Generate(pod, rng);
            }
        }

        /// <summary>Replicas generated.</summary>
        public int Pods
        {
            get;
        }

        /// <summary>Samples per feature.</summary>
        public int Samples
        {
            get;
        }

        /// <summary>Seconds between samples.</summary>
        public double ScrapeSeconds
        {
            get;
        }

        /// <summary>Pod name in the shape kube-state-metrics reports, so workload derivation behaves as in production.</summary>
        public static string PodName(int pod) => $"overfit-server-6d4b7c9f8x-{pod:d5}";

        /// <summary>One feature's full history for one pod. Not a copy — do not mutate.</summary>
        public double[] Series(int pod, MetricIndex metric) => _series[(pod * MetricCount) + (int)metric];

        /// <summary>A window of one feature, for feeding a detector.</summary>
        public ReadOnlySpan<double> Window(int pod, MetricIndex metric, int start, int length)
            => Series(pod, metric).AsSpan(start, length);

        private void Generate(int pod, Random rng)
        {
            for (var m = 0; m < MetricCount; m++)
            {
                _series[(pod * MetricCount) + m] = new double[Samples];
            }

            // Per-pod personality, drawn once. These offsets are what make a peer group non-identical, which is
            // the difference between measuring a false-positive rate and measuring nothing.
            var latencyOffset = 1.0 + ((rng.NextDouble() - 0.5) * 0.06);   // +-3%, as measured
            var cpuOffset = 1.0 + ((rng.NextDouble() - 0.5) * 0.33);       // +-16.5% -> ~33% spread across a group
            var memoryBaseline = 1.15e9 * (1.0 + ((rng.NextDouble() - 0.5) * 0.05));
            var trafficShare = (1.0 / Pods) * (1.0 + ((rng.NextDouble() - 0.5) * 0.18));
            var diurnalPhase = rng.NextDouble() * 0.2;                     // scrapers are not synchronised

            // One restart per pod per day is a quiet cluster, not a broken one: image updates, node drains,
            // evictions. Placed away from the very start so a window can contain the ramp.
            var restartAt = Samples > 200 ? rng.Next(Samples / 5, Samples) : int.MaxValue;

            var samplesPerDay = 86400.0 / ScrapeSeconds;
            var memory = memoryBaseline;

            for (var t = 0; t < Samples; t++)
            {
                var timeOfDay = ((t / samplesPerDay) + diurnalPhase) % 1.0;

                // Traffic: a daily curve between roughly a fifth and full load, never zero — a cluster with no
                // traffic exercises none of the load-sensitive paths.
                var diurnal = 0.6 + (0.4 * Math.Sin(2.0 * Math.PI * timeOfDay));
                var traffic = 40.0 * trafficShare * diurnal * (1.0 + ((rng.NextDouble() - 0.5) * 0.10));

                // Queueing: latency rises with load, sub-linearly.
                var latency = 700.0 * latencyOffset * (1.0 + (0.35 * diurnal)) * (1.0 + ((rng.NextDouble() - 0.5) * 0.08));

                // Rare scrape artefact. The reason every estimator here is rank-based rather than least-squares.
                if (rng.NextDouble() < 0.003)
                {
                    latency *= 5.0;
                }

                // Working set sawtooths between collections and steps back to near nothing on a restart.
                memory += 1.2e6 * (1.0 + ((rng.NextDouble() - 0.5) * 0.4));
                if (memory > memoryBaseline * 1.06 || t == restartAt)
                {
                    memory = t == restartAt ? 3.0e7 : memoryBaseline;
                }

                Set(pod, MetricIndex.RequestsPerSecond, t, traffic);
                Set(pod, MetricIndex.LatencyP50Ms, t, latency * 0.35);
                Set(pod, MetricIndex.LatencyP95Ms, t, latency);
                Set(pod, MetricIndex.LatencyP99Ms, t, latency * 2.4);

                // Affine, not proportional: a fixed floor plus a marginal cost per request. This is the
                // measured model, and it is why unit cost carries a residue that grows with imbalance.
                Set(pod, MetricIndex.CpuUsageRatio, t,
                    cpuOffset * (0.45 + (0.040 * traffic)) * (1.0 + ((rng.NextDouble() - 0.5) * 0.15)));

                Set(pod, MetricIndex.MemoryWorkingSetBytes, t, memory);
                Set(pod, MetricIndex.GcGen2HeapBytes, t, memory * 0.42);
                Set(pod, MetricIndex.GcPauseRatio, t, 0.004 * diurnal * (1.0 + ((rng.NextDouble() - 0.5) * 0.6)));

                // Mostly empty, occasionally a couple of items — never a starvation signal.
                Set(pod, MetricIndex.ThreadPoolQueueLength, t, rng.NextDouble() < 0.05 ? rng.Next(1, 4) : 0.0);

                // Healthy means these are zero. Exercising the zero-scale path is the point.
                Set(pod, MetricIndex.OomEventsRate, t, 0.0);
                Set(pod, MetricIndex.ErrorRate, t, 0.0);

                // No pod here carries a CPU limit, so CFS accounting does not exist for any of them.
                Set(pod, MetricIndex.CpuThrottleRatio, t, double.NaN);
            }

            PunchScrapeGaps(pod, rng);
        }

        /// <summary>
        /// Drops isolated samples to <see cref="double.NaN"/>. Prometheus misses scrapes — a busy target, a
        /// restarting exporter, a network hiccup — and "missing is not zero" is a decision the whole pipeline
        /// rests on, so the holes have to be here for the measurement to mean anything.
        /// </summary>
        private void PunchScrapeGaps(int pod, Random rng)
        {
            for (var m = 0; m < MetricCount; m++)
            {
                if ((MetricIndex)m == MetricIndex.CpuThrottleRatio)
                {
                    continue;
                }

                var series = _series[(pod * MetricCount) + m];

                for (var t = 0; t < series.Length; t++)
                {
                    if (rng.NextDouble() < 0.005)
                    {
                        series[t] = double.NaN;
                    }
                }
            }
        }

        private void Set(int pod, MetricIndex metric, int t, double value)
            => _series[(pod * MetricCount) + (int)metric][t] = value;
    }
}
