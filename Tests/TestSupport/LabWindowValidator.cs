// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Globalization;
using DevOnBike.Overfit.Anomalies.Contracts;

namespace DevOnBike.Overfit.Tests.TestSupport
{
    /// <summary>
    /// Decides whether a recorded lab window is fit to be a calibration reference.
    ///
    /// <para><b>This exists because three consecutive recordings were installed before anyone looked at
    /// them</b>, each broken in a different way, and every one of them looked like a successful run: the
    /// recorder printed a coverage table, wrote a file and exited zero. What follows is one check per failure
    /// actually observed, not a checklist imagined in advance.</para>
    ///
    /// <list type="bullet">
    /// <item><b>Phantom replicas.</b> Prometheus keeps series for pods that no longer exist. A window reaching
    /// back past a scale-down picked up eight deleted replicas, every one of them all-NaN, every one counted
    /// into "how far apart do replicas sit".</item>
    /// <item><b>An idle replica.</b> The load generator drops an endpoint that fails one <c>/health</c> probe
    /// and never retries it, so a perfectly healthy pod sat at zero requests for a whole run while Prometheus
    /// kept scraping it. Inside a peer group that is a fabricated outlier and a starved peer at once.</item>
    /// <item><b>A window longer than the traffic behind it.</b> Settle 15 minutes, record 20, and the first
    /// five minutes of the window predate the load — 22 of 81 scrapes NaN on every pod.</item>
    /// <item><b>The wrong operating point.</b> Spreading the reference load over more replicas emptied the
    /// latency histograms until p95 and p99 pinned to bucket edges; packing more replicas onto the node
    /// saturated it until the throttled replica was indistinguishable from healthy ones. Both directions
    /// produce a plausible file full of numbers that mean nothing.</item>
    /// <item><b>A restart inside the window.</b> Working set climbing from a cold start is a perfect trend
    /// signal and a perfect peer outlier, both manufactured.</item>
    /// </list>
    ///
    /// <para>The bands come from the recording this project calibrated against — healthy p95 near 860 ms and
    /// a throttled replica near 2441 ms. They are deliberately wide: the job here is to reject a window that
    /// is not measuring the reference workload at all, not to police ordinary variation. If the lab's workload
    /// is changed on purpose, these move with it, and that should be a visible edit rather than a silent
    /// drift.</para>
    /// </summary>
    public static class LabWindowValidator
    {
        /// <summary>Share of a pod's latency samples that must be finite for it to count as covered.</summary>
        public const double MinLatencyCoverage = 0.90;

        /// <summary>Median p95 across healthy replicas, in milliseconds — the reference sat at ~860.</summary>
        public const double MinHealthyP95Ms = 350.0;

        /// <inheritdoc cref="MinHealthyP95Ms"/>
        public const double MaxHealthyP95Ms = 1800.0;

        /// <summary>
        /// How much slower the throttled replica must be than its healthy siblings.
        ///
        /// <para>The reference measured 2.8x. Below this the fault is not detectable by anything downstream,
        /// so a window that fails here cannot be used to check detection — and, more quietly, it also means
        /// the healthy pods are not healthy, because the usual cause is a saturated node dragging them down
        /// to the throttled pod's level.</para>
        /// </summary>
        public const double MinFaultContrast = 1.8;

        /// <summary>Validates a loaded window. An empty problem list means it is usable.</summary>
        public static Verdict Validate(MetricWindow window, IReadOnlyList<string> faultedPods)
        {
            ArgumentNullException.ThrowIfNull(window);
            ArgumentNullException.ThrowIfNull(faultedPods);

            var problems = new List<string>();
            var healthyP95 = new List<double>();
            var faultedP95 = new List<double>();

            for (var pod = 0; pod < window.Pods.Count; pod++)
            {
                var name = window.Pods[pod];
                var faulted = faultedPods.Contains(name);

                if (!ReportsAnything(window, pod))
                {
                    problems.Add($"{name}: no finite sample on any metric — a replica that no longer exists, "
                                 + "still held by Prometheus and counted into the peer group.");

                    continue;
                }

                var traffic = FiniteAndPositive(window.Series(pod, MetricIndex.RequestsPerSecond));

                if (traffic == 0)
                {
                    problems.Add($"{name}: zero requests across the whole window — the pod was scraped but "
                                 + "never driven, which is a fabricated outlier and a starved peer at once.");
                }

                var latency = window.Series(pod, MetricIndex.LatencyP95Ms);
                var finite = CountFinite(latency);
                var coverage = latency.Length == 0 ? 0.0 : (double)finite / latency.Length;

                if (coverage < MinLatencyCoverage)
                {
                    problems.Add(
                        $"{name}: p95 finite in {finite} of {latency.Length} scrapes ({Percent(coverage)}), "
                        + $"below {Percent(MinLatencyCoverage)}.");
                }

                if (finite > 0)
                {
                    var median = Median(latency);
                    (faulted ? faultedP95 : healthyP95).Add(median);
                }

                if (AnyPositive(window.Series(pod, MetricIndex.ContainerRestarts)))
                {
                    problems.Add($"{name}: restarted inside the window — the warm-up ramp that follows is a "
                                 + "manufactured trend and a manufactured peer outlier.");
                }
            }

            CheckLeadingGap(window, problems);
            CheckOperatingPoint(healthyP95, faultedP95, problems);

            return new Verdict(problems);
        }

        /// <summary>
        /// Flags a window whose opening scrapes are blank across the fleet — the signature of recording a
        /// longer window than the load was running for. Measured per scrape rather than per pod because a
        /// leading gap hits every pod at once, which is exactly what distinguishes it from one bad replica.
        /// </summary>
        private static void CheckLeadingGap(MetricWindow window, List<string> problems)
        {
            if (window.Pods.Count == 0 || window.Length == 0)
            {
                return;
            }

            var blank = 0;

            for (var t = 0; t < window.Length; t++)
            {
                var reporting = 0;

                for (var pod = 0; pod < window.Pods.Count; pod++)
                {
                    if (double.IsFinite(window.Series(pod, MetricIndex.LatencyP95Ms)[t]))
                    {
                        reporting++;
                    }
                }

                if (reporting * 2 > window.Pods.Count)
                {
                    break;
                }

                blank++;
            }

            // A couple of blank scrapes at the head is a rate() warm-up; a fifth of the window is not.
            if (blank * 20 > window.Length)
            {
                problems.Add(
                    $"the first {blank} of {window.Length} scrapes have no latency on most pods — the window "
                    + "reaches back before the load started, so it is longer than the traffic behind it.");
            }
        }

        private static void CheckOperatingPoint(
            List<double> healthyP95,
            List<double> faultedP95,
            List<string> problems)
        {
            if (healthyP95.Count == 0)
            {
                problems.Add("no healthy replica produced a latency median — nothing to calibrate against.");

                return;
            }

            var healthy = Median(healthyP95);

            if (healthy < MinHealthyP95Ms || healthy > MaxHealthyP95Ms)
            {
                problems.Add(
                    $"healthy p95 median is {healthy:F0} ms, outside {MinHealthyP95Ms:F0}-{MaxHealthyP95Ms:F0} ms. "
                    + "Too low means the histograms are empty and the quantiles have pinned to bucket edges; "
                    + "too high means the node is saturated. Neither is the reference workload.");
            }

            if (faultedP95.Count == 0)
            {
                return;
            }

            var contrast = Median(faultedP95) / healthy;

            if (contrast < MinFaultContrast)
            {
                problems.Add(
                    $"the throttled replica is only {contrast:F2}x its healthy siblings, under {MinFaultContrast:F1}x. "
                    + "The usual cause is a saturated node dragging the healthy pods down to it.");
            }
        }

        private static bool ReportsAnything(MetricWindow window, int pod)
        {
            for (var m = 0; m < (int)MetricIndex.Count; m++)
            {
                if (CountFinite(window.Series(pod, (MetricIndex)m)) > 0)
                {
                    return true;
                }
            }

            return false;
        }

        private static int CountFinite(ReadOnlySpan<double> values)
        {
            var count = 0;

            for (var i = 0; i < values.Length; i++)
            {
                if (double.IsFinite(values[i]))
                {
                    count++;
                }
            }

            return count;
        }

        private static int FiniteAndPositive(ReadOnlySpan<double> values)
        {
            var count = 0;

            for (var i = 0; i < values.Length; i++)
            {
                if (double.IsFinite(values[i]) && values[i] > 0.0)
                {
                    count++;
                }
            }

            return count;
        }

        private static bool AnyPositive(ReadOnlySpan<double> values)
        {
            for (var i = 0; i < values.Length; i++)
            {
                if (double.IsFinite(values[i]) && values[i] > 0.0)
                {
                    return true;
                }
            }

            return false;
        }

        private static double Median(ReadOnlySpan<double> values)
        {
            var finite = new List<double>(values.Length);

            for (var i = 0; i < values.Length; i++)
            {
                if (double.IsFinite(values[i]))
                {
                    finite.Add(values[i]);
                }
            }

            return Median(finite);
        }

        private static double Median(List<double> values)
        {
            if (values.Count == 0)
            {
                return double.NaN;
            }

            values.Sort();

            return values[values.Count / 2];
        }

        private static string Percent(double share)
            => (share * 100.0).ToString("F1", CultureInfo.InvariantCulture) + "%";

        /// <summary>What the validator concluded. No problems means the window can be a reference.</summary>
        public readonly record struct Verdict(IReadOnlyList<string> Problems)
        {
            public bool IsUsable => Problems.Count == 0;

            public string Describe()
                => IsUsable
                    ? "usable as a calibration reference"
                    : string.Join("\n  - ", new[] { $"{Problems.Count} problem(s):" }.Concat(Problems));
        }
    }
}
