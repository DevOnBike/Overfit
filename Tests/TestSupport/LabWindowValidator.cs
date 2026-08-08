// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Globalization;
using DevOnBike.Overfit.Anomalies.Contracts;

namespace DevOnBike.Overfit.Tests.TestSupport
{
    /// <summary>
    /// Decides whether a recorded lab window is fit to be a calibration reference, <b>one channel at a
    /// time</b>.
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
    /// <item><b>A fleet-wide excursion.</b> Every check above is per pod, so an event that moves every replica
    /// at the same instant is invisible to all of them. See <see cref="FindFleetExcursions"/>.</item>
    /// </list>
    ///
    /// <para><b>Why the verdict is per channel.</b> A single yes/no for a whole recording forces a false
    /// choice, and the 36-hour window recorded on 2026-08-08 is the worked example. Its latency channels are
    /// worthless — <c>labapp_request_duration_seconds</c> has <c>le</c> buckets at 5, 10, 25, 50, 100 and
    /// 250 ms and higher, every request lands in <c>(25, 50]</c>, so <c>histogram_quantile</c> returns
    /// <c>25 + q x 25</c> and nothing else: p50 37.50 ms, p95 48.75 ms, p99 49.75 ms, identically on all
    /// twelve pods, with 80% of its 103 645 samples <i>exactly</i> those three values (75% on the
    /// sixty-minute recording). Three channels carrying one bit. Re-recording
    /// does not fix that; finer buckets or more load would. Meanwhile the CPU and traffic channels of the same
    /// recording are fine, and a question about work and CPU can be answered from it. Rejecting the file
    /// outright throws away good measurements; passing it silently blesses the bad ones. So the verdict names
    /// the channels, and a caller states which ones it used — <see cref="Verdict.IsUsableForAll"/> turns
    /// "valid for CpuUsageRatio and RequestsPerSecond, rejected for the three latency channels" from a
    /// judgement call into a claim the test suite checks.</para>
    ///
    /// <para>The bands come from the recording this project calibrated against — healthy p95 near 860 ms and
    /// a throttled replica near 2441 ms. They are deliberately wide: the job here is to reject a window that
    /// is not measuring the reference workload at all, not to police ordinary variation. If the lab's workload
    /// is changed on purpose, these move with it, and that should be a visible edit rather than a silent
    /// drift.</para>
    /// </summary>
    public static class LabWindowValidator
    {
        /// <summary>
        /// Share of a pod's samples that must be finite, on a channel that pod reports at all.
        ///
        /// <para>Applied per channel rather than to latency alone, because "this recording is usable for
        /// CpuUsageRatio" has to be a statement about CPU. Measured across the three recordings held here,
        /// coverage among reporting pods never drops below 95.9% on any channel, so generalising the check
        /// costs no false rejection.</para>
        ///
        /// <para>A pod that reports a channel <i>never</i> is not a coverage defect — it is an absent channel,
        /// caught by the separate "no pod reports this at all" rule.</para>
        /// </summary>
        public const double MinChannelCoverage = 0.90;

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

        /// <summary>
        /// How far above its own median a pod must sit to count towards a fleet excursion.
        ///
        /// <para><b>Measured, not chosen.</b> Take the ratio of each pod's sample to that pod's own median,
        /// then per scrape take the value 90% of the fleet exceeds. Across all three recordings held here that
        /// figure never exceeds <b>3.38x</b> outside the known excursion — that peak is the 36-hour window's
        /// diurnal maximum. Inside the excursion even the <i>least</i>-elevated pod reaches <b>90.85x</b>. The
        /// gap between 3.38 and 90.85 is where this constant lives; it is not near either edge.</para>
        /// </summary>
        public const double MinFleetExcursionFactor = 5.0;

        /// <summary>
        /// Share of reporting pods that must be elevated <i>at the same scrape</i> for it to be a fleet
        /// excursion rather than an outlier. The measured event had 12 of 12.
        /// </summary>
        public const double MinFleetExcursionShare = 0.90;

        /// <summary>
        /// The channel excursions are looked for on.
        ///
        /// <para>CPU only, deliberately. It is the channel the one real event was measured on, and the only
        /// one with evidence behind the constants above. During that event fleet traffic <i>fell</i> — the
        /// 90%-of-fleet ratio for <see cref="MetricIndex.RequestsPerSecond"/> peaked at 0.43x while CPU was at
        /// 90x — so a traffic-based detector would have found nothing. Extending this to more channels needs
        /// its own measurement, not an assumption that the same numbers transfer.</para>
        /// </summary>
        public const MetricIndex ExcursionChannel = MetricIndex.CpuUsageRatio;

        private static readonly MetricIndex[] AllChannelsStore = BuildAllChannels();

        private static readonly MetricIndex[] LatencyChannelsStore =
            [MetricIndex.LatencyP50Ms, MetricIndex.LatencyP95Ms, MetricIndex.LatencyP99Ms];

        /// <summary>Every modelled channel, in index order. Excludes the <c>Count</c> sentinel.</summary>
        public static IReadOnlyList<MetricIndex> AllChannels => AllChannelsStore;

        /// <summary>
        /// The three latency quantiles. They stand or fall together: all three are read off the same
        /// histogram, so whatever ruins one has already ruined the other two.
        /// </summary>
        public static IReadOnlyList<MetricIndex> LatencyChannels => LatencyChannelsStore;

        /// <summary>
        /// Validates a loaded window. An empty problem list means every channel is usable; otherwise ask
        /// <see cref="Verdict.IsUsableForAll"/> about the channels you actually read.
        /// </summary>
        public static Verdict Validate(MetricWindow window, IReadOnlyList<string> faultedPods)
        {
            ArgumentNullException.ThrowIfNull(window);
            ArgumentNullException.ThrowIfNull(faultedPods);

            var problems = new List<Problem>();
            var healthyP95 = new List<double>();
            var faultedP95 = new List<double>();

            for (var pod = 0; pod < window.Pods.Count; pod++)
            {
                var name = window.Pods[pod];
                var faulted = faultedPods.Contains(name);

                if (!ReportsAnything(window, pod))
                {
                    problems.Add(new Problem(
                        $"{name}: no finite sample on any metric — a replica that no longer exists, "
                        + "still held by Prometheus and counted into the peer group.",
                        AllChannelsStore));

                    continue;
                }

                var traffic = FiniteAndPositive(window.Series(pod, MetricIndex.RequestsPerSecond));

                if (traffic == 0)
                {
                    // Every channel: a pod that was never driven is idle in all of them at once, which is a
                    // fabricated outlier for the peer families and a starved peer for the rest.
                    problems.Add(new Problem(
                        $"{name}: zero requests across the whole window — the pod was scraped but "
                        + "never driven, which is a fabricated outlier and a starved peer at once.",
                        AllChannelsStore));
                }

                CheckCoverage(window, pod, name, problems);

                var latency = window.Series(pod, MetricIndex.LatencyP95Ms);

                if (CountFinite(latency) > 0)
                {
                    var median = Median(latency);
                    (faulted ? faultedP95 : healthyP95).Add(median);
                }

                if (AnyPositive(window.Series(pod, MetricIndex.ContainerRestarts)))
                {
                    problems.Add(new Problem(
                        $"{name}: restarted inside the window — the warm-up ramp that follows is a "
                        + "manufactured trend and a manufactured peer outlier.",
                        AllChannelsStore));
                }
            }

            CheckChannelsNobodyReports(window, problems);
            CheckLeadingGap(window, problems);
            CheckOperatingPoint(healthyP95, faultedP95, problems);

            var excursions = FindFleetExcursions(window);

            for (var i = 0; i < excursions.Count; i++)
            {
                problems.Add(new Problem(excursions[i].Describe(window.Length), AllChannelsStore));
            }

            return new Verdict(problems, excursions);
        }

        /// <summary>
        /// Finds stretches where <see cref="ExcursionChannel"/> leaves its operating point on essentially the
        /// whole fleet at once.
        ///
        /// <para><b>Every other check here is per pod, which is exactly why this was needed.</b> On
        /// 2026-08-07 a test suite ran on the box hosting the cluster and took all twelve replicas to 88x-157x
        /// their median CPU for about seven minutes, at the traffic trough. Thirty-seven of 8641 scrapes. Not
        /// one per-pod check noticed: no pod was an outlier, because they all moved together. Those scrapes
        /// move <c>corr(fleet traffic, fleet CPU)</c> over the recording from <b>+0.47 to -0.05</b>.</para>
        ///
        /// <para><b>It is reported, not discarded.</b> The excursion was a real event, so a trend detected
        /// inside it is not a false positive and scoring it as one corrupts the measurement in the other
        /// direction. What the caller does with it is a decision the caller has to state — see
        /// <see cref="Excise"/>, which removes it at scrape level, because that is the only level where
        /// removal does anything: seven minutes averaged into an hour never crosses any threshold.</para>
        /// </summary>
        public static IReadOnlyList<FleetExcursion> FindFleetExcursions(MetricWindow window)
        {
            ArgumentNullException.ThrowIfNull(window);

            var found = new List<FleetExcursion>();

            if (window.Pods.Count < 2 || window.Length == 0)
            {
                // A fleet of one has no "at the same time as everyone else" to be measured against.
                return found;
            }

            var medians = new double[window.Pods.Count];

            for (var pod = 0; pod < window.Pods.Count; pod++)
            {
                medians[pod] = Median(window.Series(pod, ExcursionChannel));
            }

            var ratios = new List<double>(window.Pods.Count);
            var runStart = -1;
            var runFloor = 0.0;
            var runPeak = 0.0;
            var runPods = 0;

            for (var t = 0; t <= window.Length; t++)
            {
                var floor = t == window.Length ? double.NaN : FleetFloorRatio(window, medians, ratios, t);
                var elevated = double.IsFinite(floor) && floor > MinFleetExcursionFactor;

                if (elevated)
                {
                    runStart = runStart < 0 ? t : runStart;
                    runFloor = Math.Max(runFloor, floor);
                    runPeak = Math.Max(runPeak, ratios[^1]);
                    runPods = Math.Max(runPods, ratios.Count);

                    continue;
                }

                if (runStart < 0)
                {
                    continue;
                }

                found.Add(new FleetExcursion(
                    ExcursionChannel,
                    runStart,
                    t - 1,
                    window.Start + (window.Step * runStart),
                    window.Start + (window.Step * (t - 1)),
                    runPods,
                    runFloor,
                    runPeak));

                runStart = -1;
                runFloor = 0.0;
                runPeak = 0.0;
                runPods = 0;
            }

            return found;
        }

        /// <summary>
        /// Blanks the scrapes covered by <paramref name="excursions"/> — every channel, every pod — and says
        /// what it removed.
        ///
        /// <para><b>NaN rather than deletion, on purpose.</b> Dropping the samples would leave the survivors
        /// unevenly spaced while <see cref="MetricWindow"/> carries one <c>Step</c> for the whole window, so
        /// every timestamp after the cut would be a lie. <c>NaN</c> is already this project's word for "the
        /// scrape returned nothing", and every reader handles it.</para>
        ///
        /// <para>Two knock-on effects worth knowing before calling it. Blanking enough of a window will trip
        /// <see cref="MinChannelCoverage"/> on a re-validate — the measured event is 37 of 8641 scrapes, 0.43%,
        /// nowhere near it, but a short window is a different matter. And blanking at the head of a window
        /// looks exactly like a recording that started before its load.</para>
        /// </summary>
        public static ExcisionReport Excise(MetricWindow window, IReadOnlyList<FleetExcursion> excursions)
        {
            ArgumentNullException.ThrowIfNull(window);
            ArgumentNullException.ThrowIfNull(excursions);

            var notes = new List<string>();
            var blanked = 0;

            for (var i = 0; i < excursions.Count; i++)
            {
                var excursion = excursions[i];
                var first = Math.Max(0, excursion.FirstScrape);
                var last = Math.Min(window.Length - 1, excursion.LastScrape);

                if (last < first)
                {
                    continue;
                }

                for (var pod = 0; pod < window.Pods.Count; pod++)
                {
                    for (var m = 0; m < (int)MetricIndex.Count; m++)
                    {
                        window.Series(pod, (MetricIndex)m)[first..(last + 1)].Fill(double.NaN);
                    }
                }

                blanked += last - first + 1;
                notes.Add("excised " + excursion.Describe(window.Length));
            }

            return new ExcisionReport(blanked, window.Length, notes);
        }

        /// <summary>
        /// The ratio that at least <see cref="MinFleetExcursionShare"/> of the reporting pods exceed at this
        /// scrape. Reading the low end rather than the mean is what makes this "the whole fleet moved" instead
        /// of "one pod moved a lot".
        /// </summary>
        private static double FleetFloorRatio(MetricWindow window, double[] medians, List<double> ratios, int t)
        {
            ratios.Clear();

            for (var pod = 0; pod < window.Pods.Count; pod++)
            {
                var value = window.Series(pod, ExcursionChannel)[t];

                if (!double.IsFinite(value) || !double.IsFinite(medians[pod]) || medians[pod] <= 0.0)
                {
                    continue;
                }

                ratios.Add(value / medians[pod]);
            }

            if (ratios.Count < 2)
            {
                return double.NaN;
            }

            ratios.Sort();

            var index = (int)Math.Ceiling(ratios.Count * (1.0 - MinFleetExcursionShare)) - 1;

            return ratios[Math.Max(0, index)];
        }

        /// <summary>
        /// Finite-sample coverage, per channel, for one pod. A channel the pod never reports is skipped: that
        /// is an absent channel, not a gappy one, and <see cref="CheckChannelsNobodyReports"/> owns it.
        /// </summary>
        private static void CheckCoverage(MetricWindow window, int pod, string name, List<Problem> problems)
        {
            for (var m = 0; m < AllChannelsStore.Length; m++)
            {
                var channel = AllChannelsStore[m];
                var series = window.Series(pod, channel);
                var finite = CountFinite(series);

                if (finite == 0 || series.Length == 0)
                {
                    continue;
                }

                var coverage = (double)finite / series.Length;

                if (coverage >= MinChannelCoverage)
                {
                    continue;
                }

                problems.Add(new Problem(
                    $"{name}: {channel} finite in {finite} of {series.Length} scrapes ({Percent(coverage)}), "
                    + $"below {Percent(MinChannelCoverage)}.",
                    [channel]));
            }
        }

        /// <summary>
        /// A channel no replica reports at all. It is not a broken recording — the twelve-replica lab has no
        /// CPU quota, so <see cref="MetricIndex.CpuThrottleRatio"/> is genuinely empty there — but anything
        /// reading that channel off this window reads NaN, so the channel is not usable and the verdict has to
        /// say so out loud rather than let a caller discover it as a flat line.
        /// </summary>
        private static void CheckChannelsNobodyReports(MetricWindow window, List<Problem> problems)
        {
            for (var m = 0; m < AllChannelsStore.Length; m++)
            {
                var channel = AllChannelsStore[m];

                if (window.PodsReporting(channel) > 0)
                {
                    continue;
                }

                problems.Add(new Problem(
                    $"{channel}: not reported by any of the {window.Pods.Count} replicas — the channel is "
                    + "absent from this recording, so anything reading it reads NaN.",
                    [channel]));
            }
        }

        /// <summary>
        /// Flags a window whose opening scrapes are blank across the fleet — the signature of recording a
        /// longer window than the load was running for. Measured per scrape rather than per pod because a
        /// leading gap hits every pod at once, which is exactly what distinguishes it from one bad replica.
        /// </summary>
        private static void CheckLeadingGap(MetricWindow window, List<Problem> problems)
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
                // Every channel: the window itself starts before the load, so nothing recorded in that
                // stretch is the reference workload, whichever channel you read it from.
                problems.Add(new Problem(
                    $"the first {blank} of {window.Length} scrapes have no latency on most pods — the window "
                    + "reaches back before the load started, so it is longer than the traffic behind it.",
                    AllChannelsStore));
            }
        }

        /// <summary>
        /// The operating-point band, and the one place a per-channel verdict earns its keep: everything here
        /// is read off the latency histogram, so everything here is scoped to the latency channels and says
        /// nothing about CPU or traffic in the same recording.
        /// </summary>
        private static void CheckOperatingPoint(
            List<double> healthyP95,
            List<double> faultedP95,
            List<Problem> problems)
        {
            if (healthyP95.Count == 0)
            {
                problems.Add(new Problem(
                    "no healthy replica produced a latency median — nothing to calibrate against.",
                    LatencyChannelsStore));

                return;
            }

            var healthy = Median(healthyP95);

            if (healthy < MinHealthyP95Ms || healthy > MaxHealthyP95Ms)
            {
                problems.Add(new Problem(
                    $"healthy p95 median is {healthy:F0} ms, outside {MinHealthyP95Ms:F0}-{MaxHealthyP95Ms:F0} ms. "
                    + "Too low means the histograms are empty and the quantiles have pinned to bucket edges; "
                    + "too high means the node is saturated. Neither is the reference workload.",
                    LatencyChannelsStore));
            }

            if (faultedP95.Count == 0)
            {
                return;
            }

            var contrast = Median(faultedP95) / healthy;

            if (contrast < MinFaultContrast)
            {
                problems.Add(new Problem(
                    $"the throttled replica is only {contrast:F2}x its healthy siblings, under {MinFaultContrast:F1}x. "
                    + "The usual cause is a saturated node dragging the healthy pods down to it.",
                    LatencyChannelsStore));
            }
        }

        private static MetricIndex[] BuildAllChannels()
        {
            var channels = new MetricIndex[(int)MetricIndex.Count];

            for (var m = 0; m < channels.Length; m++)
            {
                channels[m] = (MetricIndex)m;
            }

            return channels;
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

        /// <summary>
        /// One defect, and the channels it makes unusable.
        ///
        /// <para>The scope is the point. "Healthy p95 median is 49 ms" is a fact about a histogram and says
        /// nothing about the CPU series recorded beside it; a phantom replica poisons every peer comparison
        /// there is. Carrying that distinction is what lets a caller name the channels it read and be told
        /// whether it was entitled to.</para>
        /// </summary>
        public readonly record struct Problem(string Description, IReadOnlyList<MetricIndex> Channels)
        {
            /// <summary>Whether this defect makes <paramref name="channel"/> unusable.</summary>
            public bool Affects(MetricIndex channel) => Channels.Contains(channel);

            /// <inheritdoc/>
            public override string ToString() => Description;
        }

        /// <summary>
        /// A stretch in which essentially the whole fleet left its operating point at once. Scrape indices are
        /// inclusive on both ends.
        /// </summary>
        /// <param name="Channel">Channel it was detected on — see <see cref="ExcursionChannel"/>.</param>
        /// <param name="FirstScrape">First affected sample index.</param>
        /// <param name="LastScrape">Last affected sample index, inclusive.</param>
        /// <param name="FirstAt">Wall-clock time of <paramref name="FirstScrape"/>.</param>
        /// <param name="LastAt">Wall-clock time of <paramref name="LastScrape"/>.</param>
        /// <param name="PodCount">How many pods were compared.</param>
        /// <param name="FleetFloorRatio">
        /// The largest value that <see cref="MinFleetExcursionShare"/> of the fleet exceeded — i.e. how far up
        /// even the least-affected replicas went.
        /// </param>
        /// <param name="PeakRatio">The single highest pod-to-own-median ratio seen in the stretch.</param>
        public readonly record struct FleetExcursion(
            MetricIndex Channel,
            int FirstScrape,
            int LastScrape,
            DateTimeOffset FirstAt,
            DateTimeOffset LastAt,
            int PodCount,
            double FleetFloorRatio,
            double PeakRatio)
        {
            /// <summary>Number of scrapes covered, both ends inclusive.</summary>
            public int ScrapeCount => LastScrape - FirstScrape + 1;

            /// <summary>A one-line account with the numbers in it, for a report or a failing assertion.</summary>
            public string Describe(int windowLength)
                => string.Format(
                    CultureInfo.InvariantCulture,
                    "fleet excursion: scrapes {0}..{1} ({2} of {3}, {4:yyyy-MM-dd'T'HH:mm:ss'Z'}..{5:HH:mm:ss'Z'}): "
                    + "{6} on {7} replicas at once, {8:P0} of them above {9:F1}x their own median and one at "
                    + "{10:F1}x. A fleet-wide excursion is invisible to every per-pod check here, because no "
                    + "replica is an outlier when they all move together.",
                    FirstScrape,
                    LastScrape,
                    ScrapeCount,
                    windowLength,
                    FirstAt.UtcDateTime,
                    LastAt.UtcDateTime,
                    Channel,
                    PodCount,
                    MinFleetExcursionShare,
                    FleetFloorRatio,
                    PeakRatio);
        }

        /// <summary>What <see cref="Excise"/> removed. Reported, never applied silently.</summary>
        /// <param name="ScrapesExcised">Scrapes blanked across every pod and channel.</param>
        /// <param name="TotalScrapes">Length of the window it was applied to.</param>
        /// <param name="Notes">One line per excursion, saying what went and why.</param>
        public readonly record struct ExcisionReport(
            int ScrapesExcised,
            int TotalScrapes,
            IReadOnlyList<string> Notes)
        {
            /// <summary>Share of the window that was blanked.</summary>
            public double Share => TotalScrapes == 0 ? 0.0 : (double)ScrapesExcised / TotalScrapes;

            /// <inheritdoc/>
            public override string ToString()
                => ScrapesExcised == 0
                    ? "nothing excised"
                    : string.Join(
                        "\n  - ",
                        new[]
                        {
                            string.Format(
                                CultureInfo.InvariantCulture,
                                "{0} of {1} scrapes excised ({2:P2}):",
                                ScrapesExcised,
                                TotalScrapes,
                                Share)
                        }.Concat(Notes));
        }

        /// <summary>
        /// What the validator concluded, per channel. No problems means the whole recording is usable; a
        /// caller that reads a subset should ask <see cref="IsUsableForAll"/> about that subset.
        /// </summary>
        /// <param name="Problems">Every defect found, each carrying the channels it invalidates.</param>
        /// <param name="Excursions">Fleet-wide excursions found, for <see cref="Excise"/>.</param>
        public readonly record struct Verdict(
            IReadOnlyList<Problem> Problems,
            IReadOnlyList<FleetExcursion> Excursions)
        {
            /// <summary>Every channel is usable — the strictest reading, and the right one for a fixture gate.</summary>
            public bool IsUsable => Problems.Count == 0;

            /// <summary>Channels no problem touches.</summary>
            public IReadOnlyList<MetricIndex> UsableChannels => Split(true);

            /// <summary>Channels at least one problem touches.</summary>
            public IReadOnlyList<MetricIndex> RejectedChannels => Split(false);

            /// <summary>Whether this recording may be read on <paramref name="channel"/>.</summary>
            public bool IsUsableFor(MetricIndex channel)
            {
                for (var i = 0; i < Problems.Count; i++)
                {
                    if (Problems[i].Affects(channel))
                    {
                        return false;
                    }
                }

                return true;
            }

            /// <summary>
            /// Whether this recording may be read on <i>all</i> of <paramref name="channels"/>. This is the
            /// call that makes "scored on a recording valid for CPU and traffic" checkable: name what you
            /// read, and ignoring a rejected channel you actually use is still wrong.
            /// </summary>
            public bool IsUsableForAll(params MetricIndex[] channels)
            {
                ArgumentNullException.ThrowIfNull(channels);

                for (var i = 0; i < channels.Length; i++)
                {
                    if (!IsUsableFor(channels[i]))
                    {
                        return false;
                    }
                }

                return true;
            }

            /// <summary>Problems affecting <paramref name="channel"/>, and no others.</summary>
            public IReadOnlyList<Problem> ProblemsFor(MetricIndex channel)
                => Problems.Where(p => p.Affects(channel)).ToArray();

            /// <summary>The whole verdict, defects first, then the per-channel split.</summary>
            public string Describe()
            {
                if (IsUsable)
                {
                    return "usable as a calibration reference on every channel";
                }

                var lines = new[] { $"{Problems.Count} problem(s):" }
                    .Concat(Problems.Select(p => p.Description + " [" + Join(p.Channels) + "]"));

                return string.Join("\n  - ", lines)
                       + "\n  usable: " + Join(UsableChannels)
                       + "\n  rejected: " + Join(RejectedChannels);
            }

            /// <summary>The verdict restricted to the channels a caller says it read.</summary>
            public string DescribeFor(params MetricIndex[] channels)
            {
                ArgumentNullException.ThrowIfNull(channels);

                var relevant = Problems.Where(p => channels.Any(p.Affects)).ToArray();

                if (relevant.Length == 0)
                {
                    return "usable as a calibration reference for " + Join(channels);
                }

                return string.Join(
                    "\n  - ",
                    new[] { $"{relevant.Length} problem(s) on {Join(channels)}:" }
                        .Concat(relevant.Select(p => p.Description + " [" + Join(p.Channels) + "]")));
            }

            /// <summary>
            /// Channels on the wanted side of the verdict. A loop rather than a LINQ predicate because a
            /// lambda inside a struct cannot reach <c>this</c> (CS1673).
            /// </summary>
            private IReadOnlyList<MetricIndex> Split(bool usable)
            {
                var picked = new List<MetricIndex>(AllChannelsStore.Length);

                for (var i = 0; i < AllChannelsStore.Length; i++)
                {
                    if (IsUsableFor(AllChannelsStore[i]) == usable)
                    {
                        picked.Add(AllChannelsStore[i]);
                    }
                }

                return picked;
            }

            private static string Join(IReadOnlyList<MetricIndex> channels)
                => channels.Count == AllChannelsStore.Length
                    ? "every channel"
                    : channels.Count == 0
                        ? "none"
                        : string.Join(", ", channels);
        }
    }
}
