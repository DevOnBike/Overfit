// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Anomalies.Contracts;

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
    /// <para><b>Validated against the lab, and corrected twice by it</b>
    /// (<see cref="Anomalies.Diagnostics.SyntheticClusterRealismDiagnostics"/>). The first version had each pod drawing
    /// its own diurnal phase — replicas of one Deployment serve the same traffic at the same instant, so that
    /// alone pushed between-pod spread to 8.4% against a measured 4.7%. It also gave pods +-4% of internal
    /// scatter where the lab showed 52%, which is the more damaging error: Cliff's delta measures overlap, so
    /// unrealistically quiet pods separate cleanly and produce findings no real replica would.</para>
    ///
    /// <para>Both errors inflated the false-positive rate, so any figure measured before this correction is an
    /// upper bound rather than an estimate.</para>
    ///
    /// <para>Deterministic for a given seed, so a false-positive count is reproducible and a regression in it
    /// is attributable.</para>
    /// </summary>
    public sealed class SyntheticCluster
    {
        /// <summary>Features per pod — the <see cref="MetricSnapshot"/> contract.</summary>
        public const int MetricCount = (int)MetricIndex.Count;

        /// <summary>
        /// What a freshly started process holds before its caches fill, as a fraction of its settled working
        /// set. Not near-zero: the runtime, the loaded assemblies and the JIT-compiled code are there the
        /// moment the process serves its first request.
        /// </summary>
        private const double ColdStartFraction = 0.35;

        /// <summary>
        /// How much of the remaining gap to the settled working set a warm-up closes per sample. 0.154 at a
        /// 15 s scrape is a ~90 s time constant, so a restarted pod is within 5% of its siblings after about
        /// four and a half minutes.
        ///
        /// <para><b>This is a different rate from the allocation rate, and that distinction is the whole
        /// point.</b> Filling a working set is assemblies, JIT and caches populating as traffic arrives;
        /// filling a sawtooth is the process allocating garbage. The first version used the second rate for
        /// both and produced a 3.9-hour ramp — see the restart handling in <c>Generate</c>.</para>
        /// </summary>
        private const double WarmupRatePerSample = 0.154;

        /// <summary>
        /// Scrapes between one working-set reclaim and the next.
        ///
        /// <para><b>Searchable, and the reason is the retracement column.</b> This was pinned at 57 — one
        /// cycle per detector window — which made every generated window a complete sawtooth that gave back
        /// everything it gained. The lab does the opposite: three of its four replicas retraced <b>0.0%</b>
        /// over a twelve-minute window and the fourth 17.5%, so roughly one window in four contains a reclaim
        /// and the rest see nothing but climb.</para>
        ///
        /// <para>Range and interquartile spread could not see that difference — both are computed on sorted
        /// values, so a monotone climb and a sawtooth of equal amplitude score identically, and the generator
        /// sat within a tenth of a percent of the lab on both while producing the opposite signal in time.
        /// The consequence was not cosmetic: Mann-Kendall keys on monotonicity, so a generator that resets
        /// every window cannot be used to measure trend false positives at all.</para>
        /// </summary>
        internal const double MemoryCycleSamples = 92.0;

        // ---------------------------------------------------------------------------------------------
        // Scatter, FITTED to the recorded lab window by SyntheticClusterCalibrationSearch rather than chosen
        // or hand-tuned. The values below are that search's output: re-run it after any change to the fixture
        // or to the structure around them, because editing one by hand silently un-fits the rest.
        //
        // Comparison is against healthy replicas only, over a matched window length
        // (SyntheticClusterRealismDiagnostics prints the table). Each figure is the full width of a uniform
        // draw, so the interquartile spread it produces is roughly half of it — and the interquartile spread
        // is what the objective weighs, because (max - min) over a window is decided by its two most extreme
        // scrapes.
        // ---------------------------------------------------------------------------------------------

        /// <summary>
        /// Per-scrape scatter of each latency quantile, as the full width of a multiplicative draw.
        ///
        /// <para><b>Three separate draws, and that is the correction.</b> The quantiles used to be one series
        /// scaled by a constant — p50 = 0.35x, p99 = 2.4x — which forces all three to have identical relative
        /// scatter. The lab does not: interquartile spreads of <b>14% at p50, 58% at p95, 35% at p99</b>. No
        /// multiplier can produce that shape, so the previous generator was wrong in kind rather than in
        /// degree, and a peer or trend detector reading p50 and p95 saw two copies of one signal where the
        /// real thing gives two.</para>
        ///
        /// <para>A high quantile is a coarse order statistic over a few dozen requests per scrape and jumps
        /// accordingly; the median of the same requests barely moves. That p99 scatters <i>less</i> than p95
        /// is not a typo — <c>histogram_quantile</c> interpolates inside a bucket, and the top buckets are
        /// wide, so the tail is quantised.</para>
        /// </summary>
        internal const double LatencyScatterP50 = 0.2804;

        /// <inheritdoc cref="LatencyScatterP50"/>
        internal const double LatencyScatterP95 = 1.15;

        /// <inheritdoc cref="LatencyScatterP50"/>
        /// <remarks>
        /// Wider than the spread it is meant to produce, because the monotonicity clamp against p95 truncates
        /// its lower tail — p95 scatters harder, so <c>max(p99, p95)</c> bites often enough to compress p99's
        /// interquartile spread well below the width of its own draw.
        /// </remarks>
        internal const double LatencyScatterP99 = 0.6783;

        /// <summary>
        /// Scatter of the per-pod request rate. Measured interquartile spread on the lab is 13.6% — a
        /// <c>rate()</c> over a two-minute window, on one replica behind a load balancer that redistributes
        /// continuously, is not a smooth number.
        /// </summary>
        internal const double TrafficScatter = 0.2643;

        /// <summary>
        /// Scatter of CPU beyond what traffic already explains. The affine cost model carries the load-driven
        /// part; this is everything else the process does — background work, timers, collections.
        /// </summary>
        internal const double CpuScatter = 0.34;

        /// <summary>
        /// How often a scrape lands on a burst, and by how much it lifts the value.
        ///
        /// <para><b>Uniform scatter alone cannot reproduce the lab, and the ratio is how it shows.</b> A
        /// uniform draw has a range of almost exactly twice its interquartile spread. The lab's CPU has a
        /// range <b>3.3x</b> its interquartile spread and its request rate <b>3.9x</b> — fat tails, meaning a
        /// few scrapes far outside a body that is otherwise tight. Widening the uniform draw until the range
        /// matched would have inflated the body along with it, and the body is what Cliff's delta reads.</para>
        ///
        /// <para>Rare enough to sit outside the middle half of a window — so it stretches the range and leaves
        /// the interquartile spread alone, which is the shape being reproduced.</para>
        ///
        /// <para><b>Fixed, not searched, because it was measured to be unidentifiable.</b> Ten restarts of the
        /// calibration search landed on values spanning <b>70% of its allowed range</b> — 0.082 to 0.291 —
        /// with the near-optimal ones alone still spread fourfold. The reason is structural rather than a
        /// shortcoming of the search: over a window of a few dozen scrapes, once a burst is near-certain to
        /// occur at all, the range is set by how far it lifts the value and not by how often it happens; and
        /// below a quarter of the samples it never reaches the interquartile spread. Everything between those
        /// two points is a plateau, so probability and magnitude are only identifiable as a product.</para>
        ///
        /// <para>Leaving both free let the fitter hand back a precise-looking number carrying no information.
        /// The magnitude is what the data can pin, so the burst factors stay searchable and this stays put —
        /// one scrape in sixteen, about one every four minutes at a 15 s scrape, which is what a stall from a
        /// collection or a scheduling hiccup plausibly looks like. Collapsing it cost <b>nothing</b>: the
        /// search reaches the same score without it.</para>
        /// </summary>
        private const double BurstProbability = 0.0631;

        /// <inheritdoc cref="BurstProbability"/>
        internal const double TrafficBurstFactor = 1.339;

        /// <inheritdoc cref="BurstProbability"/>
        internal const double CpuBurstFactor = 1.305;

        /// <summary>
        /// Per-pod latency personality, as the full width of a uniform draw.
        ///
        /// <para><b>Deliberately below the 17% between-pod spread the lab shows on p95</b>, because that
        /// figure is not all personality. A pod's median over a 49-sample window is itself an estimate, and
        /// with p95 scattering as hard as it does the standard error of that estimate is several percent —
        /// so part of the measured between-pod spread is the noise of measuring it. Setting the offset to the
        /// observed spread would count that noise twice and hand the peer detector a group that genuinely is
        /// more separated than any real deployment.</para>
        /// </summary>
        private const double LatencyOffsetWidth = 0.14;

        /// <summary>
        /// How long a queueing episode adds to every request in that scrape, as a fraction of the latency
        /// level. Shares <see cref="BurstProbability"/> with the other bursts.
        ///
        /// <para><b>Additive, not multiplicative, and the search is what proved this had to exist.</b> The
        /// lab's p50 has a range 3.1x its interquartile spread — a fat tail — and a uniform draw gives
        /// exactly 2.0. With only its own width to move, the fitter could buy p50's range only by inflating
        /// its interquartile spread from 14% to 20%, wrecking the statistic the peer detector actually reads.
        /// A missing degree of freedom shows up as a bad trade, not as a bad number.</para>
        ///
        /// <para>The reason it must be additive is arithmetic, not taste. A stall adds the same wait to every
        /// request, so it moves each quantile by the same absolute amount — which is a large <i>relative</i>
        /// move for the median and a small one for the tail, because they sit at 0.35x and 2.4x of the same
        /// level. One term therefore fattens p50 by roughly seven times as much as p99, which is the shape
        /// the lab shows. A multiplicative burst would lift all three equally and overshoot p95 and p99,
        /// which already match.</para>
        /// </summary>
        internal const double LatencyBurstDelay = 0.0777;

        /// <summary>
        /// Height of the working-set sawtooth: how far allocation carries it above the settled floor before
        /// gen2 takes it back.
        ///
        /// <para><b>A parameter, and it is worth recording that it was first mistaken for a missing
        /// mechanism.</b> Memory was the one channel the fitter could not reach — 2.5% interquartile spread
        /// against the lab's 3.7%, 5.7% range against 7.8% — and the conclusion drawn was that some mechanism
        /// was absent, as had genuinely been the case for p50 and for the heap staircase.</para>
        ///
        /// <para>The ratio said otherwise. A uniform sawtooth has a range of exactly twice its interquartile
        /// spread; the generator sat at 2.28 and the lab at 2.11. <b>Both are uniform sawtooths and only the
        /// amplitude differed</b> — the shape was already right. That is the diagnostic worth keeping: the
        /// range-to-interquartile ratio tells you whether the mechanism is right, and the level tells you
        /// whether the number is. A missing mechanism shows up as a wrong ratio, which is what p50's 3.1
        /// against a uniform 2.0 was, and what CPU's 3.3 was.</para>
        ///
        /// <para>The value here was hardcoded at 0.06 with no source; it is now fitted like everything else
        /// around it.</para>
        /// </summary>
        internal const double MemorySawtoothAmplitude = 0.1491;

        /// <summary>
        /// How far gen2 rises above its settled level once a promotion has happened, before the next
        /// collection takes it back. Sized against the lab's 11.6% within-pod range on a heap whose
        /// interquartile spread is exactly zero — a step, held.
        /// </summary>
        internal const double HeapPromotionStep = 0.1155;

        private readonly double[][] _series;
        private readonly double _diurnalPhase;
        private readonly SyntheticClusterShape _shape;

        /// <param name="pods">Replicas in the deployment.</param>
        /// <param name="hours">Wall-clock hours to generate.</param>
        /// <param name="scrapeSeconds">Scrape interval; 15 s matches the lab's Prometheus.</param>
        /// <param name="seed">Any value; the same seed reproduces the same cluster exactly.</param>
        /// <param name="restartsPerPodPerDay">
        /// Restarts to inject, as a rate. Exists so restarts can be <b>ablated</b> rather than argued about:
        /// a restarted pod's memory ramp is a different mechanism from the GC sawtooth, and the only way to
        /// tell which one a false-positive count comes from is to turn one of them off and measure again.
        /// Zero disables them; the default is a quiet-cluster rate, not a broken-cluster one.
        /// </param>
        /// <param name="shape">
        /// Scatter parameters. Omit for <see cref="SyntheticClusterShape.Measured"/>, the values calibrated
        /// against the recorded lab window; supply one only to search them (docs/autoresearch-program.md).
        /// </param>
        public SyntheticCluster(
            int pods,
            double hours,
            double scrapeSeconds = 15.0,
            int seed = 20260729,
            double restartsPerPodPerDay = 1.0,
            SyntheticClusterShape? shape = null)
        {
            _shape = shape ?? SyntheticClusterShape.Measured;

            ArgumentOutOfRangeException.ThrowIfLessThan(pods, 3);
            ArgumentOutOfRangeException.ThrowIfLessThanOrEqual(hours, 0.0);
            ArgumentOutOfRangeException.ThrowIfLessThanOrEqual(scrapeSeconds, 0.0);
            ArgumentOutOfRangeException.ThrowIfNegative(restartsPerPodPerDay);

            Pods = pods;
            ScrapeSeconds = scrapeSeconds;
            RestartsPerPodPerDay = restartsPerPodPerDay;
            Samples = (int)(hours * 3600.0 / scrapeSeconds);

            _series = new double[pods * MetricCount][];

            var rng = new Random(seed);

            // ONE phase for the whole deployment. Drawing it per pod was a modelling error: replicas of one
            // Deployment serve the same traffic at the same instant, so they do not have independent daily
            // curves. Measured consequence — it was the dominant source of between-pod spread, pushing the
            // generator to 8.4% against the lab's 4.7% across three replicas.
            _diurnalPhase = rng.NextDouble();

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

        /// <summary>Restart rate this cluster was generated with; zero means none were injected.</summary>
        public double RestartsPerPodPerDay
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
            var latencyOffset = 1.0 + ((rng.NextDouble() - 0.5) * LatencyOffsetWidth);
            var cpuOffset = 1.0 + ((rng.NextDouble() - 0.5) * 0.33);       // +-16.5% -> ~33% spread across a group
            var memoryBaseline = 1.15e9 * (1.0 + ((rng.NextDouble() - 0.5) * 0.05));
            var trafficShare = (1.0 / Pods) * (1.0 + ((rng.NextDouble() - 0.5) * 0.18));

            // Scrapes are not synchronised, but the difference is seconds, not hours — a fraction of a percent
            // of a daily period, not a fifth of one.
            var diurnalPhase = _diurnalPhase + ((rng.NextDouble() - 0.5) * 0.002);

            var samplesPerDay = 86400.0 / ScrapeSeconds;

            // One restart per pod per day is a quiet cluster, not a broken one: image updates, node drains,
            // evictions. Placed away from the very start so a window can contain the ramp.
            // Drawn UNCONDITIONALLY, then discarded if restarts are ablated. Skipping the draw would shift
            // every subsequent random value for this pod and all later ones, so the "without restarts" arm
            // would be a different cluster rather than the same cluster minus restarts — which is not an
            // ablation, it is two unrelated populations. This cost one wrong reading already: an ablated arm
            // came out *worse* than the arm it was supposed to be a subset of.
            var restartDraw = Samples > 200 ? rng.Next(Samples / 5, Samples) : int.MaxValue;
            var restartAt = RestartsPerPodPerDay > 0.0 ? restartDraw : int.MaxValue;
            var memory = memoryBaseline;

            // The level the sawtooth rides on. Equal to the baseline except while a restarted pod warms up.
            var floor = memoryBaseline;

            // Gen2 is a staircase, not a curve: it moves when gen2 is collected and holds flat in between.
            // The lab makes this unmistakable — the interquartile spread of the heap over a twelve-minute
            // window is exactly 0.0%, because the middle half of those scrapes are the same number. Deriving
            // it from the working set, as this did, gave it the working set's continuous jitter and turned a
            // step into noise the trend detector then had to reject.
            // The fraction of the working set that is gen2 is a property of the application, so it is drawn
            // once per pod rather than per step. Drawing it per step instead put every replica on its own
            // level and pushed between-pod spread to 12% at three pods and 69% at twenty, against 2.8%
            // measured — a staircase with the right step shape and a fabricated peer group.
            var heapShare = 0.42 * Scatter(rng, 0.06);

            for (var t = 0; t < Samples; t++)
            {
                var timeOfDay = ((t / samplesPerDay) + diurnalPhase) % 1.0;

                // Traffic: a daily curve between roughly a fifth and full load, never zero — a cluster with no
                // traffic exercises none of the load-sensitive paths.
                var diurnal = 0.6 + (0.4 * Math.Sin(2.0 * Math.PI * timeOfDay));

                // Every draw below is unconditional, for the reason spelled out at restartDraw: a draw made
                // inside an `if` shifts the stream for everything after it, so an ablated arm stops being the
                // same cluster minus one feature.
                var trafficJitter = Scatter(rng, _shape.TrafficScatter);
                var trafficBurst = rng.NextDouble() < BurstProbability ? _shape.TrafficBurstFactor : 1.0;
                var traffic = 40.0 * trafficShare * diurnal * trafficJitter * trafficBurst;

                // Queueing: latency rises with load, sub-linearly. Each quantile then scatters on its own —
                // see LatencyScatterP50 for why one shared series was wrong in kind, not in degree.
                var latencyLevel = 700.0 * latencyOffset * (1.0 + (0.35 * diurnal));

                // A queueing episode: something stalls for a moment and every request in that scrape waits
                // behind it. ADDITIVE, and that is the whole point — see LatencyBurstDelay.
                var queueing = rng.NextDouble() < BurstProbability
                    ? latencyLevel * _shape.LatencyBurstDelay
                    : 0.0;

                var p50 = (latencyLevel * 0.35 * Scatter(rng, _shape.LatencyScatterP50)) + queueing;
                var p95 = (latencyLevel * Scatter(rng, _shape.LatencyScatterP95)) + queueing;
                var p99 = (latencyLevel * 2.4 * Scatter(rng, _shape.LatencyScatterP99)) + queueing;

                // Rare scrape artefact. The reason every estimator here is rank-based rather than
                // least-squares. It hits the whole histogram at once, because the slow requests behind it are
                // in every quantile's population.
                var artefact = rng.NextDouble() < 0.003 ? 5.0 : 1.0;

                p50 *= artefact;
                p95 *= artefact;
                p99 *= artefact;

                // A histogram cannot report a smaller value at a higher quantile. With independent draws the
                // bands can cross, and a crossed sample is not a rare event worth modelling — it is impossible
                // output that would teach every detector downstream something false.
                p95 = Math.Max(p95, p50);
                p99 = Math.Max(p99, p95);

                // A restart drops the working set to a cold process and the warm-up brings it back. The two
                // rates below are DIFFERENT rates, and conflating them was a measured bug: the first version
                // refilled a restarted pod at the allocation rate, so recovery from 30 MB took 933 samples —
                // 3.9 hours — leaving 3.2 of 20 pods permanently mid-ramp, each hundreds of megabytes below
                // its siblings and climbing monotonically. That is a perfect trend signal and a perfect peer
                // outlier, both entirely manufactured: ablating restarts removed 60% of all incidents and
                // 93% of every trend finding.
                if (t == restartAt)
                {
                    floor = memoryBaseline * ColdStartFraction;
                    memory = floor;
                }

                // Warm-up: the working set is dominated by assemblies, JIT-compiled code and caches, and those
                // populate in minutes as traffic arrives — not at the rate the process allocates garbage.
                floor += (memoryBaseline - floor) * WarmupRatePerSample;

                // Allocation between collections, riding on top of whatever the floor currently is.
                memory += floor * _shape.MemorySawtoothAmplitude / _shape.MemoryCycleSamples
                          * (1.0 + ((rng.NextDouble() - 0.5) * 0.4));

                // Drawn before the branch that uses it, so the stream does not depend on whether gen2 ran.
                if (memory > floor * (1.0 + _shape.MemorySawtoothAmplitude))
                {
                    memory = floor;
                }

                if (memory < floor)
                {
                    memory = floor;
                }

                // Gen2 in two discrete levels: a promotion lifts it, a collection drops it back, and it holds
                // flat in between. Both halves of the lab's measurement have to come out of this — a within-pod
                // range near 12% AND replicas sitting 2.8% apart. An earlier version drew a fresh level at each
                // collection, which reproduced the flatness and nothing else: with barely one collection inside
                // a twelve-minute window, each replica simply held whatever it drew, so the step size became
                // the between-pod spread (12% at three pods, 67% at twenty).
                // Only the top of the sawtooth, so the settled level is where most scrapes land: the lab's
                // interquartile spread is zero, meaning the middle half of the window is one number. A
                // threshold at the midpoint splits the window evenly instead and turns the step into the
                // interquartile spread — the flatness is the measurement, not the step.
                var promoted = (memory - floor) > (floor * _shape.MemorySawtoothAmplitude * (5.0 / 6.0))
                    ? _shape.HeapPromotionStep
                    : 0.0;

                Set(pod, MetricIndex.GcGen2HeapBytes, t, floor * heapShare * (1.0 + promoted));

                Set(pod, MetricIndex.RequestsPerSecond, t, traffic);
                Set(pod, MetricIndex.LatencyP50Ms, t, p50);
                Set(pod, MetricIndex.LatencyP95Ms, t, p95);
                Set(pod, MetricIndex.LatencyP99Ms, t, p99);

                // Affine, not proportional: a fixed floor plus a marginal cost per request. This is the
                // measured model, and it is why unit cost carries a residue that grows with imbalance.
                var cpuBurst = rng.NextDouble() < BurstProbability ? _shape.CpuBurstFactor : 1.0;

                Set(pod, MetricIndex.CpuUsageRatio, t,
                    cpuOffset * (0.45 + (0.040 * traffic)) * Scatter(rng, _shape.CpuScatter) * cpuBurst);

                Set(pod, MetricIndex.MemoryWorkingSetBytes, t, memory);
                Set(pod, MetricIndex.GcPauseRatio, t, 0.004 * diurnal * (1.0 + ((rng.NextDouble() - 0.5) * 0.6)));

                // Mostly empty, occasionally a couple of items — never a starvation signal.
                Set(pod, MetricIndex.ThreadPoolQueueLength, t, rng.NextDouble() < 0.05 ? rng.Next(1, 4) : 0.0);

                // Healthy means these are zero. Exercising the zero-scale path is the point.
                Set(pod, MetricIndex.OomEventsRate, t, 0.0);
                Set(pod, MetricIndex.ErrorRate, t, 0.0);

                // No pod here carries a CPU limit, so CFS accounting does not exist for any of them.
                Set(pod, MetricIndex.CpuThrottleRatio, t, double.NaN);

                // increase() over the window, so a restart reads as 1 for as long as it stays inside the
                // window and 0 afterwards. Modelling it matters: a generator that left this flat at zero would
                // be describing a cluster where pods never restart, which is not the healthy case — it is a
                // fictional one, and the false-positive rate measured against it would be worth nothing.
                var sinceRestart = t - restartAt;
                Set(pod, MetricIndex.ContainerRestarts, t, sinceRestart is >= 0 and < 80 ? 1.0 : 0.0);
            }

            PunchScrapeGaps(pod, rng);
        }

        /// <summary>
        /// Drops isolated samples to <see cref="double.NaN"/>. Prometheus misses scrapes — a busy target, a
        /// restarting exporter, a network hiccup — and "missing is not zero" is a decision the whole pipeline
        /// rests on, so the holes have to be here for the measurement to mean anything.
        /// </summary>
        /// <summary>
        /// A multiplicative draw of the given full width, centred on 1. Uniform rather than Gaussian: the body
        /// of these signals is bounded, and the tail is modelled explicitly as a burst instead of being left
        /// to a distribution's shape — see <see cref="BurstProbability"/>.
        /// </summary>
        private static double Scatter(Random rng, double width) => 1.0 + ((rng.NextDouble() - 0.5) * width);

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
