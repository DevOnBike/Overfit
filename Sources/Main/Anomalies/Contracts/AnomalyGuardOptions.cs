// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Anomalies.Contracts;
using DevOnBike.Overfit.Anomalies.Monitoring.Abstractions;
using DevOnBike.Overfit.Statistics;

namespace DevOnBike.Overfit.Anomalies.Contracts
{
    /// <summary>
    /// Everything <c>AnomalyGuard</c> needs that is not data: thresholds, topology, and the per-metric floors
    /// only the operator can supply.
    /// </summary>
    public sealed record AnomalyGuardOptions
    {
        /// <summary>Kubernetes namespace every subject is reported under.</summary>
        public string Namespace { get; init; } = string.Empty;

        /// <summary>
        /// Deployment or StatefulSet name. Used for the workload coordinate and as the subject of common-mode
        /// findings, which name no pod on purpose.
        /// </summary>
        public string Workload { get; init; } = string.Empty;

        /// <summary>Peer-comparison thresholds.</summary>
        public PeerOutlierOptions Peer { get; init; } = PeerOutlierOptions.Balanced;

        /// <summary>Trend thresholds.</summary>
        public TrendOptions Trend { get; init; } = TrendOptions.Balanced;

        /// <summary>
        /// Thresholds for the step detector, which runs on the workload's own aggregate.
        ///
        /// <para>It only runs when <see cref="DecomposeCommonMode"/> is on, because the series it judges is
        /// the cross-peer common component. Its absolute floor comes from <see cref="MinAbsoluteGap"/> — the
        /// same "how large a difference in this signal's units matters" question, asked across time rather
        /// than across replicas.</para>
        /// </summary>
        public LevelShiftOptions LevelShift { get; init; } = LevelShiftOptions.Balanced;

        /// <summary>
        /// Consecutive cycles a pod must report nothing before it is called silent. Zero disables the check.
        ///
        /// <para><b>It needs to be more than one, and the reason is not conservatism.</b> A pod created a
        /// moment before a cycle legitimately has no samples yet, and a pod being deleted stops exporting
        /// before the cluster forgets it — both would be reported at one cycle. Two cycles at the default
        /// five-minute cadence is ten minutes of silence, which is longer than either transient and far
        /// shorter than a rollout that has actually failed.</para>
        ///
        /// <para>Requires a topology that implements <see cref="IPodRoster"/>. Without one there is no list
        /// of pods that ought to be reporting, and the check is skipped rather than guessed at.</para>
        /// </summary>
        public int SilentPodCycles { get; init; } = 2;

        /// <summary>
        /// How old <see cref="IPodRoster.KnownPods"/> may be before the silent-pod check declines to use it.
        ///
        /// <para><b>A roster that stopped refreshing looks exactly like one that just did.</b> The refresh
        /// keeps the previous snapshot when Prometheus is unreachable — deliberately, so grouping does not
        /// collapse — and the consequence is that this check would keep comparing the live window against a
        /// list of pods from an hour ago. Every pod deleted since then is reported as silent, and every pod
        /// created since then is not watched at all: a fabricated incident and a missed one, from the same
        /// stale list, with nothing in the output to say so.</para>
        ///
        /// <para>The default is six times the five-minute cadence, so a couple of failed refreshes are
        /// tolerated and a sustained outage is not. <see cref="TimeSpan.Zero"/> disables the freshness gate
        /// and trusts the roster unconditionally; a roster reporting <c>null</c> is unverifiable and is
        /// trusted either way, because refusing there would silently disable the check for every existing
        /// implementation.</para>
        /// </summary>
        public TimeSpan MaxRosterAge { get; init; } = TimeSpan.FromMinutes(30);

        /// <summary>
        /// How long after a pod is created the trend family declines to judge it.
        ///
        /// <para><b>A new pod's metrics rise because it is warming up, not because anything is wrong.</b>
        /// Measured on the lab across a rollout, a manual scale-up and an HPA scale-up: a fresh replica's
        /// working set climbs <b>13-17% of typical over its first 10-20 minutes</b> at a Kendall tau of
        /// 0.70-0.94. That is a textbook trend and it is about nothing. The scale-up phases ran <b>zero quiet
        /// cycles out of seven</b> against four of six for the opposite transition, and the whole asymmetry is
        /// this. Every client deploy and every autoscale event produces one per new pod.</para>
        ///
        /// <para><b>Why not raise the floor instead.</b> On that lab the memory trend floor is 1.089 MB
        /// against pods of ~45 MB, so a floor large enough to swallow a 13% warm-up is large enough to
        /// swallow a real leak. The rise is genuine; what is missing is the knowledge that the subject is new,
        /// which no threshold can express.</para>
        ///
        /// <para><b>The trend family only.</b> Peer comparison keeps judging young pods deliberately: a
        /// replica that differs from its peers <i>right now</i> is worth reporting whatever its age, and
        /// during a rollout every pod is young, so a peer-wide grace would blind the guard exactly when a bad
        /// version is going out. The rule is about a series' history, and only the trend family reads one.</para>
        ///
        /// <para>Requires a topology that reports <see cref="PodPlacement.CreatedAt"/>. Without an age the
        /// pod is judged as before — an unknown age must not be read as "young", or a pod
        /// kube-state-metrics has not caught up with would be silently exempt.
        /// <see cref="TimeSpan.Zero"/> disables the grace.</para>
        /// </summary>
        public TimeSpan WarmUpGrace { get; init; } = TimeSpan.FromMinutes(15);

        /// <summary>
        /// Days of history a workload needs at a given hour before that hour's record is used as a seasonal
        /// expectation. Zero turns the history off entirely.
        ///
        /// <para><b>Two, because one day is a coincidence.</b> A single previous observation cannot say
        /// whether today is unusual or whether yesterday was; the comparison only means something once there
        /// is a spread to compare against. It is deliberately low all the same — a client's first week is
        /// spent in shadow mode, and a baseline that needs a month is a baseline that arrives after the
        /// decision to keep the tool has been made.</para>
        ///
        /// <para>Until the bar is met the guard behaves exactly as it did before history existed: the trend
        /// family judges the raw series. Nothing degrades on day one; it improves on day three.</para>
        /// </summary>
        public int MinimumHistoryDays { get; init; } = 2;

        /// <summary>
        /// Whether a gate with no configured floor falls back to one derived from what a healthy period
        /// actually did.
        ///
        /// <para><b>On by default, because the alternative default is worse.</b> An absent floor means the
        /// gate is off, and a guard with every absolute gate off is the configuration this project measured
        /// at <b>209 false incidents a day</b> on a lab where nothing was wrong. A calibrated floor is not a
        /// guess: it is the largest thing the cluster did while healthy, with a margin.</para>
        ///
        /// <para><b>It never overrides an explicit value</b>, even a lower one. A configured floor is a
        /// decision somebody made and may encode something the data cannot see — "we do not get up for less
        /// than fifty milliseconds" is not a statement about noise. Where it is too low, the guard says so in
        /// its proposal and leaves the fix with the operator.</para>
        ///
        /// <para><b>And it inherits the calibrator's one real hazard.</b> If the observed period contained a
        /// fault, the floor is set above that fault and the guard is blind to it at that size — permanently,
        /// and quietly. Turn this off where the observation period cannot be trusted.</para>
        /// </summary>
        public bool ApplyCalibratedFloors { get; init; } = true;

        /// <summary>
        /// Periods the operator has declared abnormal on purpose — a deployment, a node pool upgrade, a load
        /// test.
        ///
        /// <para>Inside one, findings are still made and reported but carry
        /// <see cref="IncidentLogRecord.SuppressedBy"/>, and <b>nothing observed is folded into the baseline
        /// or the floors</b>. See <see cref="MaintenanceWindow"/> for why the second half matters as much as
        /// the first.</para>
        /// </summary>
        public IReadOnlyList<MaintenanceWindow> MaintenanceWindows { get; init; } = [];

        /// <summary>
        /// Where the absolute floors come from. Null builds the default: configured values first, learned
        /// ones where those are absent.
        ///
        /// <para>Supply one to express a policy no measurement can produce — "a tenth of the container limit",
        /// or "never less than fifty milliseconds" — which is the half of the threshold question that belongs
        /// to the customer.</para>
        /// </summary>
        public IAbsoluteFloorSource? Floors { get; init; }

        /// <summary>
        /// Which moments were declared abnormal on purpose. Null builds one from
        /// <see cref="MaintenanceWindows"/>.
        ///
        /// <para>Supply one to read a deployment pipeline or a change-management system instead — those know
        /// a rollout is happening, and a ConfigMap is a second copy of that truth which somebody has to
        /// remember to update.</para>
        /// </summary>
        public IMaintenanceCalendar? Calendar { get; init; }

        /// <summary>How findings become incidents.</summary>
        public IncidentGroupingOptions Grouping { get; init; } = IncidentGroupingOptions.Balanced;

        /// <summary>
        /// Whether the trend family runs on each pod's <b>residual</b> against the group's common component,
        /// with the common component itself tested once at workload level.
        ///
        /// <para>On by default because the alternative is measurably worse and not in a subtle way: on the
        /// cluster lab, testing pods in isolation made ten of eleven healthy-replica findings the same falling
        /// latency trend on all three pods at once, during warm-up. Turn it off only for a group whose members
        /// are not expected to move together at all — at which point they are not really peers.</para>
        /// </summary>
        public bool DecomposeCommonMode { get; init; } = true;

        /// <summary>
        /// Smallest peer gap worth reporting, per metric, in that metric's own units. Indexed by
        /// <see cref="MetricIndex"/>; entries beyond its length, and a null table, mean the gate is off.
        ///
        /// <para><b>Nobody but the caller can fill this in.</b> One number cannot serve bytes, seconds, ratios
        /// and counts. Left empty, the largest single source of false peer findings on a healthy population
        /// was a GC pause difference of three tenths of a millisecond per second, which cleared a percentage
        /// gate because the metric's whole magnitude is 0.004.</para>
        /// </summary>
        public IReadOnlyList<double>? MinAbsoluteGap
        {
            get; init;
        }

        /// <summary>
        /// Smallest fitted trend change worth reporting, per metric, in that metric's own units. Same
        /// indexing and the same reasoning as <see cref="MinAbsoluteGap"/> — and the case it exists for is a
        /// series sitting at zero, where a relative gate has nothing to be relative to.
        ///
        /// <para>Defaults to <see cref="DefaultAbsoluteTrendFloors"/>. Pass a table of zeroes to turn it off.</para>
        /// </summary>
        public IReadOnlyList<double>? MinAbsoluteTrendChange { get; init; } = DefaultAbsoluteTrendFloors;

        /// <summary>
        /// The one floor that could be measured rather than left to the caller: <b>256 MiB of working set
        /// over a window</b>.
        ///
        /// <para><b>Why memory gets a default when nothing else does.</b> A byte is a byte. Everywhere else in
        /// this family the honest answer is that only whoever chose the signal knows what difference would
        /// make somebody act — three tenths of a millisecond of GC pause is meaningless or serious depending
        /// entirely on the workload. Working set is different: an operator can be asked "would you get up for
        /// this many megabytes in fifteen minutes" and give an answer that does not depend on the
        /// application.</para>
        ///
        /// <para><b>Both bounds are measured, on twenty healthy synthetic pods at the window the shadow run
        /// used.</b> False positives sit at three pods for every floor up to 150 MB and fall to <b>zero at
        /// 200 MB</b>; a 40% leak — about 430 MB on this workload — is still caught at 500 MB. 256 MiB sits
        /// between the two with room on each side, and the margin is what matters rather than the exact
        /// value: anything from roughly 200 to 400 MB behaves identically.</para>
        ///
        /// <para><b>Only working set, deliberately.</b> The gen2 heap produced zero false positives at every
        /// floor including none at all, so it gets no gate — a threshold that is not earned by a measurement
        /// is a threshold that will one day suppress something real for no recorded reason.</para>
        ///
        /// <para>The suppressed findings were <b>true</b>, which is the uncomfortable part and the reason this
        /// is a materiality gate rather than a correctness fix. On the lab the working set genuinely climbed
        /// 7.8% — about 100 MB — monotonically, with a drawdown of exactly zero. The detector was right; a
        /// process that has been serving traffic for twenty minutes is expected to do that, and waking someone
        /// for it teaches them to ignore the next one.</para>
        /// </summary>
        public static IReadOnlyList<double> DefaultAbsoluteTrendFloors { get; } = BuildTrendFloors();

        private static double[] BuildTrendFloors()
        {
            var floors = new double[(int)MetricIndex.Count];

            floors[(int)MetricIndex.MemoryWorkingSetBytes] = 256.0 * 1024 * 1024;

            return floors;
        }

        /// <summary>
        /// Where each pod sits. <b>Supply this.</b> Without it the guard falls back to deriving the workload
        /// from the pod name, which is a heuristic that gets StatefulSets, Jobs and bare pods wrong — and
        /// getting it wrong is not cosmetic: the grouper scores <c>SameWorkload</c> at 0.7 against a 0.35
        /// threshold, so a wrong answer merges unrelated pods into one incident.
        /// </summary>
        public IPodTopology? PodTopology
        {
            get; init;
        }

        /// <summary>
        /// Metrics outside the modelled set that this deployment wants watched, keyed by the name they are
        /// reported under. Evaluated by the rules, peer and trend families; <b>not</b> by the learned one —
        /// <c>MetricSnapshot.FeatureCount</c> is a trained model's input contract and cannot move per
        /// deployment.
        /// </summary>
        public IReadOnlyList<CustomMetricBinding> CustomMetrics { get; init; } = [];

        /// <summary>
        /// How old a saved incident may be and still be adopted after a restart.
        ///
        /// <para><b>A bound is required.</b> A guard restarted after a week would otherwise resurrect
        /// week-old incidents and close them immediately, producing a burst of resolutions for problems
        /// nobody remembers — the alert storm the tracker exists to prevent, with the opposite sign.</para>
        /// </summary>
        public TimeSpan MaxRestoredIncidentAge { get; init; } = TimeSpan.FromHours(2);

        /// <summary>
        /// How far back "now" reaches for the families that answer <i>what is happening</i> — the peer
        /// comparison and the absolute rules. The trend family always uses the whole window it is given.
        ///
        /// <para><b>The two questions need different amounts of time, and using one window for both was a
        /// design error rather than a threshold to tune.</b> A peer comparison asks whether replicas differ
        /// <i>at the same instant</i>: fifteen minutes is plenty, and a longer window only smears a fault
        /// that started recently into the calm before it. A trend asks where a signal is going, and a leak,
        /// a drift or a slow saturation take hours — over fifteen minutes what a trend test actually measures
        /// is fluctuation.</para>
        ///
        /// <para>Measured on the cluster lab, on replicas with nothing wrong with them: latency trends of
        /// <b>−40%</b> and <b>−75% over fifteen minutes</b>, which is not degradation but a pod recovering
        /// from a momentary load spike. No floor removes those, because the movement is real and large; only
        /// a window long enough for it to be the noise it is.</para>
        ///
        /// <para>So the caller supplies the <b>long</b> window — hours — and this trims the tail of it for the
        /// two families that want the present. Left at the default with a fifteen-minute window supplied,
        /// nothing changes for anyone.</para>
        /// </summary>
        public TimeSpan RecentWindow { get; init; } = TimeSpan.FromMinutes(15);

        /// <summary>Thresholds the absolute rules run with; empty disables that family.</summary>
        public IReadOnlyList<RuleProfile> Rules { get; init; } = DefaultRules;

        /// <summary>
        /// The rules worth running out of the box, each for a reason the relative methods cannot cover.
        ///
        /// <para>CPU throttling because the CFS counters exist only on containers carrying a limit, so the
        /// peer group can hold exactly one member — on precisely the pod being throttled. OOM kills and
        /// restarts because a single event produces a non-zero rate over a small share of the window, which
        /// Cliff's delta scores below any usable effect size: <b>the peer detector is structurally blind to
        /// one OOMKill</b>, measured, and only a rule catches it.</para>
        /// </summary>
        public static IReadOnlyList<RuleProfile> DefaultRules
        {
            get;
        } =
        [
            new(MetricIndex.CpuThrottleRatio, SustainedThresholdOptions.ForCpuThrottling),
            new(MetricIndex.OomEventsRate, SustainedThresholdOptions.ForRareEvent),
            new(MetricIndex.ContainerRestarts, SustainedThresholdOptions.ForRareEvent),
        ];

        /// <summary>
        /// The ceiling each metric is heading towards, per metric, in that metric's own units — a container
        /// memory limit, a disk capacity, an SLO. Absent entries mean no projection.
        ///
        /// <para><b>This turns a trend from an observation into something someone can act on.</b>
        /// <see cref="TrendDetector"/> has always computed a time-to-limit and put it in the finding's own
        /// words; the deployed path passed <c>NaN</c> for the limit, so it never had one to project against
        /// and the capability was dead. "Working set rose by 11% of typical" and "working set reaches its
        /// limit in 40 minutes" are the same measurement, and only one of them tells an operator whether to
        /// get up.</para>
        ///
        /// <para><b>Static and per metric, matching the two floor tables above.</b> A container's real limit
        /// varies per pod and could be read from <c>kube_pod_container_resource_limits</c>, which would be
        /// better and is not what this is; one number per metric is what an operator can state in a config
        /// file today, and a wrong-by-a-factor projection is still worth more than none. Reading the true
        /// per-pod limit is the obvious next step and is deliberately not pretended at here.</para>
        ///
        /// <para>Left empty, nothing changes: no projection is attempted and the finding reads exactly as it
        /// did before.</para>
        /// </summary>
        public IReadOnlyList<double>? SaturationLimit
        {
            get; init;
        }

        /// <summary>
        /// Looks up a per-metric ceiling. <see cref="double.NaN"/> when absent, which is what
        /// <c>TrendDetector.Detect</c> reads as "do not project" — distinct from the floor lookup below,
        /// where a missing entry means zero and therefore "gate off".
        /// </summary>
        public static double LimitFor(IReadOnlyList<double>? table, MetricIndex metric)
        {
            var index = (int)metric;

            if (table is null || index < 0 || index >= table.Count)
            {
                return double.NaN;
            }

            var value = table[index];

            return double.IsFinite(value) && value > 0.0 ? value : double.NaN;
        }

        /// <summary>Looks up a per-metric floor, treating a short or absent table as "gate off".</summary>
        public static double FloorFor(IReadOnlyList<double>? table, MetricIndex metric)
        {
            var index = (int)metric;

            if (table is null || index < 0 || index >= table.Count)
            {
                return 0.0;
            }

            var value = table[index];

            return double.IsFinite(value) && value > 0.0 ? value : 0.0;
        }
    }
}
