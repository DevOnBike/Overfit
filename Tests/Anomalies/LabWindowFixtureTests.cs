// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Anomalies.Contracts;
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
    ///
    /// <para><b>Every recording present is gated, not one named recording.</b> The previous version read
    /// <c>LabWindowFixture.Path</c>, which defaults to <c>lab-window.csv</c> — so for the whole life of the
    /// second checked-in recording, the gate that runs on every <c>dotnet test</c> never looked at it. Pointed
    /// at it, unmodified, it fails: <c>lab-window-healthy-12pod.csv</c> is the reference a published
    /// affine-trend table was scored on, and its latency channels are histogram-bucket geometry. Enumerating
    /// the directory is the difference between a gate and a gate on one file.</para>
    /// </summary>
    public sealed class LabWindowFixtureTests
    {
        private static readonly DateTimeOffset T0 = new(2026, 7, 31, 12, 0, 0, TimeSpan.Zero);

        /// <summary>
        /// What each recording present on this box is good for. <b>Checked in, so a quarantine is a written
        /// claim rather than a silence.</b>
        ///
        /// <para>An entry is required for every file found, and it is asserted in both directions: a channel
        /// declared usable must validate, and a channel declared rejected must <i>still</i> be rejected. The
        /// second half is what stops the table rotting — re-record the lab with finer histogram buckets and
        /// this test tells you to delete the quarantine rather than letting a stale one hide a fresh defect.
        /// </para>
        /// </summary>
        private static readonly FixtureExpectation[] Expected =
        [
            new FixtureExpectation(
                "lab-window.csv",
                CheckedIn: true,
                Excursions: 0,
                Rejected: [],
                Reason: "the four-replica overfit-server window this project calibrated against; clean on "
                        + "every channel, which is why it is the default."),

            new FixtureExpectation(
                "lab-window-healthy-12pod.csv",
                CheckedIn: true,
                Excursions: 0,
                Rejected:
                [
                    MetricIndex.CpuThrottleRatio,
                    MetricIndex.LatencyP50Ms,
                    MetricIndex.LatencyP95Ms,
                    MetricIndex.LatencyP99Ms
                ],
                Reason: "QUARANTINE. The three latency channels are bucket geometry, not measurement: "
                        + "labapp_request_duration_seconds has le buckets at 5/10/25/50/100/250 ms and "
                        + "higher, every request lands in (25, 50], so histogram_quantile returns "
                        + "25 + q x 25 — p50 37.50, p95 48.75, p99 49.75, identical on all twelve pods and "
                        + "exact in 2172 of this file's 2892 samples per quantile (75%). "
                        + "Three channels carrying one bit. Only finer buckets or more load fix it; "
                        + "re-recording does not. CpuThrottleRatio is simply absent — the lab-workload pods "
                        + "carry no CPU quota. CPU, traffic, memory and GC are sound and this recording may "
                        + "be scored on those."),

            new FixtureExpectation(
                "lab-window-healthy-12pod-36h.csv",
                CheckedIn: false,
                Excursions: 3,
                Rejected:
                [
                    MetricIndex.CpuThrottleRatio,
                    MetricIndex.LatencyP50Ms,
                    MetricIndex.LatencyP95Ms,
                    MetricIndex.LatencyP99Ms
                ],
                Reason: "NOT CHECKED IN — 7.54 MB, 8641 scrapes, deliberately left out of git; the entry "
                        + "exists so a local copy is gated rather than ignored. Same bucket geometry as the "
                        + "60-minute recording, plus three fleet excursions on 2026-08-07 between 08:25:30Z "
                        + "and 08:37:00Z, when a test suite running on the cluster's host took all twelve "
                        + "replicas to 90x their median CPU at the traffic trough. Those scrapes move "
                        + "corr(fleet traffic, fleet CPU) from +0.47 to -0.05, and they are excised here."),
        ];

        private readonly ITestOutputHelper _output;

        public LabWindowFixtureTests(ITestOutputHelper output)
        {
            _output = output;
        }

        /// <summary>Every recording found in the fixture directory, by file name.</summary>
        public static TheoryData<string> RecordingsPresent()
        {
            var data = new TheoryData<string>();

            foreach (var path in LabWindowFixture.AllPaths())
            {
                data.Add(Path.GetFileName(path));
            }

            return data;
        }

        [Theory]
        [MemberData(nameof(RecordingsPresent))]
        public void EveryRecordingIsUsableForTheChannelsItDeclares(string fileName)
        {
            var expectation = Find(fileName);

            Assert.True(
                expectation is not null,
                $"{fileName} is an undeclared recording. Add an entry to {nameof(Expected)} saying which "
                + "channels it may be scored on and why any are quarantined — an ungated recording in this "
                + "directory is exactly how lab-window-healthy-12pod.csv went unchecked.");

            var path = Path.Combine(LabWindowFixture.Directory, fileName);
            var (window, faulted) = LabWindowFixture.Load(path);
            var excursions = LabWindowValidator.FindFleetExcursions(window);

            Assert.True(
                excursions.Count == expectation!.Excursions,
                $"{fileName}: {excursions.Count} fleet excursion(s), declared {expectation.Excursions}."
                + Describe(excursions, window.Length));

            var report = LabWindowValidator.Excise(window, excursions);
            _output.WriteLine($"{fileName}: {report}");

            var verdict = LabWindowValidator.Validate(window, faulted);

            for (var i = 0; i < LabWindowValidator.AllChannels.Count; i++)
            {
                var channel = LabWindowValidator.AllChannels[i];
                var quarantined = Array.IndexOf(expectation.Rejected, channel) >= 0;

                if (!quarantined)
                {
                    Assert.True(
                        verdict.IsUsableFor(channel),
                        $"{fileName} is declared usable for {channel} and is not — "
                        + verdict.DescribeFor(channel));

                    continue;
                }

                Assert.False(
                    verdict.IsUsableFor(channel),
                    $"{fileName} quarantines {channel}, but the validator now passes it. If the recording "
                    + "was fixed, delete the quarantine; a stale one hides the next real defect. Stated "
                    + $"reason was: {expectation.Reason}");
            }
        }

        /// <summary>
        /// Catches a checked-in recording that was renamed or deleted while its declaration stayed behind —
        /// the failure mode the theory above cannot see, because it only enumerates what is there.
        /// </summary>
        [Fact]
        public void EveryDeclaredCheckedInRecordingIsPresent()
        {
            var present = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

            foreach (var path in LabWindowFixture.AllPaths())
            {
                present.Add(Path.GetFileName(path));
            }

            foreach (var expectation in Expected)
            {
                if (!expectation.CheckedIn)
                {
                    continue;
                }

                Assert.True(
                    present.Contains(expectation.FileName),
                    $"{expectation.FileName} is declared as checked in but is not in "
                    + $"{LabWindowFixture.Directory}. Either it moved and the declaration did not, or it was "
                    + "dropped and this table is now describing a file nobody has.");
            }
        }

        private static FixtureExpectation? Find(string fileName)
        {
            foreach (var expectation in Expected)
            {
                if (string.Equals(expectation.FileName, fileName, StringComparison.OrdinalIgnoreCase))
                {
                    return expectation;
                }
            }

            return null;
        }

        private static string Describe(
            IReadOnlyList<LabWindowValidator.FleetExcursion> excursions,
            int windowLength)
        {
            var text = string.Empty;

            for (var i = 0; i < excursions.Count; i++)
            {
                text += "\n  - " + excursions[i].Describe(windowLength);
            }

            return text;
        }

        /// <summary>One recording, and the claim this suite makes about it.</summary>
        /// <param name="FileName">File name inside the fixture directory.</param>
        /// <param name="CheckedIn">Whether git carries it — a local-only recording must not be required.</param>
        /// <param name="Excursions">Fleet excursions expected before excision.</param>
        /// <param name="Rejected">Channels that must still fail validation after excision.</param>
        /// <param name="Reason">Why, in enough detail to decide whether it is still true.</param>
        private sealed record FixtureExpectation(
            string FileName,
            bool CheckedIn,
            int Excursions,
            MetricIndex[] Rejected,
            string Reason);

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
            Assert.Contains(verdict.Problems, p => p.Description.Contains("pod-phantom", StringComparison.Ordinal));
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
            Assert.Contains(verdict.Problems, p => p.Description.Contains("zero requests", StringComparison.Ordinal));
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
            Assert.Contains(
                verdict.Problems,
                p => p.Description.Contains("before the load started", StringComparison.Ordinal));
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
            Assert.Contains(
                verdict.Problems,
                p => p.Description.Contains("healthy siblings", StringComparison.Ordinal));
        }

        /// <summary>Too little traffic per pod: the quantiles pin to histogram bucket edges.</summary>
        [Fact]
        public void AWindowRecordedOnAnUnderloadedCluster_IsRejected()
        {
            var window = Recording(healthyP95: 120.0, faultedP95: 340.0);
            var verdict = LabWindowValidator.Validate(window, ["pod-faulted"]);

            Assert.False(verdict.IsUsable);
            Assert.Contains(verdict.Problems, p => p.Description.Contains("outside", StringComparison.Ordinal));
        }

        /// <summary>
        /// The 2026-08-07 event, in miniature: every replica leaves its operating point at the same instant.
        ///
        /// <para>This is the one defect none of the checks above can see, and the reason is structural — they
        /// are all per pod, and no pod is an outlier when the whole fleet moves together.</para>
        /// </summary>
        [Fact]
        public void AFleetWideExcursion_IsRejected()
        {
            var window = Recording();

            for (var pod = 0; pod < window.Pods.Count; pod++)
            {
                var cpu = window.Series(pod, MetricIndex.CpuUsageRatio);

                for (var t = 20; t < 26; t++)
                {
                    cpu[t] *= 40.0;
                }
            }

            var verdict = LabWindowValidator.Validate(window, ["pod-faulted"]);

            Assert.False(verdict.IsUsable);
            Assert.Single(verdict.Excursions);
            Assert.Equal(20, verdict.Excursions[0].FirstScrape);
            Assert.Equal(25, verdict.Excursions[0].LastScrape);
            Assert.Contains(
                verdict.Problems,
                p => p.Description.Contains("fleet excursion", StringComparison.Ordinal));
        }

        /// <summary>
        /// The other half of the excursion check, and the half that keeps it from swallowing the peer signal:
        /// one replica running hot is an ordinary outlier, which the peer families exist to find. Only a
        /// simultaneous move across the fleet is invisible to them, so only that is reported here.
        /// </summary>
        [Fact]
        public void ASinglePodRunningHot_IsNotAFleetExcursion()
        {
            var window = Recording();
            var cpu = window.Series(0, MetricIndex.CpuUsageRatio);

            for (var t = 20; t < 26; t++)
            {
                cpu[t] *= 40.0;
            }

            var verdict = LabWindowValidator.Validate(window, ["pod-faulted"]);

            Assert.Empty(verdict.Excursions);
            Assert.True(verdict.IsUsable, verdict.Describe());
        }

        /// <summary>
        /// Excision removes the excursion at scrape level and says what it took.
        ///
        /// <para>Scrape level is the only level where it does anything: the real event is seven minutes inside
        /// a thirty-six hour recording, and averaged into any aggregate it never crosses a threshold.</para>
        /// </summary>
        [Fact]
        public void ExcisingAFleetExcursion_ClearsItAndReportsWhatWent()
        {
            var window = Recording();

            for (var pod = 0; pod < window.Pods.Count; pod++)
            {
                var cpu = window.Series(pod, MetricIndex.CpuUsageRatio);

                for (var t = 20; t < 26; t++)
                {
                    cpu[t] *= 40.0;
                }
            }

            var report = LabWindowValidator.Excise(window, LabWindowValidator.FindFleetExcursions(window));

            Assert.Equal(6, report.ScrapesExcised);
            Assert.Contains("excised", report.ToString(), StringComparison.Ordinal);
            Assert.Contains("CpuUsageRatio", report.ToString(), StringComparison.Ordinal);

            for (var pod = 0; pod < window.Pods.Count; pod++)
            {
                Assert.All(
                    window.Series(pod, MetricIndex.CpuUsageRatio)[20..26].ToArray(),
                    v => Assert.True(double.IsNaN(v)));
            }

            Assert.Empty(LabWindowValidator.FindFleetExcursions(window));
        }

        /// <summary>
        /// The per-channel verdict, both directions at once: a latency defect must not condemn the CPU series
        /// recorded beside it, and a caller that names a rejected channel must still be told no.
        /// </summary>
        [Fact]
        public void ALatencyDefect_RejectsOnlyTheLatencyChannels()
        {
            var window = Recording(healthyP95: 120.0, faultedP95: 340.0);
            var verdict = LabWindowValidator.Validate(window, ["pod-faulted"]);

            Assert.False(verdict.IsUsable);
            Assert.True(verdict.IsUsableForAll(MetricIndex.CpuUsageRatio, MetricIndex.RequestsPerSecond));
            Assert.False(verdict.IsUsableFor(MetricIndex.LatencyP95Ms));

            // Naming a rejected channel alongside good ones is still wrong — that is the whole point.
            Assert.False(verdict.IsUsableForAll(MetricIndex.CpuUsageRatio, MetricIndex.LatencyP95Ms));
            Assert.Equal(LabWindowValidator.LatencyChannels, verdict.RejectedChannels);
        }

        /// <summary>
        /// A phantom replica is not a latency problem, so per-channel scoping must not turn it into one: it
        /// invalidates every peer comparison there is, and the verdict has to keep saying so.
        /// </summary>
        [Fact]
        public void APhantomReplica_RejectsEveryChannel()
        {
            var window = Recording(extraPods: ["pod-phantom"]);
            var verdict = LabWindowValidator.Validate(window, ["pod-faulted"]);

            Assert.Empty(verdict.UsableChannels);
            Assert.False(verdict.IsUsableFor(MetricIndex.CpuUsageRatio));
        }

        [Fact]
        public void ARestartInsideTheWindow_IsRejected()
        {
            var window = Recording();
            window.Series(2, MetricIndex.ContainerRestarts)[20] = 1.0;

            var verdict = LabWindowValidator.Validate(window, ["pod-faulted"]);

            Assert.False(verdict.IsUsable);
            Assert.Contains(
                verdict.Problems,
                p => p.Description.Contains("restarted inside", StringComparison.Ordinal));
        }

        /// <summary>
        /// Four healthy replicas at the operating point the project calibrated against — healthy p95 near
        /// 860 ms, throttled near 2441 ms — with every modelled channel populated.
        ///
        /// <para>Every channel, because the verdict is per channel: a window that carries only latency,
        /// traffic and restarts is not "well formed", it is a window with ten empty channels, and the baseline
        /// this file's negative cases are measured against has to be the thing it claims to be.</para>
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

                var faulted = names[pod] == "pod-faulted";
                var level = faulted ? faultedP95 : healthyP95;

                Fill(window, pod, MetricIndex.LatencyP95Ms, level, 0.30, rng);
                Fill(window, pod, MetricIndex.LatencyP50Ms, level * 0.55, 0.30, rng);
                Fill(window, pod, MetricIndex.LatencyP99Ms, level * 1.20, 0.30, rng);
                Fill(window, pod, MetricIndex.RequestsPerSecond, 4.0, 0.20, rng);
                Fill(window, pod, MetricIndex.CpuUsageRatio, 0.42, 0.20, rng);
                Fill(window, pod, MetricIndex.CpuThrottleRatio, faulted ? 0.31 : 0.0, 0.20, rng);
                Fill(window, pod, MetricIndex.MemoryWorkingSetBytes, 268_435_456.0, 0.05, rng);
                Fill(window, pod, MetricIndex.GcGen2HeapBytes, 41_943_040.0, 0.05, rng);
                Fill(window, pod, MetricIndex.GcPauseRatio, 0.004, 0.30, rng);
                Fill(window, pod, MetricIndex.ThreadPoolQueueLength, 1.0, 0.50, rng);
                Fill(window, pod, MetricIndex.ErrorRate, 0.0, 0.0, rng);
                Fill(window, pod, MetricIndex.OomEventsRate, 0.0, 0.0, rng);
                Fill(window, pod, MetricIndex.ContainerRestarts, 0.0, 0.0, rng);
            }

            return window;
        }

        private static void Fill(
            MetricWindow window,
            int pod,
            MetricIndex channel,
            double level,
            double spread,
            Random rng)
        {
            var series = window.Series(pod, channel);

            for (var t = 0; t < series.Length; t++)
            {
                series[t] = level * (1.0 + ((rng.NextDouble() - 0.5) * spread));
            }
        }
    }
}
