// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Globalization;
using System.Text;
using DevOnBike.Overfit.Anomalies.Contracts;
using DevOnBike.Overfit.Anomalies.Monitoring;
using DevOnBike.Overfit.Tests.TestSupport;

namespace DevOnBike.Overfit.Tests.Anomalies.Diagnostics
{
    /// <summary>
    /// Records a window of the cluster lab's real Prometheus data into a checked-in fixture.
    ///
    /// <para><b>Why this exists: every calibration decision in the guard currently rests on a simulator, and
    /// the simulator has been wrong three times.</b> Its diurnal phase was drawn per pod, its within-pod
    /// scatter was 6.5× too tight, and its post-restart memory ramp took 3.9 hours instead of minutes — each
    /// of those inflated the false-positive rate, and each was found only by comparing against six numbers
    /// copied by hand off a Grafana panel. Six numbers found three bugs. A recorded window turns that into a
    /// comparison that can be repeated, extended and run in CI.</para>
    ///
    /// <para><b>The fixture is also the first labelled data this project has.</b> The lab runs one replica
    /// carrying <c>overfit.dev/fault=cpu-throttle</c> alongside three healthy ones, and the recorder writes
    /// that annotation into the header. Validating the learned stack needs labels and has never had any;
    /// this is a small start on the M0 gate rather than a realism tool only.</para>
    ///
    /// <para><b>Recording an idle cluster is worthless</b>, and worse than worthless because it looks like
    /// data. The lab's RED signals sit at zero until something drives the server, so this must run while load
    /// is flowing — see <c>k8s/overfit/forward-replicas.cmd</c> and <c>AnomalyLabLoadGeneratorTests</c>. The
    /// recorder refuses to write a fixture whose request-rate channel is empty rather than producing a file
    /// that would silently misrepresent the cluster.</para>
    ///
    /// <para><b>REQUIRES A PORT-FORWARD TO PROMETHEUS ON :9090.</b> Without it this fails with a bare
    /// <c>HttpRequestException: connection refused (127.0.0.1:9090)</c>, which names no cause and reads
    /// like a defect in the recorder. It is not one — measured 2026-08-07, five of these lab diagnostics
    /// failed that way in the first release-gate run purely because nothing was forwarding.</para>
    /// <code>
    ///   k8s\monitoring\forward.cmd                                            (all three ports at once)
    ///   kubectl port-forward -n monitoring svc/overfit-lab-prometheus 9090:9090   (this one only)
    /// </code>
    /// <para>Verify it before believing a red result:
    /// <c>curl "http://127.0.0.1:9090/api/v1/query?query=up"</c>.</para>
    ///
    /// <para>It is <c>[LongFact]</c>, so it never runs in an ordinary <c>dotnet test</c>. Knobs, read from
    /// the code rather than from memory: <c>OVERFIT_LAB_PROM</c> — <b>not</b>
    /// <c>OVERFIT_LAB_PROMETHEUS</c>, which is what every other diagnostic in this directory uses —
    /// <c>OVERFIT_LAB_CONFIG</c>, <c>OVERFIT_LAB_FIXTURE</c>, <c>OVERFIT_LAB_FAULT_PODS</c>
    /// (comma-separated, annotated in the header), <c>OVERFIT_LAB_MINUTES</c>,
    /// <c>OVERFIT_LAB_STEP_SECONDS</c>, and the two that decide where the window ends —
    /// <c>OVERFIT_LAB_END_OFFSET_MINUTES</c> and <c>OVERFIT_LAB_END_UTC</c>, see
    /// <see cref="EndOfWindow"/>.</para>
    ///
    /// <para>An earlier version of that list asserted that <c>OVERFIT_LAB_MINUTES</c> and
    /// <c>OVERFIT_LAB_STEP_SECONDS</c> were <i>not</i> read anywhere in this file and that setting them did
    /// nothing. Both have been read since the first version of the method — they are the first two lines of
    /// it — so the warning was false, and following it would have meant recording the default twenty minutes
    /// while believing a longer window was impossible.</para>
    /// </summary>
    public sealed class LabFixtureRecorderDiagnostics
    {
        private const string DefaultPrometheus = "http://127.0.0.1:9090";

        /// <summary>
        /// Which pods to record, and under what metric names.
        ///
        /// <para><b>Read from the guard's own configuration rather than hardcoded, and that was a
        /// correction.</b> The first version pinned <c>overfit-server-.*</c> in namespace <c>overfit</c> with
        /// the built-in query templates — which stopped matching the moment the lab was rebuilt around an
        /// application that names its metrics <c>labapp_*</c>. A recorder that silently records nothing
        /// produces an empty fixture, and an empty fixture is worse than none because it looks like data.</para>
        /// </summary>
        private static string ConfigPath =>
            Environment.GetEnvironmentVariable("OVERFIT_LAB_CONFIG")
            ?? RepositoryPaths.TestsBin("lab-guard.json");

        private readonly ITestOutputHelper _output;

        public LabFixtureRecorderDiagnostics(ITestOutputHelper output)
        {
            _output = output;
        }

        [LabFact(LabEndpoint.Prometheus)]
        public async Task RecordsAWindowOfTheLabIntoAFixture()
        {
            var minutes = Setting("OVERFIT_LAB_MINUTES", 20);
            var stepSeconds = Setting("OVERFIT_LAB_STEP_SECONDS", 15);
            var baseUrl = Environment.GetEnvironmentVariable("OVERFIT_LAB_PROM") ?? DefaultPrometheus;
            var faultPods = (Environment.GetEnvironmentVariable("OVERFIT_LAB_FAULT_PODS") ?? string.Empty)
                .Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);

            var end = EndOfWindow(stepSeconds);
            var start = end.AddMinutes(-minutes);

            // The same mapping the deployed guard runs, so the fixture holds the series the guard actually
            // judges rather than a parallel set that happens to share names.
            Assert.True(File.Exists(ConfigPath),
                $"no guard configuration at {ConfigPath} — pull it from the running ConfigMap first, or the "
                + "recorder will use built-in templates that no longer match this lab.");

            var file = System.Text.Json.JsonSerializer.Deserialize<AnomalyGuardConfigFile>(
                await File.ReadAllTextAsync(ConfigPath),
                new System.Text.Json.JsonSerializerOptions { PropertyNameCaseInsensitive = true });

            Assert.NotNull(file);

            var map = AnomalyGuardConfigReader.ReadMap(file, out _);

            var config = PrometheusHistoricalSourceConfig.ForOverfitServer(
                baseUrl, file.PodRegex, file.Namespace, start, end,
                step: TimeSpan.FromSeconds(stepSeconds),
                window: TimeSpan.FromMinutes(2)) with
            {
                QueryOverrides = map.ToQueryOverrides(),
            };

            using var source = new PrometheusHistoricalSource(config);
            var batches = await source.FetchAsync();

            // FetchAsync returns one entry per scrape step, but every entry holds THE SAME series list — the
            // batching exists for TimeSeriesAligner, which windows the shared list around each timestamp.
            // Reading a value per batch therefore yields the same number repeated once per step; the first
            // version of this recorder did exactly that and wrote 61 copies of one sample. The real time
            // series is inside each RawMetricSeries.Samples, with its own timestamps.
            var timestamps = new List<long>(batches.Count);
            foreach (var (scrapeMs, _) in batches)
            {
                timestamps.Add(scrapeMs);
            }

            var series = new SortedDictionary<string, SortedDictionary<string, List<float>>>(StringComparer.Ordinal);
            var grid = new Dictionary<long, int>(timestamps.Count);
            for (var i = 0; i < timestamps.Count; i++)
            {
                grid[timestamps[i]] = i;
            }

            foreach (var s in batches.Count > 0 ? batches[0].Series : [])
            {
                var metric = ((MetricIndex)s.MetricTypeId).ToString();

                if (!series.TryGetValue(metric, out var perPod))
                {
                    perPod = new SortedDictionary<string, List<float>>(StringComparer.Ordinal);
                    series[metric] = perPod;
                }

                if (!perPod.TryGetValue(s.Pod.PodName, out var values))
                {
                    // Missing is NaN, never zero — the whole pipeline's rule, and a fixture that broke it
                    // would teach whatever reads it the opposite. Pre-filling means a scrape the series has
                    // no sample for stays a gap instead of silently shifting later samples left.
                    values = new List<float>(timestamps.Count);
                    for (var i = 0; i < timestamps.Count; i++)
                    {
                        values.Add(float.NaN);
                    }

                    perPod[s.Pod.PodName] = values;
                }

                foreach (var sample in s.Samples)
                {
                    if (grid.TryGetValue(sample.Timestamp, out var slot))
                    {
                        values[slot] = sample.Value;
                    }
                }
            }

            var pods = new SortedSet<string>(StringComparer.Ordinal);
            foreach (var perPod in series.Values)
            {
                foreach (var pod in perPod.Keys)
                {
                    pods.Add(pod);
                }
            }

            _output.WriteLine(Coverage(source, series, pods, timestamps.Count, minutes, stepSeconds));

            Assert.True(pods.Count >= 3,
                $"{pods.Count} pods returned series; a peer group needs at least three. Check the port-forward "
                + $"to Prometheus at {baseUrl}, the namespace '{file.Namespace}' and the regex '{file.PodRegex}'.");

            AssertTrafficWasFlowing(series);

            var path = Environment.GetEnvironmentVariable("OVERFIT_LAB_FIXTURE")
                       ?? Path.Combine(AppContext.BaseDirectory, "lab-window.csv");

            Directory.CreateDirectory(Path.GetDirectoryName(path)!);
            // No BOM: this is a checked-in data file that ordinary tools and scripts will read, and a BOM
            // turns the first header line into something that does not compare equal to "#...".
            File.WriteAllText(
                path,
                Render(series, pods, timestamps, faultPods, minutes, stepSeconds, start, end),
                new UTF8Encoding(encoderShouldEmitUTF8Identifier: false));

            _output.WriteLine($"\nwritten: {path}  ({new FileInfo(path).Length / 1024} KB)");
        }

        /// <summary>
        /// The instant the recorded window ends at.
        ///
        /// <para>By default a step short of <b>now</b>: the most recent point is often still being filled in,
        /// and a partial last sample would read as a dip in every series at once.
        /// <c>OVERFIT_LAB_END_OFFSET_MINUTES</c> pushes that further back, which is what recording after a
        /// load run needs — the <c>rate()</c> window is two minutes wide, so samples taken right after traffic
        /// stops are still decaying and would put a false downward trend in every RED signal at once.</para>
        ///
        /// <para><b><c>OVERFIT_LAB_END_UTC</c> pins it to an absolute instant instead</b>, which is what
        /// recording a <i>past</i> stretch out of Prometheus retention needs. The offset knob can reach the
        /// past too, and doing it that way is a trap: the offset is counted from whenever the test happens to
        /// start, so the same command run twenty minutes later records a different window and the two results
        /// are not comparable. A recording used as a calibration reference has to name the stretch it covers,
        /// not a distance from an unrecorded present.</para>
        /// </summary>
        private static DateTime EndOfWindow(int stepSeconds)
        {
            var pinned = Environment.GetEnvironmentVariable("OVERFIT_LAB_END_UTC");

            if (!string.IsNullOrWhiteSpace(pinned))
            {
                var parsed = DateTime.TryParse(
                    pinned,
                    CultureInfo.InvariantCulture,
                    DateTimeStyles.AdjustToUniversal | DateTimeStyles.AssumeUniversal,
                    out var instant);

                Assert.True(parsed,
                    $"OVERFIT_LAB_END_UTC='{pinned}' is not a parseable instant. Use ISO-8601 UTC, "
                    + "e.g. 2026-08-07T20:00:00Z.");

                return instant;
            }

            var offset = Setting("OVERFIT_LAB_END_OFFSET_MINUTES", 0);

            return DateTime.UtcNow.AddSeconds(-stepSeconds).AddMinutes(-offset);
        }

        /// <summary>
        /// Per-pod, per-metric coverage — <b>not</b> "did this metric return any series", which is what the
        /// existing lab diagnostic checks and which passes when a metric arrives for one pod out of twenty.
        /// </summary>
        private static string Coverage(
            PrometheusHistoricalSource source,
            SortedDictionary<string, SortedDictionary<string, List<float>>> series,
            SortedSet<string> pods,
            int scrapes,
            int minutes,
            int stepSeconds)
        {
            var report = new StringBuilder();

            report.Append($"window       {minutes} min at {stepSeconds} s = {scrapes} scrapes\n");
            report.Append($"pods         {pods.Count}\n\n");
            report.Append($"{"metric",-26}{"mapped",8}{"pods",6}{"finite %",10}   note\n");

            for (var m = 0; m < (int)MetricIndex.Count; m++)
            {
                var metric = (MetricIndex)m;
                var name = metric.ToString();
                var mapped = source.IsMapped(metric);

                series.TryGetValue(name, out var perPod);
                var podCount = perPod?.Count ?? 0;

                var total = 0;
                var finite = 0;

                if (perPod is not null)
                {
                    foreach (var values in perPod.Values)
                    {
                        foreach (var v in values)
                        {
                            total++;
                            finite += float.IsFinite(v) ? 1 : 0;
                        }
                    }
                }

                var note = !mapped ? "no query for this deployment"
                    : podCount == 0 ? "<-- query matched nothing"
                    : podCount < pods.Count ? $"<-- only {podCount} of {pods.Count} pods"
                    : string.Empty;

                report.Append($"{name,-26}{(mapped ? "yes" : "no"),8}{podCount,6}"
                              + $"{(total > 0 ? finite * 100.0 / total : 0.0),9:F1}%   {note}\n");
            }

            return report.ToString();
        }

        /// <summary>
        /// An idle cluster records flat zeros in every RED signal, and a fixture like that is worse than no
        /// fixture: it looks like data and would recalibrate the guard against a cluster nobody was using.
        /// </summary>
        private static void AssertTrafficWasFlowing(
            SortedDictionary<string, SortedDictionary<string, List<float>>> series)
        {
            var name = MetricIndex.RequestsPerSecond.ToString();
            var moving = 0;

            if (series.TryGetValue(name, out var perPod))
            {
                foreach (var values in perPod.Values)
                {
                    foreach (var v in values)
                    {
                        if (float.IsFinite(v) && v > 0.01f)
                        {
                            moving++;
                            break;
                        }
                    }
                }
            }

            Assert.True(moving >= 3,
                $"only {moving} pods showed any request traffic. The lab's RED signals are zero until "
                + "something drives the server — start the load generator and record while it runs, or the "
                + "fixture will describe an idle cluster.");
        }

        private static string Render(
            SortedDictionary<string, SortedDictionary<string, List<float>>> series,
            SortedSet<string> pods,
            List<long> timestamps,
            string[] faultPods,
            int minutes,
            int stepSeconds,
            DateTime start,
            DateTime end)
        {
            var sb = new StringBuilder();

            sb.Append("# Overfit cluster lab — recorded Prometheus window\n");
            sb.Append("#\n");
            sb.Append("# One row per (metric, pod). Values are ordered oldest first and aligned to the\n");
            sb.Append("# timestamps row. 'nan' means the scrape returned nothing for that pod and metric —\n");
            sb.Append("# missing, NOT zero. Do not substitute zero when reading this file.\n");
            sb.Append("#\n");
            sb.Append(string.Create(CultureInfo.InvariantCulture,
                $"# recorded_utc={DateTime.UtcNow:O}\n"));

            // When the recording is historical, recorded_utc says nothing about what is in the file — the
            // range does, and it is the only thing that makes the recording citable as evidence about a
            // particular stretch of the cluster's life.
            sb.Append(string.Create(CultureInfo.InvariantCulture,
                $"# range_start_utc={start:yyyy-MM-ddTHH:mm:ssZ} range_end_utc={end:yyyy-MM-ddTHH:mm:ssZ}\n"));
            sb.Append(string.Create(CultureInfo.InvariantCulture,
                $"# window_minutes={minutes} step_seconds={stepSeconds} scrapes={timestamps.Count}\n"));
            sb.Append("# load=even (SKEW=1), driven by AnomalyLabLoadGeneratorTests\n");

            foreach (var pod in pods)
            {
                var faulted = Array.IndexOf(faultPods, pod) >= 0;
                sb.Append($"# pod={pod} label={(faulted ? "FAULT:cpu-throttle" : "healthy")}\n");
            }

            sb.Append("#\n");
            sb.Append("timestamps_ms,").Append(string.Join(',', timestamps)).Append('\n');

            foreach (var (metric, perPod) in series)
            {
                foreach (var (pod, values) in perPod)
                {
                    sb.Append(metric).Append(',').Append(pod);

                    foreach (var v in values)
                    {
                        sb.Append(',');
                        sb.Append(float.IsFinite(v)
                            ? v.ToString("R", CultureInfo.InvariantCulture)
                            : "nan");
                    }

                    sb.Append('\n');
                }
            }

            return sb.ToString();
        }

        private static int Setting(string name, int fallback)
        {
            var raw = Environment.GetEnvironmentVariable(name);

            return int.TryParse(raw, NumberStyles.Integer, CultureInfo.InvariantCulture, out var value)
                ? value
                : fallback;
        }
    }
}
