// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Text.Json;
using DevOnBike.Overfit.Anomalies.Contracts;
using DevOnBike.Overfit.Anomalies.Incidents;
using DevOnBike.Overfit.Anomalies.Incidents.Abstractions;
using DevOnBike.Overfit.Anomalies.Monitoring;
using DevOnBike.Overfit.Server.AspNet.Services;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;

namespace DevOnBike.Overfit.Cli
{
    /// <summary>
    /// Runs the anomaly guard as a long-lived process: <c>overfit anomaly-guard --config guard.json</c>.
    ///
    /// <para><b>This is the deployable artefact, and until it existed there was none.</b> Every measurement
    /// this project has of the guard came from a test method calling <c>RunCycle</c> directly — which
    /// exercised the detectors and nothing around them. The registration, the hosted loop, the logging sink
    /// and the durable state had never been executed together, so "it works" was a statement about a library,
    /// not about a thing anyone could deploy.</para>
    ///
    /// <para><b>Shadow by default.</b> The registration installs <c>IncidentLogOptions.Shadow</c>, so this
    /// counts, explains and wakes nobody. That is the intended first week at any new cluster: the coverage
    /// counters are read before any finding is, because a metric nobody exports produces silence that looks
    /// exactly like health.</para>
    ///
    /// <para><b>Durable state is opt-in but strongly advised.</b> Without <c>--state</c> a restart reopens
    /// every incident that was running, so a rollout of this process pages an operator for problems they were
    /// already told about. The file is small; an <c>emptyDir</c> costs one duplicate burst per restart, a few
    /// megabytes of volume costs none.</para>
    /// </summary>
    internal static class AnomalyGuardCommand
    {
        public static async Task<int> RunAsync(
            string configPath, string? statePath, int cadenceSeconds, int windowMinutes, int metricsPort,
            CancellationToken ct)
        {
            if (!File.Exists(configPath))
            {
                Console.Error.WriteLine($"Configuration file not found: {configPath}");

                return 1;
            }

            AnomalyGuardConfigFile? file;

            try
            {
                await using var stream = File.OpenRead(configPath);

                file = await JsonSerializer.DeserializeAsync(
                    stream, AnomalyGuardJsonContext.Default.AnomalyGuardConfigFile, ct).ConfigureAwait(false);
            }
            catch (JsonException ex)
            {
                Console.Error.WriteLine($"{configPath} is not valid JSON: {ex.Message}");

                return 1;
            }

            if (file is null)
            {
                Console.Error.WriteLine($"{configPath} is empty.");

                return 1;
            }

            // Refused rather than defaulted. A guard pointed at no Prometheus starts happily and reports
            // nothing, which is indistinguishable from a healthy cluster — the failure mode this whole
            // subsystem is built to avoid, arriving through its own configuration.
            if (string.IsNullOrWhiteSpace(file.Prometheus))
            {
                Console.Error.WriteLine($"{configPath}: 'prometheus' is required (the HTTP API base URL).");

                return 1;
            }

            var builder = Host.CreateApplicationBuilder();

            builder.Logging.ClearProviders();
            builder.Logging.AddSimpleConsole(options =>
            {
                options.SingleLine = false;
                options.TimestampFormat = "yyyy-MM-dd HH:mm:ss ";
                options.UseUtcTimestamp = true;
            });

            if (!string.IsNullOrWhiteSpace(statePath))
            {
                // Registered before the guard so TryAddSingletonSink-style "caller wins" resolution applies.
                builder.Services.AddSingleton<IIncidentStore>(_ => new FileIncidentStore(statePath));

                // The learned state lives beside the incidents, in its own file. Separate payloads: the
                // incidents are small, change every cycle and matter for hours; the baseline and the floor
                // calibration are large, change slowly and matter for days. Losing the second costs a week of
                // learning and leaves the guard quieter than it should be, which looks like success.
                var learnedPath = Path.Combine(
                    Path.GetDirectoryName(statePath) is { Length: > 0 } dir ? dir : ".",
                    "learned-state.txt");

                builder.Services.AddSingleton<ILearnedStateStore>(_ => new FileLearnedStateStore(learnedPath));
            }

            var problems = 0;

            // Left at the measured defaults unless the operator overrides them. Window especially: a
            // sweep on a healthy population gave 234 false incidents a day at 20 minutes, 93 at 60, and
            // 2583 at 240 — a four-hour window sits on the slope of the daily traffic curve, so longer is
            // emphatically not safer.
            var schedule = new AnomalyGuardServiceOptions
            {
                Cadence = cadenceSeconds > 0
                    ? TimeSpan.FromSeconds(cadenceSeconds)
                    : new AnomalyGuardServiceOptions().Cadence,
                Window = windowMinutes > 0
                    ? TimeSpan.FromMinutes(windowMinutes)
                    : new AnomalyGuardServiceOptions().Window,
            };

            builder.Services.AddOverfitAnomalyGuard(
                file,
                schedule,
                onProblem: line =>
                {
                    problems++;
                    Console.Error.WriteLine($"config: {line}");
                });

            using var host = builder.Build();

            var logger = host.Services.GetRequiredService<ILoggerFactory>().CreateLogger("overfit.guard");

            logger.LogInformation(
                "anomaly guard starting: prometheus={Prometheus} namespace={Namespace} pods=/{PodRegex}/ "
                + "cadence={Cadence} window={Window} state={State} configProblems={Problems}",
                file.Prometheus,
                string.IsNullOrEmpty(file.Namespace) ? "(all)" : file.Namespace,
                string.IsNullOrEmpty(file.PodRegex) ? ".*" : file.PodRegex,
                schedule.Cadence,
                schedule.Window,
                string.IsNullOrWhiteSpace(statePath) ? "(none — incidents reopen on restart)" : statePath,
                problems);

            // What this deployment will be blind to, stated at startup rather than inferred later from a
            // cycle that saw nothing. An unmapped feature has its query suppressed on purpose — issuing one
            // built from a metric the cluster does not export returns an empty result that Prometheus reports
            // as success, and empty is indistinguishable from healthy.
            var map = host.Services.GetRequiredService<MetricMap>();

            logger.LogInformation(
                "metric coverage: {Mapped} of {Total} known features mapped, {Custom} custom",
                map.MappedCount,
                (int)MetricIndex.Count,
                map.Custom.Count);

            for (var i = 0; i < map.Unmapped.Count; i++)
            {
                logger.LogWarning(
                    "blind: {Metric} has no binding — no query will be issued. Every cycle still COUNTS it "
                    + "blind, but this line is the only warning you get about it: a missing binding cannot "
                    + "change without editing this configuration, and repeating it each cycle taught "
                    + "operators to skip the line that a BOUND metric's silence shares. That case — an "
                    + "exporter that broke while the workload kept serving — is still warned about by name, "
                    + "every cycle.",
                    map.Unmapped[i]);
            }

            // The guard's own metrics, so "this has stopped" is detectable from outside. Without a scrape the
            // counters are a property nobody reads, and an alert written against a series that never arrives
            // reads as healthy in most alerting rules.
            var service = host.Services.GetRequiredService<AnomalyGuardService>();

            // The guard goes with the telemetry, so the same port also serves /ack and /suppressions. One
            // port rather than two: it is already scraped, already in the Service, and an operator endpoint
            // nobody exposed is an operator endpoint nobody can use.
            using var metrics = GuardMetricsEndpoint.TryStart(
                service.Telemetry, logger, metricsPort, service.Guard);

            if (metrics is not null)
            {
                logger.LogInformation(
                    "serving guard metrics on :{Port}/metrics — alert on "
                    + "overfit_guard_last_cycle_timestamp_seconds going stale", metricsPort);
            }

            await host.RunAsync(ct).ConfigureAwait(false);

            return 0;
        }
    }
}
