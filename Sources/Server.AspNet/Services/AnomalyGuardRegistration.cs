// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Anomalies.Incidents.Abstractions;
using DevOnBike.Overfit.Anomalies.Incidents.Contracts;
using DevOnBike.Overfit.Anomalies.Monitoring;
using DevOnBike.Overfit.Anomalies.Monitoring.Abstractions;
using DevOnBike.Overfit.Anomalies.Monitoring.Contracts;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;

namespace DevOnBike.Overfit.Server.AspNet.Services
{
    /// <summary>
    /// Wires the anomaly guard into a host: a Prometheus window source, a sink, and the loop that drives them.
    /// </summary>
    public static class AnomalyGuardRegistration
    {
        /// <summary>
        /// Registers the guard as a background service.
        ///
        /// <para><b>Off unless it is asked for.</b> The guard is not a property of running the server — it
        /// watches a cluster, needs a reachable Prometheus and a pod regex that matches something, and on a
        /// host with neither it would log a failed cycle every five minutes forever. So there is no
        /// "enabled by default": a caller that wants it says so, and supplies the two things only it knows.</para>
        ///
        /// <para><b>The sink defaults to <see cref="LoggerIncidentSink"/> in shadow mode</b> — everything at
        /// <c>Information</c>, nobody woken. That is the shape the rollout needs: run it against a real
        /// cluster, record what it says, and let an operator mark which of it was real. That produces the
        /// false-positive rate on somebody else's data <i>and</i> the labels the learned path has never had,
        /// as a by-product of the first step rather than as a prerequisite for it.</para>
        /// </summary>
        /// <param name="services">Host container.</param>
        /// <param name="prometheus">
        /// Query configuration: base URL, namespace, pod regex, and the per-deployment
        /// <c>QueryOverrides</c>. <b>The overrides are the part that will not be right by default</b> — metric
        /// names belong to whoever wrote the exporter, and the built-in templates use OpenTelemetry naming
        /// while this project's own server exports <c>overfit_chat_*</c>. A mismatch returns an empty result
        /// that Prometheus reports as success, which is why the guard counts blind metrics per cycle.
        /// </param>
        /// <param name="options">Cadence, window, thresholds; the defaults are the measured ones.</param>
        /// <remarks>
        /// Registers a <see cref="PrometheusTopologySource"/> too, and it is not optional in practice: the
        /// grouper scores <c>SameWorkload</c> at 0.7 against a 0.35 threshold, so without real ownership the
        /// guard falls back to a name heuristic and any pod it guesses wrong is merged into the wrong
        /// incident. It needs kube-state-metrics reachable through the same Prometheus.
        /// </remarks>
        public static IServiceCollection AddOverfitAnomalyGuard(
            this IServiceCollection services,
            PrometheusHistoricalSourceConfig prometheus,
            AnomalyGuardServiceOptions? options = null)
        {
            ArgumentNullException.ThrowIfNull(services);
            ArgumentNullException.ThrowIfNull(prometheus);

            // One client for the lifetime of the host, lent to a source per cycle. The source no longer
            // disposes what it is lent — it used to, which would have broken the second cycle.
            services.AddSingleton(_ => new PrometheusMetricWindowSource(prometheus));

            return services.AddGuardCore(prometheus, options ?? new AnomalyGuardServiceOptions());
        }

        /// <summary>
        /// Registers the guard from a configuration section — a ConfigMap, an <c>appsettings</c> file, an
        /// environment override — instead of from code.
        ///
        /// <para><b>The point is that a client fills this in, not us.</b> They name their metrics and their
        /// thresholds; the PromQL, the query overrides and the per-feature floor tables are derived. The
        /// alternative, which is what existed until now, is that somebody on our side edits C# per customer.
        /// </para>
        ///
        /// <para><b>Unreadable entries are dropped and reported through <paramref name="onProblem"/>, never
        /// defaulted.</b> A threshold that quietly became zero is a gate that quietly stopped gating. Pass a
        /// callback that logs — or one that throws, if a deployment would rather refuse to start than run
        /// half-configured. That is a defensible choice and deliberately the caller's.</para>
        /// </summary>
        /// <param name="services">Host container.</param>
        /// <param name="section">
        /// The section holding an <see cref="AnomalyGuardConfigFile"/>, conventionally <c>AnomalyGuard</c>.
        /// </param>
        /// <param name="options">Cadence, window and tracking; the defaults are the measured ones.</param>
        /// <param name="onProblem">
        /// Receives one line per configuration entry that could not be used. Called during registration, so
        /// the operator learns at startup rather than from a cycle that saw nothing.
        /// </param>
        public static IServiceCollection AddOverfitAnomalyGuard(
            this IServiceCollection services,
            IConfiguration section,
            AnomalyGuardServiceOptions? options = null,
            Action<string>? onProblem = null)
        {
            ArgumentNullException.ThrowIfNull(services);
            ArgumentNullException.ThrowIfNull(section);

            var file = new AnomalyGuardConfigFile();
            section.Bind(file);

            var map = AnomalyGuardConfigReader.ReadMap(file, out var mapProblems);
            var (gap, trendChange) = AnomalyGuardConfigReader.ReadThresholds(file, out var floorProblems);

            if (onProblem is not null)
            {
                Report(mapProblems, onProblem);
                Report(floorProblems, onProblem);
            }

            var prometheus = PrometheusHistoricalSourceConfig.ForOverfitServer(
                file.Prometheus,
                podRegex: file.PodRegex,
                namespaceName: file.Namespace,
                rangeStart: DateTime.UtcNow.AddMinutes(-20),
                rangeEnd: DateTime.UtcNow,
                step: TimeSpan.FromSeconds(15)) with
            {
                // The client's mapping replaces the built-in templates entirely. An unmapped feature gets an
                // empty template, which is how a source is told not to issue that query at all.
                QueryOverrides = map.ToQueryOverrides(),
            };

            var given = options ?? new AnomalyGuardServiceOptions();
            var resolved = new AnomalyGuardServiceOptions
            {
                Cadence = given.Cadence,
                Window = given.Window,
                EndOffset = given.EndOffset,
                Tracking = given.Tracking,
                Guard = given.Guard with
                {
                    Namespace = file.Namespace,
                    MinAbsoluteGap = gap,
                    MinAbsoluteTrendChange = trendChange,
                    CustomMetrics = map.Custom,
                },
            };

            services.AddSingleton(map);

            // The custom channels travel outside QueryOverrides, which is keyed by MetricIndex and therefore
            // cannot carry a name the enum does not have.
            services.AddSingleton(_ => new PrometheusMetricWindowSource(
                prometheus, httpClient: null, customQueries: map.CustomQueries()));

            return services.AddGuardCore(prometheus, resolved);
        }

        private static void Report(IReadOnlyList<string> problems, Action<string> onProblem)
        {
            for (var i = 0; i < problems.Count; i++)
            {
                onProblem(problems[i]);
            }
        }

        /// <summary>The parts both entry points share, once the configuration has been settled.</summary>
        private static IServiceCollection AddGuardCore(
            this IServiceCollection services,
            PrometheusHistoricalSourceConfig prometheus,
            AnomalyGuardServiceOptions options)
        {
            services.AddSingleton(options);

            // One instance, registered under the refreshable interface: the loop refreshes it and the guard
            // reads it. Two instances would leave the guard resolving against a snapshot nobody updates,
            // which fails silently — every pod would fall back to the name heuristic while the logs said
            // topology was fine.
            services.AddSingleton<IRefreshablePodTopology>(_ =>
                new PrometheusTopologySource(prometheus.PrometheusBaseUrl, prometheus));

            services.TryAddSingletonSink();
            services.AddHostedService<AnomalyGuardService>();

            return services;
        }

        /// <summary>
        /// Registers the logging sink unless the caller has already chosen one, so a host that wants
        /// Prometheus counters or a store instead simply registers its own first.
        /// </summary>
        private static void TryAddSingletonSink(this IServiceCollection services)
        {
            for (var i = 0; i < services.Count; i++)
            {
                if (services[i].ServiceType == typeof(IIncidentSink))
                {
                    return;
                }
            }

            services.AddSingleton<IIncidentSink>(provider =>
                new LoggerIncidentSink(
                    provider.GetRequiredService<Microsoft.Extensions.Logging.ILogger<LoggerIncidentSink>>(),
                    IncidentLogOptions.Shadow));
        }
    }
}
