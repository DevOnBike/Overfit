// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Anomalies.Contracts;
using DevOnBike.Overfit.Anomalies.Hosting;
using DevOnBike.Overfit.Anomalies.Monitoring;
using DevOnBike.Overfit.Anomalies.Monitoring.Abstractions;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;

namespace DevOnBike.Overfit.Tests.Anomalies
{
    /// <summary>
    /// That the container both entry points build can actually produce the guard.
    ///
    /// <para><b>Nothing checked this before.</b> <c>WorkloadIdentityTests</c> resolves
    /// <see cref="AnomalyGuardServiceOptions"/> and stops there, so every dependency of
    /// <see cref="AnomalyGuardService"/> itself was unverified: a constructor parameter whose service type
    /// nobody registered fails at <c>GetRequiredService</c>, which in a real deployment is host startup —
    /// after the image is built and pushed. These are the cheapest possible tests for the failure with the
    /// longest feedback loop in this subsystem.</para>
    ///
    /// <para>The two overloads are covered separately on purpose. They register the window source at two
    /// different call sites with two different argument lists, and a change applied to one of them is a
    /// container that works from code and throws from a ConfigMap — which is the deployment nobody runs
    /// locally.</para>
    /// </summary>
    public sealed class AnomalyGuardRegistrationTests
    {
        /// <summary>
        /// The code-first overload. Resolving the guard is the assertion; the hosted-service identity check
        /// is the half a non-null assertion would miss, and it is load-bearing — a host scrapes
        /// <see cref="AnomalyGuardService.Telemetry"/> off the singleton, and if the hosted service were a
        /// second instance those counters would belong to an object that never runs a cycle.
        /// </summary>
        [Fact]
        public void TheGuardResolvesFromTheContainerTheCodeOverloadBuilds()
        {
            var services = new ServiceCollection();

            services.AddLogging();
            services.AddOverfitAnomalyGuard(Prometheus());

            using var provider = services.BuildServiceProvider();
            var guard = provider.GetRequiredService<AnomalyGuardService>();

            Assert.NotNull(guard);
            Assert.Contains(guard, provider.GetServices<IHostedService>());
        }

        /// <summary>
        /// The configuration-file overload, which is the one a deployed guard actually takes — the CLI
        /// deserialises the file itself and calls it, because the <c>IConfiguration</c> overload binds with
        /// reflection and cannot go into a Native-AOT image.
        /// </summary>
        [Fact]
        public void TheGuardResolvesFromTheContainerTheConfigFileOverloadBuilds()
        {
            var services = new ServiceCollection();

            services.AddLogging();
            services.AddOverfitAnomalyGuard(
                new AnomalyGuardConfigFile
                {
                    Prometheus = "http://127.0.0.1:9090",
                    Namespace = "lab",
                    Workload = "lab-workload",
                    PodRegex = "lab-workload-.*",
                });

            using var provider = services.BuildServiceProvider();

            Assert.NotNull(provider.GetRequiredService<AnomalyGuardService>());
        }

        /// <summary>
        /// The window source resolves under the interface the loop asks for, is the Prometheus one, and has
        /// singleton lifetime.
        ///
        /// <para><b>What this does NOT catch, stated because the obvious reading of it is wrong.</b> A stray
        /// <c>AddSingleton&lt;PrometheusMetricWindowSource&gt;(...)</c> left alongside the interface
        /// registration — the natural thing for somebody who later wants <c>SeriesReturned</c> or
        /// <c>CustomChannels</c>, which are deliberately not on the interface — is <i>invisible here</i>.
        /// Measured rather than reasoned: with both registrations present, resolving the interface returns the
        /// interface registration's instance every time and never consults the concrete descriptor, so the
        /// second source (and its second <c>HttpClient</c>) exists, resolves independently, and no test in this
        /// file goes red. The <c>Assert.Same</c> below therefore proves the interface registration is a
        /// singleton — which <c>AddSingleton</c> guarantees for free — and nothing about how many sources the
        /// container can build.</para>
        ///
        /// <para>Catching that would mean asserting over the <see cref="ServiceCollection"/>'s descriptors
        /// before the provider is built (exactly one with <c>ServiceType</c> <see cref="IMetricWindowSource"/>,
        /// none with <c>ServiceType</c> <see cref="PrometheusMetricWindowSource"/>) rather than over resolved
        /// instances. Not done here: it is a different kind of test — registration shape rather than
        /// resolution — and adding it is a scope decision, not part of extracting the seam.</para>
        /// </summary>
        [Fact]
        public void TheWindowSourceIsASingleInstanceBehindTheInterface()
        {
            var services = new ServiceCollection();

            services.AddLogging();
            services.AddOverfitAnomalyGuard(Prometheus());

            using var provider = services.BuildServiceProvider();
            var source = provider.GetRequiredService<IMetricWindowSource>();

            Assert.IsType<PrometheusMetricWindowSource>(source);
            Assert.Same(source, provider.GetRequiredService<IMetricWindowSource>());
        }

        /// <summary>
        /// A syntactically valid target that nothing is listening on. Registration issues no query, and
        /// neither does resolution — the first request happens on the first cycle, which these tests never
        /// run.
        /// </summary>
        private static PrometheusHistoricalSourceConfig Prometheus()
        {
            return PrometheusHistoricalSourceConfig.ForOverfitServer(
                "http://127.0.0.1:9090",
                "lab-workload-.*",
                "lab",
                DateTime.UtcNow.AddMinutes(-20),
                DateTime.UtcNow,
                step: TimeSpan.FromSeconds(15));
        }
    }
}
