// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Anomalies.Contracts;
using DevOnBike.Overfit.Anomalies.Monitoring;

namespace DevOnBike.Overfit.Tests.Anomalies
{
    /// <summary>
    /// Proposing a metric mapping instead of asking a human to write one.
    ///
    /// <para>The case that motivated this is embarrassing and worth stating: the mapping for this project's
    /// own lab was written by the author of the system and left <b>two channels of thirteen unbound</b>,
    /// reporting blind for hours before anyone read the log. Hand-authoring is the wrong default.</para>
    /// </summary>
    public sealed class MetricDiscoveryTests
    {
        [Fact]
        public async Task ADotNetWorkloadResolvesItsRuntimeChannels()
        {
            var available = Names(
                "container_cpu_usage_seconds_total",
                "container_memory_working_set_bytes",
                "dotnet_gc_heap_size_bytes",
                "dotnet_gc_pause_seconds_total",
                "dotnet_threadpool_queue_length");

            var found = await MetricDiscovery.ProposeAsync(available, _ => Task.FromResult(12), TestContext.Current.CancellationToken);

            Assert.Equal(DiscoveryOutcome.Resolved, Outcome(found, MetricIndex.GcGen2HeapBytes));
            Assert.Equal("dotnet_gc_heap_size_bytes", Chosen(found, MetricIndex.GcGen2HeapBytes).Source);
            Assert.Equal("dotnet", Chosen(found, MetricIndex.GcGen2HeapBytes).Stack);
            Assert.Equal(DiscoveryOutcome.Resolved, Outcome(found, MetricIndex.CpuUsageRatio));
        }

        [Fact]
        public async Task AJvmWorkloadResolvesTheJvmNamesForTheSameChannels()
        {
            var available = Names(
                "container_cpu_usage_seconds_total",
                "jvm_memory_used_bytes",
                "jvm_gc_pause_seconds_sum",
                "http_server_requests_seconds_bucket");

            var found = await MetricDiscovery.ProposeAsync(available, _ => Task.FromResult(4), TestContext.Current.CancellationToken);

            Assert.Equal("jvm_memory_used_bytes", Chosen(found, MetricIndex.GcGen2HeapBytes).Source);
            Assert.Equal("jvm_gc_pause_seconds_sum", Chosen(found, MetricIndex.GcPauseRatio).Source);

            // A histogram is exposed as three series and the name index carries the buckets, so that is what
            // has to be probed — looking for the bare family name finds nothing and reports a false blind.
            Assert.Equal(DiscoveryOutcome.Resolved, Outcome(found, MetricIndex.LatencyP95Ms));
            Assert.Equal(0.95, Chosen(found, MetricIndex.LatencyP95Ms).Quantile);
        }

        /// <summary>
        /// <b>The test that carries the design.</b> A cluster running one Java service and twelve .NET ones
        /// has <c>jvm_*</c> in its name index either way. Binding on the name alone would produce a query that
        /// returns nothing for ever and a channel that reports blind while looking configured.
        /// </summary>
        [Fact]
        public async Task ANameThatExistsButIsNotExportedByThesePodsIsRejected()
        {
            var available = Names("dotnet_gc_heap_size_bytes", "jvm_memory_used_bytes");

            var found = await MetricDiscovery.ProposeAsync(
                available,
                name => Task.FromResult(name.StartsWith("dotnet_", StringComparison.Ordinal) ? 12 : 0), TestContext.Current.CancellationToken);

            var heap = Find(found, MetricIndex.GcGen2HeapBytes);

            Assert.Equal(DiscoveryOutcome.Resolved, heap.Outcome);
            Assert.Equal("dotnet_gc_heap_size_bytes", heap.Chosen.Source);

            // The rejected one is still listed, so the report can show what was considered rather than
            // presenting a verdict out of nowhere.
            Assert.Contains(heap.Candidates, c => c.Source == "jvm_memory_used_bytes" && !c.IsEvidenced);
        }

        /// <summary>
        /// Two evidenced candidates must not be resolved by picking the first. Whether a 4xx is an error is a
        /// business decision, and a guess here is a guard confidently measuring the wrong thing.
        /// </summary>
        [Fact]
        public async Task SeveralEvidencedCandidatesStayAmbiguous()
        {
            var available = Names("http_requests_total", "nginx_http_requests_total");
            var found = await MetricDiscovery.ProposeAsync(available, _ => Task.FromResult(6), TestContext.Current.CancellationToken);

            var errors = Find(found, MetricIndex.ErrorRate);

            Assert.Equal(DiscoveryOutcome.Ambiguous, errors.Outcome);
            Assert.Equal(string.Empty, errors.Chosen.Source);
            Assert.Equal(2, errors.Candidates.Count);
        }

        [Fact]
        public async Task AChannelNobodyExportsIsReportedAsNotFound()
        {
            var found = await MetricDiscovery.ProposeAsync(Names("container_cpu_usage_seconds_total"), _ => Task.FromResult(12), TestContext.Current.CancellationToken);

            Assert.Equal(DiscoveryOutcome.NotFound, Outcome(found, MetricIndex.GcGen2HeapBytes));
            Assert.Empty(Find(found, MetricIndex.GcGen2HeapBytes).Candidates);
        }

        [Fact]
        public async Task EveryChannelIsAccountedFor()
        {
            var found = await MetricDiscovery.ProposeAsync(Names("container_cpu_usage_seconds_total"), _ => Task.FromResult(1), TestContext.Current.CancellationToken);

            // One entry per channel, always — a discovery report that silently omits a channel is a report
            // that says nothing about the thing the guard will be blind to.
            Assert.Equal((int)MetricIndex.Count, found.Count);
        }

        /// <summary>
        /// <b>The case this feature exists for.</b> An application nobody has met, instrumented perfectly
        /// conventionally under its own name. Matching whole names left four of thirteen channels blind on
        /// this project's own lab for exactly this reason.
        /// </summary>
        [Fact]
        public async Task ABespokeApplicationIsMatchedByTheShapeOfItsNames()
        {
            var available = Names(
                "labapp_requests_total",
                "labapp_errors_total",
                "labapp_request_duration_seconds_bucket",
                "container_cpu_usage_seconds_total");

            var found = await MetricDiscovery.ProposeAsync(available, _ => Task.FromResult(12), TestContext.Current.CancellationToken);

            var rps = Find(found, MetricIndex.RequestsPerSecond);
            Assert.Equal(DiscoveryOutcome.Resolved, rps.Outcome);
            Assert.Equal("labapp_requests_total", rps.Chosen.Source);
            Assert.Equal(MetricSourceKind.Counter, rps.Chosen.Kind);
            Assert.True(rps.Chosen.Inferred);

            var errors = Find(found, MetricIndex.ErrorRate);
            Assert.Equal(DiscoveryOutcome.Resolved, errors.Outcome);
            Assert.Equal("labapp_errors_total", errors.Chosen.Source);

            // The histogram binds to its FAMILY name; _bucket is only how it was found. Binding the bucket
            // series itself would make every quantile query malformed.
            var p95 = Find(found, MetricIndex.LatencyP95Ms);
            Assert.Equal(DiscoveryOutcome.Resolved, p95.Outcome);
            Assert.Equal("labapp_request_duration_seconds", p95.Chosen.Source);
            Assert.Equal(MetricSourceKind.HistogramSeconds, p95.Chosen.Kind);
            Assert.Equal(0.95, p95.Chosen.Quantile);
        }

        /// <summary>
        /// A name this project has seen must beat one that merely ends like the right thing, or a stray
        /// series would displace a binding whose meaning is actually known.
        /// </summary>
        [Fact]
        public async Task AKnownNameWinsOverASuffixMatch()
        {
            var available = Names("dotnet_gc_heap_size_bytes", "some_other_heap_used_bytes");
            var found = await MetricDiscovery.ProposeAsync(available, _ => Task.FromResult(12), TestContext.Current.CancellationToken);

            var heap = Find(found, MetricIndex.GcGen2HeapBytes);

            Assert.Equal("dotnet_gc_heap_size_bytes", heap.Chosen.Source);
            Assert.False(heap.Chosen.Inferred);

            // The suffix pass never ran, so the stray series is not even listed — the known match settled it.
            Assert.DoesNotContain(heap.Candidates, c => c.Source == "some_other_heap_used_bytes");
        }

        [Fact]
        public async Task ASuffixMatchThesePodsDoNotExportIsStillRejected()
        {
            var available = Names("otherapp_requests_total");
            var found = await MetricDiscovery.ProposeAsync(available, _ => Task.FromResult(0), TestContext.Current.CancellationToken);

            var rps = Find(found, MetricIndex.RequestsPerSecond);

            Assert.Equal(DiscoveryOutcome.NotFound, rps.Outcome);

            // Listed anyway: "exists but your pods do not export it" and "nothing like this exists" are
            // different problems with different fixes.
            Assert.Contains(rps.Candidates, c => c.Source == "otherapp_requests_total" && !c.IsEvidenced);
        }

        [Fact]
        public async Task TwoBespokeCountersStayAmbiguous()
        {
            var available = Names("orders_requests_total", "payments_requests_total");
            var found = await MetricDiscovery.ProposeAsync(available, _ => Task.FromResult(3), TestContext.Current.CancellationToken);

            Assert.Equal(DiscoveryOutcome.Ambiguous, Outcome(found, MetricIndex.RequestsPerSecond));
        }

        /// <summary>
        /// Container-level channels get no suffix rules, because cAdvisor does not vary its names per
        /// application — a rule there could only invent false candidates.
        /// </summary>
        [Fact]
        public async Task ContainerChannelsAreNotSuffixMatched()
        {
            var found = await MetricDiscovery.ProposeAsync(Names("myapp_memory_working_set_bytes"), _ => Task.FromResult(12), TestContext.Current.CancellationToken);

            Assert.Equal(DiscoveryOutcome.NotFound, Outcome(found, MetricIndex.MemoryWorkingSetBytes));
        }

        /// <summary>
        /// Found on a real cluster, not by reasoning: <c>_failures_total</c> matched cAdvisor's
        /// <c>container_memory_failures_total</c> — a page-fault counter — and offered it beside the
        /// application's genuine error counter.
        /// </summary>
        [Fact]
        public async Task InfrastructureSeriesAreNotOfferedAsApplicationSignals()
        {
            var available = Names("container_memory_failures_total", "labapp_errors_total");
            var found = await MetricDiscovery.ProposeAsync(available, _ => Task.FromResult(12), TestContext.Current.CancellationToken);

            var errors = Find(found, MetricIndex.ErrorRate);

            Assert.Equal(DiscoveryOutcome.Resolved, errors.Outcome);
            Assert.Equal("labapp_errors_total", errors.Chosen.Source);
            Assert.DoesNotContain(errors.Candidates, c => c.Source.StartsWith("container_", StringComparison.Ordinal));
        }

        [Fact]
        public async Task StacksAreNamedFromWhatIsPresent()
        {
            var stacks = MetricDiscovery.Stacks(Names(
                "jvm_memory_used_bytes", "nginx_connections_waiting", "container_cpu_usage_seconds_total"));

            Assert.Equal(["jvm", "nginx"], stacks);
        }

        /// <summary>
        /// The generated config must contain only resolved channels. Writing an ambiguous one would encode a
        /// guess in a file that then looks like a decision somebody made.
        /// </summary>
        [Fact]
        public async Task OnlyResolvedChannelsReachTheConfig()
        {
            var available = Names(
                "container_cpu_usage_seconds_total", "http_requests_total", "nginx_http_requests_total");

            var json = MetricDiscovery.ToConfigJson(await MetricDiscovery.ProposeAsync(available, _ => Task.FromResult(5), TestContext.Current.CancellationToken));

            Assert.Contains("\"CpuUsageRatio\"", json, StringComparison.Ordinal);
            Assert.DoesNotContain("\"ErrorRate\"", json, StringComparison.Ordinal);
        }

        [Fact]
        public async Task HistogramChannelsCarryTheirQuantileIntoTheConfig()
        {
            var json = MetricDiscovery.ToConfigJson(
                await MetricDiscovery.ProposeAsync(Names("http_request_duration_seconds_bucket"), _ => Task.FromResult(3), TestContext.Current.CancellationToken));

            Assert.Contains("\"quantile\": 0.99", json, StringComparison.Ordinal);
            Assert.Contains("HistogramSeconds", json, StringComparison.Ordinal);
        }

        private static HashSet<string> Names(params string[] names)
            => new(names, StringComparer.Ordinal);

        private static ChannelDiscovery Find(IReadOnlyList<ChannelDiscovery> found, MetricIndex metric)
        {
            for (var i = 0; i < found.Count; i++)
            {
                if (found[i].Metric == metric)
                {
                    return found[i];
                }
            }

            throw new InvalidOperationException($"no entry for {metric}");
        }

        private static DiscoveryOutcome Outcome(IReadOnlyList<ChannelDiscovery> found, MetricIndex metric)
            => Find(found, metric).Outcome;

        private static MetricCandidate Chosen(IReadOnlyList<ChannelDiscovery> found, MetricIndex metric)
            => Find(found, metric).Chosen;
    }
}
