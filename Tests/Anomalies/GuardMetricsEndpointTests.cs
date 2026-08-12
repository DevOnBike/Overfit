// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Net;
using System.Net.Sockets;
using DevOnBike.Overfit.Anomalies.Contracts;
using DevOnBike.Overfit.Anomalies.Hosting;
using DevOnBike.Overfit.Anomalies.Incidents;
using DevOnBike.Overfit.Anomalies.Incidents.Abstractions;
using DevOnBike.Overfit.Anomalies.Monitoring;
using DevOnBike.Overfit.Anomalies.Monitoring.Abstractions;
using DevOnBike.Overfit.Runtime;
using DevOnBike.Overfit.Tests.TestSupport;
using Microsoft.Extensions.Logging;

namespace DevOnBike.Overfit.Tests.Anomalies
{
    /// <summary>
    /// The guard's own metrics channel, driven over a real socket.
    ///
    /// <para><b>The property under test is that one bad request cannot end the channel.</b> The serve loop
    /// used to catch two transport exception types and nothing else, so anything else escaped the loop, and
    /// the loop's task was discarded — the endpoint stopped answering, the process kept running, and nothing
    /// said so. From outside that is indistinguishable from a healthy guard with nothing to report, which is
    /// the one failure this whole subsystem exists to make impossible.</para>
    ///
    /// <para><b>What these tests do NOT reproduce, deliberately.</b> The comment on <c>ServeAsync</c> names a
    /// concrete cause — a scrape enumerating guard state while a cycle mutates it. That race is not reachable:
    /// <c>RunCycle</c>, <c>Acknowledge</c> and <c>ActiveSuppressions</c> all take the same
    /// <c>lock (_gate)</c>, so a concurrent scrape blocks rather than enumerating a mutating list. A test
    /// written against it could not fail. The injection here is instead an <see cref="IClock"/> whose getter
    /// throws — a non-transport exception raised at exactly the point inside the request handler that the
    /// shipped defect escaped from, deterministically and with no race.</para>
    /// </summary>
    public sealed class GuardMetricsEndpointTests
    {
        private static readonly DateTimeOffset T0 = new(2026, 8, 2, 12, 0, 0, TimeSpan.Zero);

        /// <summary>
        /// One request fails for a reason that is not a transport fault; the next scrape must still be
        /// answered.
        ///
        /// <para>Step one is the capability check and it is not decoration: without it, a passing step three
        /// would be indistinguishable from an endpoint that never served anything at all.</para>
        /// </summary>
        [Fact]
        public async Task AnUnexpectedFailureOnOneRequestDoesNotKillTheMetricsChannel()
        {
            var port = FreePort();
            var guard = Guard();
            var logger = new CapturingLogger();
            var clock = new ThrowingClock();

            using var endpoint = GuardMetricsEndpoint.TryStart(guard.Telemetry, logger, port, guard, clock);

            Assert.NotNull(endpoint);

            var baseUrl = $"http://127.0.0.1:{port}";
            using var scrape = new HttpClient { Timeout = TimeSpan.FromSeconds(5) };

            // 1. It serves at all.
            using (var before = await scrape.GetAsync($"{baseUrl}/metrics"))
            {
                Assert.Equal(HttpStatusCode.OK, before.StatusCode);
                Assert.Contains(
                    "overfit_guard_last_cycle_timestamp_seconds",
                    await before.Content.ReadAsStringAsync(),
                    StringComparison.Ordinal);
            }

            // 2. One request throws an InvalidOperationException inside the handler. It is never answered —
            //    the handler dies before writing a response — so this request is abandoned rather than
            //    awaited, and its own client is torn down below.
            var poison = new HttpClient { Timeout = TimeSpan.FromSeconds(5) };
            var poisoned = poison.GetAsync($"{baseUrl}/suppressions");

            // The clock counts its own reads, so "the handler ran" is observable WITHOUT depending on the
            // fix being present. A fixed sleep here would let a slow box turn step 3 into a test that passes
            // because nothing was ever injected.
            await WaitUntil(
                () => clock.Reads > 0,
                "the /suppressions handler was never reached, so nothing was injected and step 3 proves "
                + "nothing");

            // 3. The channel must still be there.
            HttpResponseMessage? after = null;

            try
            {
                after = await scrape.GetAsync($"{baseUrl}/metrics");
            }
            catch (Exception ex) when (ex is TaskCanceledException or HttpRequestException)
            {
                Assert.Fail(
                    "The metrics channel stopped serving after ONE non-transport request failure: the second "
                    + $"/metrics scrape ended in {ex.GetType().Name}. That is the defect this endpoint's "
                    + "catch-all exists to prevent — the guard keeps cycling while its own observability "
                    + "channel is dead, which reads from outside as a healthy guard.");
            }

            using (after)
            {
                Assert.Equal(HttpStatusCode.OK, after!.StatusCode);
                Assert.Contains(
                    "overfit_guard_last_cycle_timestamp_seconds",
                    await after.Content.ReadAsStringAsync(),
                    StringComparison.Ordinal);
            }

            // The failure is announced rather than swallowed. Error, not Debug: a client hanging up mid-write
            // is the client's business, and this was not that.
            var errors = logger.Errors();

            Assert.Single(errors);
            Assert.Contains("not a transport fault", errors[0], StringComparison.Ordinal);

            poison.Dispose();
            await ObserveAsync(poisoned);
        }

        /// <summary>
        /// The control. Without it, the test above would pass against an endpoint that answers 200 to
        /// everything including the request that was supposed to fail.
        /// </summary>
        [Fact]
        public async Task AHealthyScrapeSequenceKeepsAnsweringAndLogsNoError()
        {
            var port = FreePort();
            var guard = Guard();
            var logger = new CapturingLogger();
            var clock = new ManualClock(new DateTimeOffset(2026, 1, 1, 0, 0, 0, TimeSpan.Zero));

            using var endpoint = GuardMetricsEndpoint.TryStart(guard.Telemetry, logger, port, guard, clock);

            Assert.NotNull(endpoint);

            var baseUrl = $"http://127.0.0.1:{port}";
            using var scrape = new HttpClient { Timeout = TimeSpan.FromSeconds(5) };

            using (var first = await scrape.GetAsync($"{baseUrl}/metrics"))
            {
                Assert.Equal(HttpStatusCode.OK, first.StatusCode);
            }

            using (var suppressions = await scrape.GetAsync($"{baseUrl}/suppressions"))
            {
                Assert.Equal(HttpStatusCode.OK, suppressions.StatusCode);

                // The instant is the injected clock's, not the wall clock's — a leaked real timestamp is
                // visible here rather than plausible.
                Assert.Contains(
                    "0 active suppression(s) at 2026-01-01 00:00:00Z",
                    await suppressions.Content.ReadAsStringAsync(),
                    StringComparison.Ordinal);
            }

            using (var second = await scrape.GetAsync($"{baseUrl}/metrics"))
            {
                Assert.Equal(HttpStatusCode.OK, second.StatusCode);
            }

            Assert.Empty(logger.Errors());
        }

        /// <summary>
        /// <c>POST /ack</c> silences a finding, so it fails closed: an unauthenticated call is refused and
        /// nothing is muted.
        ///
        /// <para>The token is a process-wide static read once from the environment, so the expected refusal
        /// code is read from the same environment rather than assumed. Both refusals name the variable, and
        /// the two assertions that matter — not 200, nothing suppressed — hold either way.</para>
        /// </summary>
        [Fact]
        public async Task AckWithoutATokenIsRefusedAndSuppressesNothing()
        {
            var port = FreePort();
            var guard = Guard();
            var logger = new CapturingLogger();
            var clock = new ManualClock(new DateTimeOffset(2026, 1, 1, 0, 0, 0, TimeSpan.Zero));

            using var endpoint = GuardMetricsEndpoint.TryStart(guard.Telemetry, logger, port, guard, clock);

            Assert.NotNull(endpoint);

            var baseUrl = $"http://127.0.0.1:{port}";
            using var client = new HttpClient { Timeout = TimeSpan.FromSeconds(5) };
            using var content = new StringContent(string.Empty);
            using var response = await client.PostAsync($"{baseUrl}/ack?id=1&kind=noise", content);

            var configured = Environment.GetEnvironmentVariable(OverfitEnvironment.GuardAckToken)?.Trim()
                             is { Length: > 0 };

            Assert.Equal(
                configured ? HttpStatusCode.Unauthorized : HttpStatusCode.ServiceUnavailable,
                response.StatusCode);

            Assert.Contains(
                OverfitEnvironment.GuardAckToken,
                await response.Content.ReadAsStringAsync(),
                StringComparison.Ordinal);

            Assert.Empty(guard.ActiveSuppressions(clock.UtcNow));
        }

        /// <summary>
        /// The transparency guarantee, end to end: something an operator muted is enumerable over HTTP, with
        /// the subject, the reason and the expiry.
        ///
        /// <para>A mute nobody can read back is indistinguishable from a detector that stopped working, and
        /// that equivalence is the whole argument for letting an operator silence anything at all.</para>
        /// </summary>
        [Fact]
        public async Task SuppressionsListsWhatIsMuted()
        {
            var sink = new CapturingSink();
            var guard = GuardThatReports(sink);

            guard.RunCycle(OneReplicaHigh(), T0.AddMinutes(20));

            Assert.NotEmpty(sink.Ids);

            var incident = sink.Ids[0];

            guard.Acknowledge(
                incident, OperatorLabelKind.Noise, TimeSpan.FromHours(1), "known sawtooth",
                T0.AddMinutes(21));

            var port = FreePort();
            var clock = new ManualClock(T0.AddMinutes(22));

            using var endpoint = GuardMetricsEndpoint.TryStart(
                guard.Telemetry, new CapturingLogger(), port, guard, clock);

            Assert.NotNull(endpoint);

            using var client = new HttpClient { Timeout = TimeSpan.FromSeconds(5) };
            using var response = await client.GetAsync($"http://127.0.0.1:{port}/suppressions");

            Assert.Equal(HttpStatusCode.OK, response.StatusCode);

            var body = await response.Content.ReadAsStringAsync();

            Assert.Contains("1 active suppression(s) at 2026-08-02 12:22:00Z", body, StringComparison.Ordinal);
            Assert.Contains($"incident {incident}", body, StringComparison.Ordinal);
            Assert.Contains("pod-0", body, StringComparison.Ordinal);
            Assert.Contains("until 2026-08-02 13:21:00Z", body, StringComparison.Ordinal);
            Assert.Contains("\"known sawtooth\"", body, StringComparison.Ordinal);
        }

        private static AnomalyGuard Guard()
        {
            return new AnomalyGuard(
                new AnomalyGuardOptions
                {
                    Namespace = "lab",
                    Workload = "svc",
                    MinimumHistoryDays = 0,
                    Grouping = IncidentGroupingOptions.Balanced with { Topology = TopologyWeights.SingleNode },
                },
                new NullSink(),
                IncidentTrackingOptions.Balanced);
        }

        /// <summary>
        /// The shape 49 tests in this folder use to make one replica visibly different from its peers, with
        /// floors sized to the fixture rather than to the shipped 256 MiB default.
        /// </summary>
        private static AnomalyGuard GuardThatReports(CapturingSink sink)
        {
            var floors = new double[(int)MetricIndex.Count];

            floors[(int)MetricIndex.MemoryWorkingSetBytes] = 1.089e6;

            return new AnomalyGuard(
                new AnomalyGuardOptions
                {
                    Namespace = "lab",
                    Workload = "svc",
                    PodTopology = new FakeTopology(),
                    MinimumHistoryDays = 0,
                    DecomposeCommonMode = false,
                    MinAbsoluteGap = floors,
                    MinAbsoluteTrendChange = floors,
                    Grouping = IncidentGroupingOptions.Balanced with { Topology = TopologyWeights.SingleNode },
                },
                sink,
                IncidentTrackingOptions.Balanced);
        }

        /// <summary>Flat memory, one replica far above the rest — a peer question with an absolute gap.</summary>
        private static MetricWindow OneReplicaHigh()
        {
            var names = new List<string> { "pod-0", "pod-1", "pod-2", "pod-3" };
            var window = new MetricWindow(names, 80, T0, TimeSpan.FromSeconds(15));
            var rng = new Random(20260802);

            for (var pod = 0; pod < names.Count; pod++)
            {
                var memory = window.Series(pod, MetricIndex.MemoryWorkingSetBytes);
                var cpu = window.Series(pod, MetricIndex.CpuUsageRatio);
                var level = pod == 0 ? 90e6 : 40e6;

                for (var i = 0; i < window.Length; i++)
                {
                    memory[i] = level * (1.0 + ((rng.NextDouble() - 0.5) * 0.01));
                    cpu[i] = 0.2;
                }
            }

            return window;
        }

        private static int FreePort()
        {
            var l = new TcpListener(IPAddress.Loopback, 0);

            l.Start();

            var port = ((IPEndPoint)l.LocalEndpoint).Port;

            l.Stop();

            return port;
        }

        /// <summary>Waits for something another thread does, or fails the test by name after ten seconds.</summary>
        private static async Task WaitUntil(Func<bool> condition, string because)
        {
            var deadline = Environment.TickCount64 + 10_000;

            // #pragma BOUND: the loop cannot run past the deadline above — ten seconds of 10 ms waits.
            while (Environment.TickCount64 < deadline)
            {
                if (condition())
                {
                    return;
                }

                await Task.Delay(10);
            }

            Assert.Fail(because);
        }

        /// <summary>
        /// Observes a request that was never going to be answered, so it does not end as an unobserved task
        /// exception after the test has finished.
        /// </summary>
        private static async Task ObserveAsync(Task<HttpResponseMessage> request)
        {
            try
            {
                (await request).Dispose();
            }
            catch (Exception ex)
                when (ex is TaskCanceledException or HttpRequestException or ObjectDisposedException)
            {
            }
        }

        private sealed class NullSink : IIncidentSink
        {
            public void Report(ReadOnlySpan<IncidentLogRecord> rows)
            {
            }
        }

        private sealed class CapturingSink : IIncidentSink
        {
            public List<long> Ids { get; } = [];

            public void Report(ReadOnlySpan<IncidentLogRecord> rows)
            {
                for (var i = 0; i < rows.Length; i++)
                {
                    Ids.Add(rows[i].IncidentId);
                }
            }
        }

        private sealed class FakeTopology : IPodTopology
        {
            public bool TryResolve(string pod, out PodPlacement placement)
            {
                placement = new PodPlacement("svc", "rs-1", "node-0", string.Empty, T0.AddDays(-1));

                return true;
            }
        }

        /// <summary>
        /// A clock that throws where the endpoint reads it, and counts the reads.
        ///
        /// <para><c>InvalidOperationException</c> on purpose: it is neither of the two transport types the
        /// serve loop filters, so it is exactly the class of failure that used to escape the loop. The count
        /// is what makes "the handler actually ran" observable to the test without relying on the fix.</para>
        /// </summary>
        private sealed class ThrowingClock : IClock
        {
            private int _reads;

            public int Reads => Volatile.Read(ref _reads);

            public DateTimeOffset UtcNow
            {
                get
                {
                    Interlocked.Increment(ref _reads);

                    throw new InvalidOperationException(
                        "Collection was modified; enumeration operation may not execute.");
                }
            }
        }

        /// <summary>Records what the endpoint said, from the serving thread.</summary>
        private sealed class CapturingLogger : ILogger
        {
            private readonly List<(LogLevel Level, string Message)> _records = [];

            public IDisposable? BeginScope<TState>(TState state)
                where TState : notnull
            {
                return null;
            }

            public bool IsEnabled(LogLevel logLevel) => true;

            public void Log<TState>(
                LogLevel logLevel,
                EventId eventId,
                TState state,
                Exception? exception,
                Func<TState, Exception?, string> formatter)
            {
                var message = formatter(state, exception);

                lock (_records)
                {
                    _records.Add((logLevel, message));
                }
            }

            public List<string> Errors()
            {
                var errors = new List<string>();

                lock (_records)
                {
                    for (var i = 0; i < _records.Count; i++)
                    {
                        if (_records[i].Level == LogLevel.Error)
                        {
                            errors.Add(_records[i].Message);
                        }
                    }
                }

                return errors;
            }
        }
    }
}
