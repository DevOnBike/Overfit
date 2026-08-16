// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Anomalies.Contracts;
using DevOnBike.Overfit.Anomalies.Hosting;
using DevOnBike.Overfit.Anomalies.Incidents;
using DevOnBike.Overfit.Anomalies.Incidents.Abstractions;
using DevOnBike.Overfit.Anomalies.Monitoring.Abstractions;
using Microsoft.Extensions.Logging.Abstractions;

namespace DevOnBike.Overfit.Tests.Anomalies
{
    /// <summary>
    /// The <see cref="BackgroundService"/> loop itself — <c>ExecuteAsync</c>, not the cycle inside it.
    ///
    /// <para><b>Why this is a separate subject from <see cref="AnomalyGuardServiceCycleTests"/>.</b> Those
    /// drive <c>RunCycleAsync</c> directly, which is the right way to test what one cycle decides and lets
    /// them pass a historical <c>now</c>. Nothing drove the loop around it, and the loop owns properties the
    /// cycle cannot have: that there is a next cycle at all, that a failing one does not end the run, and
    /// that cancellation stops it rather than hanging a pod until the kubelet loses patience.</para>
    ///
    /// <para><b>The failure these exist against is the one this subsystem keeps producing.</b> A guard that
    /// has stopped cycling and a cluster with nothing wrong are the same observation from outside: no
    /// incidents, a Ready pod, a container that has not exited. The loop is the only thing standing between
    /// those two, and it had nothing checking it.</para>
    ///
    /// <para>Windows are never built here. The loop does not care what a window contains — a source
    /// returning <see langword="null"/> drives a complete cycle to <see cref="GuardCycleKind.Blind"/>, which
    /// is a legitimate outcome and keeps these tests about cadence, survival and shutdown.</para>
    /// </summary>
    public sealed class AnomalyGuardLoopTests
    {
        /// <summary>
        /// Short enough that a handful of cycles fit in a fast test, long enough that a loop which ignored it
        /// entirely would still be distinguishable from one that honours it.
        /// </summary>
        private static readonly TimeSpan Cadence = TimeSpan.FromMilliseconds(50);

        /// <summary>
        /// Generous against the cadence. These assert that cycles HAPPEN, never how many land in a period —
        /// a timing test that counts iterations on a shared CI box is a flake with a schedule.
        /// </summary>
        private static readonly TimeSpan Patience = TimeSpan.FromSeconds(10);

        [Fact]
        public async Task TheLoopKeepsCyclingUntilItIsStopped()
        {
            using var source = new CountingSource();
            using var service = Service(source);

            await service.StartAsync(CancellationToken.None);

            var reached = await source.WaitForReads(3, Patience);

            await Stop(service);

            Assert.True(reached, $"the loop produced {source.Reads} cycle(s) in {Patience}, expected 3");
        }

        [Fact]
        public async Task ASourceThatThrowsEveryTimeDoesNotEndTheLoop()
        {
            // The operationally important one. A guard whose source is broken must keep failing LOUDLY —
            // counted by GuardTelemetry, logged every cycle — rather than exit its loop and leave a Ready pod
            // reporting nothing, which is indistinguishable from a healthy cluster at every layer above.
            using var source = new CountingSource { Throw = true };
            using var service = Service(source);

            await service.StartAsync(CancellationToken.None);

            var reached = await source.WaitForReads(3, Patience);

            await Stop(service);

            Assert.True(reached, $"the loop stopped after {source.Reads} failing cycle(s)");
        }

        [Fact]
        public async Task StoppingEndsTheLoopAndNoFurtherCyclesRun()
        {
            using var source = new CountingSource();
            using var service = Service(source);

            await service.StartAsync(CancellationToken.None);
            await source.WaitForReads(2, Patience);
            await Stop(service);

            var settled = source.Reads;

            // Several cadences' worth. If the token were not threaded into WaitForNextTickAsync the loop
            // would still be running here and this count would move.
            await Task.Delay(TimeSpan.FromMilliseconds(400));

            Assert.Equal(settled, source.Reads);
        }

        [Fact]
        public async Task StoppingCompletesRatherThanWaitingOutTheCadence()
        {
            // A cadence far longer than the test's patience: if shutdown waited for the next tick instead of
            // observing the token, this could only pass by taking a minute.
            // Long enough that "returned before the next tick" is a real distinction, short enough that
            // waiting for the first one costs seconds rather than a minute.
            var cadence = TimeSpan.FromSeconds(2);

            using var source = new CountingSource();
            using var service = Service(source, cadence);

            await service.StartAsync(CancellationToken.None);

            // Wait for the first cycle, so the loop is demonstrably PARKED on the timer when it is stopped.
            // Without this the test passed under a mutation that removed the stopping token — it was
            // stopping a loop that had not reached the wait yet, which is not the property it names.
            Assert.True(await source.WaitForReads(1, Patience), "the loop never reached its first cycle");

            var stopping = service.StopAsync(CancellationToken.None);
            var promptly = await Task.WhenAny(stopping, Task.Delay(cadence / 2)) == stopping;

            Assert.True(promptly, "StopAsync did not return before half a cadence — it waited for the tick");

            await stopping;
        }

        /// <summary>
        /// Stops the service, or gives up after <see cref="Patience"/> and says so.
        ///
        /// <para><b>Bounded because an unbounded await was measured to HANG rather than fail.</b> With the
        /// stopping token removed from <c>WaitForNextTickAsync</c> — the regression these tests exist to
        /// catch — <c>StopAsync</c> never returns, so a plain <c>await</c> turned every one of these into a
        /// test that blocks the suite instead of reddening it. A hang is a worse failure than a red: it has
        /// no message, it stops everything behind it, and on CI it reads as an infrastructure problem.</para>
        /// </summary>
        private static async Task<bool> Stop(AnomalyGuardService service)
        {
            var stopping = service.StopAsync(CancellationToken.None);

            if (await Task.WhenAny(stopping, Task.Delay(Patience)) != stopping)
            {
                return false;
            }

            await stopping;

            return true;
        }

        private static AnomalyGuardService Service(IMetricWindowSource source, TimeSpan? cadence = null)
        {
            return new AnomalyGuardService(
                new AnomalyGuardServiceOptions
                {
                    Cadence = cadence ?? Cadence,
                    Guard = new AnomalyGuardOptions
                    {
                        Namespace = "overfit",
                        Workload = "overfit-server",
                        Grouping = IncidentGroupingOptions.Balanced with
                        {
                            Topology = TopologyWeights.SingleNode
                        },
                    },
                    Tracking = IncidentTrackingOptions.Balanced,
                },
                source,
                new NullSink(),
                NullLogger<AnomalyGuardService>.Instance);
        }

        /// <summary>
        /// Counts reads and, optionally, fails every one of them. Returns <see langword="null"/> otherwise,
        /// which the cycle treats as a cluster it cannot see — a real outcome, and one that needs no window.
        /// </summary>
        private sealed class CountingSource : IMetricWindowSource
        {
            private readonly SemaphoreSlim _read = new(0);
            private int _reads;

            public bool Throw
            {
                get; init;
            }

            public int Reads => Volatile.Read(ref _reads);

            public IReadOnlyList<string> StalePodsExcluded => [];

            public Task<MetricWindow?> ReadAsync(DateTimeOffset end, TimeSpan window, CancellationToken ct)
            {
                Interlocked.Increment(ref _reads);
                _read.Release();

                if (Throw)
                {
                    throw new InvalidOperationException("the source is broken, deliberately");
                }

                return Task.FromResult<MetricWindow?>(null);
            }

            /// <summary>
            /// Waits for <paramref name="count"/> reads rather than sleeping for a duration: a fixed sleep
            /// either flakes on a loaded box or wastes the time it was padded with.
            /// </summary>
            public async Task<bool> WaitForReads(int count, TimeSpan patience)
            {
                using var deadline = new CancellationTokenSource(patience);

                try
                {
                    for (var i = 0; i < count; i++)
                    {
                        await _read.WaitAsync(deadline.Token);
                    }
                }
                catch (OperationCanceledException)
                {
                    return false;
                }

                return true;
            }

            public void Dispose()
            {
                _read.Dispose();
            }
        }

        private sealed class NullSink : IIncidentSink
        {
            public void Report(ReadOnlySpan<IncidentLogRecord> rows)
            {
            }
        }
    }
}
