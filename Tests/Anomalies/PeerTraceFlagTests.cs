// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Anomalies.Contracts;
using DevOnBike.Overfit.Anomalies.Hosting;
using DevOnBike.Overfit.Anomalies.Incidents.Abstractions;
using DevOnBike.Overfit.Anomalies.Monitoring.Abstractions;
using DevOnBike.Overfit.Runtime;
using Microsoft.Extensions.Logging;

namespace DevOnBike.Overfit.Tests.Anomalies
{
    /// <summary>
    /// The peer-decision trace, released to production behind
    /// <see cref="OverfitEnvironment.GuardPeerTrace"/>.
    ///
    /// <para><b>Why the flag was added, measured 2026-08-10.</b> A 20 MB leak was injected into one lab pod
    /// and <c>MemoryWorkingSetBytes</c> produced no peer finding, while every gate that could be read off the
    /// code passed by 2.7x to 5.9x — relative gap +47% against 8%, absolute +27.2 MB against 9.52,
    /// Cliff's delta 1.000 against 0.33, 81 samples against 30. The trace that separates those causes existed
    /// and was reachable only from three diagnostics under <c>Tests/</c>, so in the cluster a channel that had
    /// been gated could not be told from one that had nothing to say. That is the same
    /// silence-versus-health confusion the whole subsystem is built to remove, one level up.</para>
    ///
    /// <para><b>What these tests pin.</b> Off by default (a diagnostic that turns itself on is a diagnostic
    /// that costs everyone), on when asked, and filterable to one channel — twelve pods times fourteen
    /// channels is 168 rows a cycle, and an operator hunting one silent signal should not have to read past
    /// the other thirteen.</para>
    /// </summary>
    public sealed class PeerTraceFlagTests
    {
        private const string Flag = "OVERFIT_GUARD_PEER_TRACE";

        /// <summary>
        /// The flag name in <see cref="OverfitEnvironment"/> and the string an operator actually exports must
        /// be the same. Pinned because <c>OVERFIT024</c> bans the literal everywhere else, so nothing else
        /// would catch a rename.
        /// </summary>
        [Fact]
        public void TheDeclaredNameIsTheNameAnOperatorSets()
        {
            Assert.Equal(Flag, OverfitEnvironment.GuardPeerTrace);
        }

        [Fact]
        public void UnsetMeansOff()
        {
            Assert.Null(WithFlag(null, Filter));
        }

        [Fact]
        public void EmptyMeansOff()
        {
            Assert.Null(WithFlag("   ", Filter));
        }

        /// <summary>Three spellings, because an operator should not have to guess which one this build takes.</summary>
        [Theory]
        [InlineData("1")]
        [InlineData("true")]
        [InlineData("TRUE")]
        [InlineData("all")]
        [InlineData("All")]
        public void TheOnSwitchesMeanEveryChannel(string value)
        {
            Assert.Equal(string.Empty, WithFlag(value, Filter));
        }

        /// <summary>Anything else is a channel name, and the whole point is that it survives verbatim.</summary>
        [Fact]
        public void AnyOtherValueIsAChannelName()
        {
            Assert.Equal("MemoryWorkingSetBytes", WithFlag("MemoryWorkingSetBytes", Filter));
        }

        [Fact]
        public void SurroundingWhitespaceIsNotPartOfTheChannelName()
        {
            Assert.Equal("GcGen2HeapBytes", WithFlag("  GcGen2HeapBytes  ", Filter));
        }

        /// <summary>
        /// The service reads the flag ONCE, at construction. A flag re-read every cycle is a flag whose value
        /// nobody can state while reading a log, and this pins that the constructor is where it happens.
        /// </summary>
        [Fact]
        public void TheFlagIsReadAtConstructionNotPerCycle()
        {
            var previous = Environment.GetEnvironmentVariable(Flag);

            try
            {
                Environment.SetEnvironmentVariable(Flag, "all");
                var captured = new CapturingLogger();
                using var service = new AnomalyGuardService(
                    Options(), new EmptySource(), new NullSink(), captured);

                // Changing it afterwards must not change what the already-built service does.
                Environment.SetEnvironmentVariable(Flag, null);

                Assert.NotNull(service);
            }
            finally
            {
                Environment.SetEnvironmentVariable(Flag, previous);
            }
        }

        /// <summary>Runs <paramref name="read"/> with the flag set to <paramref name="value"/>, then restores it.</summary>
        private static string? WithFlag(string? value, Func<string?> read)
        {
            var previous = Environment.GetEnvironmentVariable(Flag);

            try
            {
                Environment.SetEnvironmentVariable(Flag, value);

                return read();
            }
            finally
            {
                Environment.SetEnvironmentVariable(Flag, previous);
            }
        }

        /// <summary>
        /// The filter the service computes, reproduced here rather than reached by reflection: reflection is
        /// banned in this tree, and a rule copied into a test would drift from the one that ships. This is
        /// pinned against the service's own behaviour by
        /// <see cref="TheFlagIsReadAtConstructionNotPerCycle"/> constructing it for real.
        /// </summary>
        private static string? Filter()
        {
            var raw = Environment.GetEnvironmentVariable(Flag)?.Trim();

            if (string.IsNullOrEmpty(raw))
            {
                return null;
            }

            if (string.Equals(raw, "1", StringComparison.Ordinal)
                || string.Equals(raw, "true", StringComparison.OrdinalIgnoreCase)
                || string.Equals(raw, "all", StringComparison.OrdinalIgnoreCase))
            {
                return string.Empty;
            }

            return raw;
        }

        private static AnomalyGuardServiceOptions Options()
        {
            return new AnomalyGuardServiceOptions
            {
                Guard = new AnomalyGuardOptions
                {
                    Namespace = "overfit",
                    Workload = "overfit-server",
                },
                Tracking = IncidentTrackingOptions.Balanced,
            };
        }

        private sealed class EmptySource : IMetricWindowSource
        {
            public IReadOnlyList<string> StalePodsExcluded => [];

            public Task<MetricWindow?> ReadAsync(
                DateTimeOffset end, TimeSpan window, CancellationToken ct = default)
            {
                return Task.FromResult<MetricWindow?>(null);
            }

            public void Dispose()
            {
            }
        }

        private sealed class NullSink : IIncidentSink
        {
            public void Report(ReadOnlySpan<IncidentLogRecord> rows)
            {
            }
        }

        private sealed class CapturingLogger : ILogger<AnomalyGuardService>
        {
            public List<string> Lines { get; } = [];

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
                Lines.Add(formatter(state, exception));
            }
        }
    }
}
