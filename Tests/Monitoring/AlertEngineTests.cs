// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Anomalies.Alerting;
using DevOnBike.Overfit.Anomalies.Monitoring.Abstractions;
using DevOnBike.Overfit.Anomalies.Monitoring.Contracts;

namespace DevOnBike.Overfit.Tests.Monitoring
{
    public sealed class AlertEngineTests
    {
        // -------------------------------------------------------------------------
        // Test sink implementations
        // -------------------------------------------------------------------------

        private sealed class CapturingSink : IAlertSink
        {
            private readonly List<AlertEvent> _received = [];
            public IReadOnlyList<AlertEvent> Received => _received;
            public int SendCount => _received.Count;

            public Task SendAsync(AlertEvent alert, CancellationToken ct = default)
            {
                lock (_received) { _received.Add(alert); }
                return Task.CompletedTask;
            }
        }

        private sealed class ThrowingSink : IAlertSink
        {
            public Task SendAsync(AlertEvent alert, CancellationToken ct = default)
                => Task.FromException(new InvalidOperationException("Sink failure"));
        }

        private sealed class SlowSink(int delayMs = 50) : IAlertSink
        {
            public Task SendAsync(AlertEvent alert, CancellationToken ct = default)
                => Task.Delay(delayMs, ct);
        }

        // -------------------------------------------------------------------------
        // Helpers
        // -------------------------------------------------------------------------

        private static AlertEngineConfig MakeConfig(
            float threshold = 0.8f,
            float criticalThreshold = 0.95f,
            TimeSpan? cooldown = null)
            => new()
            {
                AlertThreshold = threshold,
                CriticalThreshold = criticalThreshold,
                CooldownDuration = cooldown ?? TimeSpan.FromMilliseconds(50)
            };

        // -------------------------------------------------------------------------
        // Constructor validation
        // -------------------------------------------------------------------------

        [Fact]
        public void Constructor_WhenNoSinks_ThenThrowsArgumentException()
            => Assert.Throws<ArgumentException>(() => new AlertEngine(null, Array.Empty<IAlertSink>()));

        [Fact]
        public void Constructor_WhenAlertThresholdIsZero_ThenThrowsArgumentException()
        {
            var config = new AlertEngineConfig
            {
                AlertThreshold = 0f
            };
            Assert.Throws<ArgumentException>(() => new AlertEngine(config, new CapturingSink()));
        }

        [Fact]
        public void Constructor_WhenAlertThresholdExceeds1_ThenThrowsArgumentException()
        {
            var config = new AlertEngineConfig
            {
                AlertThreshold = 1.1f
            };
            Assert.Throws<ArgumentException>(() => new AlertEngine(config, new CapturingSink()));
        }

        [Fact]
        public void Constructor_WhenCriticalThresholdBelowAlertThreshold_ThenThrowsArgumentException()
        {
            var config = new AlertEngineConfig
            {
                AlertThreshold = 0.9f,
                CriticalThreshold = 0.8f
            };
            Assert.Throws<ArgumentException>(() => new AlertEngine(config, new CapturingSink()));
        }

        // -------------------------------------------------------------------------
        // TryAlert — threshold
        // -------------------------------------------------------------------------

        [Fact]
        public async Task TryAlert_WhenScoreBelowThreshold_ThenReturnsFalse()
        {
            await using var engine = new AlertEngine(MakeConfig(threshold: 0.8f), new CapturingSink());
            Assert.False(engine.TryAlert("pod-1", anomalyScore: 0.79f, reconstructionMse: 0.01f));
        }

        [Fact]
        public async Task TryAlert_WhenScoreAtThreshold_ThenReturnsTrue()
        {
            await using var engine = new AlertEngine(MakeConfig(threshold: 0.8f), new CapturingSink());
            Assert.True(engine.TryAlert("pod-1", anomalyScore: 0.8f, reconstructionMse: 0.01f));
        }

        [Fact]
        public async Task TryAlert_WhenScoreAboveThreshold_ThenReturnsTrue()
        {
            await using var engine = new AlertEngine(MakeConfig(threshold: 0.8f), new CapturingSink());
            Assert.True(engine.TryAlert("pod-1", anomalyScore: 0.99f, reconstructionMse: 0.05f));
        }

        [Fact]
        public async Task TryAlert_WhenScoreAtExactly1_ThenReturnsTrue()
        {
            await using var engine = new AlertEngine(MakeConfig(threshold: 0.8f), new CapturingSink());
            Assert.True(engine.TryAlert("pod-1", anomalyScore: 1.0f, reconstructionMse: 0.1f));
        }

        // -------------------------------------------------------------------------
        // TryAlert — severity
        // -------------------------------------------------------------------------

        [Fact]
        public async Task TryAlert_WhenScoreBetweenThresholds_ThenSeverityIsWarning()
        {
            var sink = new CapturingSink();
            var config = MakeConfig(threshold: 0.8f, criticalThreshold: 0.95f);
            await using var engine = new AlertEngine(config, sink);

            engine.TryAlert("pod-1", anomalyScore: 0.85f, reconstructionMse: 0.02f);
            await engine.DisposeAsync();

            Assert.Equal(AlertSeverity.Warning, sink.Received[0].Severity);
        }

        [Fact]
        public async Task TryAlert_WhenScoreAtCriticalThreshold_ThenSeverityIsCritical()
        {
            var sink = new CapturingSink();
            var config = MakeConfig(threshold: 0.8f, criticalThreshold: 0.95f);
            await using var engine = new AlertEngine(config, sink);

            engine.TryAlert("pod-1", anomalyScore: 0.95f, reconstructionMse: 0.05f);
            await engine.DisposeAsync();

            Assert.Equal(AlertSeverity.Critical, sink.Received[0].Severity);
        }

        [Fact]
        public async Task TryAlert_WhenScoreAboveCriticalThreshold_ThenSeverityIsCritical()
        {
            var sink = new CapturingSink();
            var config = MakeConfig(threshold: 0.8f, criticalThreshold: 0.95f);
            await using var engine = new AlertEngine(config, sink);

            engine.TryAlert("pod-1", anomalyScore: 1.0f, reconstructionMse: 0.1f);
            await engine.DisposeAsync();

            Assert.Equal(AlertSeverity.Critical, sink.Received[0].Severity);
        }

        // -------------------------------------------------------------------------
        // TryAlert — sink delivery
        // -------------------------------------------------------------------------

        [Fact]
        public async Task TryAlert_WhenAlertFired_ThenSinkReceivesEvent()
        {
            var sink = new CapturingSink();
            await using var engine = new AlertEngine(MakeConfig(), sink);

            engine.TryAlert("pod-1", anomalyScore: 0.9f, reconstructionMse: 0.03f);
            await engine.DisposeAsync();

            Assert.Equal(1, sink.SendCount);
        }

        [Fact]
        public async Task TryAlert_WhenAlertFired_ThenEventHasCorrectPodName()
        {
            var sink = new CapturingSink();
            await using var engine = new AlertEngine(MakeConfig(), sink);

            engine.TryAlert("my-pod", anomalyScore: 0.9f, reconstructionMse: 0.03f);
            await engine.DisposeAsync();

            Assert.Equal("my-pod", sink.Received[0].PodName);
        }

        [Fact]
        public async Task TryAlert_WhenAlertFired_ThenEventHasCorrectScore()
        {
            var sink = new CapturingSink();
            await using var engine = new AlertEngine(MakeConfig(), sink);

            engine.TryAlert("pod-1", anomalyScore: 0.87f, reconstructionMse: 0.025f);
            await engine.DisposeAsync();

            Assert.Equal(0.87f, sink.Received[0].AnomalyScore);
            Assert.Equal(0.025f, sink.Received[0].ReconstructionMse);
        }

        [Fact]
        public async Task TryAlert_WhenAlertFired_ThenEventHasUtcTimestamp()
        {
            var sink = new CapturingSink();
            var before = DateTime.UtcNow;
            await using var engine = new AlertEngine(MakeConfig(), sink);

            engine.TryAlert("pod-1", anomalyScore: 0.9f, reconstructionMse: 0.01f);
            await engine.DisposeAsync();

            var after = DateTime.UtcNow;
            Assert.InRange(sink.Received[0].DetectedAt, before, after);
            Assert.Equal(DateTimeKind.Utc, sink.Received[0].DetectedAt.Kind);
        }

        [Fact]
        public async Task TryAlert_WhenMultipleSinks_ThenAllSinksReceiveEvent()
        {
            var sink1 = new CapturingSink();
            var sink2 = new CapturingSink();
            await using var engine = new AlertEngine(MakeConfig(), sink1, sink2);

            engine.TryAlert("pod-1", anomalyScore: 0.9f, reconstructionMse: 0.03f);
            await engine.DisposeAsync();

            Assert.Equal(1, sink1.SendCount);
            Assert.Equal(1, sink2.SendCount);
        }

        [Fact]
        public async Task TryAlert_WhenOneSinkThrows_ThenOtherSinkStillReceivesEvent()
        {
            var goodSink = new CapturingSink();
            await using var engine = new AlertEngine(MakeConfig(), new ThrowingSink(), goodSink);

            engine.TryAlert("pod-1", anomalyScore: 0.9f, reconstructionMse: 0.03f);
            await engine.DisposeAsync();

            Assert.Equal(1, goodSink.SendCount);
        }

        // -------------------------------------------------------------------------
        // TryAlert — cooldown
        // -------------------------------------------------------------------------

        [Fact]
        public async Task TryAlert_WhenInCooldown_ThenReturnsFalse()
        {
            var config = MakeConfig(cooldown: TimeSpan.FromHours(1)); // very long cooldown
            await using var engine = new AlertEngine(config, new CapturingSink());

            engine.TryAlert("pod-1", anomalyScore: 0.9f, reconstructionMse: 0.03f); // first — fires
            var second = engine.TryAlert("pod-1", anomalyScore: 0.9f, reconstructionMse: 0.03f);

            Assert.False(second);
        }

        [Fact]
        public async Task TryAlert_WhenCooldownExpired_ThenReturnsTrue()
        {
            var config = MakeConfig(cooldown: TimeSpan.FromMilliseconds(30));
            await using var engine = new AlertEngine(config, new CapturingSink());

            engine.TryAlert("pod-1", anomalyScore: 0.9f, reconstructionMse: 0.03f); // first — fires
            await Task.Delay(60); // wait for cooldown to expire

            var second = engine.TryAlert("pod-1", anomalyScore: 0.9f, reconstructionMse: 0.03f);
            Assert.True(second);
        }

        [Fact]
        public async Task TryAlert_WhenDifferentPods_ThenCooldownIsIndependent()
        {
            var config = MakeConfig(cooldown: TimeSpan.FromHours(1));
            await using var engine = new AlertEngine(config, new CapturingSink());

            engine.TryAlert("pod-A", anomalyScore: 0.9f, reconstructionMse: 0.03f); // fires for pod-A
            var podB = engine.TryAlert("pod-B", anomalyScore: 0.9f, reconstructionMse: 0.03f);

            Assert.True(podB); // pod-B has its own cooldown
        }

        // -------------------------------------------------------------------------
        // Counters
        // -------------------------------------------------------------------------

        [Fact]
        public async Task AlertsFired_WhenAlertDispatched_ThenIncrements()
        {
            await using var engine = new AlertEngine(MakeConfig(), new CapturingSink());

            engine.TryAlert("pod-1", 0.9f, 0.03f);
            engine.TryAlert("pod-1", 0.9f, 0.03f); // suppressed by cooldown

            Assert.Equal(1L, engine.AlertsFired);
        }

        [Fact]
        public async Task AlertsSuppressed_WhenBelowThreshold_ThenIncrements()
        {
            await using var engine = new AlertEngine(MakeConfig(threshold: 0.8f), new CapturingSink());

            engine.TryAlert("pod-1", 0.5f, 0.01f); // below threshold
            engine.TryAlert("pod-1", 0.6f, 0.01f); // below threshold

            Assert.Equal(2L, engine.AlertsSuppressed);
        }

        // -------------------------------------------------------------------------
        // Dispose
        // -------------------------------------------------------------------------

        [Fact]
        public async Task DisposeAsync_WhenCalled_ThenPendingAlertsAreDelivered()
        {
            var sink = new CapturingSink();
            await using var engine = new AlertEngine(MakeConfig(), sink);

            for (var i = 0; i < 5; i++)
            {
                engine.TryAlert($"pod-{i}", 0.9f, 0.03f);
            }

            await engine.DisposeAsync();

            Assert.Equal(5, sink.SendCount);
        }
    }
}