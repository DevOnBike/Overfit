// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Anomalies.Monitoring;
using DevOnBike.Overfit.Anomalies.Contracts;
using DevOnBike.Overfit.Statistics;

namespace DevOnBike.Overfit.Tests.Anomalies
{
    /// <summary>
    /// Pins the per-signal claims in <see cref="PeerSignalCatalog"/>.
    ///
    /// <para>Each assertion here is a physical statement about the metric, not a preference, so a change to
    /// one should have to be argued for rather than absorbed silently — which is exactly what happened when
    /// memory sat in the load-sensitive set unexamined and became the largest single source of false peer
    /// findings.</para>
    /// </summary>
    public sealed class PeerSignalCatalogTests
    {
        [Theory]
        [InlineData(MetricIndex.CpuUsageRatio)]
        [InlineData(MetricIndex.GcPauseRatio)]
        [InlineData(MetricIndex.ThreadPoolQueueLength)]
        public void CostsWithARealPerRequestComponent_AreNormalisedByWork(MetricIndex metric)
        {
            Assert.Equal(PeerSignalKind.LoadSensitive, PeerSignalCatalog.Classify(metric));
            Assert.True(PeerSignalCatalog.RequiresWork(metric));
        }

        /// <summary>
        /// The regression this catalog exists for. A working set is assemblies, JIT'd code, caches and the
        /// live set — dividing it by request rate reports the traffic imbalance as a memory anomaly.
        /// </summary>
        [Theory]
        [InlineData(MetricIndex.MemoryWorkingSetBytes)]
        [InlineData(MetricIndex.GcGen2HeapBytes)]
        public void FixedCosts_AreComparedRaw(MetricIndex metric)
        {
            Assert.Equal(PeerSignalKind.LoadIndependent, PeerSignalCatalog.Classify(metric));
            Assert.False(PeerSignalCatalog.RequiresWork(metric));
        }

        [Theory]
        [InlineData(MetricIndex.ContainerRestarts)]
        [InlineData(MetricIndex.OomEventsRate)]
        [InlineData(MetricIndex.CpuThrottleRatio)]
        [InlineData(MetricIndex.ErrorRate)]
        public void EventsAndFractions_AreComparedRaw(MetricIndex metric)
        {
            Assert.Equal(PeerSignalKind.LoadIndependent, PeerSignalCatalog.Classify(metric));
        }

        /// <summary>
        /// Every channel must classify without throwing, so adding a metric cannot leave the peer path
        /// undefined for it — and the unknown case must land on the side that compares what was reported
        /// rather than dividing it by something unrelated.
        /// </summary>
        [Fact]
        public void EveryChannelClassifies_AndTheDefaultIsToNotDivide()
        {
            for (var m = 0; m < (int)MetricIndex.Count; m++)
            {
                var kind = PeerSignalCatalog.Classify((MetricIndex)m);

                Assert.True(kind is PeerSignalKind.LoadIndependent or PeerSignalKind.LoadSensitive);
            }

            Assert.Equal(PeerSignalKind.LoadIndependent, PeerSignalCatalog.Classify(unchecked((MetricIndex)9999)));
        }
    }
}
