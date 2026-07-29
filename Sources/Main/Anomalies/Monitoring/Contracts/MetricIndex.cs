// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Anomalies.Monitoring.Contracts
{
    public enum MetricIndex : byte
    {
        CpuUsageRatio = 0,
        CpuThrottleRatio = 1,
        MemoryWorkingSetBytes = 2,
        OomEventsRate = 3,
        LatencyP50Ms = 4,
        LatencyP95Ms = 5,
        LatencyP99Ms = 6,
        RequestsPerSecond = 7,
        ErrorRate = 8,
        GcGen2HeapBytes = 9,
        GcPauseRatio = 10,
        ThreadPoolQueueLength = 11,

        /// <summary>
        /// Container restarts <b>during the window</b> (an <c>increase()</c>, not the raw counter).
        ///
        /// <para>Added last on purpose. <see cref="MetricIndex"/> values are persisted as the
        /// <c>MetricTypeId</c> byte on every <c>RawMetricSeries</c> and in historical CSV, so inserting one in
        /// the middle would silently reinterpret every stored sample.</para>
        ///
        /// <para><b>Why it had to be added at all.</b> An OOM kill's strongest evidence is the restart it
        /// causes, and this channel did not exist — so the guard could not see it. Measured:
        /// <c>OomDetectionTests.PeerComparison_MissesASingleOomKill</c> shows the peer family is structurally
        /// blind to a single kill, because one event yields a non-zero rate across only ~10% of the window and
        /// Cliff's delta then lands under the 0.33 materiality gate.</para>
        ///
        /// <para><b>Scraped, but not a model feature.</b> This channel is deliberately absent from
        /// <see cref="MetricSnapshot"/>, so <c>MetricIndex.Count</c> is larger than
        /// <c>MetricSnapshot.FeatureCount</c>. The rules, peer and trend families read raw series keyed by this
        /// enum and see it; the token vocabulary does not, because moving that vocabulary invalidates every
        /// trained checkpoint and it must not move each time the guard learns to scrape one more thing.</para>
        /// </summary>
        ContainerRestarts = 12,

        Count = 13 // sentinel — number of features
    }
}