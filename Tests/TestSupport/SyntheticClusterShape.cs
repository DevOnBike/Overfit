// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Tests.TestSupport
{
    /// <summary>
    /// The scatter parameters of <see cref="SyntheticCluster"/>, gathered so they can be searched rather than
    /// edited.
    ///
    /// <para><b>Exactly the parameters a calibration loop is allowed to move</b> — see
    /// <c>docs/autoresearch-program.md</c>. Everything else in the generator (the diurnal curve, the affine
    /// CPU cost model, restart and warm-up rates, the sawtooth, scrape gaps, identically-zero counters) is
    /// structure that was measured or reasoned about rather than fitted, and a search allowed to move it
    /// would quietly trade structure for score.</para>
    ///
    /// <para>The burst probability is absent for a different reason: it was measured to be
    /// <b>unidentifiable</b> — ten restarts of the calibration search spread it over 70% of its allowed
    /// range at the same score, because only its product with the burst magnitudes reaches the data. A
    /// parameter a search cannot pin does not belong in the genome; it belongs in the structure, with the
    /// reason written down.</para>
    ///
    /// <para>The per-pod personality widths are absent on purpose. They only move the between-pod column,
    /// which is a range over three draws on the lab side and is deliberately not scored; a search given a
    /// parameter with no gradient random-walks it and then reports convergence.</para>
    ///
    /// <para>Defaults are <see cref="Measured"/> — the values calibrated against the recorded lab window,
    /// each documented on the corresponding constant in <see cref="SyntheticCluster"/>. Constructing a
    /// cluster without a shape uses them, so nothing changes for existing callers.</para>
    /// </summary>
    public readonly record struct SyntheticClusterShape(
        double LatencyScatterP50,
        double LatencyScatterP95,
        double LatencyScatterP99,
        double TrafficScatter,
        double CpuScatter,
        double TrafficBurstFactor,
        double CpuBurstFactor,
        double HeapPromotionStep,
        double LatencyBurstDelay,
        double MemorySawtoothAmplitude,
        double MemoryCycleSamples)
    {
        /// <summary>The calibrated values. Documented individually on <see cref="SyntheticCluster"/>.</summary>
        public static SyntheticClusterShape Measured => new(
            SyntheticCluster.LatencyScatterP50,
            SyntheticCluster.LatencyScatterP95,
            SyntheticCluster.LatencyScatterP99,
            SyntheticCluster.TrafficScatter,
            SyntheticCluster.CpuScatter,
            SyntheticCluster.TrafficBurstFactor,
            SyntheticCluster.CpuBurstFactor,
            SyntheticCluster.HeapPromotionStep,
            SyntheticCluster.LatencyBurstDelay,
            SyntheticCluster.MemorySawtoothAmplitude,
            SyntheticCluster.MemoryCycleSamples);

        /// <summary>Reads one parameter by index, so a search can iterate without a switch at every site.</summary>
        public double this[int index] => index switch
        {
            0 => LatencyScatterP50,
            1 => LatencyScatterP95,
            2 => LatencyScatterP99,
            3 => TrafficScatter,
            4 => CpuScatter,
            5 => TrafficBurstFactor,
            6 => CpuBurstFactor,
            7 => HeapPromotionStep,
            8 => LatencyBurstDelay,
            9 => MemorySawtoothAmplitude,
            10 => MemoryCycleSamples,
            _ => throw new ArgumentOutOfRangeException(nameof(index)),
        };

        /// <summary>Returns a copy with one parameter replaced.</summary>
        public SyntheticClusterShape With(int index, double value) => index switch
        {
            0 => this with { LatencyScatterP50 = value },
            1 => this with { LatencyScatterP95 = value },
            2 => this with { LatencyScatterP99 = value },
            3 => this with { TrafficScatter = value },
            4 => this with { CpuScatter = value },
            5 => this with { TrafficBurstFactor = value },
            6 => this with { CpuBurstFactor = value },
            7 => this with { HeapPromotionStep = value },
            8 => this with { LatencyBurstDelay = value },
            9 => this with { MemorySawtoothAmplitude = value },
            10 => this with { MemoryCycleSamples = value },
            _ => throw new ArgumentOutOfRangeException(nameof(index)),
        };

        /// <summary>Parameter names, index-aligned with the indexer, for reports.</summary>
        public static string Name(int index) => index switch
        {
            0 => nameof(LatencyScatterP50),
            1 => nameof(LatencyScatterP95),
            2 => nameof(LatencyScatterP99),
            3 => nameof(TrafficScatter),
            4 => nameof(CpuScatter),
            5 => nameof(TrafficBurstFactor),
            6 => nameof(CpuBurstFactor),
            7 => nameof(HeapPromotionStep),
            8 => nameof(LatencyBurstDelay),
            9 => nameof(MemorySawtoothAmplitude),
            10 => nameof(MemoryCycleSamples),
            _ => throw new ArgumentOutOfRangeException(nameof(index)),
        };

        /// <summary>How many parameters the search may move.</summary>
        public const int Count = 11;
    }
}
