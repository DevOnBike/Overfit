// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Anomalies.Monitoring
{
    /// <summary>
    /// Everything a guard remembers between restarts, restored together.
    ///
    /// <para><b>A named type rather than a tuple, and the reason is the fourth member.</b> Two of these were
    /// tolerable positionally; four are not, and the pair that must not be confused —
    /// <see cref="Labels"/> and <see cref="Suppressions"/> — are both operator feedback and both stores of
    /// small records. Swapping them silently would leave a guard muting what it was told was real.</para>
    ///
    /// <para>They travel together because they share one lifetime: wiping the calibration and keeping the
    /// labels would leave a guard constrained by evidence about numbers it no longer holds.</para>
    /// </summary>
    /// <param name="History">Per workload, per metric, per hour-of-day baseline.</param>
    /// <param name="Calibrator">What a healthy period turned out to look like, already wired to the labels.</param>
    /// <param name="Labels">What operators said about past incidents.</param>
    /// <param name="Suppressions">What operators asked not to hear, and until when.</param>
    public readonly record struct LearnedStateSnapshot(
        MetricHistory History,
        FloorCalibrator Calibrator,
        OperatorLabelStore Labels,
        SuppressionStore Suppressions)
    {
        /// <summary>
        /// The per-pod peer-novelty section, <b>as raw text</b>, or empty when the payload predates it.
        ///
        /// <para><b>Not a restored object, unlike the other four, and the reason is a dependency rather than
        /// an oversight.</b> <c>PeerNoveltyTracker.Read</c> needs the cadence profile the operator chose and
        /// the roster's current creation times — the latter to refuse a row whose pod name has been reused.
        /// Neither is known here, and inventing a profile so this type could hand back a tracker would put a
        /// guessed threshold inside a parser. The guard restores it once it has both.</para>
        ///
        /// <para><b>A property rather than a fifth positional member, deliberately.</b> Adding one would
        /// change this record's <c>Deconstruct</c> arity and break every existing
        /// <c>var (history, calibrator, ...) =</c> at once — a source-breaking change to buy nothing.</para>
        /// </summary>
        public string PeerNovelty
        {
            get; init;
        } = string.Empty;
    }
}
