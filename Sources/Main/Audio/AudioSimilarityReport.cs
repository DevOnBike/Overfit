// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Globalization;

namespace DevOnBike.Overfit.Audio
{
    /// <summary>
    /// The objective similarity of a candidate waveform to a reference ("ideal") waveform — the read-out of
    /// <see cref="AudioSimilarity.Compare"/>. Two complementary views:
    /// <list type="bullet">
    ///   <item><b>Waveform domain</b> (<see cref="SignalToNoiseRatioDb"/>, <see cref="Correlation"/>,
    ///   <see cref="RootMeanSquareError"/>) — sample-aligned; the right gate when the candidate is a
    ///   <i>deterministic decode</i> of the same input (e.g. a codec decoder vs. a reference decode), where the
    ///   two signals should line up sample-for-sample.</item>
    ///   <item><b>Spectral domain</b> (<see cref="MelSpectralDistance"/>, <see cref="MelDistanceDtw"/>) —
    ///   alignment-tolerant; the right gate when timing/length differ (e.g. generated speech vs. a reference clip)
    ///   so a small phase or duration drift does not read as "totally wrong".</item>
    /// </list>
    /// All metrics are <i>reference-based</i>: closeness to a supplied ideal, not a reference-free MOS prediction.
    /// </summary>
    public readonly struct AudioSimilarityReport
    {
        public AudioSimilarityReport(
            double signalToNoiseRatioDb,
            double correlation,
            double rootMeanSquareError,
            double melSpectralDistance,
            double melDistanceDtw,
            int referenceSampleCount,
            int candidateSampleCount)
        {
            SignalToNoiseRatioDb = signalToNoiseRatioDb;
            Correlation = correlation;
            RootMeanSquareError = rootMeanSquareError;
            MelSpectralDistance = melSpectralDistance;
            MelDistanceDtw = melDistanceDtw;
            ReferenceSampleCount = referenceSampleCount;
            CandidateSampleCount = candidateSampleCount;
        }

        /// <summary>Sample-aligned signal-to-noise ratio in dB: <c>10·log10(Σref² / Σ(ref−cand)²)</c>. Higher is
        /// closer; <see cref="double.PositiveInfinity"/> means bit-identical (zero residual).</summary>
        public double SignalToNoiseRatioDb { get; }

        /// <summary>Pearson correlation of the sample-aligned waveforms, in <c>[-1, 1]</c>; <c>1</c> = identical
        /// shape. Sensitive to phase/timing, so it is most meaningful for aligned decodes.</summary>
        public double Correlation { get; }

        /// <summary>Root-mean-square error of the sample-aligned waveforms (amplitude units). <c>0</c> =
        /// identical.</summary>
        public double RootMeanSquareError { get; }

        /// <summary>Frame-aligned RMS distance in the log-mel domain (over the overlapping frames). Cheap and
        /// timing-tolerant to within a frame; <c>0</c> = identical spectra.</summary>
        public double MelSpectralDistance { get; }

        /// <summary>DTW-aligned (dynamic-time-warping) mean log-mel frame distance — the timing-robust spectral
        /// metric. Warps the two mel sequences onto a common path before measuring, so a duration or onset drift
        /// between generated and reference speech does not dominate. Lower = closer; <c>0</c> = identical.</summary>
        public double MelDistanceDtw { get; }

        public int ReferenceSampleCount { get; }
        public int CandidateSampleCount { get; }

        /// <summary>True when the candidate is (numerically) the reference: zero residual and an identical length.
        /// A convenience for bit-parity-style gates.</summary>
        public bool IsIdentical =>
            ReferenceSampleCount == CandidateSampleCount
            && double.IsPositiveInfinity(SignalToNoiseRatioDb);

        public override string ToString()
        {
            var snr = double.IsPositiveInfinity(SignalToNoiseRatioDb)
                ? "inf"
                : SignalToNoiseRatioDb.ToString("0.0", CultureInfo.InvariantCulture);
            return string.Create(CultureInfo.InvariantCulture,
                $"SNR={snr} dB, corr={Correlation:0.000}, rmse={RootMeanSquareError:0.0000}, "
                + $"mel={MelSpectralDistance:0.0000}, melDtw={MelDistanceDtw:0.0000} "
                + $"(ref {ReferenceSampleCount} / cand {CandidateSampleCount} samples)");
        }
    }
}
