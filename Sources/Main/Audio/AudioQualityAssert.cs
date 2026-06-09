// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Globalization;
using System.Text;

namespace DevOnBike.Overfit.Audio
{
    /// <summary>
    /// Turns <see cref="AudioSimilarity"/> metrics into a CI gate: assert that a candidate waveform stays within
    /// objective tolerance of a reference, throwing <see cref="AudioQualityException"/> (with the failing metric,
    /// its value and the threshold) otherwise. The audio-side counterpart of the RAG <c>RagAssert</c> harness — a
    /// measurable regression guard for the codec decoder (S2) and end-to-end TTS (S4), where there is otherwise no
    /// byte-parity check.
    /// <para>
    /// Unspecified bounds are not checked. Typical uses: a <i>deterministic decode</i> gate asserts the waveform
    /// (<c>minSignalToNoiseRatioDb</c> + <c>minCorrelation</c>); a <i>generated-speech</i> gate asserts the
    /// timing-robust spectral distance (<c>maxMelDistanceDtw</c>) against a reference clip.
    /// </para>
    /// </summary>
    public static class AudioQualityAssert
    {
        /// <summary>
        /// Asserts that <paramref name="candidate"/> is within tolerance of <paramref name="reference"/>. Each
        /// bound left at its permissive default (±∞) is skipped. Throws <see cref="AudioQualityException"/>
        /// listing every breached bound; returns the computed <see cref="AudioSimilarityReport"/> on success.
        /// </summary>
        public static AudioSimilarityReport Matches(
            ReadOnlySpan<float> reference, int referenceRate,
            ReadOnlySpan<float> candidate, int candidateRate,
            double minSignalToNoiseRatioDb = double.NegativeInfinity,
            double minCorrelation = double.NegativeInfinity,
            double maxRootMeanSquareError = double.PositiveInfinity,
            double maxMelSpectralDistance = double.PositiveInfinity,
            double maxMelDistanceDtw = double.PositiveInfinity,
            string? label = null)
        {
            var report = AudioSimilarity.Compare(reference, referenceRate, candidate, candidateRate);

            StringBuilder? failures = null;
            Check(ref failures, report.SignalToNoiseRatioDb >= minSignalToNoiseRatioDb,
                "SNR", report.SignalToNoiseRatioDb, ">=", minSignalToNoiseRatioDb, "dB");
            Check(ref failures, report.Correlation >= minCorrelation,
                "correlation", report.Correlation, ">=", minCorrelation, null);
            Check(ref failures, report.RootMeanSquareError <= maxRootMeanSquareError,
                "rmse", report.RootMeanSquareError, "<=", maxRootMeanSquareError, null);
            Check(ref failures, report.MelSpectralDistance <= maxMelSpectralDistance,
                "mel distance", report.MelSpectralDistance, "<=", maxMelSpectralDistance, null);
            Check(ref failures, report.MelDistanceDtw <= maxMelDistanceDtw,
                "mel distance (DTW)", report.MelDistanceDtw, "<=", maxMelDistanceDtw, null);

            if (failures is not null)
            {
                var prefix = label is null ? "Audio quality gate failed" : $"Audio quality gate '{label}' failed";
                throw new AudioQualityException($"{prefix}: {failures}(actual: {report})");
            }

            return report;
        }

        private static void Check(
            ref StringBuilder? failures, bool ok, string metric, double actual, string op, double threshold, string? unit)
        {
            if (ok)
            {
                return;
            }

            failures ??= new StringBuilder();
            var u = unit is null ? string.Empty : " " + unit;
            failures.Append(CultureInfo.InvariantCulture,
                $"{metric} {actual:0.0000}{u} not {op} {threshold:0.0000}{u}; ");
        }
    }
}
