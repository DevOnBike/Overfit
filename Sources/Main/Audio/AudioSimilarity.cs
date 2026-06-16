// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Audio
{
    /// <summary>
    /// Objective, reference-based audio-quality metrics — "how close is this waveform to the ideal one?". The
    /// counterpart, for generated audio, of the RAG stability harness: a measurable gate instead of a subjective
    /// listen. Pure managed, model-free; reuses <see cref="MelSpectrogram"/> (log-mel) and
    /// <see cref="AudioResampler"/>.
    /// <para>
    /// Use it to gate a codec/vocoder decode against a reference decode (waveform metrics — they should align
    /// sample-for-sample) and to track how close synthesized speech is to a reference clip (the DTW mel distance —
    /// it tolerates timing/length drift). See <see cref="AudioSimilarityReport"/> for which metric fits which case.
    /// </para>
    /// </summary>
    public static class AudioSimilarity
    {
        // MelSpectrogram is hardwired to 16 kHz; both signals are resampled to this before the spectral metrics so
        // the filterbank's frequency mapping is consistent for reference and candidate alike.
        private const int MelRate = MelSpectrogram.SampleRate;

        /// <summary>
        /// Compares a <paramref name="candidate"/> waveform against a <paramref name="reference"/> ("ideal") one,
        /// resampling internally so the two need not share a rate. Waveform metrics are measured at the reference
        /// rate over the overlapping prefix; spectral metrics at 16 kHz over the full clips (DTW handles length
        /// mismatch). Both inputs are mono float PCM in <c>[-1, 1]</c>.
        /// </summary>
        public static AudioSimilarityReport Compare(
            ReadOnlySpan<float> reference, int referenceRate,
            ReadOnlySpan<float> candidate, int candidateRate)
        {
            if (referenceRate <= 0 || candidateRate <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(referenceRate), "Sample rates must be positive.");
            }

            var refCount = reference.Length;
            var candCount = candidate.Length;

            // ── waveform domain: bring the candidate onto the reference rate, then align by prefix ──
            float[]? candAtRefRented = null;
            ReadOnlySpan<float> candAtRef = candidate;
            if (candidateRate != referenceRate)
            {
                candAtRefRented = AudioResampler.Resample(candidate, candidateRate, referenceRate);
                candAtRef = candAtRefRented;
            }

            var snr = SignalToNoiseRatioDb(reference, candAtRef);
            var corr = Correlation(reference, candAtRef);
            var rmse = RootMeanSquareError(reference, candAtRef);

            // ── spectral domain: both at 16 kHz → log-mel → frame-aligned RMS + DTW path cost ──
            var refMelSamples = ResampleTo(reference, referenceRate, MelRate);
            var candMelSamples = ResampleTo(candidate, candidateRate, MelRate);

            var mel = new MelSpectrogram();
            var refMel = CopyLogMel(mel, refMelSamples, out var refFrames);
            var candMel = CopyLogMel(mel, candMelSamples, out var candFrames);
            var nMels = mel.MelCount;

            var melDist = FrameAlignedMelDistance(refMel, refFrames, candMel, candFrames, nMels);
            var melDtw = DtwMelDistance(refMel, refFrames, candMel, candFrames, nMels);

            return new AudioSimilarityReport(snr, corr, rmse, melDist, melDtw, refCount, candCount);
        }

        /// <summary>Sample-aligned signal-to-noise ratio in dB over the overlapping prefix:
        /// <c>10·log10(Σref² / Σ(ref−cand)²)</c>. <see cref="double.PositiveInfinity"/> when the residual is zero;
        /// <see cref="double.NegativeInfinity"/> when the reference is silent but the candidate is not.</summary>
        public static double SignalToNoiseRatioDb(ReadOnlySpan<float> reference, ReadOnlySpan<float> candidate)
        {
            var n = Math.Min(reference.Length, candidate.Length);
            double signal = 0.0;
            double noise = 0.0;
            for (var i = 0; i < n; i++)
            {
                double r = reference[i];
                var d = r - candidate[i];
                signal += r * r;
                noise += d * d;
            }

            if (noise <= 0.0)
            {
                return double.PositiveInfinity;
            }
            if (signal <= 0.0)
            {
                return double.NegativeInfinity;
            }
            return 10.0 * Math.Log10(signal / noise);
        }

        /// <summary>Pearson correlation of the sample-aligned prefixes, in <c>[-1, 1]</c>. Returns <c>0</c> when
        /// either signal is constant over the window (variance undefined).</summary>
        public static double Correlation(ReadOnlySpan<float> reference, ReadOnlySpan<float> candidate)
        {
            var n = Math.Min(reference.Length, candidate.Length);
            if (n == 0)
            {
                return 0.0;
            }

            double meanR = 0.0;
            double meanC = 0.0;
            for (var i = 0; i < n; i++)
            {
                meanR += reference[i];
                meanC += candidate[i];
            }
            meanR /= n;
            meanC /= n;

            double cov = 0.0;
            double varR = 0.0;
            double varC = 0.0;
            for (var i = 0; i < n; i++)
            {
                var dr = reference[i] - meanR;
                var dc = candidate[i] - meanC;
                cov += dr * dc;
                varR += dr * dr;
                varC += dc * dc;
            }

            var denom = Math.Sqrt(varR * varC);
            if (denom <= 0.0)
            {
                return 0.0;
            }
            return cov / denom;
        }

        /// <summary>Root-mean-square error of the sample-aligned prefixes (amplitude units).</summary>
        public static double RootMeanSquareError(ReadOnlySpan<float> reference, ReadOnlySpan<float> candidate)
        {
            var n = Math.Min(reference.Length, candidate.Length);
            if (n == 0)
            {
                return 0.0;
            }

            double sum = 0.0;
            for (var i = 0; i < n; i++)
            {
                var d = (double)reference[i] - candidate[i];
                sum += d * d;
            }
            return Math.Sqrt(sum / n);
        }

        private static float[] ResampleTo(ReadOnlySpan<float> samples, int srcRate, int dstRate)
        {
            if (srcRate == dstRate)
            {
                return samples.ToArray();
            }
            return AudioResampler.Resample(samples, srcRate, dstRate);
        }

        // Snapshots the mel-major [nMels × frames] log-mel out of the reused MelSpectrogram buffer (the next call
        // overwrites it), so reference and candidate spectra coexist.
        private static float[] CopyLogMel(MelSpectrogram mel, ReadOnlySpan<float> samples, out int frames)
        {
            var produced = mel.LogMel(samples, out frames);
            var size = mel.MelCount * frames;
            var copy = new float[size];
            produced.AsSpan(0, size).CopyTo(copy);
            return copy;
        }

        // RMS distance over the overlapping frames, mel-major layout (index = bin*frames + frame). Cheap; assumes
        // the two are roughly time-aligned (true for a deterministic decode).
        private static double FrameAlignedMelDistance(
            ReadOnlySpan<float> refMel, int refFrames, ReadOnlySpan<float> candMel, int candFrames, int nMels)
        {
            var frames = Math.Min(refFrames, candFrames);
            if (frames == 0)
            {
                return 0.0;
            }

            double sum = 0.0;
            for (var f = 0; f < frames; f++)
            {
                for (var b = 0; b < nMels; b++)
                {
                    double d = refMel[b * refFrames + f] - candMel[b * candFrames + f];
                    sum += d * d;
                }
            }
            return Math.Sqrt(sum / (frames * (double)nMels));
        }

        // Dynamic-time-warping mean frame distance: warps the two mel sequences onto a least-cost monotone path,
        // then normalizes by the path length. Timing/length-robust — the right metric for generated-vs-reference
        // speech. Rolling two-row DP (no jagged arrays), O(refFrames · candFrames).
        private static double DtwMelDistance(
            ReadOnlySpan<float> refMel, int refFrames, ReadOnlySpan<float> candMel, int candFrames, int nMels)
        {
            if (refFrames == 0 || candFrames == 0)
            {
                return 0.0;
            }

            var prev = new double[candFrames + 1];
            var cur = new double[candFrames + 1];
            for (var j = 0; j <= candFrames; j++)
            {
                prev[j] = double.PositiveInfinity;
            }
            prev[0] = 0.0;

            for (var i = 1; i <= refFrames; i++)
            {
                cur[0] = double.PositiveInfinity;
                for (var j = 1; j <= candFrames; j++)
                {
                    var cost = FrameDistance(refMel, refFrames, i - 1, candMel, candFrames, j - 1, nMels);
                    var best = prev[j - 1];          // diagonal (match)
                    if (prev[j] < best)
                    {
                        best = prev[j];               // insertion
                    }
                    if (cur[j - 1] < best)
                    {
                        best = cur[j - 1];            // deletion
                    }
                    cur[j] = cost + best;
                }

                var swap = prev;
                prev = cur;
                cur = swap;
            }

            // Normalize by the warping-path length so longer clips are not penalized.
            var pathLength = refFrames + candFrames;
            return prev[candFrames] / pathLength;
        }

        private static double FrameDistance(
            ReadOnlySpan<float> a, int aFrames, int ai, ReadOnlySpan<float> b, int bFrames, int bi, int nMels)
        {
            double sum = 0.0;
            for (var k = 0; k < nMels; k++)
            {
                double d = a[k * aFrames + ai] - b[k * bFrames + bi];
                sum += d * d;
            }
            return Math.Sqrt(sum);
        }
    }
}
