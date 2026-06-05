// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Audio;

namespace DevOnBike.Overfit.Tests.Audio
{
    /// <summary>The objective audio-quality evaluator (the "how close to ideal?" gate for generated speech /
    /// codec decodes): identical signals score perfectly, degradations lower the metrics monotonically, and the
    /// DTW spectral distance tolerates a sample-rate mismatch. Model-free and deterministic.</summary>
    public sealed class AudioSimilarityTests
    {
        private const int Rate = 24_000;

        [Fact]
        public void Identical_ScoresPerfect()
        {
            var s = Sine(Rate, 0.5, 220.0, 0.5f);

            var r = AudioSimilarity.Compare(s, Rate, s, Rate);

            Assert.True(r.IsIdentical);
            Assert.True(double.IsPositiveInfinity(r.SignalToNoiseRatioDb));
            Assert.True(r.Correlation > 0.9999);
            Assert.Equal(0.0, r.RootMeanSquareError, 6);
            Assert.Equal(0.0, r.MelSpectralDistance, 5);
            Assert.Equal(0.0, r.MelDistanceDtw, 5);
        }

        [Fact]
        public void Correlation_OfNegatedSignal_IsMinusOne()
        {
            var s = Sine(Rate, 0.25, 330.0, 0.4f);
            var negated = new float[s.Length];
            for (var i = 0; i < s.Length; i++)
            {
                negated[i] = -s[i];
            }

            Assert.Equal(-1.0, AudioSimilarity.Correlation(s, negated), 4);
        }

        [Fact]
        public void SmallNoise_LowersScores_ButStaysClose()
        {
            var s = Sine(Rate, 0.5, 220.0, 0.5f);
            var noisy = AddNoise(s, amplitude: 0.01f, seed: 1234);

            var r = AudioSimilarity.Compare(s, Rate, noisy, Rate);

            Assert.False(double.IsPositiveInfinity(r.SignalToNoiseRatioDb));
            Assert.InRange(r.SignalToNoiseRatioDb, 20.0, 50.0);
            Assert.InRange(r.Correlation, 0.99, 1.0);
            Assert.True(r.RootMeanSquareError > 0.0);
        }

        [Fact]
        public void DifferentTone_HasLargerMelDistance_ThanNearIdentical()
        {
            var reference = Sine(Rate, 0.5, 220.0, 0.5f);
            var nearIdentical = AddNoise(reference, amplitude: 0.005f, seed: 7);
            var differentTone = Sine(Rate, 0.5, 660.0, 0.5f);

            var near = AudioSimilarity.Compare(reference, Rate, nearIdentical, Rate);
            var far = AudioSimilarity.Compare(reference, Rate, differentTone, Rate);

            Assert.True(far.MelSpectralDistance > near.MelSpectralDistance);
            Assert.True(far.MelDistanceDtw > near.MelDistanceDtw);
        }

        [Fact]
        public void ResampledCandidate_StaysSpectrallyClose()
        {
            var reference = Sine(Rate, 0.5, 300.0, 0.5f);
            var downsampled = AudioResampler.Resample(reference, Rate, 16_000);

            // Compared across rates — Compare resamples internally; the tone survives the round trip.
            var r = AudioSimilarity.Compare(reference, Rate, downsampled, 16_000);

            Assert.True(r.Correlation > 0.9, $"corr={r.Correlation}");
            Assert.True(r.MelDistanceDtw < 0.5, $"melDtw={r.MelDistanceDtw}");
        }

        [Fact]
        public void Snr_OfSilentReferenceWithSignal_IsNegativeInfinity()
        {
            var silent = new float[12_000];
            var signal = Sine(Rate, 0.5, 220.0, 0.3f);

            Assert.True(double.IsNegativeInfinity(AudioSimilarity.SignalToNoiseRatioDb(silent, signal)));
        }

        private static float[] Sine(int rate, double seconds, double freq, float amplitude)
        {
            var n = (int)(rate * seconds);
            var s = new float[n];
            for (var i = 0; i < n; i++)
            {
                s[i] = amplitude * MathF.Sin((float)(2.0 * Math.PI * freq * i / rate));
            }
            return s;
        }

        // Deterministic LCG noise so the test is reproducible without Math.Random.
        private static float[] AddNoise(float[] s, float amplitude, uint seed)
        {
            var noisy = new float[s.Length];
            var state = seed;
            for (var i = 0; i < s.Length; i++)
            {
                state = (state * 1664525u) + 1013904223u;
                var u = (state >> 8) / (float)(1 << 24); // [0,1)
                noisy[i] = s[i] + amplitude * (2f * u - 1f);
            }
            return noisy;
        }
    }
}
