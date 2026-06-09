// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Audio;

namespace DevOnBike.Overfit.Tests.Audio
{
    /// <summary>The CI-gate facade: a bit-close decode passes a strict gate; a degraded one throws with the
    /// breached metric named (so a regression points at itself). Model-free and deterministic.</summary>
    public sealed class AudioQualityAssertTests
    {
        private const int Rate = 24_000;

        [Fact]
        public void Matches_IdenticalSignal_PassesStrictGate()
        {
            var s = Sine(440.0, 0.4f);

            var report = AudioQualityAssert.Matches(
                s, Rate, s, Rate,
                minSignalToNoiseRatioDb: 60.0,
                minCorrelation: 0.999,
                maxMelDistanceDtw: 0.01);

            Assert.True(report.IsIdentical);
        }

        [Fact]
        public void Matches_DegradedSignal_Throws_NamingTheMetric()
        {
            var reference = Sine(440.0, 0.4f);
            var degraded = AddNoise(reference, amplitude: 0.3f, seed: 99);

            var ex = Assert.Throws<AudioQualityException>(() => AudioQualityAssert.Matches(
                reference, Rate, degraded, Rate,
                minSignalToNoiseRatioDb: 30.0,
                minCorrelation: 0.999,
                label: "decode"));

            Assert.Contains("decode", ex.Message);
            Assert.Contains("SNR", ex.Message);
        }

        private static float[] Sine(double freq, float amplitude)
        {
            var n = (int)(Rate * 0.4);
            var s = new float[n];
            for (var i = 0; i < n; i++)
            {
                s[i] = amplitude * MathF.Sin((float)(2.0 * Math.PI * freq * i / Rate));
            }
            return s;
        }

        private static float[] AddNoise(float[] s, float amplitude, uint seed)
        {
            var noisy = new float[s.Length];
            var state = seed;
            for (var i = 0; i < s.Length; i++)
            {
                state = (state * 1664525u) + 1013904223u;
                var u = (state >> 8) / (float)(1 << 24);
                noisy[i] = s[i] + amplitude * (2f * u - 1f);
            }
            return noisy;
        }
    }
}
