// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels.Rope;

namespace DevOnBike.Overfit.Tests.LanguageModels.Runtime
{
    /// <summary>
    /// Unit tests for the Llama-3 "llama3" RoPE frequency scaling (<see cref="RopeScaling.Apply"/>),
    /// using the Llama-3.2 config (factor 32, low/high freq factors 1/4, original ctx 8192):
    /// high-frequency dims unchanged, low-frequency dims divided by the factor, medium band
    /// smoothly interpolated. Port-of-HF math, checked at the three branch boundaries.
    /// </summary>
    public sealed class RopeScalingTests
    {
        // Llama-3.2-1B: high_freq_wavelen = 8192/4 = 2048; low_freq_wavelen = 8192/1 = 8192.
        private static readonly RopeScaling Llama32 = new(Factor: 32f, LowFreqFactor: 1f, HighFreqFactor: 4f, OriginalContextLength: 8192);

        private static float FreqForWavelen(float wavelen) => 2f * MathF.PI / wavelen;

        [Fact]
        public void HighFrequency_ShortWavelength_Unchanged()
        {
            var freq = FreqForWavelen(1000f);   // < 2048 → high freq, untouched
            Assert.Equal(freq, Llama32.Apply(freq), 5);
        }

        [Fact]
        public void LowFrequency_LongWavelength_DividedByFactor()
        {
            var freq = FreqForWavelen(20_000f); // > 8192 → low freq, /32
            Assert.Equal(freq / 32f, Llama32.Apply(freq), 5);
        }

        [Fact]
        public void MediumFrequency_SmoothlyInterpolated()
        {
            const float wavelen = 4096f;         // between 2048 and 8192
            var freq = FreqForWavelen(wavelen);
            var smooth = (8192f / wavelen - 1f) / (4f - 1f);   // (2-1)/3 = 1/3
            var expected = ((1f - smooth) * (freq / 32f)) + (smooth * freq);
            Assert.Equal(expected, Llama32.Apply(freq), 5);
        }

        [Fact]
        public void Boundaries_AreContinuous()
        {
            // At the band edges the piecewise function is continuous (no jump).
            var atHigh = FreqForWavelen(2048f);
            var atLow = FreqForWavelen(8192f);
            // Just inside vs at the boundary differ only marginally.
            Assert.True(MathF.Abs(Llama32.Apply(atHigh) - Llama32.Apply(FreqForWavelen(2047f))) < 1e-4f);
            Assert.True(MathF.Abs(Llama32.Apply(atLow) - Llama32.Apply(FreqForWavelen(8191f))) < 1e-4f);
        }
    }
}
