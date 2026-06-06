// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Audio.Tts;

namespace DevOnBike.Overfit.Tests.Audio
{
    /// <summary>Post-synthesis silence trimming: dead air at the ends is removed, with padding so onsets/decays
    /// survive; all-silent and already-tight inputs are handled. Model-free.</summary>
    public sealed class AudioPostProcessingTests
    {
        [Fact]
        public void TrimSilence_RemovesLeadingAndTrailing_KeepingPadding()
        {
            // 100 silent, 10 loud, 100 silent. With keepPadding=5 → keep [95..114] = 20 samples.
            var s = new float[210];
            for (var i = 100; i < 110; i++)
            {
                s[i] = 0.5f;
            }

            var trimmed = AudioPostProcessing.TrimSilence(s, amplitudeThreshold: 0.01f, keepPadding: 5);

            Assert.Equal(20, trimmed.Length);
            // The loud region is preserved.
            Assert.Equal(0.5f, trimmed[5]);
        }

        [Fact]
        public void TrimSilence_AllSilence_ReturnsAsIs()
        {
            var s = new float[50];
            var trimmed = AudioPostProcessing.TrimSilence(s);
            Assert.Equal(50, trimmed.Length);
        }

        [Fact]
        public void TrimSilence_NoSilence_Unchanged()
        {
            var s = new float[] { 0.3f, -0.4f, 0.5f };
            var trimmed = AudioPostProcessing.TrimSilence(s, keepPadding: 0);
            Assert.Equal(3, trimmed.Length);
        }
    }
}
