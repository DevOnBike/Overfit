// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Audio.Tts
{
    /// <summary>
    /// Small waveform clean-ups applied after synthesis. Today: leading/trailing <b>silence trimming</b> — neural
    /// vocoders often leave dead air at the ends, which reads as latency and makes concatenated sentences ragged.
    /// Pure, model-free, deterministic.
    /// </summary>
    public static class AudioPostProcessing
    {
        /// <summary>
        /// Trims leading and trailing near-silence (|sample| below <paramref name="amplitudeThreshold"/>), keeping
        /// <paramref name="keepPadding"/> samples of headroom on each side so speech onsets/decays are not clipped.
        /// Returns the original array unchanged if it is entirely silent or nothing needs trimming.
        /// </summary>
        public static float[] TrimSilence(ReadOnlySpan<float> samples, float amplitudeThreshold = 0.01f, int keepPadding = 720)
        {
            var first = -1;
            var last = -1;
            for (var i = 0; i < samples.Length; i++)
            {
                if (MathF.Abs(samples[i]) >= amplitudeThreshold)
                {
                    if (first < 0)
                    {
                        first = i;
                    }
                    last = i;
                }
            }

            if (first < 0)
            {
                // All silence — return a copy of the input (caller owns it either way).
                return samples.ToArray();
            }

            var start = Math.Max(0, first - keepPadding);
            var end = Math.Min(samples.Length - 1, last + keepPadding);
            if (start == 0 && end == samples.Length - 1)
            {
                return samples.ToArray();
            }

            return samples[start..(end + 1)].ToArray();
        }
    }
}
