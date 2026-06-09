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
        /// Scales the whole signal so its loudest sample reaches <paramref name="targetPeak"/> — fixes quiet
        /// recordings (low input gain) so silence detection and the model both see a healthy level. Returns a new
        /// array; a fully silent input is returned unchanged.
        /// </summary>
        public static float[] PeakNormalize(ReadOnlySpan<float> samples, float targetPeak = 0.95f)
        {
            var peak = 0f;
            for (var i = 0; i < samples.Length; i++)
            {
                var a = MathF.Abs(samples[i]);
                if (a > peak)
                {
                    peak = a;
                }
            }
            var output = new float[samples.Length];
            if (peak <= 1e-9f)
            {
                samples.CopyTo(output);
                return output;
            }
            var gain = targetPeak / peak;
            for (var i = 0; i < samples.Length; i++)
            {
                output[i] = samples[i] * gain;
            }
            return output;
        }

        /// <summary>
        /// Trims leading and trailing near-silence (|sample| below <paramref name="amplitudeThreshold"/>), keeping
        /// <paramref name="keepPadding"/> samples of headroom on each side so speech onsets/decays are not clipped.
        /// Returns the original array unchanged if it is entirely silent or nothing needs trimming.
        /// <para>
        /// Set <paramref name="trimLeading"/> false to keep the start intact and only trim the tail — the right
        /// choice for TTS <b>output</b>: a generated clip's first phoneme often has a soft onset (nasal/fricative)
        /// whose attack sits below the gate, and clipping it mangles the first word. (Trimming both ends is correct
        /// for <i>recorded training</i> clips, which carry a real lead-in pause.)
        /// </para>
        /// </summary>
        public static float[] TrimSilence(ReadOnlySpan<float> samples, float amplitudeThreshold = 0.01f, int keepPadding = 720, bool trimLeading = true)
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

            var start = trimLeading ? Math.Max(0, first - keepPadding) : 0;
            var end = Math.Min(samples.Length - 1, last + keepPadding);
            if (start == 0 && end == samples.Length - 1)
            {
                return samples.ToArray();
            }

            return samples[start..(end + 1)].ToArray();
        }
    }
}
