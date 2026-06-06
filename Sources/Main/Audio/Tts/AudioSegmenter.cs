// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Audio.Tts
{
    /// <summary>
    /// Splits one long recording into utterance segments on silence — so a voice-clone dataset can be recorded as a
    /// single take (read each line with a clear pause between) instead of many separate files. A short pause inside
    /// a sentence is kept; only a gap of at least <c>minSilence</c> ends a segment. Pure, model-free.
    /// </summary>
    public static class AudioSegmenter
    {
        /// <summary>
        /// Returns the <c>[start, end)</c> sample ranges of the spoken segments. <paramref name="amplitudeThreshold"/>
        /// is the silence floor; <paramref name="minSilenceSeconds"/> is the gap that separates segments;
        /// <paramref name="minSegmentSeconds"/> drops blips shorter than that; <paramref name="keepPadding"/> samples
        /// of headroom are kept on each side.
        /// </summary>
        public static List<(int Start, int End)> SplitOnSilence(
            ReadOnlySpan<float> samples,
            int sampleRate,
            float amplitudeThreshold = 0.015f,
            float minSilenceSeconds = 0.3f,
            float minSegmentSeconds = 0.2f,
            int keepPadding = 480)
        {
            var minSilence = (int)(minSilenceSeconds * sampleRate);
            var minSegment = (int)(minSegmentSeconds * sampleRate);
            var n = samples.Length;
            var segments = new List<(int, int)>();

            var i = 0;
            while (i < n)
            {
                // Skip leading silence.
                while (i < n && MathF.Abs(samples[i]) < amplitudeThreshold)
                {
                    i++;
                }
                if (i >= n)
                {
                    break;
                }

                var start = i;
                var lastLoud = i;
                var j = i;
                while (j < n)
                {
                    if (MathF.Abs(samples[j]) >= amplitudeThreshold)
                    {
                        lastLoud = j;
                        j++;
                        continue;
                    }

                    // Measure the silence run; only a long-enough one ends the segment.
                    var silenceStart = j;
                    while (j < n && MathF.Abs(samples[j]) < amplitudeThreshold)
                    {
                        j++;
                    }
                    if (j - silenceStart >= minSilence || j >= n)
                    {
                        break;
                    }
                }

                var segStart = Math.Max(0, start - keepPadding);
                var segEnd = Math.Min(n, lastLoud + 1 + keepPadding);
                if (lastLoud + 1 - start >= minSegment)
                {
                    segments.Add((segStart, segEnd));
                }
                i = j;
            }

            return segments;
        }
    }
}
