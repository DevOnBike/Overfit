// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Audio.Tts;

namespace DevOnBike.Overfit.Tests.Audio
{
    /// <summary>Silence-based segmentation for single-take dataset recording: separate utterances split on long
    /// pauses, short in-utterance pauses do not, and blips below the minimum are dropped. Model-free.</summary>
    public sealed class AudioSegmenterTests
    {
        private const int Rate = 16000;

        [Fact]
        public void SplitOnSilence_ThreeUtterances_SeparatedByPauses()
        {
            // 3 one-second tones separated by 0.5 s silences.
            var audio = Build(
                (1.0, true), (0.5, false), (1.0, true), (0.5, false), (1.0, true));

            var segs = AudioSegmenter.SplitOnSilence(audio, Rate, minSilenceSeconds: 0.3f, keepPadding: 0);

            Assert.Equal(3, segs.Count);
            foreach (var (start, end) in segs)
            {
                Assert.True(end - start >= (int)(0.8 * Rate)); // roughly the 1 s tone
            }
        }

        [Fact]
        public void SplitOnSilence_ShortPauseInsideUtterance_NotSplit()
        {
            // A 0.1 s gap is shorter than minSilence (0.3 s) → stays one segment.
            var audio = Build((0.6, true), (0.1, false), (0.6, true));

            var segs = AudioSegmenter.SplitOnSilence(audio, Rate, minSilenceSeconds: 0.3f, keepPadding: 0);

            Assert.Single(segs);
        }

        [Fact]
        public void SplitOnSilence_AllSilence_NoSegments()
        {
            Assert.Empty(AudioSegmenter.SplitOnSilence(new float[Rate], Rate));
        }

        // Builds audio from (seconds, isTone) spans: a 220 Hz tone or silence.
        private static float[] Build(params (double Seconds, bool Tone)[] parts)
        {
            var total = 0;
            foreach (var p in parts)
            {
                total += (int)(p.Seconds * Rate);
            }
            var audio = new float[total];
            var pos = 0;
            foreach (var (seconds, tone) in parts)
            {
                var len = (int)(seconds * Rate);
                if (tone)
                {
                    for (var i = 0; i < len; i++)
                    {
                        audio[pos + i] = 0.3f * MathF.Sin((float)(2.0 * Math.PI * 220.0 * i / Rate));
                    }
                }
                pos += len;
            }
            return audio;
        }
    }
}
