// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Audio.Tts.Orpheus;

namespace DevOnBike.Overfit.Tests.Audio
{
    /// <summary>Voice-clone training-sequence assembly: prompt + audio tokens + end-of-speech, with the loss-mask
    /// boundary at the prompt end, and audio-token ids that decode back to the original codes (so training targets
    /// match the generation path exactly). Model-free.</summary>
    public sealed class OrpheusTrainingSequenceTests
    {
        [Fact]
        public void Build_LaysOutPromptThenAudioThenEnd()
        {
            int[] prompt = [1, 2, 3];
            int[][] codes = [[5], [6, 7], [8, 9, 10, 11]]; // F=1 frame
            const int audioBase = 1000;
            const int eos = 2;

            var ex = OrpheusTrainingSequence.Build(prompt, codes, audioBase, eos);

            Assert.Equal(3, ex.PromptLength);
            Assert.Equal(3 + 7 + 1, ex.InputIds.Length);
            Assert.Equal(8, ex.TargetLength);
            Assert.Equal([1, 2, 3], ex.InputIds[..3]);     // prompt preserved
            Assert.Equal(eos, ex.InputIds[^1]);            // ends with end-of-speech
            // First audio token: base + code(5) + 10 + 0 = 1015.
            Assert.Equal(1015, ex.InputIds[3]);
        }

        [Fact]
        public void Build_AudioTokens_DecodeBackToTheOriginalCodes()
        {
            int[] prompt = [42, 43];
            int[][] codes =
            [
                [100, 101],
                [200, 201, 202, 203],
                [300, 301, 302, 303, 304, 305, 306, 307],
            ];
            const int audioBase = 128256; // Orpheus-style custom-token base
            const int eos = 128258;

            var ex = OrpheusTrainingSequence.Build(prompt, codes, audioBase, eos);

            // Strip the prompt + EOS, undo the base/offset → custom-token numbers → codes → levels.
            var audioCount = ex.InputIds.Length - prompt.Length - 1;
            var decoded = new int[audioCount];
            for (var i = 0; i < audioCount; i++)
            {
                var n = ex.InputIds[prompt.Length + i] - audioBase;
                decoded[i] = OrpheusSnacBridge.DecodeCustomToken(n, i);
            }

            var levels = OrpheusSnacBridge.Redistribute(decoded);
            Assert.Equal(codes[0], levels[0]);
            Assert.Equal(codes[1], levels[1]);
            Assert.Equal(codes[2], levels[2]);
        }
    }
}
