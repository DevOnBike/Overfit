// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Audio.Tts.Orpheus;

namespace DevOnBike.Overfit.Tests.Audio
{
    /// <summary>The Orpheus prompt wrapping (voice + text in the audio-start/end-of-turn special tokens) and the
    /// preset-voice list. Model-free.</summary>
    public sealed class OrpheusPromptTests
    {
        [Fact]
        public void Format_WrapsVoiceAndText_WithSpecialTokens()
        {
            var prompt = OrpheusPrompt.Format("Hello there.", "tara");
            Assert.Equal("<|audio|>tara: Hello there.<|eot_id|>", prompt);
        }

        [Fact]
        public void Format_DefaultsToTara()
        {
            Assert.StartsWith("<|audio|>tara: ", OrpheusPrompt.Format("hi"));
        }

        [Fact]
        public void IsKnownVoice_RecognizesPresets_CaseInsensitive()
        {
            Assert.True(OrpheusPrompt.IsKnownVoice("leo"));
            Assert.True(OrpheusPrompt.IsKnownVoice("ZOE"));
            Assert.False(OrpheusPrompt.IsKnownVoice("nonexistent"));
        }
    }
}
