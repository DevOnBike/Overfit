// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Audio.Tts.Orpheus
{
    /// <summary>
    /// One voice-clone training example: the full token sequence (prompt then audio tokens then end-of-speech) the
    /// LM should learn, plus <see cref="PromptLength"/> — the number of leading prompt tokens that are *context*,
    /// so the loss is taken only on the audio-token continuation (completion-only fine-tuning).
    /// </summary>
    public sealed class OrpheusTrainingExample
    {
        public OrpheusTrainingExample(int[] inputIds, int promptLength)
        {
            InputIds = inputIds;
            PromptLength = promptLength;
        }

        /// <summary>Full sequence: <c>[prompt…][audio tokens…][end-of-speech]</c>.</summary>
        public int[] InputIds
        {
            get;
        }

        /// <summary>Index where the trainable continuation (audio tokens) begins; tokens before it are masked.</summary>
        public int PromptLength
        {
            get;
        }

        /// <summary>Number of trainable (audio + end) tokens.</summary>
        public int TargetLength => InputIds.Length - PromptLength;
    }
}
