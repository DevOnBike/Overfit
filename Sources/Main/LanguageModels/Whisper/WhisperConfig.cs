// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.LanguageModels.Whisper
{
    /// <summary>
    /// Whisper hyper-parameters (the 10 ints + float flag at the head of a whisper.cpp ggml file).
    /// </summary>
    public sealed record WhisperConfig(
        int NVocab,
        int NAudioCtx,
        int NAudioState,
        int NAudioHead,
        int NAudioLayer,
        int NTextCtx,
        int NTextState,
        int NTextHead,
        int NTextLayer,
        int NMels,
        bool F16)
    {
        /// <summary>Multilingual models carry the language tokens (n_vocab ≥ 51865); English-only = 51864.</summary>
        public bool IsMultilingual => NVocab >= 51865;

        /// <summary>Number of language tokens (whisper.cpp: <c>n_vocab − 51765 − (multilingual ? 1 : 0)</c>).</summary>
        public int NumLanguages => NVocab - 51765 - (IsMultilingual ? 1 : 0);
    }
}
