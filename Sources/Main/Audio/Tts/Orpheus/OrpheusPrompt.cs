// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Audio.Tts.Orpheus
{
    /// <summary>
    /// Builds the Orpheus TTS input prompt. The model is conditioned on a named preset voice and the text, wrapped
    /// in the audio-start / end-of-turn special tokens the model was trained with — the exact form the GGUF
    /// inference path uses: <c>&lt;|audio|&gt;{voice}: {text}&lt;|eot_id|&gt;</c>.
    /// </summary>
    public static class OrpheusPrompt
    {
        public const string DefaultVoice = "tara";

        /// <summary>The preset voices shipped with <c>orpheus-3b-0.1-ft</c> (best-quality first).</summary>
        public static string[] AvailableVoices { get; } =
            ["tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe"];

        private const string AudioStart = "<|audio|>";
        private const string EndOfTurn = "<|eot_id|>";

        /// <summary>Formats <paramref name="text"/> for <paramref name="voice"/> into the model's prompt string.</summary>
        public static string Format(string text, string voice = DefaultVoice)
            => $"{AudioStart}{voice}: {text}{EndOfTurn}";

        /// <summary>True if <paramref name="voice"/> is one of the model's preset voices.</summary>
        public static bool IsKnownVoice(string voice)
        {
            foreach (var v in AvailableVoices)
            {
                if (string.Equals(v, voice, StringComparison.OrdinalIgnoreCase))
                {
                    return true;
                }
            }
            return false;
        }
    }
}
