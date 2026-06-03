// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.LanguageModels.Whisper
{
    /// <summary>
    /// Whisper's fixed language order (index 0 = English). The language token id is
    /// <c>StartOfTranscript + 1 + index</c>; this maps a language code (e.g. "en", "pl") to that index.
    /// </summary>
    public static class WhisperLanguages
    {
        /// <summary>ISO codes in Whisper's canonical order.</summary>
        public static readonly string[] Codes =
        {
            "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl", "ca", "nl", "ar", "sv", "it",
            "id", "hi", "fi", "vi", "he", "uk", "el", "ms", "cs", "ro", "da", "hu", "ta", "no", "th", "ur",
            "hr", "bg", "lt", "la", "mi", "ml", "cy", "sk", "te", "fa", "lv", "bn", "sr", "az", "sl", "kn",
            "et", "mk", "br", "eu", "is", "hy", "ne", "mn", "bs", "kk", "sq", "sw", "gl", "mr", "pa", "si",
            "km", "sn", "yo", "so", "af", "oc", "ka", "be", "tg", "sd", "gu", "am", "yi", "lo", "uz", "fo",
            "ht", "ps", "tk", "nn", "mt", "sa", "lb", "my", "bo", "tl", "mg", "as", "tt", "haw", "ln", "ha",
            "ba", "jw", "su",
        };

        /// <summary>Index of <paramref name="code"/> in Whisper's order, or −1 if unknown.</summary>
        public static int IndexOf(string code)
        {
            for (var i = 0; i < Codes.Length; i++)
            {
                if (string.Equals(Codes[i], code, StringComparison.OrdinalIgnoreCase)) { return i; }
            }
            return -1;
        }
    }
}
