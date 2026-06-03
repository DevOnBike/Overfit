// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels.Whisper;

namespace DevOnBike.Overfit.Tests.LanguageModels.Whisper
{
    /// <summary>Whisper language order → language token ids (no model needed).</summary>
    public sealed class WhisperLanguagesTests
    {
        [Fact]
        public void LanguageOrder_AndTokens()
        {
            Assert.Equal(0, WhisperLanguages.IndexOf("en"));   // English is index 0
            Assert.Equal(10, WhisperLanguages.IndexOf("pl"));  // Polish
            Assert.Equal(2, WhisperLanguages.IndexOf("de"));
            Assert.Equal(-1, WhisperLanguages.IndexOf("xx"));  // unknown

            var cfg = new WhisperConfig(51865, 1500, 384, 6, 4, 448, 384, 6, 4, 80, true);
            var tok = new WhisperTokenizer(cfg, Array.Empty<string>());
            Assert.Equal(tok.StartOfTranscript + 1, tok.LanguageToken("en"));        // 50259
            Assert.Equal(tok.StartOfTranscript + 1 + 10, tok.LanguageToken("pl"));   // 50269
            Assert.Throws<ArgumentException>(() => tok.LanguageToken("xx"));
        }
    }
}
