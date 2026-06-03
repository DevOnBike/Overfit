// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels.Whisper;

namespace DevOnBike.Overfit.Tests.LanguageModels.Whisper
{
    /// <summary>
    /// Validates <see cref="WhisperTokenizer"/>: the special-token ids match whisper.cpp's arithmetic for
    /// both multilingual and English-only models, and byte-level decode round-trips UTF-8 text (including a
    /// space and a multi-byte character) while skipping special tokens.
    /// </summary>
    public sealed class WhisperTokenizerTests
    {
        [Fact]
        public void SpecialTokenIds_Multilingual_MatchWhisperCpp()
        {
            var cfg = new WhisperConfig(51865, 1500, 384, 6, 4, 448, 384, 6, 4, 80, true);
            var tok = new WhisperTokenizer(cfg, Array.Empty<string>());

            Assert.True(cfg.IsMultilingual);
            Assert.Equal(99, cfg.NumLanguages);
            Assert.Equal(50257, tok.EndOfTranscript);
            Assert.Equal(50258, tok.StartOfTranscript);
            Assert.Equal(50358, tok.Translate);
            Assert.Equal(50359, tok.Transcribe);
            Assert.Equal(50363, tok.NoTimestamps);
            Assert.Equal(50364, tok.TimestampBegin);
            Assert.Equal(50259, tok.LanguageTokenAt(0)); // English
        }

        [Fact]
        public void SpecialTokenIds_EnglishOnly_MatchWhisperCpp()
        {
            var cfg = new WhisperConfig(51864, 1500, 384, 6, 4, 448, 384, 6, 4, 80, true);
            var tok = new WhisperTokenizer(cfg, Array.Empty<string>());

            Assert.False(cfg.IsMultilingual);
            Assert.Equal(50256, tok.EndOfTranscript);
            Assert.Equal(50257, tok.StartOfTranscript);
            Assert.Equal(50358, tok.Transcribe);
            Assert.Equal(50362, tok.NoTimestamps);
            Assert.Equal(50363, tok.TimestampBegin);
        }

        [Fact]
        public void Decode_RawPieces_JoinUtf8_AndSkipsSpecials()
        {
            var cfg = new WhisperConfig(51865, 1500, 384, 6, 4, 448, 384, 6, 4, 80, true);

            // whisper.cpp ggml vocab = RAW UTF-8 token pieces (leading space is a literal 0x20).
            var vocab = new[] { "Z", "orv", "ex", " metal", " café" };
            var tok = new WhisperTokenizer(cfg, vocab);

            // [sot, en, transcribe, notimestamps, "Z","orv","ex"," metal"," café", eot]
            var ids = new[]
            {
                tok.StartOfTranscript, tok.LanguageTokenAt(0), tok.Transcribe, tok.NoTimestamps,
                0, 1, 2, 3, 4, tok.EndOfTranscript,
            };
            Assert.Equal("Zorvex metal café", tok.Decode(ids));
        }
    }
}
