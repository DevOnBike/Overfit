// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Audio.Tts;

namespace DevOnBike.Overfit.Tests.Audio
{
    /// <summary>TTS text normalization: numbers → words, symbols/abbreviations expanded, and the pronunciation
    /// lexicon (incl. the "Overfit" fix) applied. Model-free and deterministic.</summary>
    public sealed class TtsTextNormalizerTests
    {
        private readonly TtsTextNormalizer _n = new();

        [Theory]
        [InlineData(0, "zero")]
        [InlineData(7, "seven")]
        [InlineData(13, "thirteen")]
        [InlineData(42, "forty two")]
        [InlineData(100, "one hundred")]
        [InlineData(234, "two hundred thirty four")]
        [InlineData(1000, "one thousand")]
        [InlineData(1234, "one thousand two hundred thirty four")]
        [InlineData(1000000, "one million")]
        [InlineData(-5, "minus five")]
        public void NumberToWords_Cardinals(long value, string expected)
        {
            Assert.Equal(expected, EnglishNumberToWords.Convert(value));
        }

        [Fact]
        public void Normalize_IntegerInSentence_BecomesWords()
        {
            Assert.Equal("I have twelve apples", _n.Normalize("I have 12 apples"));
        }

        [Fact]
        public void Normalize_Decimal_SpokenAsPointDigits()
        {
            Assert.Equal("pi is three point one four", _n.Normalize("pi is 3.14"));
        }

        [Fact]
        public void Normalize_ThousandsSeparators_Stripped()
        {
            Assert.Equal("one thousand two hundred thirty four", _n.Normalize("1,234"));
        }

        [Fact]
        public void Normalize_OverfitBrandWord_RespelledForPronunciation()
        {
            // The exact issue the user hit: a brand/OOV word the model mangles.
            Assert.Equal("welcome to over fit", _n.Normalize("welcome to Overfit"));
        }

        [Fact]
        public void Normalize_Symbols_Expanded()
        {
            Assert.Equal("built in dot net and Rust", _n.Normalize("built in .NET & Rust"));
            Assert.Equal("fifty percent done", _n.Normalize("50% done"));
        }

        [Fact]
        public void Normalize_Abbreviation_Expanded()
        {
            Assert.Equal("use a model for example tara", _n.Normalize("use a model e.g. tara"));
        }

        [Fact]
        public void Normalize_DecimalNotConfusedWithSentenceEnd()
        {
            // The '.' after "done" is a sentence end (no digit follows); the one in 3.5 is a decimal.
            Assert.Equal("it is three point five. done", _n.Normalize("it is 3.5. done"));
        }

        [Fact]
        public void Normalize_CollapsesWhitespace()
        {
            Assert.Equal("a b c", _n.Normalize("a   b\t\n c"));
        }

        [Fact]
        public void Normalize_ExtraLexicon_OverridesAndExtends()
        {
            var n = new TtsTextNormalizer(new Dictionary<string, string> { ["zorvex"] = "zor vex" });
            Assert.Equal("the metal zor vex", n.Normalize("the metal Zorvex"));
        }
    }
}
