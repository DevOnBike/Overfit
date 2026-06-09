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

        [Theory]
        [InlineData("AI", "A I")]
        [InlineData("ai", "A I")]   // case-insensitive
        [InlineData("Ai", "A I")]
        [InlineData("GPU", "G P U")]
        [InlineData("GGUF", "G G U F")]
        [InlineData("LLM", "L L M")]
        [InlineData("GPT", "G P T")]
        [InlineData("HTTP", "H T T P")]
        [InlineData("HTTPS", "H T T P S")]
        [InlineData("IoT", "I O T")]
        [InlineData("JSON", "jason")]  // pronounced, not spelled
        public void Normalize_Acronym_SpelledOrPronounced(string input, string expected)
        {
            // Acronyms the model reads wrong raw — the lexicon spells them so Orpheus says the letter names.
            Assert.Equal(expected, _n.Normalize(input));
        }

        [Fact]
        public void Normalize_AcronymsInSentence_EachExpanded()
        {
            Assert.Equal(
                "the A I runs on the C P U not the G P U",
                _n.Normalize("the AI runs on the CPU not the GPU"));
        }

        [Theory]
        [InlineData("brain", "brain")]     // contains "ai" but is one word — must NOT become "br A I n"
        [InlineData("mlops", "mlops")]     // contains "ml" — whole-word only
        [InlineData("biostatus", "biostatus")]
        [InlineData("vmware", "vmware")]   // contains "vm" — whole-word only
        public void Normalize_AcronymSubstringInsideWord_NotReplaced(string input, string expected)
        {
            // Whole-word matching: an acronym that appears as a substring of a real word is left alone.
            Assert.Equal(expected, _n.Normalize(input));
        }
    }
}
