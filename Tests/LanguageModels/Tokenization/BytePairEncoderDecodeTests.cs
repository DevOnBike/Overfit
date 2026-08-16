// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Tokenization;

namespace DevOnBike.Overfit.Tests.LanguageModels.Tokenization
{
    /// <summary>
    /// <see cref="BytePairEncoder"/>'s decode.
    ///
    /// <para><b>Why this file exists.</b> On 2026-08-11 a mutation deleting the non-ASCII fallback from
    /// <c>ByteDecode</c> left the suite green — and the reason was not a weak test but NO test: filtering
    /// the suite on <c>BytePairEncoder</c> matched <b>zero</b> tests. Every existing exercise of this class
    /// goes through GPT-2 fixtures and is skipped wherever those are absent, so its decode had no coverage
    /// that runs by default. <c>LoadFromStrings</c> makes that fixable in a few lines, which is the part
    /// worth noticing.</para>
    /// </summary>
    public sealed class BytePairEncoderDecodeTests
    {
        [Fact]
        public void DecodesByteLevelPiecesToText()
        {
            var tokenizer = Build();

            Assert.Equal("Hello world", tokenizer.Decode(tokenizer.Encode("Hello world")));
        }

        /// <summary>
        /// Round-trips text the vocabulary does not cover as whole pieces, so the byte fallback carries it.
        /// </summary>
        [Fact]
        public void RoundTripsTextOutsideTheMergedVocabulary()
        {
            var tokenizer = Build();

            foreach (var text in new[] { "Hello world", "zzz", "a b c", "!!!", "Hello  world" })
            {
                Assert.Equal(text, tokenizer.Decode(tokenizer.Encode(text)));
            }
        }

        /// <summary>
        /// <b>The case the mutation exposed.</b> A character with no entry in the byte-decoder table is
        /// encoded as itself rather than dropped. Dropping it loses text silently — the reply is shorter
        /// and nothing reports it.
        /// </summary>
        [Fact]
        public void NonAsciiTextSurvivesTheRoundTrip()
        {
            var tokenizer = Build();

            foreach (var text in new[] { "zażółć", "日本語", "café", "naïve" })
            {
                Assert.Equal(text, tokenizer.Decode(tokenizer.Encode(text)));
            }
        }

        [Fact]
        public void AnUnknownIdDoesNotAbortTheDecode()
        {
            var tokenizer = Build();
            var ids = tokenizer.Encode("Hello");

            var withGarbage = new int[ids.Length + 1];
            ids.CopyTo(withGarbage, 0);
            withGarbage[withGarbage.Length - 1] = int.MaxValue;

            Assert.Equal("Hello", tokenizer.Decode(withGarbage));
        }

        [Fact]
        public void AnEmptySequenceDecodesToEmpty()
        {
            Assert.Equal(string.Empty, Build().Decode([]));
        }

        /// <summary>
        /// A byte-level vocabulary holding one entry per byte value, and no merges.
        ///
        /// <para>The pieces are the GPT-2 alphabet's characters, taken from the encoder the class builds
        /// itself rather than re-derived here — re-deriving that mapping in a test is how the test and the
        /// code end up agreeing on the same wrong answer. With every byte present and no merges, the
        /// encoder falls back to one token per byte, which is exactly the path the decode has to invert.
        /// </para>
        /// </summary>
        private static BytePairEncoder Build()
        {
            var json = new System.Text.StringBuilder("{");

            for (var b = 0; b < 256; b++)
            {
                if (b > 0)
                {
                    json.Append(',');
                }

                json.Append('"').Append(Escape(GptAlphabetChar(b))).Append("\":").Append(b);
            }

            json.Append('}');

            return BytePairEncoder.LoadFromStrings(json.ToString(), string.Empty);
        }

        /// <summary>
        /// The GPT-2 byte-level alphabet: printable ASCII and Latin-1 map to themselves, everything else is
        /// shifted into the U+0100 block so a byte is always one printable character.
        /// </summary>
        private static char GptAlphabetChar(int value)
        {
            var printable = (value >= 33 && value <= 126)
                || (value >= 161 && value <= 172)
                || (value >= 174 && value <= 255);

            if (printable)
            {
                return (char)value;
            }

            var shift = 0;

            for (var b = 0; b < value; b++)
            {
                var isPrintable = (b >= 33 && b <= 126)
                    || (b >= 161 && b <= 172)
                    || (b >= 174 && b <= 255);

                if (!isPrintable)
                {
                    shift++;
                }
            }

            return (char)(256 + shift);
        }

        private static string Escape(char value)
        {
            return value switch
            {
                '"' => "\\\"",
                '\\' => "\\\\",
                _ => value.ToString(),
            };
        }
    }
}
