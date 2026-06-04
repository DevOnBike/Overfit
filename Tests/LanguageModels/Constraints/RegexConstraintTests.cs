// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Text;
using DevOnBike.Overfit.LanguageModels.Constraints;
using DevOnBike.Overfit.LanguageModels.Contracts;

namespace DevOnBike.Overfit.Tests.LanguageModels.Constraints
{
    /// <summary>
    /// Fast tests (no model) for <see cref="RegexConstraint"/> token masking, driven by a one-char-per-token
    /// ASCII tokenizer (logit index == char code): the constraint allows exactly the characters that keep the
    /// automaton alive, gates end-of-text on an accepting state, and accepts a full matching string.
    /// </summary>
    public sealed class RegexConstraintTests
    {
        private const float Masked = float.NegativeInfinity;
        private const int Eos = 127;

        [Fact]
        public void DatePattern_AllowsOnlyDigitsAndDashes_AndCompletesAtFullMatch()
        {
            var c = new RegexConstraint(new AsciiTokenizer(), @"\d{4}-\d{2}-\d{2}");
            var logits = new float[128];

            // At the start only a digit may begin the date.
            Array.Clear(logits);
            c.ApplyMask(logits);
            Assert.NotEqual(Masked, logits['2']);
            Assert.Equal(Masked, logits['a']);
            Assert.Equal(Masked, logits['-']);   // a dash cannot be first
            Assert.Equal(Masked, logits[Eos]);   // not a complete match yet

            // Drive a full conforming date; every character must be allowed.
            foreach (var ch in "2026-06-04")
            {
                Array.Clear(logits);
                c.ApplyMask(logits);
                Assert.NotEqual(Masked, logits[ch]);
                c.Accept(ch);
            }

            Assert.True(c.IsComplete);

            // Now end-of-text is allowed, and no further digit may extend the (complete) date.
            Array.Clear(logits);
            c.ApplyMask(logits);
            Assert.NotEqual(Masked, logits[Eos]);
            Assert.Equal(Masked, logits['0']);
        }

        [Fact]
        public void Alternation_MasksTokensOffEveryBranch()
        {
            var c = new RegexConstraint(new AsciiTokenizer(), "(cat|dog)");
            var logits = new float[128];
            c.ApplyMask(logits);

            Assert.NotEqual(Masked, logits['c']);   // "cat"
            Assert.NotEqual(Masked, logits['d']);   // "dog"
            Assert.Equal(Masked, logits['x']);      // neither branch
        }

        [Fact]
        public void MidPattern_MasksCharactersThatLeaveTheAutomaton()
        {
            var c = new RegexConstraint(new AsciiTokenizer(), @"[A-Z]{3}");
            c.Accept('A');
            c.Accept('B');

            var logits = new float[128];
            c.ApplyMask(logits);
            Assert.NotEqual(Masked, logits['C']);   // a third letter completes it
            Assert.Equal(Masked, logits['1']);      // a digit is not in [A-Z]
            Assert.Equal(Masked, logits[Eos]);      // only two letters so far — not a full match
        }

        [Fact]
        public void DeadEnd_GracefullyAllowsEndOfText_InsteadOfDegenerating()
        {
            // Only two-character tokens exist, so after "AB" (2 of 3 letters) no token can add exactly one
            // more without overshooting — a BPE dead-end. The constraint must then un-mask end-of-text so
            // generation stops (with the valid prefix) rather than repeating a masked token forever.
            var c = new RegexConstraint(new TwoCharTokenizer(), "[A-Z]{3}");
            c.Accept(0);   // "AB"

            var logits = new float[3];
            c.ApplyMask(logits);

            Assert.Equal(Masked, logits[0]);      // "AB" → would make four letters
            Assert.Equal(Masked, logits[1]);      // "CD" → four letters
            Assert.NotEqual(Masked, logits[2]);   // end-of-text un-masked — graceful escape
        }

        // Vocabulary of two two-char tokens ("AB", "CD") + an end-of-text token (id 2).
        private sealed class TwoCharTokenizer : ITokenizer
        {
            public int VocabularySize => 3;
            public int EndOfTextTokenId => 2;
            public int UnknownTokenId => 0;
            public bool SupportsZeroAllocationEncode => false;
            public bool SupportsZeroAllocationDecode => false;

            public int CountTokens(ReadOnlySpan<char> text) => text.Length;
            public int Encode(ReadOnlySpan<char> text, Span<int> destination) => 0;
            public int Decode(ReadOnlySpan<int> tokens, Span<char> destination) => 0;

            public string DecodeToString(ReadOnlySpan<int> tokens)
            {
                var sb = new StringBuilder();
                foreach (var t in tokens)
                {
                    sb.Append(t switch { 0 => "AB", 1 => "CD", _ => string.Empty });
                }
                return sb.ToString();
            }
        }

        // One token per ASCII char (token id == char code); 127 is the end-of-text sentinel (decodes to "").
        private sealed class AsciiTokenizer : ITokenizer
        {
            public int VocabularySize => 128;
            public int EndOfTextTokenId => 127;
            public int UnknownTokenId => 0;
            public bool SupportsZeroAllocationEncode => true;
            public bool SupportsZeroAllocationDecode => true;

            public int CountTokens(ReadOnlySpan<char> text) => text.Length;

            public int Encode(ReadOnlySpan<char> text, Span<int> destination)
            {
                for (var i = 0; i < text.Length; i++) { destination[i] = text[i]; }
                return text.Length;
            }

            public int Decode(ReadOnlySpan<int> tokens, Span<char> destination)
            {
                for (var i = 0; i < tokens.Length; i++) { destination[i] = (char)tokens[i]; }
                return tokens.Length;
            }

            public string DecodeToString(ReadOnlySpan<int> tokens)
            {
                var sb = new StringBuilder(tokens.Length);
                foreach (var t in tokens)
                {
                    if (t is >= 0 and < 127) { sb.Append((char)t); }
                }
                return sb.ToString();
            }
        }
    }
}
