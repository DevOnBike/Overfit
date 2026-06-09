// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels.Constraints.Regex;

namespace DevOnBike.Overfit.Tests.LanguageModels.Constraints.Regex
{
    /// <summary>
    /// Fast tests for the regex → NFA → DFA compiler (<see cref="RegexDfa"/>): literals, char classes,
    /// predefined classes, quantifiers (<c>* + ? {n} {n,m}</c>), alternation, groups — verified by anchored
    /// full-match of accepted / rejected strings. No model.
    /// </summary>
    public sealed class RegexDfaTests
    {
        private static bool FullMatch(string pattern, string input)
        {
            var dfa = RegexDfa.Compile(pattern);
            var state = dfa.Start;

            foreach (var c in input)
            {
                state = dfa.Next(state, c);
                if (state < 0) { return false; }
            }

            return dfa.IsAccepting(state);
        }

        [Theory]
        [InlineData(@"\d{4}-\d{2}-\d{2}", "2026-06-04", true)]
        [InlineData(@"\d{4}-\d{2}-\d{2}", "2026-6-04", false)]   // month must be two digits
        [InlineData(@"\d{4}-\d{2}-\d{2}", "abcd-06-04", false)]
        [InlineData(@"[A-Z]{3}", "ABC", true)]
        [InlineData(@"[A-Z]{3}", "AB", false)]
        [InlineData(@"[A-Z]{3}", "ABCD", false)]
        [InlineData(@"[A-Z]{3}", "abc", false)]
        [InlineData(@"(cat|dog)s?", "cat", true)]
        [InlineData(@"(cat|dog)s?", "cats", true)]
        [InlineData(@"(cat|dog)s?", "dog", true)]
        [InlineData(@"(cat|dog)s?", "cow", false)]
        [InlineData(@"a*b+", "b", true)]
        [InlineData(@"a*b+", "aaabbb", true)]
        [InlineData(@"a*b+", "a", false)]
        [InlineData(@"a*b+", "", false)]
        [InlineData(@"\w+@\w+\.\w+", "alice@example.com", true)]
        [InlineData(@"\w+@\w+\.\w+", "alice@example", false)]
        [InlineData(@"\d{2,4}", "1", false)]
        [InlineData(@"\d{2,4}", "12", true)]
        [InlineData(@"\d{2,4}", "1234", true)]
        [InlineData(@"\d{2,4}", "12345", false)]
        public void FullMatch_Behaves(string pattern, string input, bool expected)
        {
            Assert.Equal(expected, FullMatch(pattern, input));
        }

        [Fact]
        public void Dot_ExcludesNewline()
        {
            Assert.True(FullMatch(".", "x"));
            Assert.False(FullMatch(".", "\n"));
        }
    }
}
