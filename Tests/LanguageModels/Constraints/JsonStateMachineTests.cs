// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels.Constraints;

namespace DevOnBike.Overfit.Tests.LanguageModels.Constraints
{
    /// <summary>
    /// Unit tests for the well-formed-JSON acceptor that backs JSON-mode constrained generation:
    /// every valid document is accepted and reported complete, every malformed one is rejected at
    /// the offending character, and valid-but-unfinished prefixes are accepted yet not complete.
    /// </summary>
    public sealed class JsonStateMachineTests
    {
        [Theory]
        [InlineData("{}")]
        [InlineData("[]")]
        [InlineData("\"hello\"")]
        [InlineData("123")]
        [InlineData("0")]
        [InlineData("-1.5e10")]
        [InlineData("-0.25E+3")]
        [InlineData("true")]
        [InlineData("false")]
        [InlineData("null")]
        [InlineData("{\"a\":1}")]
        [InlineData("[1,2,3]")]
        [InlineData("{\"a\":[1,{\"b\":\"c\"}],\"d\":null}")]
        [InlineData("  { \"x\" : 1 }  ")]
        [InlineData("\"esc \\\" \\n \\u00e9\"")]
        public void Accepts_CompleteValidJson(string json)
        {
            Assert.True(Feed(json, out var complete), $"rejected valid JSON: {json}");
            Assert.True(complete, $"valid JSON not reported complete: {json}");
        }

        [Theory]
        [InlineData("{")]
        [InlineData("[1,")]
        [InlineData("\"unclosed")]
        [InlineData("1.")]
        [InlineData("-")]
        [InlineData("tru")]
        [InlineData("{\"a\":")]
        public void Accepts_ValidPrefix_ButNotComplete(string json)
        {
            Assert.True(Feed(json, out var complete), $"rejected valid prefix: {json}");
            Assert.False(complete, $"incomplete JSON wrongly reported complete: {json}");
        }

        [Theory]
        [InlineData("{,}")]
        [InlineData("[,]")]
        [InlineData("[1,]")]
        [InlineData("{\"a\"}")]      // key with no colon/value
        [InlineData("{\"a\":1,}")]   // trailing comma
        [InlineData("01")]           // leading zero
        [InlineData("1.e5")]         // no fraction digit
        [InlineData(".5")]           // no integer part
        [InlineData("+1")]
        [InlineData("nulll")]        // junk after complete value
        [InlineData("[1 2]")]        // missing comma
        [InlineData("\"a\\x\"")]     // invalid escape
        [InlineData("}")]
        public void Rejects_MalformedJson(string json)
        {
            Assert.False(Feed(json, out _), $"accepted malformed JSON: {json}");
        }

        private static bool Feed(string s, out bool complete)
        {
            var sm = new JsonStateMachine();
            foreach (var c in s)
            {
                if (!sm.TryAdvance(c))
                {
                    complete = false;
                    return false;
                }
            }
            complete = sm.IsComplete;
            return true;
        }
    }
}
