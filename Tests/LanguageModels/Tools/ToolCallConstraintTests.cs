// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Text;
using DevOnBike.Overfit.LanguageModels.Contracts;
using DevOnBike.Overfit.LanguageModels.Tools;

namespace DevOnBike.Overfit.Tests.LanguageModels.Tools
{
    /// <summary>
    /// Tests the tool-call envelope constraint with a hand-built character vocabulary: the model can
    /// only ever produce <c>{"name": "&lt;registered tool&gt;", "arguments": &lt;valid json&gt;}</c>.
    /// Driven character-by-character (each "token" is one char) so it needs no model.
    /// </summary>
    public sealed class ToolCallConstraintTests
    {
        private static readonly ToolDefinition[] Tools =
        [
            new("get_weather", "weather by city"),
            new("get_time", "current time"),
        ];

        // Same tools, now with declared parameter schemas — the constraint must force the exact keys,
        // order and value types (not just well-formed JSON).
        private static readonly ToolDefinition[] SchemaTools =
        [
            new("get_weather", "weather by city", [new ToolParameter("city")]),
            new("add", "add two integers",
                [new ToolParameter("a", ToolParameterKind.Integer), new ToolParameter("b", ToolParameterKind.Integer)]),
        ];

        [Fact]
        public void ForcesEnvelope_AndAcceptsExactlyOneToolName()
        {
            // The full canonical call must be accepted end-to-end and reported complete only at the end.
            var ok = Drive("{\"name\": \"get_weather\", \"arguments\": {\"city\": \"paris\"}}", out var complete);
            Assert.True(ok, "valid tool call was rejected");
            Assert.True(complete, "valid tool call not reported complete");
        }

        [Fact]
        public void RejectsUnknownToolName()
        {
            // "get_zzz" diverges from every registered name → rejected mid-name.
            Assert.False(Drive("{\"name\": \"get_zzz\", \"arguments\": {}}", out _));
        }

        [Fact]
        public void RejectsMalformedArguments()
        {
            // arguments value is not well-formed JSON (trailing comma) → rejected.
            Assert.False(Drive("{\"name\": \"get_time\", \"arguments\": {\"a\":1,}}", out _));
        }

        [Fact]
        public void RejectsBrokenEnvelope()
        {
            // Missing the "arguments" key entirely.
            Assert.False(Drive("{\"name\": \"get_time\"}", out _));
        }

        [Fact]
        public void FirstAllowedCharIsOpeningBrace()
        {
            var c = new ToolCallConstraint(Tools, new CharTokenizer(Alphabet()));
            var logits = new float[Alphabet().Length];
            c.ApplyMask(logits);

            Assert.False(float.IsNegativeInfinity(logits[IndexOf('{')]));   // envelope must start with {
            Assert.True(float.IsNegativeInfinity(logits[IndexOf('"')]));    // anything else is masked
            Assert.True(float.IsNegativeInfinity(logits[IndexOf('x')]));
        }

        [Fact]
        public void Schema_AcceptsExactKeysAndStringValue()
        {
            var ok = Drive(SchemaTools, "{\"name\": \"get_weather\", \"arguments\": {\"city\": \"paris\"}}", out var complete);
            Assert.True(ok, "valid schema'd tool call was rejected");
            Assert.True(complete, "valid schema'd tool call not reported complete");
        }

        [Fact]
        public void Schema_RejectsWrongKeyName()
        {
            // schema declares "city"; the model trying "town" diverges at the very first key character.
            Assert.False(Drive(SchemaTools, "{\"name\": \"get_weather\", \"arguments\": {\"town\": \"paris\"}}", out _));
        }

        [Fact]
        public void Schema_RejectsWrongValueType()
        {
            // "city" is a String parameter; a numeric value must be rejected at the value's first char.
            Assert.False(Drive(SchemaTools, "{\"name\": \"get_weather\", \"arguments\": {\"city\": 5}}", out _));
        }

        [Fact]
        public void Schema_EnforcesMultiKeyOrderAndIntegerValues()
        {
            var ok = Drive(SchemaTools, "{\"name\": \"add\", \"arguments\": {\"a\": 12, \"b\": -3}}", out var complete);
            Assert.True(ok, "valid two-integer call was rejected");
            Assert.True(complete);
        }

        [Fact]
        public void Schema_RejectsKeysOutOfOrder()
        {
            // schema order is a, then b; emitting "b" first diverges at the first key character.
            Assert.False(Drive(SchemaTools, "{\"name\": \"add\", \"arguments\": {\"b\": 1, \"a\": 2}}", out _));
        }

        [Fact]
        public void Schema_RejectsMissingSecondKey()
        {
            // After the first value, the only continuation is the ', "b": ' segment — closing early fails.
            Assert.False(Drive(SchemaTools, "{\"name\": \"add\", \"arguments\": {\"a\": 1}}", out _));
        }

        // Drives the constraint one character at a time via Accept, masking before each to confirm
        // the character was actually permitted. Returns false at the first masked character.
        private static bool Drive(string text, out bool complete) => Drive(Tools, text, out complete);

        private static bool Drive(ToolDefinition[] tools, string text, out bool complete)
        {
            var alphabet = Alphabet();
            var c = new ToolCallConstraint(tools, new CharTokenizer(alphabet));
            var logits = new float[alphabet.Length];

            foreach (var ch in text)
            {
                logits.AsSpan().Clear();
                c.ApplyMask(logits);
                var idx = alphabet.IndexOf(ch);
                if (idx < 0 || float.IsNegativeInfinity(logits[idx]))
                {
                    complete = false;
                    return false;
                }
                c.Accept(idx);
            }

            complete = c.IsComplete;
            return true;
        }

        private static string Alphabet() => "{}[]\":, \tabcdefghijklmnopqrstuvwxyz0123456789_-";

        private static int IndexOf(char c) => Alphabet().IndexOf(c);

        // One token per character; EndOfText is a sentinel id past the alphabet.
        private sealed class CharTokenizer : ITokenizer
        {
            private readonly string _alphabet;

            public CharTokenizer(string alphabet) => _alphabet = alphabet;

            public int VocabularySize => _alphabet.Length;
            public int EndOfTextTokenId => _alphabet.Length;   // outside the alphabet ⇒ never a char
            public int UnknownTokenId => -1;
            public bool SupportsZeroAllocationEncode => false;
            public bool SupportsZeroAllocationDecode => false;

            public int CountTokens(ReadOnlySpan<char> text) => throw new NotSupportedException();
            public int Encode(ReadOnlySpan<char> text, Span<int> destination) => throw new NotSupportedException();
            public int Decode(ReadOnlySpan<int> tokens, Span<char> destination) => throw new NotSupportedException();

            public string DecodeToString(ReadOnlySpan<int> tokens)
            {
                var sb = new StringBuilder();
                foreach (var t in tokens)
                {
                    if (t >= 0 && t < _alphabet.Length)
                    {
                        sb.Append(_alphabet[t]);
                    }
                }
                return sb.ToString();
            }
        }
    }
}
