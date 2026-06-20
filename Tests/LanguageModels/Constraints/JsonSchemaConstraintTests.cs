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
    /// Fast tests (no model) for <see cref="JsonSchemaConstraint"/> token masking, driven by a one-char-per-token
    /// ASCII tokenizer (logit index == char code). Verifies the masked vocabulary at key positions — object
    /// open, required-field gating of <c>}</c>, string enums — and an end-to-end conforming generation.
    /// </summary>
    public sealed class JsonSchemaConstraintTests
    {
        private const string PersonSchema = """
        {
          "type": "object",
          "properties": { "name": { "type": "string" }, "age": { "type": "integer" } },
          "required": ["name"],
          "additionalProperties": false
        }
        """;

        private const float Masked = float.NegativeInfinity;

        [Fact]
        public void AtStart_OnlyObjectOpenAllowed()
        {
            var c = new JsonSchemaConstraint(new AsciiTokenizer(), PersonSchema);
            var logits = Fresh();
            c.ApplyMask(logits);

            Assert.NotEqual(Masked, logits['{']);   // an object may open
            Assert.Equal(Masked, logits['x']);      // a bare value cannot
            Assert.Equal(Masked, logits['[']);      // root type is object, not array
        }

        [Fact]
        public void RequiredField_BlocksClosing_AllowsContinuation()
        {
            var c = new JsonSchemaConstraint(new AsciiTokenizer(), PersonSchema);
            AcceptAll(c, """{"age":5""");
            var logits = Fresh();
            c.ApplyMask(logits);

            Assert.Equal(Masked, logits['}']);       // cannot close — "name" is required and not yet emitted
            Assert.NotEqual(Masked, logits[',']);    // may continue to another property
        }

        [Fact]
        public void WrongType_IsMasked_AtValueStart()
        {
            var c = new JsonSchemaConstraint(new AsciiTokenizer(), PersonSchema);
            AcceptAll(c, """{"name":"Bob","age":""");
            var logits = Fresh();
            c.ApplyMask(logits);

            Assert.Equal(Masked, logits['"']);       // "age" is an integer — a string value is forbidden
            Assert.NotEqual(Masked, logits['5']);    // a digit is allowed
        }

        [Fact]
        public void Enum_MasksNonMemberCharacters()
        {
            var c = new JsonSchemaConstraint(new AsciiTokenizer(), """{ "type": "string", "enum": ["low", "high"] }""");
            c.Accept('"');
            var logits = Fresh();
            c.ApplyMask(logits);

            Assert.NotEqual(Masked, logits['l']);    // "low"
            Assert.NotEqual(Masked, logits['h']);    // "high"
            Assert.Equal(Masked, logits['m']);       // no enum value starts with 'm'
        }

        [Fact]
        public void ConformingDocument_GeneratesEndToEnd_AndCompletes()
        {
            var c = new JsonSchemaConstraint(new AsciiTokenizer(), PersonSchema);
            var logits = new float[128];
            foreach (var ch in """{"name":"Bob","age":5}""")
            {
                Reset(logits);
                c.ApplyMask(logits);
                Assert.NotEqual(Masked, logits[ch]);   // every character of a conforming document is allowed
                c.Accept(ch);
            }
            Assert.True(c.IsComplete);
        }

        private static float[] Fresh()
        {
            var l = new float[128];
            return l;   // already all-zero
        }

        private static void Reset(float[] logits) => Array.Clear(logits);

        private static void AcceptAll(JsonSchemaConstraint c, string text)
        {
            foreach (var ch in text)
            {
                c.Accept(ch);
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
                for (var i = 0; i < text.Length; i++)
                {
                    destination[i] = text[i];
                }
                return text.Length;
            }

            public int Decode(ReadOnlySpan<int> tokens, Span<char> destination)
            {
                for (var i = 0; i < tokens.Length; i++)
                {
                    destination[i] = (char)tokens[i];
                }
                return tokens.Length;
            }

            public string DecodeToString(ReadOnlySpan<int> tokens)
            {
                var sb = new StringBuilder(tokens.Length);
                foreach (var t in tokens)
                {
                    if (t is >= 0 and < 127)
                    {
                        sb.Append((char)t);
                    }   // 127 = EOS → empty
                }
                return sb.ToString();
            }
        }
    }
}
