// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels.Constraints;
using DevOnBike.Overfit.LanguageModels.Constraints.Schema;

namespace DevOnBike.Overfit.Tests.LanguageModels.Constraints.Schema
{
    /// <summary>
    /// Fast char-level tests for <see cref="JsonSchemaTracker"/> + <see cref="JsonStateMachine"/> in lockstep
    /// (Stage 2 of the JSON-Schema constraint) — exactly the per-character loop the token constraint runs.
    /// Validates type restriction, required-property gating of <c>}</c>, declared-key restriction under
    /// <c>additionalProperties:false</c>, string enums, and integer-only number rules. No model.
    /// </summary>
    public sealed class JsonSchemaTrackerTests
    {
        // Feeds `json` char-by-char; returns false at the first character the schema/parser rejects.
        private static bool Feed(CompiledJsonSchema schema, string json, out bool complete)
        {
            var machine = new JsonStateMachine();
            var tracker = new JsonSchemaTracker(schema);
            foreach (var c in json)
            {
                if (!tracker.IsCharAllowedBySchema(c, in machine)) { complete = false; return false; }
                if (!machine.TryAdvance(c)) { complete = false; return false; }
                tracker.OnCharAdvanced(c, in machine);
            }
            complete = machine.IsComplete;
            return true;
        }

        private const string PersonSchema = """
        {
          "type": "object",
          "properties": {
            "name": { "type": "string" },
            "age":  { "type": "integer" }
          },
          "required": ["name"],
          "additionalProperties": false
        }
        """;

        [Fact]
        public void Accepts_ConformingObject()
        {
            var schema = JsonSchemaCompiler.Compile(PersonSchema);
            Assert.True(Feed(schema, """{"name":"Bob","age":5}""", out var complete));
            Assert.True(complete);
        }

        [Fact]
        public void Rejects_WrongPropertyType()
        {
            var schema = JsonSchemaCompiler.Compile(PersonSchema);
            // "age" must be an integer — a string value is rejected at the opening quote.
            Assert.False(Feed(schema, """{"name":"Bob","age":"five"}""", out _));
        }

        [Fact]
        public void Rejects_ClosingBeforeRequiredEmitted()
        {
            var schema = JsonSchemaCompiler.Compile(PersonSchema);
            // "name" is required, so an object with only "age" cannot close.
            Assert.False(Feed(schema, """{"age":5}""", out _));
        }

        [Fact]
        public void Rejects_UndeclaredKey_WhenAdditionalForbidden()
        {
            var schema = JsonSchemaCompiler.Compile(PersonSchema);
            Assert.False(Feed(schema, """{"name":"Bob","color":"red"}""", out _));
        }

        [Fact]
        public void Enum_AcceptsMember_RejectsNonMember()
        {
            var schema = JsonSchemaCompiler.Compile("""{ "type": "string", "enum": ["low", "high", "urgent"] }""");
            Assert.True(Feed(schema, "\"high\"", out var complete));
            Assert.True(complete);
            Assert.False(Feed(schema, "\"medium\"", out _));
        }

        [Fact]
        public void IntegerType_RejectsFraction()
        {
            var schema = JsonSchemaCompiler.Compile("""{ "type": "integer" }""");
            Assert.True(Feed(schema, "42", out var complete));
            Assert.True(complete);
            Assert.False(Feed(schema, "42.5", out _));   // fraction not allowed for integer
        }

        [Fact]
        public void NestedObject_RequiredAndTypes_Enforced()
        {
            var schema = JsonSchemaCompiler.Compile("""
            {
              "type": "object",
              "properties": {
                "meta": { "type": "object", "properties": { "id": { "type": "integer" } }, "required": ["id"], "additionalProperties": false }
              },
              "required": ["meta"],
              "additionalProperties": false
            }
            """);
            Assert.True(Feed(schema, """{"meta":{"id":7}}""", out var complete));
            Assert.True(complete);
            Assert.False(Feed(schema, """{"meta":{}}""", out _));        // nested "id" required
        }
    }
}
