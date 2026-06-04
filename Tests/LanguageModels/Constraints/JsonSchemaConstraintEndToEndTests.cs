// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Text.Json;
using DevOnBike.Overfit.LanguageModels;
using DevOnBike.Overfit.LanguageModels.Constraints;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests.LanguageModels.Constraints
{
    /// <summary>
    /// End-to-end: the real Qwen2.5-3B, decoded under a <see cref="JsonSchemaConstraint"/>, emits JSON that
    /// conforms to the supplied schema by construction — parses, has every required field with the right
    /// type, and (additionalProperties:false) carries no extra keys. [LongFact] (needs C:\qwen3b).
    /// </summary>
    public sealed class JsonSchemaConstraintEndToEndTests
    {
        private const string Gguf = @"C:\qwen3b\qwen.q4km.gguf";

        private readonly ITestOutputHelper _out;
        public JsonSchemaConstraintEndToEndTests(ITestOutputHelper output) => _out = output;

        [LongFact]
        public void RealQwen_EmitsSchemaConformingJson()
        {
            if (!File.Exists(Gguf)) { _out.WriteLine($"missing {Gguf}"); return; }

            using var client = OverfitClient.LoadGguf(Gguf, mmap: true);

            const string schema = """
            {
              "type": "object",
              "properties": {
                "sentiment": { "type": "string", "enum": ["positive", "negative", "neutral"] }
              },
              "required": ["sentiment"],
              "additionalProperties": false
            }
            """;

            var constraint = new JsonSchemaConstraint(client.Tokenizer, schema);
            var reply = client.Send(
                "Classify the sentiment of this review as JSON: \"I absolutely love this product, it is fantastic!\"",
                onText: null, constraint: constraint);

            _out.WriteLine($"reply: {reply}");

            using var doc = JsonDocument.Parse(reply);   // schema-valid → always parses
            var root = doc.RootElement;
            Assert.Equal(JsonValueKind.Object, root.ValueKind);

            Assert.True(root.TryGetProperty("sentiment", out var sentiment) && sentiment.ValueKind == JsonValueKind.String, "sentiment:string");
            Assert.Contains(sentiment.GetString(), new[] { "positive", "negative", "neutral" });   // enum honoured

            // additionalProperties:false → exactly the one declared key.
            var count = 0;
            foreach (var _ in root.EnumerateObject()) { count++; }
            Assert.Equal(1, count);
        }
    }
}
