// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels.Constraints.Schema;

namespace DevOnBike.Overfit.Tests.LanguageModels.Constraints.Schema
{
    /// <summary>
    /// Fast unit tests for the JSON-Schema compiler + string trie (Stage 1 of the JSON-Schema constraint):
    /// type flags, object properties / required bitmask / additionalProperties, string enums, nested
    /// objects, array items, and the trie's character-walk + terminal detection. No model.
    /// </summary>
    public sealed class JsonSchemaCompilerTests
    {
        [Fact]
        public void Compile_Object_TypesPropertiesRequiredAndAdditional()
        {
            var schema = """
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

            var compiled = JsonSchemaCompiler.Compile(schema);
            ref readonly var root = ref compiled.Nodes[0];

            Assert.Equal(JsonSchemaType.Object, root.AllowedTypes);
            Assert.True(root.AdditionalPropertiesForbidden);
            Assert.NotNull(root.PropertyNames);
            Assert.Equal(["name", "age"], root.PropertyNames);

            // "name" is required (bit 0), "age" is not (bit 1).
            Assert.Equal(1UL, root.RequiredBitmask);

            // Property value nodes carry their own types.
            Assert.NotNull(root.Properties);
            Assert.Equal(JsonSchemaType.String, compiled.Nodes[root.Properties!["name"]].AllowedTypes);
            Assert.Equal(JsonSchemaType.Integer, compiled.Nodes[root.Properties["age"]].AllowedTypes);
        }

        [Fact]
        public void Compile_StringEnum_BuildsEnumTrie()
        {
            var compiled = JsonSchemaCompiler.Compile("""{ "type": "string", "enum": ["low", "high", "urgent"] }""");
            ref readonly var root = ref compiled.Nodes[0];

            Assert.Equal(JsonSchemaType.String, root.AllowedTypes);
            Assert.Equal(["low", "high", "urgent"], root.EnumValues);
            Assert.True(root.EnumTrieIndex >= 0);

            // The enum trie accepts "low" and marks it terminal; rejects a wrong char.
            var trie = compiled.Tries[root.EnumTrieIndex];
            Assert.True(trie.TryGetChild(0, 'l', out var n1));
            Assert.True(trie.TryGetChild(n1, 'o', out var n2));
            Assert.True(trie.TryGetChild(n2, 'w', out var n3));
            Assert.True(trie.IsTerminal(n3));
            Assert.Equal("low", trie.GetCompleteName(n3));
            Assert.False(trie.TryGetChild(0, 'z', out _));   // no enum starts with 'z'
        }

        [Fact]
        public void Compile_NestedObject_And_Array()
        {
            var schema = """
            {
              "type": "object",
              "properties": {
                "tags": { "type": "array", "items": { "type": "string" } },
                "meta": { "type": "object", "properties": { "id": { "type": "integer" } }, "required": ["id"] }
              }
            }
            """;

            var compiled = JsonSchemaCompiler.Compile(schema);
            ref readonly var root = ref compiled.Nodes[0];

            var tags = compiled.Nodes[root.Properties!["tags"]];
            Assert.Equal(JsonSchemaType.Array, tags.AllowedTypes);
            Assert.True(tags.ItemsNodeIndex >= 0);
            Assert.Equal(JsonSchemaType.String, compiled.Nodes[tags.ItemsNodeIndex].AllowedTypes);

            var meta = compiled.Nodes[root.Properties["meta"]];
            Assert.Equal(JsonSchemaType.Object, meta.AllowedTypes);
            Assert.Equal(1UL, meta.RequiredBitmask);   // "id" required
        }

        [Fact]
        public void Compile_TypeUnion_OrsTheFlags()
        {
            var compiled = JsonSchemaCompiler.Compile("""{ "type": ["string", "null"] }""");
            Assert.Equal(JsonSchemaType.String | JsonSchemaType.Null, compiled.Nodes[0].AllowedTypes);
        }

        [Fact]
        public void Compile_NoType_IsAny()
        {
            var compiled = JsonSchemaCompiler.Compile("{}");
            Assert.Equal(JsonSchemaType.Any, compiled.Nodes[0].AllowedTypes);
        }

        [Fact]
        public void Trie_PrefixSharing_IsTerminalOnlyAtWholeWords()
        {
            var trie = new JsonStringTrie(["car", "cart"]);
            // c-a-r is terminal ("car"); r→t continues to terminal ("cart").
            Assert.True(trie.TryGetChild(0, 'c', out var c));
            Assert.True(trie.TryGetChild(c, 'a', out var a));
            Assert.True(trie.TryGetChild(a, 'r', out var r));
            Assert.True(trie.IsTerminal(r));
            Assert.True(trie.TryGetChild(r, 't', out var t));
            Assert.True(trie.IsTerminal(t));
            Assert.Equal("cart", trie.GetCompleteName(t));
        }
    }
}
