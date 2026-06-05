// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Collections.Frozen;

namespace DevOnBike.Overfit.LanguageModels.Constraints.Schema
{
    /// <summary>
    /// One compiled schema position in the flat <see cref="CompiledJsonSchema.Nodes"/> array. Immutable;
    /// references to sub-schemas (object property values, array items) are indices into that same array.
    /// (Design ported from dotLLM's <c>SchemaNode</c>, trimmed to the MVP feature set.)
    /// </summary>
    public readonly record struct JsonSchemaNode
    {
        /// <summary>Allowed value types at this position.</summary>
        public JsonSchemaType AllowedTypes { get; init; }

        /// <summary>For an object: property name → child node index. Null for non-objects.</summary>
        public FrozenDictionary<string, int>? Properties { get; init; }

        /// <summary>For an object: property names, ordered to match <see cref="RequiredBitmask"/> bit positions.</summary>
        public string[]? PropertyNames { get; init; }

        /// <summary>For an object: bitmask of required properties (bit i ↔ <see cref="PropertyNames"/>[i]).</summary>
        public ulong RequiredBitmask { get; init; }

        /// <summary>For an object: whether properties outside <see cref="Properties"/> are forbidden.</summary>
        public bool AdditionalPropertiesForbidden { get; init; }

        /// <summary>For an array: node index of the items schema, or -1 if unconstrained.</summary>
        public int ItemsNodeIndex { get; init; }

        /// <summary>For a string enum: the allowed values, or null if unconstrained.</summary>
        public string[]? EnumValues { get; init; }

        /// <summary>Index of this object's property-name trie in <see cref="CompiledJsonSchema.Tries"/>, or -1.</summary>
        public int PropertyTrieIndex { get; init; }

        /// <summary>Index of this string's enum-value trie in <see cref="CompiledJsonSchema.Tries"/>, or -1.</summary>
        public int EnumTrieIndex { get; init; }

        /// <summary>An unconstrained position — any JSON value.</summary>
        public static JsonSchemaNode Unconstrained => new()
        {
            AllowedTypes = JsonSchemaType.Any,
            ItemsNodeIndex = -1,
            PropertyTrieIndex = -1,
            EnumTrieIndex = -1,
        };
    }
}
