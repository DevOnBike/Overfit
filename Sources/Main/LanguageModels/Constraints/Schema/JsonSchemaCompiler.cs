// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Collections.Frozen;
using System.Text.Json;

namespace DevOnBike.Overfit.LanguageModels.Constraints.Schema
{
    /// <summary>
    /// Compiles a JSON-Schema document (text) into a flat <see cref="CompiledJsonSchema"/> for
    /// constrained decoding. Reflection-free (hand-walked <see cref="JsonDocument"/>), AOT-safe.
    ///
    /// MVP subset: <c>type</c> (single or union array), object <c>properties</c> + <c>required</c> +
    /// <c>additionalProperties:false</c>, string <c>enum</c>, array <c>items</c>, and arbitrary nesting of
    /// those. Unsupported keywords (<c>anyOf</c>, <c>const</c>, numeric <c>enum</c>, <c>$ref</c>,
    /// min/maxItems, pattern) are ignored — the position falls back to its declared (or any) type.
    /// </summary>
    public static class JsonSchemaCompiler
    {
        /// <summary>Compiles <paramref name="schemaJson"/>; the root schema becomes node 0.</summary>
        public static CompiledJsonSchema Compile(string schemaJson)
        {
            ArgumentNullException.ThrowIfNull(schemaJson);

            using var doc = JsonDocument.Parse(schemaJson);
            var nodes = new List<JsonSchemaNode>();
            var tries = new List<JsonStringTrie>();
            CompileNode(doc.RootElement, nodes, tries);

            // A shared all-types node for values under keys not declared in the schema (additional
            // properties when not forbidden) — see CompiledJsonSchema.UnconstrainedNodeIndex.
            var unconstrained = nodes.Count;
            nodes.Add(JsonSchemaNode.Unconstrained);

            return new CompiledJsonSchema(nodes.ToArray(), tries.ToArray(), unconstrained);
        }

        // Appends the compiled node(s) for `el` and returns this node's index. Reserves its own slot first
        // so recursively-compiled children get later indices (a parent may reference its children).
        private static int CompileNode(JsonElement el, List<JsonSchemaNode> nodes, List<JsonStringTrie> tries)
        {
            var idx = nodes.Count;
            nodes.Add(default);

            var types = ParseTypes(el);
            FrozenDictionary<string, int>? properties = null;
            string[]? propertyNames = null;
            ulong requiredMask = 0;
            var additionalForbidden = false;
            var itemsIndex = -1;
            string[]? enumValues = null;
            var propertyTrieIndex = -1;
            var enumTrieIndex = -1;

            var isObject = el.ValueKind == JsonValueKind.Object;

            if (isObject && el.TryGetProperty("properties", out var props) && props.ValueKind == JsonValueKind.Object)
            {
                var dict = new Dictionary<string, int>(StringComparer.Ordinal);
                var names = new List<string>();
                foreach (var p in props.EnumerateObject())
                {
                    var childIndex = CompileNode(p.Value, nodes, tries);
                    dict[p.Name] = childIndex;
                    names.Add(p.Name);
                }
                properties = dict.ToFrozenDictionary(StringComparer.Ordinal);
                propertyNames = names.ToArray();

                if (el.TryGetProperty("required", out var required) && required.ValueKind == JsonValueKind.Array)
                {
                    foreach (var r in required.EnumerateArray())
                    {
                        if (r.ValueKind != JsonValueKind.String) { continue; }
                        var bit = names.IndexOf(r.GetString() ?? string.Empty);
                        if (bit >= 0 && bit < 64) { requiredMask |= 1UL << bit; }
                    }
                }

                if (el.TryGetProperty("additionalProperties", out var ap) && ap.ValueKind == JsonValueKind.False)
                {
                    additionalForbidden = true;
                }

                if (names.Count > 0)
                {
                    propertyTrieIndex = tries.Count;
                    tries.Add(new JsonStringTrie(names));
                }

                if (types == JsonSchemaType.None) { types = JsonSchemaType.Object; }
            }

            if (isObject && el.TryGetProperty("enum", out var en) && en.ValueKind == JsonValueKind.Array)
            {
                var values = new List<string>();
                var allStrings = true;
                foreach (var e in en.EnumerateArray())
                {
                    if (e.ValueKind == JsonValueKind.String) { values.Add(e.GetString() ?? string.Empty); }
                    else { allStrings = false; }
                }
                if (values.Count > 0 && allStrings)
                {
                    enumValues = values.ToArray();
                    enumTrieIndex = tries.Count;
                    tries.Add(new JsonStringTrie(values));
                    if (types == JsonSchemaType.None) { types = JsonSchemaType.String; }
                }
            }

            if (isObject && el.TryGetProperty("items", out var items) && items.ValueKind == JsonValueKind.Object)
            {
                itemsIndex = CompileNode(items, nodes, tries);
                if (types == JsonSchemaType.None) { types = JsonSchemaType.Array; }
            }

            if (types == JsonSchemaType.None) { types = JsonSchemaType.Any; }

            nodes[idx] = new JsonSchemaNode
            {
                AllowedTypes = types,
                Properties = properties,
                PropertyNames = propertyNames,
                RequiredBitmask = requiredMask,
                AdditionalPropertiesForbidden = additionalForbidden,
                ItemsNodeIndex = itemsIndex,
                EnumValues = enumValues,
                PropertyTrieIndex = propertyTrieIndex,
                EnumTrieIndex = enumTrieIndex,
            };
            return idx;
        }

        private static JsonSchemaType ParseTypes(JsonElement el)
        {
            if (el.ValueKind != JsonValueKind.Object || !el.TryGetProperty("type", out var t))
            {
                return JsonSchemaType.None;
            }
            if (t.ValueKind == JsonValueKind.String)
            {
                return MapType(t.GetString());
            }
            if (t.ValueKind == JsonValueKind.Array)
            {
                var types = JsonSchemaType.None;
                foreach (var e in t.EnumerateArray())
                {
                    if (e.ValueKind == JsonValueKind.String) { types |= MapType(e.GetString()); }
                }
                return types;
            }
            return JsonSchemaType.None;
        }

        private static JsonSchemaType MapType(string? type) => type switch
        {
            "object" => JsonSchemaType.Object,
            "array" => JsonSchemaType.Array,
            "string" => JsonSchemaType.String,
            "number" => JsonSchemaType.Number,
            "integer" => JsonSchemaType.Integer,
            "boolean" => JsonSchemaType.Boolean,
            "null" => JsonSchemaType.Null,
            _ => JsonSchemaType.None,
        };
    }
}
