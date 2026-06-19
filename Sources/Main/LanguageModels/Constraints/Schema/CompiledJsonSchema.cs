// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.LanguageModels.Constraints.Schema
{
    /// <summary>
    /// An immutable compiled JSON schema: a flat array of <see cref="JsonSchemaNode"/> (node 0 is the root)
    /// plus the shared string tries they reference. Produced by <see cref="JsonSchemaCompiler"/> and shared
    /// (read-only) across every speculative copy of the schema tracker.
    /// </summary>
    public sealed class CompiledJsonSchema
    {
        public CompiledJsonSchema(JsonSchemaNode[] nodes, JsonStringTrie[] tries, int unconstrainedNodeIndex)
        {
            Nodes = nodes ?? throw new ArgumentNullException(nameof(nodes));
            Tries = tries ?? throw new ArgumentNullException(nameof(tries));

            if (nodes.Length == 0)
            {
                throw new ArgumentException("A compiled schema must have at least the root node.", nameof(nodes));
            }
            if ((uint)unconstrainedNodeIndex >= (uint)nodes.Length)
            {
                throw new ArgumentOutOfRangeException(nameof(unconstrainedNodeIndex));
            }

            UnconstrainedNodeIndex = unconstrainedNodeIndex;
        }

        /// <summary>Flat node array; index 0 is the root.</summary>
        public JsonSchemaNode[] Nodes
        {
            get;
        }

        /// <summary>Shared property-name / enum-value tries referenced by node trie indices.</summary>
        public JsonStringTrie[] Tries
        {
            get;
        }

        /// <summary>Index of a shared all-types node, used for values under a key not present in the schema
        /// (an additional property when <c>additionalProperties</c> is not forbidden).</summary>
        public int UnconstrainedNodeIndex
        {
            get;
        }
    }
}
