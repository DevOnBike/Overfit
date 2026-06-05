// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Runtime.CompilerServices;

namespace DevOnBike.Overfit.LanguageModels.Constraints.Schema
{
    /// <summary>
    /// A character trie over a set of strings (object property names, or string-enum values), used during
    /// key/value string generation to restrict which characters may appear next and to detect a complete
    /// (terminal) string. Immutable after construction and shared across all speculative constraint copies.
    /// Children are a sorted flat <c>(char, int)[]</c> for cache locality — typical nodes have 1–5 children,
    /// where a linear scan beats a dictionary. (Ported from dotLLM's <c>PropertyNameTrie</c>.)
    /// </summary>
    public sealed class JsonStringTrie
    {
        private readonly TrieNode[] _nodes;

        // Per node, a bitmask of value indices (construction order, capped at 64) whose path passes through
        // it — i.e. which values are still reachable from here. For a property-name trie this aligns with
        // PropertyNames bit positions, so the tracker can mask a key character that leads only to
        // already-emitted properties (preventing a duplicate-key dead-end). Unused for enum tries.
        private readonly ulong[] _reachable;

        /// <summary>Builds a trie from <paramref name="values"/> (property names or enum values).</summary>
        public JsonStringTrie(IReadOnlyList<string> values)
        {
            ArgumentNullException.ThrowIfNull(values);

            // Build with dictionaries for convenience, then freeze each node's children to a sorted array.
            var children = new List<Dictionary<char, int>> { new() };  // node 0 = root
            var terminal = new List<bool> { false };
            var names = new List<string?> { null };

            for (var v = 0; v < values.Count; v++)
            {
                var value = values[v];
                var current = 0;
                foreach (var c in value)
                {
                    if (!children[current].TryGetValue(c, out var child))
                    {
                        child = children.Count;
                        children[current][c] = child;
                        children.Add(new Dictionary<char, int>());
                        terminal.Add(false);
                        names.Add(null);
                    }
                    current = child;
                }
                terminal[current] = true;
                names[current] = value;
            }

            _nodes = new TrieNode[children.Count];
            for (var i = 0; i < children.Count; i++)
            {
                var sorted = new (char Key, int Child)[children[i].Count];
                var j = 0;
                foreach (var kvp in children[i]) { sorted[j++] = (kvp.Key, kvp.Value); }
                Array.Sort(sorted, static (a, b) => a.Key.CompareTo(b.Key));
                _nodes[i] = new TrieNode(sorted, terminal[i], names[i]);
            }

            // Reachability bitmasks: for each value v (≤ 64), OR bit v into every node on its path (root
            // included), so ReachableMask(node) = the set of values still completable from that node.
            _reachable = new ulong[_nodes.Length];
            for (var v = 0; v < values.Count && v < 64; v++)
            {
                var node = 0;
                _reachable[node] |= 1UL << v;
                foreach (var c in values[v])
                {
                    TryGetChild(node, c, out node);
                    _reachable[node] |= 1UL << v;
                }
            }
        }

        /// <summary>Bitmask of value indices (construction order) still completable from
        /// <paramref name="nodeIndex"/>. See <see cref="_reachable"/>.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public ulong ReachableMask(int nodeIndex) => _reachable[nodeIndex];

        /// <summary>Advances from <paramref name="nodeIndex"/> by <paramref name="c"/>; false if no edge.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public bool TryGetChild(int nodeIndex, char c, out int childIndex)
        {
            var children = _nodes[nodeIndex].Children;
            for (var i = 0; i < children.Length; i++)
            {
                if (children[i].Key == c) { childIndex = children[i].Child; return true; }
                if (children[i].Key > c) { break; }   // sorted — no later match possible
            }
            childIndex = 0;
            return false;
        }

        /// <summary>Whether <paramref name="nodeIndex"/> is a complete (terminal) string.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public bool IsTerminal(int nodeIndex) => _nodes[nodeIndex].IsTerminal;

        /// <summary>The complete string at a terminal node, or null.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public string? GetCompleteName(int nodeIndex) => _nodes[nodeIndex].CompleteName;

        private readonly record struct TrieNode(
            (char Key, int Child)[] Children,
            bool IsTerminal,
            string? CompleteName);
    }
}
