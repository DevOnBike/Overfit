// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Runtime.CompilerServices;
using Phase = DevOnBike.Overfit.LanguageModels.Constraints.JsonStateMachine.Phase;

namespace DevOnBike.Overfit.LanguageModels.Constraints.Schema
{
    /// <summary>
    /// A schema overlay that advances in lockstep with a <see cref="JsonStateMachine"/> to enforce a
    /// <see cref="CompiledJsonSchema"/> on top of well-formedness: value-type restriction, required-property
    /// gating of the closing <c>}</c>, property-name restriction (only declared keys when
    /// <c>additionalProperties:false</c>), and string-enum restriction. Logic ported from dotLLM's
    /// <c>SchemaTracker</c>, retargeted to this engine's parser phases.
    ///
    /// Value type — a by-value copy is a deep clone of the (inline) stacks, so the constraint can probe a
    /// candidate token speculatively then keep the copy only if every character is accepted. The
    /// <see cref="CompiledJsonSchema"/> is shared by reference (immutable). Max nesting 64.
    /// </summary>
    public struct JsonSchemaTracker
    {
        private const int MaxDepth = 64;
        private const int MaxKeyLength = 128;

        private readonly CompiledJsonSchema _schema;

        private NodeIdxStack _nodeStack;     // schema node index of each containing object/array
        private int _stackDepth;
        private int _currentNodeIndex;       // schema node for the value currently being generated
        private PropBitStack _emittedProps;  // emitted-property bitmask per object nesting level
        private KeyBuffer _keyBuffer;        // current key characters (to match against the schema)
        private int _keyLength;
        private int _trieNodeIndex;          // position in the property-name trie while in a key string
        private int _enumTrieNodeIndex;      // position in the enum-value trie while in an enum value string
        private bool _inKeyString;
        private bool _inEnumString;

        public JsonSchemaTracker(CompiledJsonSchema schema)
        {
            _schema = schema ?? throw new ArgumentNullException(nameof(schema));
            _currentNodeIndex = 0;   // root
        }

        /// <summary>Whether the candidate character is allowed by the schema at the current position. Call
        /// BEFORE <see cref="JsonStateMachine.TryAdvance"/> (i.e. with the machine in its pre-advance state).</summary>
        public readonly bool IsCharAllowedBySchema(char c, in JsonStateMachine machine)
        {
            return machine.CurrentPhase switch
            {
                Phase.ExpectValue => IsValueStartAllowed(c, _currentNodeIndex),
                Phase.ExpectValueOrClose => c == ']' || IsValueStartAllowed(c, _currentNodeIndex),
                Phase.ObjectStart => IsObjectStartAllowed(c),
                Phase.ObjectExpectKey => IsObjectNextKeyAllowed(c),
                Phase.ObjectAfterPair => IsObjectCommaOrCloseAllowed(c),
                Phase.InString => IsInStringAllowed(c),
                Phase.InNumber => IsNumberCharAllowed(c, in machine),
                _ => true,   // ObjectAfterKey/ArrayAfterElement/Done/StringEscape/StringUnicode/InLiteral: parser-handled syntax
            };
        }

        /// <summary>Updates schema position from a structural event. Call AFTER the character has been
        /// accepted by the machine (i.e. with the machine in its post-advance state).</summary>
        public void OnCharAdvanced(char c, in JsonStateMachine machine)
        {
            var phase = machine.CurrentPhase;

            if (c == '{' && phase == Phase.ObjectStart) { PushObject(); return; }
            if (c == '[' && phase == Phase.ExpectValueOrClose) { PushArray(); return; }

            if (c == '"' && phase == Phase.InString)
            {
                if (machine.CurrentStringIsKey) { StartKeyString(); }
                else { StartValueString(); }
                return;
            }

            if (_inKeyString && phase == Phase.InString) { AppendKeyChar(c); return; }
            if (_inEnumString && phase == Phase.InString) { AdvanceEnumTrie(c); return; }

            if (_inKeyString && phase == Phase.ObjectAfterKey) { FinishKeyString(); return; }

            // Enum value string closed (transitioned out of the string phases) — clear the flag and fall
            // through to the value-complete restore below.
            if (_inEnumString && phase is not (Phase.InString or Phase.StringEscape or Phase.StringUnicode))
            {
                _inEnumString = false;
            }

            if (c == '}' && machine.Depth < _stackDepth) { PopContainer(); return; }
            if (c == ']' && machine.Depth < _stackDepth) { PopContainer(); return; }

            if (c == ',' && phase == Phase.ExpectValue && machine.TopIsArray) { AdvanceArrayItem(); return; }

            // A value finished inside a container: restore the current node to the parent container so the
            // following key/comma/close is checked against the container's schema (required mask, props).
            if (_stackDepth > 0 && phase is Phase.ObjectAfterPair or Phase.ObjectExpectKey or Phase.ArrayAfterElement)
            {
                _currentNodeIndex = _nodeStack[_stackDepth - 1];
            }
        }

        /// <summary>Resets to the initial (root) position.</summary>
        public void Reset()
        {
            _currentNodeIndex = 0;
            _stackDepth = 0;
            _keyLength = 0;
            _trieNodeIndex = 0;
            _enumTrieNodeIndex = 0;
            _inKeyString = false;
            _inEnumString = false;
        }

        // ── Value-start type restriction ─────────────────────────────────────
        private readonly bool IsValueStartAllowed(char c, int nodeIndex)
        {
            if (IsWhitespace(c)) { return true; }

            var types = _schema.Nodes[nodeIndex].AllowedTypes;
            return c switch
            {
                '{' => Has(types, JsonSchemaType.Object),
                '[' => Has(types, JsonSchemaType.Array),
                '"' => Has(types, JsonSchemaType.String),
                '-' => Has(types, JsonSchemaType.Number) || Has(types, JsonSchemaType.Integer),
                >= '0' and <= '9' => Has(types, JsonSchemaType.Number) || Has(types, JsonSchemaType.Integer),
                't' or 'f' => Has(types, JsonSchemaType.Boolean),
                'n' => Has(types, JsonSchemaType.Null),
                _ => false,
            };
        }

        // ── Object restrictions ──────────────────────────────────────────────
        private readonly bool IsObjectStartAllowed(char c)
        {
            if (IsWhitespace(c)) { return true; }
            ref readonly var node = ref _schema.Nodes[_currentNodeIndex];
            if (c == '}') { return node.RequiredBitmask == 0; }                       // empty only if nothing required
            if (c == '"') { return node.Properties != null || !node.AdditionalPropertiesForbidden; }
            return false;
        }

        private readonly bool IsObjectNextKeyAllowed(char c)
        {
            if (IsWhitespace(c)) { return true; }
            if (c != '"') { return false; }
            ref readonly var node = ref _schema.Nodes[_currentNodeIndex];
            if (node.AdditionalPropertiesForbidden && node.PropertyNames != null)
            {
                return RemainingProps(node) != 0;   // some declared property is still unemitted
            }
            return true;
        }

        private readonly bool IsObjectCommaOrCloseAllowed(char c)
        {
            if (IsWhitespace(c)) { return true; }
            ref readonly var node = ref _schema.Nodes[_currentNodeIndex];
            var emitted = _stackDepth > 0 ? _emittedProps[_stackDepth - 1] : 0;
            if (c == '}') { return (node.RequiredBitmask & ~emitted) == 0; }          // all required emitted
            if (c == ',')
            {
                if (node.AdditionalPropertiesForbidden && node.PropertyNames != null)
                {
                    return RemainingProps(node) != 0;
                }
                return true;
            }
            return false;
        }

        private readonly ulong RemainingProps(in JsonSchemaNode node)
        {
            var emitted = _stackDepth > 0 ? _emittedProps[_stackDepth - 1] : 0;
            var count = node.PropertyNames!.Length;
            var all = count < 64 ? (1UL << count) - 1 : ~0UL;
            return all & ~emitted;
        }

        // ── String content restrictions ──────────────────────────────────────
        private readonly bool IsInStringAllowed(char c)
        {
            if (_inKeyString)
            {
                ref readonly var node = ref _schema.Nodes[_currentNodeIndex];
                if (node.PropertyTrieIndex < 0) { return true; }   // no declared names — unconstrained key
                var trie = _schema.Tries[node.PropertyTrieIndex];
                if (c == '\\') { return true; }
                if (c == '"')
                {
                    if (trie.IsTerminal(_trieNodeIndex)) { return !IsKeyAlreadyEmitted(node, trie); }
                    return !node.AdditionalPropertiesForbidden;   // incomplete declared name only OK if extras allowed
                }
                if (!trie.TryGetChild(_trieNodeIndex, c, out var child))
                {
                    return !node.AdditionalPropertiesForbidden;   // off-trie char only OK if extra keys allowed
                }
                if (node.AdditionalPropertiesForbidden)
                {
                    // Reject a char that only leads to already-emitted properties (a duplicate-key dead-end).
                    var emitted = _stackDepth > 0 ? _emittedProps[_stackDepth - 1] : 0;
                    return (trie.ReachableMask(child) & ~emitted) != 0;
                }
                return true;
            }

            if (_inEnumString)
            {
                ref readonly var node = ref _schema.Nodes[_currentNodeIndex];
                if (node.EnumTrieIndex < 0) { return true; }
                var trie = _schema.Tries[node.EnumTrieIndex];
                if (c == '\\') { return true; }
                if (c == '"') { return trie.IsTerminal(_enumTrieNodeIndex); }
                return trie.TryGetChild(_enumTrieNodeIndex, c, out _);
            }

            return true;   // unconstrained string value
        }

        private readonly bool IsKeyAlreadyEmitted(in JsonSchemaNode node, JsonStringTrie trie)
        {
            if (!node.AdditionalPropertiesForbidden || node.PropertyNames == null) { return false; }
            var name = trie.GetCompleteName(_trieNodeIndex);
            if (name is null) { return false; }
            var bit = Array.IndexOf(node.PropertyNames, name);
            if (bit < 0 || bit >= 64 || _stackDepth == 0) { return false; }
            return (_emittedProps[_stackDepth - 1] & (1UL << bit)) != 0;
        }

        // ── Number restrictions ──────────────────────────────────────────────
        // A number's end is only known when a non-number character arrives, and the machine TERMINATES the
        // number then re-dispatches that character internally (TerminateNumberAndReprocess). So a closing
        // '}' / ']' / ',' is seen here in the InNumber phase, not in the parent container phase — we must
        // apply the parent container's close / separator rule here, or required-property gating after a
        // numeric value would be skipped. `machine` is pre-advance, so its container stack is still intact.
        private readonly bool IsNumberCharAllowed(char c, in JsonStateMachine machine)
        {
            if (c == '}')
            {
                if (_stackDepth == 0) { return false; }
                ref readonly var obj = ref _schema.Nodes[_nodeStack[_stackDepth - 1]];
                return (obj.RequiredBitmask & ~_emittedProps[_stackDepth - 1]) == 0;   // all required emitted
            }
            if (c == ']') { return true; }   // closing an array after a number (no minItems in the MVP)
            if (c == ',')
            {
                if (machine.TopIsArray || _stackDepth == 0) { return true; }
                ref readonly var obj = ref _schema.Nodes[_nodeStack[_stackDepth - 1]];
                if (obj.AdditionalPropertiesForbidden && obj.PropertyNames != null)
                {
                    return RemainingProps(obj) != 0;
                }
                return true;
            }

            // Continuation character: integer-only rejects a fraction / exponent.
            var types = _schema.Nodes[_currentNodeIndex].AllowedTypes;
            if (Has(types, JsonSchemaType.Integer) && !Has(types, JsonSchemaType.Number))
            {
                if (c is '.' or 'e' or 'E') { return false; }
            }
            return true;
        }

        // ── Structural transitions ───────────────────────────────────────────
        private void PushObject()
        {
            if (_stackDepth >= MaxDepth) { return; }
            _nodeStack[_stackDepth] = _currentNodeIndex;
            _emittedProps[_stackDepth] = 0;
            _stackDepth++;
        }

        private void PushArray()
        {
            if (_stackDepth >= MaxDepth) { return; }
            _nodeStack[_stackDepth] = _currentNodeIndex;
            _stackDepth++;
            var items = _schema.Nodes[_currentNodeIndex].ItemsNodeIndex;
            _currentNodeIndex = items >= 0 ? items : _schema.UnconstrainedNodeIndex;
        }

        private void PopContainer()
        {
            if (_stackDepth <= 0) { return; }
            _stackDepth--;
            _currentNodeIndex = _stackDepth > 0 ? _nodeStack[_stackDepth - 1] : 0;
        }

        private void AdvanceArrayItem()
        {
            if (_stackDepth <= 0) { return; }
            var items = _schema.Nodes[_nodeStack[_stackDepth - 1]].ItemsNodeIndex;
            _currentNodeIndex = items >= 0 ? items : _schema.UnconstrainedNodeIndex;
        }

        private void StartKeyString()
        {
            _inKeyString = true;
            _keyLength = 0;
            _trieNodeIndex = 0;
        }

        private void AppendKeyChar(char c)
        {
            if (_keyLength < MaxKeyLength) { _keyBuffer[_keyLength++] = c; }
            var trieIndex = _schema.Nodes[_currentNodeIndex].PropertyTrieIndex;
            if (trieIndex >= 0 && _schema.Tries[trieIndex].TryGetChild(_trieNodeIndex, c, out var child))
            {
                _trieNodeIndex = child;
            }
        }

        private void FinishKeyString()
        {
            _inKeyString = false;
            var keyName = new string(((ReadOnlySpan<char>)_keyBuffer).Slice(0, _keyLength));

            ref readonly var objectNode = ref _schema.Nodes[_currentNodeIndex];
            if (objectNode.Properties != null && objectNode.Properties.TryGetValue(keyName, out var valueNodeIndex))
            {
                if (objectNode.PropertyNames != null)
                {
                    var bit = Array.IndexOf(objectNode.PropertyNames, keyName);
                    if (bit >= 0 && bit < 64 && _stackDepth > 0) { _emittedProps[_stackDepth - 1] |= 1UL << bit; }
                }
                _currentNodeIndex = valueNodeIndex;
            }
            else
            {
                _currentNodeIndex = _schema.UnconstrainedNodeIndex;   // additional property — value unconstrained
            }
        }

        private void StartValueString()
        {
            if (_schema.Nodes[_currentNodeIndex].EnumTrieIndex >= 0)
            {
                _inEnumString = true;
                _enumTrieNodeIndex = 0;
            }
        }

        private void AdvanceEnumTrie(char c)
        {
            var trieIndex = _schema.Nodes[_currentNodeIndex].EnumTrieIndex;
            if (trieIndex >= 0 && _schema.Tries[trieIndex].TryGetChild(_enumTrieNodeIndex, c, out var child))
            {
                _enumTrieNodeIndex = child;
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static bool Has(JsonSchemaType types, JsonSchemaType flag) => (types & flag) != 0;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static bool IsWhitespace(char c) => c is ' ' or '\t' or '\n' or '\r';

        [InlineArray(MaxDepth)]
        private struct NodeIdxStack { private int _e0; }

        [InlineArray(MaxDepth)]
        private struct PropBitStack { private ulong _e0; }

        [InlineArray(MaxKeyLength)]
        private struct KeyBuffer { private char _e0; }
    }
}
