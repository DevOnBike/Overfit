// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels.Constraints.Schema;
using DevOnBike.Overfit.LanguageModels.Contracts;

namespace DevOnBike.Overfit.LanguageModels.Constraints
{
    /// <summary>
    /// An <see cref="ITokenConstraint"/> that forces the generated text to be a JSON document conforming to a
    /// supplied <b>JSON Schema</b> — not merely well-formed (that is <see cref="JsonGrammarConstraint"/>), but
    /// schema-valid: correct value types, all <c>required</c> properties present before the object may close,
    /// only declared keys under <c>additionalProperties:false</c>, and string <c>enum</c> values. The model
    /// therefore cannot emit a structurally-valid-but-schema-wrong object (e.g. a missing required field or a
    /// number where a string is expected) — guaranteed by construction, no post-hoc validation/repair.
    ///
    /// Built on the same machinery as <see cref="JsonGrammarConstraint"/>: a value-type
    /// <see cref="JsonStateMachine"/> (well-formedness) plus a value-type <see cref="JsonSchemaTracker"/>
    /// (the schema overlay), advanced in lockstep. At each step every vocabulary token is replayed through a
    /// copy of the committed pair and masked if any character breaks either; the end-of-text token is masked
    /// until the document is complete (which, given the required-property gating, means schema-satisfied).
    ///
    /// MVP schema subset (see <see cref="JsonSchemaCompiler"/>): typed object properties, required/optional,
    /// <c>additionalProperties:false</c>, string enums, nested objects, simple arrays. A cheap first-character
    /// schema check prunes most tokens before the full per-token replay (the dotLLM <c>FirstCharBuckets</c>
    /// idea); a per-state mask cache is the documented follow-on if profiles need it.
    ///
    /// LIMITATION (shared by all token-level constrained decoders): masking operates on whole vocabulary
    /// tokens, so a model can commit a partial property name (e.g. <c>"i</c>) whose only schema-valid
    /// continuation is a character sequence its BPE has no single token for — a dead-end where every token is
    /// masked. It is most likely with multiple free-form string keys on a weak model; enum-valued and
    /// simple-keyed schemas are robust. The fix is token healing (re-tokenizing across the boundary), a
    /// documented follow-on.
    /// </summary>
    public sealed class JsonSchemaConstraint : ITokenConstraint
    {
        private readonly string[] _tokenText;
        private readonly int _eosTokenId;
        private JsonStateMachine _committed;
        private JsonSchemaTracker _tracker;

        /// <param name="tokenizer">Tokenizer whose vocabulary text is scanned to enforce the schema.</param>
        /// <param name="schemaJson">A JSON-Schema document (text) the output must conform to.</param>
        public JsonSchemaConstraint(ITokenizer tokenizer, string schemaJson)
        {
            ArgumentNullException.ThrowIfNull(tokenizer);
            ArgumentNullException.ThrowIfNull(schemaJson);

            _tracker = new JsonSchemaTracker(JsonSchemaCompiler.Compile(schemaJson));
            _eosTokenId = tokenizer.EndOfTextTokenId;

            var vocab = tokenizer.VocabularySize;
            _tokenText = new string[vocab];
            Span<int> one = stackalloc int[1];
            for (var t = 0; t < vocab; t++)
            {
                one[0] = t;
                _tokenText[t] = tokenizer.DecodeToString(one);
            }
        }

        public bool IsComplete => _committed.IsComplete;

        public void ApplyMask(Span<float> logits)
        {
            // The model's logit vector can be longer than the tokenizer vocabulary (GGUF padding) — those
            // trailing slots have no text and are always masked.
            if (logits.Length < _tokenText.Length)
            {
                throw new ArgumentException(
                    $"Logits length ({logits.Length}) is smaller than the tokenizer vocabulary ({_tokenText.Length}).",
                    nameof(logits));
            }

            var anyAllowed = false;
            for (var t = 0; t < logits.Length; t++)
            {
                if (t == _eosTokenId) { continue; }   // handled after the loop (needs anyAllowed)

                var text = t < _tokenText.Length ? _tokenText[t] : string.Empty;
                if (text.Length == 0)
                {
                    logits[t] = float.NegativeInfinity;
                    continue;
                }

                // Cheap prune: reject on the first character against the committed state (no copy) before
                // the full replay — kills most of the vocabulary at each position (wrong type / wrong char).
                if (!_tracker.IsCharAllowedBySchema(text[0], in _committed) || !Accepts(text))
                {
                    logits[t] = float.NegativeInfinity;
                }
                else
                {
                    anyAllowed = true;
                }
            }

            // End-of-text is allowed once the document is complete, OR as a graceful escape from a BPE
            // dead-end (no other token is schema-valid) — so generation terminates with the valid prefix
            // instead of degenerating. (The real fix for the dead-end itself is token healing.)
            if (_eosTokenId >= 0 && _eosTokenId < logits.Length && !_committed.IsComplete && anyAllowed)
            {
                logits[_eosTokenId] = float.NegativeInfinity;
            }
        }

        public void Accept(int token)
        {
            if (token == _eosTokenId) { return; }
            if ((uint)token >= (uint)_tokenText.Length) { return; }

            var text = _tokenText[token];
            for (var i = 0; i < text.Length; i++)
            {
                // The token was unmasked, so every character advances both committed machines.
                _committed.TryAdvance(text[i]);
                _tracker.OnCharAdvanced(text[i], in _committed);
            }
        }

        // Would feeding the whole token text keep the document schema-valid (from the committed state)?
        private bool Accepts(string text)
        {
            var machine = _committed;   // value-type copies — speculative, no allocation
            var tracker = _tracker;
            for (var i = 0; i < text.Length; i++)
            {
                var c = text[i];
                if (!tracker.IsCharAllowedBySchema(c, in machine)) { return false; }
                if (!machine.TryAdvance(c)) { return false; }
                tracker.OnCharAdvanced(c, in machine);
            }
            return true;
        }
    }
}
