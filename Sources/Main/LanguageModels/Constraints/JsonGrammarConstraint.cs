// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels.Contracts;

namespace DevOnBike.Overfit.LanguageModels.Constraints
{
    /// <summary>
    /// An <see cref="ITokenConstraint"/> that forces the generated text to be a single well-formed
    /// JSON document (JSON-mode). At each step it walks every vocabulary token through a copy of the
    /// committed <see cref="JsonStateMachine"/> and masks out (sets to
    /// <see cref="float.NegativeInfinity"/>) any token whose characters would break well-formedness;
    /// the end-of-text token is masked until the JSON is complete. The model therefore <b>cannot</b>
    /// emit invalid JSON — no prompt-engineering, no post-hoc repair.
    ///
    /// The per-token text table is built once from the tokenizer (<see cref="ITokenizer.DecodeToString"/>);
    /// tokens that decode to nothing (most special/control tokens) are disallowed, so only the
    /// end-of-text token can terminate generation.
    ///
    /// Cost: O(vocab × token length) per generated token. Fine for short structured outputs; a
    /// per-state cache / token prefix-trie is the documented follow-on if it shows up in profiles.
    /// </summary>
    public sealed class JsonGrammarConstraint : ITokenConstraint
    {
        private readonly string[] _tokenText;
        private readonly int _eosTokenId;
        private readonly bool _requireObject;
        private bool _rootStarted;
        private JsonStateMachine _committed;

        /// <param name="tokenizer">Tokenizer whose vocabulary token text is scanned to enforce the JSON grammar.</param>
        /// <param name="requireObject">
        /// When true, the root value must be a JSON object: until the opening <c>{</c> is emitted, only
        /// whitespace or a token whose first non-whitespace character is <c>{</c> is allowed. Stops a
        /// model from satisfying "valid JSON" with a bare string/number/array (e.g. a quoted string that
        /// merely contains JSON). Defaults to false (any well-formed JSON value).
        /// </param>
        public JsonGrammarConstraint(ITokenizer tokenizer, bool requireObject = false)
        {
            ArgumentNullException.ThrowIfNull(tokenizer);

            _requireObject = requireObject;
            _eosTokenId = tokenizer.EndOfTextTokenId;
            _tokenText = TokenTextTable.For(tokenizer);
        }

        public bool IsComplete => _committed.IsComplete;

        public void ApplyMask(Span<float> logits)
        {
            // The model's logit vector can be LONGER than the tokenizer vocabulary — GGUF models pad
            // the vocab/embedding to a round number (e.g. Qwen: 151936 logits vs 151665 real tokens).
            // Those trailing slots are padding tokens with no text and are always masked.
            if (logits.Length < _tokenText.Length)
            {
                throw new ArgumentException(
                    $"Logits length ({logits.Length}) is smaller than the tokenizer vocabulary ({_tokenText.Length}).",
                    nameof(logits));
            }

            var anyAllowed = false;

            for (var t = 0; t < logits.Length; t++)
            {
                if (t == _eosTokenId)
                {
                    // Decided after the loop: whether it may terminate depends on whether anything else
                    // survived, which is not known yet.
                    continue;
                }

                // Padding slots beyond the tokenizer vocab, and special/control tokens that render
                // empty, never belong inside JSON output.
                var text = t < _tokenText.Length ? _tokenText[t] : string.Empty;

                // One evaluation, one branch. Whether the token is allowed and whether anything is allowed
                // are the same question asked twice, and asking it twice is how the sibling constraint came
                // to replay every token through its state machine two times per decode step.
                var allowed = text.Length > 0
                              && (!_requireObject || _rootStarted || OpensObject(text))
                              && Accepts(text);

                if (!allowed)
                {
                    logits[t] = float.NegativeInfinity;

                    continue;
                }

                anyAllowed = true;
            }

            // End-of-text is allowed once the document is complete, OR as an escape from a BPE dead-end
            // where nothing else is grammar-valid.
            //
            // WITHOUT the second condition every logit went to negative infinity, which is not a refusal:
            // it is a degenerate distribution handed to the sampler, and softmax over all -inf is NaN. The
            // sibling JsonSchemaConstraint already had this escape and this one did not — the same
            // one-of-a-pair-carries-the-guard shape as four other findings the same week.
            //
            // The dead-end itself is a tokenizer problem: a BPE vocabulary need not contain any token that
            // continues a valid prefix. Token healing is the real repair; terminating on the valid prefix
            // is the honest interim answer.
            if (_eosTokenId >= 0 && _eosTokenId < logits.Length && !_committed.IsComplete && anyAllowed)
            {
                logits[_eosTokenId] = float.NegativeInfinity;
            }
        }

        public void Accept(int token)
        {
            if (token == _eosTokenId)
            {
                return;
            }
            if ((uint)token >= (uint)_tokenText.Length)
            {
                return;
            }

            var text = _tokenText[token];

            for (var i = 0; i < text.Length; i++)
            {
                // The invariant is that this token was unmasked, so every character advances. It was
                // asserted in a comment and the result discarded — NASA rule 7 exactly. If it ever breaks
                // (a caller accepting a token it did not mask, or the mask and this loop drifting apart in
                // a later edit) the machine desynchronises from the text and every subsequent mask is
                // computed from a state that does not describe the document. That is silent and permanent;
                // this turns it into one exception at the moment it happens.
                if (!_committed.TryAdvance(text[i]))
                {
                    throw new OverfitRuntimeException(
                        $"Token {token} ('{text}') was accepted but character '{text[i]}' does not advance "
                        + "the JSON state machine. The constraint's mask and its committed state have "
                        + "diverged; continuing would compute every later mask from a wrong state.");
                }

                if (!IsJsonWhitespace(text[i]))
                {
                    _rootStarted = true;
                }
            }
        }

        // True if the token is all whitespace (still at the document start) or its first non-whitespace
        // character is '{'. Used only while the root object has not yet opened.
        private static bool OpensObject(string text)
        {
            foreach (var c in text)
            {
                if (IsJsonWhitespace(c))
                {
                    continue;
                }
                return c == '{';
            }
            return true;
        }

        private static bool IsJsonWhitespace(char c) => c is ' ' or '\t' or '\n' or '\r';

        // Would feeding the whole token text keep the document well-formed (from the committed state)?
        private bool Accepts(string text)
        {
            var probe = _committed;   // value-type copy — speculative, no allocation
            for (var i = 0; i < text.Length; i++)
            {
                if (!probe.TryAdvance(text[i]))
                {
                    return false;
                }
            }
            return true;
        }
    }
}
