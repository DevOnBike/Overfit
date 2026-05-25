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
        private JsonStateMachine _committed;

        public JsonGrammarConstraint(ITokenizer tokenizer)
        {
            ArgumentNullException.ThrowIfNull(tokenizer);

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
            // The model's logit vector can be LONGER than the tokenizer vocabulary — GGUF models pad
            // the vocab/embedding to a round number (e.g. Qwen: 151936 logits vs 151665 real tokens).
            // Those trailing slots are padding tokens with no text and are always masked.
            if (logits.Length < _tokenText.Length)
            {
                throw new ArgumentException(
                    $"Logits length ({logits.Length}) is smaller than the tokenizer vocabulary ({_tokenText.Length}).",
                    nameof(logits));
            }

            for (var t = 0; t < logits.Length; t++)
            {
                if (t == _eosTokenId)
                {
                    // The end-of-text token is allowed only once the JSON is complete.
                    if (!_committed.IsComplete) { logits[t] = float.NegativeInfinity; }
                    continue;
                }

                // Padding slots beyond the tokenizer vocab, and special/control tokens that render
                // empty, never belong inside JSON output.
                var text = t < _tokenText.Length ? _tokenText[t] : string.Empty;
                if (text.Length == 0)
                {
                    logits[t] = float.NegativeInfinity;
                    continue;
                }

                if (!Accepts(text))
                {
                    logits[t] = float.NegativeInfinity;
                }
            }
        }

        public void Accept(int token)
        {
            if (token == _eosTokenId) { return; }
            if ((uint)token >= (uint)_tokenText.Length) { return; }

            var text = _tokenText[token];
            for (var i = 0; i < text.Length; i++)
            {
                // The token was unmasked, so every character must advance the committed machine.
                _committed.TryAdvance(text[i]);
            }
        }

        // Would feeding the whole token text keep the document well-formed (from the committed state)?
        private bool Accepts(string text)
        {
            var probe = _committed;   // value-type copy — speculative, no allocation
            for (var i = 0; i < text.Length; i++)
            {
                if (!probe.TryAdvance(text[i])) { return false; }
            }
            return true;
        }
    }
}
