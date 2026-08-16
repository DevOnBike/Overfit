// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels.Constraints.Regex;
using DevOnBike.Overfit.LanguageModels.Contracts;

namespace DevOnBike.Overfit.LanguageModels.Constraints
{
    /// <summary>
    /// An <see cref="ITokenConstraint"/> that forces the entire generated text to match a regular expression
    /// — phone numbers, dates, IDs, currency, custom codes — by construction. At each step every vocabulary
    /// token is replayed through a copy of the committed <see cref="RegexDfa"/> state and masked if any of its
    /// characters would leave the automaton (a dead end); the end-of-text token is masked until the automaton
    /// is in an accepting (full-match) state. The model therefore cannot emit a string the pattern would
    /// reject. Pair with a prompt that asks for ONLY the value (the whole output is matched).
    ///
    /// Same machinery as <see cref="JsonGrammarConstraint"/> — a tiny value-type state (here a single DFA
    /// state index) advanced in lockstep, with a cheap first-character prune before the full per-token replay.
    /// Supported regex syntax is the MVP subset of <see cref="RegexDfa"/> (ASCII; literals, classes,
    /// <c>\d \w \s</c>, <c>* + ? {n} {n,m}</c>, alternation, groups).
    /// </summary>
    public sealed class RegexConstraint : ITokenConstraint
    {
        private readonly string[] _tokenText;
        private readonly int _eosTokenId;
        private readonly RegexDfa _dfa;
        private int _state;

        /// <param name="tokenizer">Tokenizer whose vocabulary text is scanned to enforce the pattern.</param>
        /// <param name="pattern">A regular expression the whole output must match.</param>
        public RegexConstraint(ITokenizer tokenizer, string pattern)
        {
            ArgumentNullException.ThrowIfNull(tokenizer);
            ArgumentNullException.ThrowIfNull(pattern);

            _dfa = RegexDfa.Compile(pattern);
            _state = _dfa.Start;
            _eosTokenId = tokenizer.EndOfTextTokenId;
            _tokenText = TokenTextTable.For(tokenizer);
        }

        public bool IsComplete => _dfa.IsAccepting(_state);

        public void ApplyMask(Span<float> logits)
        {
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
                    continue;
                }   // handled after the loop (needs anyAllowed)

                var text = t < _tokenText.Length ? _tokenText[t] : string.Empty;
                if (text.Length == 0)
                {
                    logits[t] = float.NegativeInfinity;
                    continue;
                }

                // Cheap prune: reject on the first character (no replay) before the full walk.
                //
                // Evaluated ONCE. The second `if` was the literal negation of the first, recomputed rather
                // than remembered, so `Accepts` — which walks the whole token text through the DFA — ran
                // twice for every surviving token at every decode step, over the entire vocabulary. The
                // same defect was fixed in JsonSchemaConstraint on 2026-08-03 and left here because the
                // review that found it named only the two JSON files.
                var allowed = _dfa.Next(_state, text[0]) >= 0 && Accepts(text);

                if (!allowed)
                {
                    logits[t] = float.NegativeInfinity;

                    continue;
                }

                anyAllowed = true;
            }

            // End-of-text is allowed once the pattern fully matches, OR as a graceful escape from a BPE
            // dead-end (no other token keeps the automaton alive) — so generation terminates instead of
            // degenerating into a masked-token repeat. (The real fix for the dead-end itself is token healing.)
            if (_eosTokenId >= 0 && _eosTokenId < logits.Length && !_dfa.IsAccepting(_state) && anyAllowed)
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
                // Asserted rather than assumed, and here the assumption was more dangerous than in the two
                // JSON constraints. `Next` returns -1 for a dead transition, and that -1 was stored
                // straight into the state field: `IsAccepting(-1)` indexes an array at -1, and
                // `Next(-1, c)` computes `-1 * Alphabet + c`, also negative. So a broken invariant did not
                // desynchronise quietly — it produced an IndexOutOfRangeException from inside the
                // constraint on some later call, with nothing pointing at the token that caused it.
                var next = _dfa.Next(_state, text[i]);

                if (next < 0)
                {
                    throw new OverfitRuntimeException(
                        $"Token {token} ('{text}') was accepted but character '{text[i]}' has no transition "
                        + "from the current state. The constraint's mask and its committed state have "
                        + "diverged; continuing would compute every later mask from a wrong state.");
                }

                _state = next;
            }
        }

        // Would feeding the whole token text keep the automaton alive (from the committed state)?
        private bool Accepts(string text)
        {
            var state = _state;
            for (var i = 0; i < text.Length; i++)
            {
                state = _dfa.Next(state, text[i]);
                if (state < 0)
                {
                    return false;
                }
            }
            return true;
        }
    }
}
