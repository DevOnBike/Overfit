// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels.Contracts;

namespace DevOnBike.Overfit.LanguageModels.Runtime
{
    /// <summary>
    /// Draft-model speculative drafter: a small, fast <see cref="CachedLlamaSession"/> (e.g. Qwen2.5-0.5B)
    /// proposes continuation tokens for a larger target session of the SAME tokenizer / vocabulary. Unlike
    /// prompt-lookup it PREDICTS (rather than echoes), so speculation wins on novel text too. It keeps its
    /// own KV cache in lockstep with the target's committed sequence: <see cref="Draft"/> conditions on the
    /// target's chosen token then greedily rolls out proposals; <see cref="Sync"/> rolls the draft KV back
    /// to the accepted prefix and re-feeds only the correction (never recomputing accepted drafts).
    ///
    /// Vocabulary compatibility (identical token ids) is the caller's responsibility — a mismatched draft
    /// would propose ids the target cannot verify meaningfully.
    /// </summary>
    internal sealed class DraftModelSpeculativeDrafter : ISpeculativeDrafter
    {
        private readonly CachedLlamaSession _draft;
        private SamplingOptions _greedy = SamplingOptions.Greedy;
        private int _syncedLen;       // draft KV length known to equal prompt + committed-so-far
        private int _draftBase = -1;  // draft KV length when the live Draft ran (-1 = none this step)

        /// <summary>Wraps <paramref name="draftSession"/> and primes its KV with <paramref name="prompt"/>
        /// (the same prompt token ids the target was reset with).</summary>
        public DraftModelSpeculativeDrafter(CachedLlamaSession draftSession, ReadOnlySpan<int> prompt)
        {
            _draft = draftSession ?? throw new ArgumentNullException(nameof(draftSession));
            _draft.Reset(prompt);
            _syncedLen = _draft.Position;
        }

        public int Draft(int seed, Span<int> draftOut)
        {
            // Must start exactly at the synced length — discard any leftover speculative tail.
            if (_draft.Position != _syncedLen)
            {
                _draft.RollbackTo(_syncedLen);
            }
            _draftBase = _syncedLen;

            // Condition the draft model on the target's seed token, then greedily roll out proposals.
            _draft.Feed(seed);

            var produced = 0;
            // Leave one slot of headroom so the next single forward never trips the full-cache guard.
            var cap = _draft.MaxContextLength - 1;
            for (var i = 0; i < draftOut.Length && _draft.Position < cap; i++)
            {
                draftOut[i] = _draft.GenerateNextToken(in _greedy);
                produced++;
            }
            return produced;
        }

        public void Sync(ReadOnlySpan<int> committedThisStep)
        {
            if (_draftBase < 0)
            {
                // No live draft this step (target took the single-token fallback before Draft ran):
                // append every committed token from the last synced point.
                ApplyTail(_syncedLen, committedThisStep, 0);
            }
            else
            {
                // The draft fed [seed, d0, d1, …]; committedThisStep = [seed, d0..d(a-1), correction].
                // The accepted drafts already sit in the draft KV, so keep [base, base + keep) and re-feed
                // only the trailing correction / bonus (which differs from the rejected draft). keep is
                // committed.Length-1 and never exceeds what was fed, so accepted drafts are not recomputed.
                var keep = committedThisStep.Length - 1;
                ApplyTail(_draftBase + keep, committedThisStep, keep);
            }
            _draftBase = -1;
        }

        private void ApplyTail(int rollbackTo, ReadOnlySpan<int> committed, int from)
        {
            if (_draft.Position != rollbackTo)
            {
                _draft.RollbackTo(rollbackTo);
            }
            for (var i = from; i < committed.Length; i++)
            {
                _draft.Feed(committed[i]);
            }
            _syncedLen = _draft.Position;
        }
    }
}
