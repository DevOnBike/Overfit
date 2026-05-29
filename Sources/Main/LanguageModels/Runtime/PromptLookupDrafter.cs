// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.LanguageModels.Runtime
{
    /// <summary>
    /// Prompt-lookup drafting for speculative decoding (no draft model): predicts the next tokens by
    /// matching the most recent suffix n-gram of the generated sequence against an earlier occurrence in
    /// the same sequence and proposing the tokens that followed it last time. Cheap and allocation-free;
    /// effective on repetitive text (code, JSON / structured output, quoting the prompt) — exactly the
    /// in-process-agentic workloads Overfit targets. The proposed tokens are only a guess: they are
    /// verified in one batched forward, so accepting them never changes the (greedy) output.
    /// </summary>
    public static class PromptLookupDrafter
    {
        /// <summary>
        /// Writes up to <paramref name="draft"/>.Length candidate continuation tokens for
        /// <paramref name="history"/> and returns the count (0 if no n-gram match). Tries the longest
        /// suffix n-gram first (<paramref name="ngramMax"/> down to <paramref name="ngramMin"/>); within a
        /// length, the most recent earlier occurrence wins.
        /// </summary>
        public static int Draft(
            ReadOnlySpan<int> history,
            Span<int> draft,
            int ngramMin = 1,
            int ngramMax = 3)
        {
            if (draft.IsEmpty) { return 0; }
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(ngramMin);
            if (ngramMax < ngramMin) { ngramMax = ngramMin; }

            var len = history.Length;
            var maxN = Math.Min(ngramMax, len - 1);   // need at least one token before the suffix to match

            for (var n = maxN; n >= ngramMin; n--)
            {
                var suffixStart = len - n;
                // Most recent earlier occurrence: scan candidate starts from just-before-the-suffix down.
                for (var i = suffixStart - 1; i >= 0; i--)
                {
                    var match = true;
                    for (var j = 0; j < n; j++)
                    {
                        if (history[i + j] != history[suffixStart + j]) { match = false; break; }
                    }
                    if (!match) { continue; }

                    // Propose the tokens that followed this occurrence (bounded by what exists + draft room).
                    var followStart = i + n;
                    var available = len - followStart;
                    var count = Math.Min(draft.Length, available);
                    for (var c = 0; c < count; c++) { draft[c] = history[followStart + c]; }
                    return count;
                }
            }

            return 0;
        }
    }
}
