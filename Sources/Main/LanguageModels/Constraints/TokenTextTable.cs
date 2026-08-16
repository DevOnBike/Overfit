// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Runtime.CompilerServices;
using DevOnBike.Overfit.LanguageModels.Contracts;

namespace DevOnBike.Overfit.LanguageModels.Constraints
{
    /// <summary>
    /// The decoded text of every token in a vocabulary, built once per tokenizer.
    ///
    /// <para><b>It was built once per constraint instance, in the constructor, by both constraints.</b>
    /// That is roughly 152 000 <see cref="ITokenizer.DecodeToString"/> calls and as many string
    /// allocations on Qwen — paid before the first token of a request, in identical code duplicated across
    /// two classes. The API shape invites a constraint per request, which is exactly the pattern that makes
    /// it hurt: every JSON-constrained call re-decodes the whole vocabulary to produce a table that cannot
    /// have changed.</para>
    ///
    /// <para><b>Keyed on the tokenizer instance, and weakly.</b> The table is a pure function of the
    /// vocabulary, so two constraints over one tokenizer must see the same strings; a static dictionary
    /// would instead keep every tokenizer a process ever loaded alive for its lifetime, which on this
    /// project's target hardware is the more expensive mistake.
    /// <see cref="ConditionalWeakTable{TKey, TValue}"/> holds no strong reference, so the table dies with
    /// the tokenizer.</para>
    ///
    /// <para>The array is handed out directly rather than copied. Callers inside this assembly only read
    /// it, and copying 152 000 references per constraint would reintroduce a smaller version of the cost
    /// being removed.</para>
    /// </summary>
    internal static class TokenTextTable
    {
        private static readonly ConditionalWeakTable<ITokenizer, string[]> Cache = new();

        /// <summary>Decoded text per token id, for every id below the tokenizer's vocabulary size.</summary>
        public static string[] For(ITokenizer tokenizer)
        {
            ArgumentNullException.ThrowIfNull(tokenizer);

            return Cache.GetValue(tokenizer, Build);
        }

        private static string[] Build(ITokenizer tokenizer)
        {
            var vocab = tokenizer.VocabularySize;
            var text = new string[vocab];
            Span<int> one = stackalloc int[1];

            for (var t = 0; t < vocab; t++)
            {
                one[0] = t;
                text[t] = tokenizer.DecodeToString(one);
            }

            return text;
        }
    }
}
