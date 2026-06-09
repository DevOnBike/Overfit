// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Audio.Tts
{
    /// <summary>
    /// Splits normalized text into sentences so long input can be synthesized chunk-by-chunk (one model run per
    /// sentence, audio concatenated) instead of one long generation that drifts or overruns the context. Intended
    /// to run <i>after</i> <see cref="TtsTextNormalizer"/>, which has already removed decimal points and
    /// abbreviation dots — so the only remaining <c>. ! ?</c> are real sentence ends. Model-free.
    /// </summary>
    public static class SentenceSplitter
    {
        /// <summary>Returns the trimmed, non-empty sentences of <paramref name="text"/> (terminal punctuation kept).</summary>
        public static IReadOnlyList<string> Split(string text)
        {
            var result = new List<string>();
            if (string.IsNullOrWhiteSpace(text))
            {
                return result;
            }

            var start = 0;
            var i = 0;
            while (i < text.Length)
            {
                var c = text[i];
                if (c is '.' or '!' or '?')
                {
                    var j = i + 1;
                    while (j < text.Length && text[j] is '.' or '!' or '?')
                    {
                        j++;
                    }
                    if (j >= text.Length || char.IsWhiteSpace(text[j]))
                    {
                        var sentence = text[start..j].Trim();
                        if (sentence.Length > 0)
                        {
                            result.Add(sentence);
                        }
                        start = j;
                        i = j;
                        continue;
                    }
                    i = j;
                    continue;
                }
                i++;
            }

            var tail = text[start..].Trim();
            if (tail.Length > 0)
            {
                result.Add(tail);
            }
            return result;
        }
    }
}
