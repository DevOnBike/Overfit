// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Ops
{
    /// <summary>
    /// A ready-made label-level <b>n-gram</b> <see cref="ICtcLanguageModel"/> for CTC beam-search
    /// rescoring — train it on a corpus of label sequences and hand it to
    /// <see cref="CtcDecoder.BeamSearchDecode(ReadOnlySpan{float}, int, int, int, int, ICtcLanguageModel, double, double)"/>.
    /// <c>P(next | context)</c> uses the last <c>order−1</c> labels with add-k (Laplace) smoothing and
    /// <b>back-off to the longest seen context</b> (shorten the context until one has been observed, else
    /// fall back to the unigram). Smoothing keeps every probability positive, so unseen continuations get
    /// a finite (low) score rather than <c>−∞</c>.
    ///
    /// <para>Minimal use: <c>var lm = new NGramCtcLanguageModel(classCount); lm.Train(corpus);</c> then
    /// <c>CtcDecoder.BeamSearchDecode(logits, T, C, blank, beamWidth, lm, weight)</c>. Counts are dense
    /// per context (<c>classCount</c> ints), so this suits modest class counts (OCR / phoneme sets), not
    /// huge vocabularies.</para>
    /// </summary>
    public sealed class NGramCtcLanguageModel : ICtcLanguageModel
    {
        private sealed class Context
        {
            public int[] Counts = [];
            public int Total;
        }

        private readonly Dictionary<string, Context> _contexts = new();
        private readonly int[] _unigram;
        private long _unigramTotal;
        private readonly int _classCount;
        private readonly int _contextLength;   // order − 1
        private readonly double _smoothing;

        /// <param name="classCount">Number of distinct labels (the CTC blank is never part of a labeling).</param>
        /// <param name="order">n-gram order: 1 = unigram, 2 = bigram, 3 = trigram (default).</param>
        /// <param name="smoothing">Add-k Laplace constant (&gt; 0). Larger = closer to uniform.</param>
        public NGramCtcLanguageModel(int classCount, int order = 3, double smoothing = 0.1)
        {
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(classCount);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(order);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(smoothing);

            _classCount = classCount;
            _contextLength = order - 1;
            _smoothing = smoothing;
            _unigram = new int[classCount];
        }

        /// <summary>n-gram order (= context length + 1).</summary>
        public int Order => _contextLength + 1;

        /// <summary>Number of labels the model is defined over.</summary>
        public int ClassCount => _classCount;

        /// <summary>Accumulates n-gram counts from one label sequence.</summary>
        public void Train(ReadOnlySpan<int> sequence)
        {
            for (var i = 0; i < sequence.Length; i++)
            {
                var label = sequence[i];
                ValidateLabel(label);

                var start = Math.Max(0, i - _contextLength);
                if (i > start)
                {
                    AddContextCount(sequence.Slice(start, i - start), label);
                }

                _unigram[label]++;
                _unigramTotal++;
            }
        }

        /// <summary>Accumulates n-gram counts from a corpus of label sequences.</summary>
        public void Train(IEnumerable<int[]> sequences)
        {
            ArgumentNullException.ThrowIfNull(sequences);
            foreach (var sequence in sequences)
            {
                if (sequence is not null)
                {
                    Train(sequence);
                }
            }
        }

        /// <inheritdoc/>
        public double LogProbability(ReadOnlySpan<int> prefix, int nextLabel)
        {
            ValidateLabel(nextLabel);

            var denom = _smoothing * _classCount;
            var maxLen = Math.Min(prefix.Length, _contextLength);
            for (var length = maxLen; length >= 1; length--)
            {
                var context = prefix.Slice(prefix.Length - length, length);
                if (_contexts.TryGetValue(Encode(context), out var counts) && counts.Total > 0)
                {
                    return Math.Log((counts.Counts[nextLabel] + _smoothing) / (counts.Total + denom));
                }
            }

            // Back-off to the unigram (also the untrained case ⇒ uniform).
            return Math.Log((_unigram[nextLabel] + _smoothing) / (_unigramTotal + denom));
        }

        private void AddContextCount(ReadOnlySpan<int> context, int label)
        {
            var key = Encode(context);
            if (!_contexts.TryGetValue(key, out var counts))
            {
                counts = new Context { Counts = new int[_classCount] };
                _contexts[key] = counts;
            }
            counts.Counts[label]++;
            counts.Total++;
        }

        private void ValidateLabel(int label)
        {
            if ((uint)label >= (uint)_classCount)
            {
                throw new ArgumentOutOfRangeException(nameof(label), $"label {label} out of range [0,{_classCount}).");
            }
        }

        private static string Encode(ReadOnlySpan<int> context)
        {
            if (context.IsEmpty)
            {
                return string.Empty;
            }

            Span<char> chars = context.Length <= 64 ? stackalloc char[context.Length] : new char[context.Length];
            for (var i = 0; i < context.Length; i++)
            {
                chars[i] = (char)context[i];
            }
            return new string(chars);
        }
    }
}
