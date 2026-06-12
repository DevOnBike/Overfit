// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.LanguageModels.Contracts;

namespace DevOnBike.Overfit.LanguageModels.Runtime
{
    /// <summary>
    /// Stateful single-sequence SLM session.
    ///
    /// This is the first runtime shell before KV cache.
    ///
    /// Current implementation:
    /// - stores the rolling context in a preallocated token buffer,
    /// - exposes GenerateNextToken(...),
    /// - samples from logits using TokenSampler,
    /// - still calls GPT1Model.GenerateLogits(...) for every step.
    ///
    /// Therefore this class stabilizes the public API but does not yet solve the
    /// current performance problem. The next PR should replace the inside of
    /// GenerateNextToken with a KV-cache decode path.
    /// </summary>
    public sealed class SlmSession : ISlmSession
    {
        private readonly GPT1Model _model;
        private readonly int[] _contextTokens;
        private readonly float[] _lastLogits;
        private readonly int[] _sampleIndexScratch;
        private readonly float[] _sampleScoreScratch;

        private Random _random = new(1);
        private int _currentSeed;
        private int _contextLength;
        private int _totalPosition;
        private bool _disposed;

        public SlmSession(GPT1Model model, int maxContextLength)
        {
            _model = model ?? throw new ArgumentNullException(nameof(model));

            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(maxContextLength);

            if (maxContextLength > model.Config.ContextLength)
            {
                throw new ArgumentException(
                    $"maxContextLength={maxContextLength} exceeds model context length {model.Config.ContextLength}.",
                    nameof(maxContextLength));
            }

            MaxContextLength = maxContextLength;
            VocabularySize = model.Config.VocabSize;

            _contextTokens = new int[maxContextLength];
            _lastLogits = new float[VocabularySize];
            _sampleIndexScratch = new int[VocabularySize];
            _sampleScoreScratch = new float[VocabularySize];
        }

        public int CurrentPosition => _totalPosition;

        public int MaxContextLength { get; }

        public int VocabularySize { get; }

        public bool HasKeyValueCache => false;

        public void Reset()
        {
            ThrowIfDisposed();

            Array.Clear(_contextTokens);
            Array.Clear(_lastLogits);

            _contextLength = 0;
            _totalPosition = 0;
        }

        public void Reset(ReadOnlySpan<int> promptTokens)
        {
            ThrowIfDisposed();

            Reset();

            if (promptTokens.IsEmpty)
            {
                return;
            }

            var tokensToCopy = Math.Min(promptTokens.Length, MaxContextLength);
            var sourceStart = promptTokens.Length - tokensToCopy;

            promptTokens
                .Slice(sourceStart, tokensToCopy)
                .CopyTo(_contextTokens);

            _contextLength = tokensToCopy;
            _totalPosition = promptTokens.Length;
        }

        public int GenerateNextToken(in SamplingOptions sampling)
        {
            ThrowIfDisposed();

            if (_contextLength == 0)
            {
                throw new OverfitRuntimeException("Cannot generate a token from an empty session. Call Reset(promptTokens) first.");
            }

            EnsureRandom(in sampling);

            // GPT1Model.GenerateLogits currently requires an int[] with exact
            // sequence length. This allocation is intentional in the skeleton and
            // should disappear when the KV-cache decode path lands.
#pragma warning disable OVERFIT001 // documented skeleton debt: GenerateLogits requires an exact-length int[]; goes away with the KV-cache decode path (see comment above)
            var context = new int[_contextLength];
#pragma warning restore OVERFIT001
            _contextTokens.AsSpan(0, _contextLength).CopyTo(context);

            var logits = _model.GenerateLogits(context);

            if (logits.Length != VocabularySize)
            {
                throw new OverfitRuntimeException(
                    $"Model returned {logits.Length} logits, expected {VocabularySize}.");
            }

            logits.CopyTo(_lastLogits);

            var nextToken = TokenSampler.Sample(
                _lastLogits,
                in sampling,
                _random,
                _sampleIndexScratch,
                _sampleScoreScratch);

            AppendToken(nextToken);

            return nextToken;
        }

        public int Generate(
            ReadOnlySpan<int> promptTokens,
            Span<int> outputTokens,
            in GenerationOptions options)
        {
            ThrowIfDisposed();

            Reset(promptTokens);

            var sampling = options.Sampling;
            var generated = 0;

            for (; generated < outputTokens.Length && generated < options.MaxNewTokens; generated++)
            {
                var token = GenerateNextToken(in sampling);

                outputTokens[generated] = token;

                if (options.StopOnEndOfTextToken &&
                    options.EndOfTextTokenId >= 0 &&
                    token == options.EndOfTextTokenId)
                {
                    generated++;
                    break;
                }
            }

            return generated;
        }

        public void GetLastLogits(Span<float> destination)
        {
            ThrowIfDisposed();

            if (destination.Length < _lastLogits.Length)
            {
                throw new ArgumentException("Destination span is too small for last logits.", nameof(destination));
            }

            _lastLogits.CopyTo(destination);
        }

        public void Dispose()
        {
            _disposed = true;
        }

        private void AppendToken(int token)
        {
            if (_contextLength < MaxContextLength)
            {
                _contextTokens[_contextLength] = token;
                _contextLength++;
            }
            else
            {
                // Overlapping in-place left-shift — Span.CopyTo has memmove semantics.
                _contextTokens.AsSpan(1, MaxContextLength - 1)
                    .CopyTo(_contextTokens.AsSpan(0, MaxContextLength - 1));

                _contextTokens[MaxContextLength - 1] = token;
            }

            _totalPosition++;
        }

        private void EnsureRandom(in SamplingOptions sampling)
        {
            if (sampling.Seed == 0 || sampling.Seed == _currentSeed)
            {
                return;
            }

            _currentSeed = sampling.Seed;
            _random = new Random(sampling.Seed);
        }

        private void ThrowIfDisposed()
        {
            if (_disposed)
            {
                throw new ObjectDisposedException(nameof(SlmSession));
            }
        }
    }
}
