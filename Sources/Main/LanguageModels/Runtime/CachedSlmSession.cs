// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
//
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.LanguageModels.Contracts;

namespace DevOnBike.Overfit.LanguageModels.Runtime
{
    /// <summary>
    /// KV-cache backed SLM session for GPT1Model.
    ///
    /// This is the first runtime session that uses the cached decode path:
    ///
    /// prompt token(s)
    /// -> CachedGpt1ModelAdapter.DecodeNextToken(...)
    /// -> cached logits
    /// -> TokenSampler.Sample(...)
    /// -> decode sampled token to advance cache
    ///
    /// Scope:
    /// - GPT1Model only,
    /// - batch = 1,
    /// - fixed context length,
    /// - no sliding-window cache eviction yet,
    /// - no tokenizer.
    ///
    /// Current behavior:
    /// - Reset(promptTokens) pre-fills the KV cache by decoding prompt tokens.
    /// - GenerateNextToken(...) samples from the last available logits.
    /// - After sampling, the sampled token is immediately decoded so logits are
    ///   ready for the next generation step.
    ///
    /// This class intentionally does not replace the old SlmSession yet. It lets
    /// tests and benchmarks compare cached vs legacy paths side-by-side.
    /// </summary>
    public class CachedSlmSession : ISlmSession
    {
        private readonly CachedGpt1ModelAdapter _adapter;
        private readonly float[] _lastLogits;
        private readonly int[] _sampleIndexScratch;
        private readonly float[] _sampleScoreScratch;

        private Random _random = new(1);
        private int _currentSeed;
        private bool _hasLogits;
        private bool _disposed;

        public CachedSlmSession(GPT1Model model)
            : this(new CachedGpt1ModelAdapter(model))
        {
        }

        public CachedSlmSession(CachedGpt1ModelAdapter adapter)
        {
            _adapter = adapter ?? throw new ArgumentNullException(nameof(adapter));

            MaxContextLength = adapter.MaxContextLength;
            VocabularySize = adapter.VocabSize;

            _lastLogits = new float[VocabularySize];
            _sampleIndexScratch = new int[VocabularySize];
            _sampleScoreScratch = new float[VocabularySize];
        }

        public int CurrentPosition => _adapter.CurrentPosition;

        public int MaxContextLength { get; }

        public int VocabularySize { get; }

        public bool HasKeyValueCache => true;

        public void Reset()
        {
            ThrowIfDisposed();

            _adapter.Reset();
            Array.Clear(_lastLogits);
            Array.Clear(_sampleIndexScratch);
            Array.Clear(_sampleScoreScratch);

            _hasLogits = false;
        }

        public void Reset(ReadOnlySpan<int> promptTokens)
        {
            ThrowIfDisposed();

            Reset();

            if (promptTokens.IsEmpty)
            {
                return;
            }

            if (promptTokens.Length > MaxContextLength)
            {
                throw new ArgumentException(
                    $"Prompt length {promptTokens.Length} exceeds cached session max context length {MaxContextLength}. Sliding-window cache eviction is not implemented yet.",
                    nameof(promptTokens));
            }

            for (var i = 0; i < promptTokens.Length; i++)
            {
                _adapter.DecodeNextToken(
                    promptTokens[i],
                    _lastLogits);
            }

            _hasLogits = true;
        }

        public int GenerateNextToken(in SamplingOptions sampling)
        {
            ThrowIfDisposed();

            if (!_hasLogits)
            {
                throw new InvalidOperationException("Cannot generate from an empty cached session. Call Reset(promptTokens) first.");
            }

            if (_adapter.IsFull)
            {
                throw new InvalidOperationException(
                    $"Cannot generate because KV cache is full. MaxContextLength={MaxContextLength}.");
            }

            EnsureRandom(in sampling);

            var nextToken = TokenSampler.Sample(
                _lastLogits,
                in sampling,
                _random,
                _sampleIndexScratch,
                _sampleScoreScratch);

            _adapter.DecodeNextToken(
                nextToken,
                _lastLogits);

            _hasLogits = true;

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

            while (generated < outputTokens.Length &&
                   generated < options.MaxNewTokens)
            {
                var token = GenerateNextToken(in sampling);

                outputTokens[generated] = token;
                generated++;

                if (options.StopOnEndOfTextToken &&
                    options.EndOfTextTokenId >= 0 &&
                    token == options.EndOfTextTokenId)
                {
                    break;
                }
            }

            return generated;
        }

        public void GetLastLogits(Span<float> destination)
        {
            ThrowIfDisposed();

            if (destination.Length < VocabularySize)
            {
                throw new ArgumentException("Destination span is too small for logits.", nameof(destination));
            }

            _lastLogits.AsSpan().CopyTo(destination);
        }

        public void RefreshWeightsFromModel()
        {
            ThrowIfDisposed();

            _adapter.RefreshWeightsFromModel();
        }

        public void Dispose()
        {
            if (_disposed)
            {
                return;
            }

            _disposed = true;
            _adapter.Dispose();
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
                throw new ObjectDisposedException(nameof(CachedSlmSession));
            }
        }
    }
}
