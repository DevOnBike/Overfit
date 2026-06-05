// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
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
    public sealed class CachedSlmSession : ISlmSession
    {
        private readonly CachedGpt1ModelAdapter _adapter;
        private readonly float[] _lastLogits;
        private readonly int[] _sampleIndexScratch;
        private readonly float[] _sampleScoreScratch;

        private Random _random = new(1);
        private int _currentSeed;
        private bool _hasLogits;
        private bool _disposed;

        /// <summary>
        /// Prompt length at/above which <see cref="Prefill"/> uses the batched
        /// (head-parallel) stack pass instead of the single-token loop. Below this
        /// the loop's lower per-call overhead wins. Conservative default; tunable.
        /// </summary>
        private const int BatchedPrefillThreshold = 16;

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

        /// <summary>
        /// Feeds prompt tokens into the KV cache one at a time. Each token goes
        /// through embedding → transformer stack → cache write, leaving the
        /// session ready for <see cref="GenerateNextToken"/>.
        ///
        /// Can be called multiple times after a single <see cref="Reset()"/> to
        /// append context incrementally (chat history) provided the cumulative
        /// token count stays within <see cref="MaxContextLength"/>.
        ///
        /// **Performance note:** prompts at/above <see cref="BatchedPrefillThreshold"/>
        /// tokens take the batched (head-parallel) stack pass — one weight read per
        /// layer over the whole prompt instead of per token — measured ~3.5× faster
        /// TTFT on GPT-2-Small dims. Shorter prompts use the single-token loop.
        /// </summary>
        public void Prefill(ReadOnlySpan<int> promptTokens)
        {
            ThrowIfDisposed();

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

            // Batched prefill: for prompts at/above the threshold, run the whole
            // prompt through one batched (head-parallel) pass per layer instead of
            // the single-token loop. Bit-identical to the loop (CachedGptStackTests /
            // CachedSlmPrefillBatchedTests), and measured ~3.5× faster TTFT on
            // GPT-2-Small dims at 64 tokens. Below the threshold the loop's lower
            // per-call overhead wins; non-GPT-2 stacks (RoPE/GQA/SwiGLU) fall back.
            if (promptTokens.Length >= BatchedPrefillThreshold && _adapter.SupportsBatchedPrefill)
            {
                _adapter.PrefillBatched(promptTokens, _lastLogits);
                _hasLogits = true;
                return;
            }

            // Skip the LM-head projection for every prompt token except the last:
            // their logits would be immediately overwritten by the next decode,
            // so computing them is wasted work (~27 % of per-token cost on GPT-2
            // Small). The final token uses the full path so _lastLogits reflects
            // the prediction at the end of the prompt.
            var lastIndex = promptTokens.Length - 1;

            for (var i = 0; i < lastIndex; i++)
            {
                _adapter.PrefillToken(promptTokens[i]);
            }

            _adapter.DecodeNextToken(promptTokens[lastIndex], _lastLogits);

            _hasLogits = true;
        }

        /// <summary>
        /// Convenience overload: clears the cache and prefills it with the
        /// supplied prompt. Equivalent to <see cref="Reset()"/> followed by
        /// <see cref="Prefill"/>.
        /// </summary>
        public void Reset(ReadOnlySpan<int> promptTokens)
        {
            Reset();
            Prefill(promptTokens);
        }

        public int GenerateNextToken(in SamplingOptions sampling)
        {
            ThrowIfDisposed();

            if (!_hasLogits)
            {
                throw new OverfitRuntimeException("Cannot generate from an empty cached session. Call Reset(promptTokens) first.");
            }

            if (_adapter.IsFull)
            {
                throw new OverfitRuntimeException(
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
