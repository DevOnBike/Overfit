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
    /// KV-cache backed inference engine for GPT1Model.
    ///
    /// This class intentionally sits next to the existing SlmInferenceEngine
    /// instead of replacing it.
    ///
    /// Why:
    ///
    /// - legacy SlmInferenceEngine remains stable,
    /// - cached runtime can be benchmarked and tested side-by-side,
    /// - API users can explicitly opt into the KV-cache path,
    /// - final migration can happen after parity tests.
    ///
    /// Scope:
    ///
    /// - GPT1Model only,
    /// - batch = 1,
    /// - single-sequence autoregressive generation,
    /// - fixed context length,
    /// - no tokenizer,
    /// - no sliding-window KV eviction.
    /// </summary>
    public class CachedSlmInferenceEngine : IDisposable
    {
        private readonly GPT1Model _model;
        private bool _disposed;

        private CachedSlmInferenceEngine(GPT1Model model)
        {
            _model = model ?? throw new ArgumentNullException(nameof(model));

            VocabularySize = model.Config.VocabSize;
            MaxContextLength = model.Config.ContextLength;
            DModel = model.Config.DModel;
            LayerCount = model.Config.NLayers;
            HeadCount = model.Config.NHeads;
        }

        public int VocabularySize { get; }

        public int MaxContextLength { get; }

        public int DModel { get; }

        public int LayerCount { get; }

        public int HeadCount { get; }

        public bool HasKeyValueCache => true;

        public static CachedSlmInferenceEngine FromGpt1(GPT1Model model)
        {
            return new CachedSlmInferenceEngine(model);
        }

        public CachedSlmSession CreateSession()
        {
            ThrowIfDisposed();

            return new CachedSlmSession(_model);
        }

        public int Generate(
            ReadOnlySpan<int> promptTokens,
            Span<int> outputTokens,
            in GenerationOptions options)
        {
            ThrowIfDisposed();

            using var session = CreateSession();

            return session.Generate(
                promptTokens,
                outputTokens,
                in options);
        }

        public int GenerateGreedy(
            ReadOnlySpan<int> promptTokens,
            Span<int> outputTokens,
            int maxNewTokens)
        {
            ThrowIfDisposed();

            var options = new GenerationOptions(
                maxNewTokens,
                MaxContextLength,
                SamplingOptions.Greedy,
                stopOnEndOfTextToken: false);

            return Generate(
                promptTokens,
                outputTokens,
                in options);
        }

        public void Dispose()
        {
            _disposed = true;
        }

        private void ThrowIfDisposed()
        {
            if (_disposed)
            {
                throw new ObjectDisposedException(nameof(CachedSlmInferenceEngine));
            }
        }
    }
}
