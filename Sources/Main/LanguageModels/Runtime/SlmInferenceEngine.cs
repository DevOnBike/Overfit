// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
//
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Diagnostics;
using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.LanguageModels.Contracts;

namespace DevOnBike.Overfit.LanguageModels.Runtime
{
    /// <summary>
    /// First SLM inference-engine shell for GPT-style models.
    ///
    /// This class deliberately does not implement KV cache yet. It gives the
    /// roadmap a stable public surface:
    ///
    /// - CreateSession()
    /// - Generate(...)
    /// - GenerateStreaming(...)
    /// - GenerationStats
    ///
    /// The implementation still delegates per token to GPT1Model.GenerateLogits(...).
    /// That means it inherits the current allocation-heavy autoregressive path.
    /// The next performance PR should keep this public API and replace the inside
    /// of SlmSession.GenerateNextToken with cached decode.
    /// </summary>
    public sealed class SlmInferenceEngine : ISlmInferenceEngine
    {
        private readonly GPT1Model _model;
        private readonly Gpt1SlmModelAdapter _modelAdapter;
        private readonly bool _disposeModelWithEngine;

        private GenerationStats _lastGenerationStats;
        private bool _disposed;

        public SlmInferenceEngine(
            GPT1Model model,
            bool disposeModelWithEngine = false)
        {
            _model = model ?? throw new ArgumentNullException(nameof(model));
            _modelAdapter = new Gpt1SlmModelAdapter(model);
            _disposeModelWithEngine = disposeModelWithEngine;

            _model.Eval();
        }

        public ISlmModel Model => _modelAdapter;

        public int VocabularySize => _model.Config.VocabSize;

        public int MaxContextLength => _model.Config.ContextLength;

        public bool SupportsKeyValueCache => false;

        public bool SupportsStreaming => true;

        public static SlmInferenceEngine FromGpt1(
            GPT1Model model,
            bool disposeModelWithEngine = false)
        {
            return new SlmInferenceEngine(model, disposeModelWithEngine);
        }

        public ISlmSession CreateSession()
        {
            ThrowIfDisposed();

            return new SlmSession(_model, _model.Config.ContextLength);
        }

        public ISlmSession CreateSession(int maxContextLength)
        {
            ThrowIfDisposed();

            return new SlmSession(_model, maxContextLength);
        }

        public int Generate(
            ReadOnlySpan<int> promptTokens,
            Span<int> outputTokens,
            in GenerationOptions options)
        {
            ThrowIfDisposed();

            var allocatedBefore = GC.GetAllocatedBytesForCurrentThread();
            var start = Stopwatch.GetTimestamp();

            using var session = CreateSession(options.MaxContextLength > 0
                ? Math.Min(options.MaxContextLength, MaxContextLength)
                : MaxContextLength);

            var generated = session.Generate(
                promptTokens,
                outputTokens,
                in options);

            var elapsedNanoseconds = GetElapsedNanoseconds(start);
            var allocatedBytes = GC.GetAllocatedBytesForCurrentThread() - allocatedBefore;

            _lastGenerationStats = new GenerationStats(
                promptTokens: promptTokens.Length,
                generatedTokens: generated,
                elapsedNanoseconds: elapsedNanoseconds,
                allocatedBytes: allocatedBytes,
                usedKeyValueCache: false);

            return generated;
        }

        public GenerationStats GenerateStreaming(
            ReadOnlySpan<int> promptTokens,
            in GenerationOptions options,
            TokenGeneratedHandler onToken)
        {
            ThrowIfDisposed();

            if (onToken is null)
            {
                throw new ArgumentNullException(nameof(onToken));
            }

            var allocatedBefore = GC.GetAllocatedBytesForCurrentThread();
            var start = Stopwatch.GetTimestamp();

            using var session = CreateSession(options.MaxContextLength > 0
                ? Math.Min(options.MaxContextLength, MaxContextLength)
                : MaxContextLength);

            session.Reset(promptTokens);

            var sampling = options.Sampling;
            var generated = 0;
            var logits = new float[VocabularySize];

            while (generated < options.MaxNewTokens)
            {
                var token = session.GenerateNextToken(in sampling);
                session.GetLastLogits(logits);

                var shouldContinue = onToken(
                    token,
                    session.CurrentPosition,
                    logits);

                generated++;

                if (!shouldContinue)
                {
                    break;
                }

                if (options.StopOnEndOfTextToken &&
                    options.EndOfTextTokenId >= 0 &&
                    token == options.EndOfTextTokenId)
                {
                    break;
                }
            }

            var elapsedNanoseconds = GetElapsedNanoseconds(start);
            var allocatedBytes = GC.GetAllocatedBytesForCurrentThread() - allocatedBefore;

            _lastGenerationStats = new GenerationStats(
                promptTokens: promptTokens.Length,
                generatedTokens: generated,
                elapsedNanoseconds: elapsedNanoseconds,
                allocatedBytes: allocatedBytes,
                usedKeyValueCache: false);

            return _lastGenerationStats;
        }

        public void ResetMetrics()
        {
            _lastGenerationStats = default;
        }

        public GenerationStats GetLastGenerationStats()
        {
            return _lastGenerationStats;
        }

        public void Dispose()
        {
            if (_disposed)
            {
                return;
            }

            _disposed = true;
            _modelAdapter.Dispose();

            if (_disposeModelWithEngine)
            {
                _model.Dispose();
            }
        }

        private static long GetElapsedNanoseconds(long startTimestamp)
        {
            var elapsedTicks = Stopwatch.GetTimestamp() - startTimestamp;
            return (long)(elapsedTicks * (1_000_000_000.0 / Stopwatch.Frequency));
        }

        private void ThrowIfDisposed()
        {
            if (_disposed)
            {
                throw new ObjectDisposedException(nameof(SlmInferenceEngine));
            }
        }
    }
}
