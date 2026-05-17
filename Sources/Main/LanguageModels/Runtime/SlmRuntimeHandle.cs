// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels.Contracts;

namespace DevOnBike.Overfit.LanguageModels.Runtime
{
    /// <summary>
    /// Owns an SLM runtime session and, when needed, the engine that created it.
    ///
    /// This keeps legacy and cached runtimes selectable without deleting or
    /// changing the existing SlmInferenceEngine.
    ///
    /// Disposal order:
    /// - session first,
    /// - engine second.
    /// </summary>
    public sealed class SlmRuntimeHandle : IDisposable
    {
        private readonly IDisposable? _engine;
        private readonly ISlmSession _session;
        private bool _disposed;

        public SlmRuntimeHandle(
            SlmRuntimeMode mode,
            ISlmSession session,
            IDisposable? engine)
        {
            _session = session ?? throw new ArgumentNullException(nameof(session));

            Mode = mode;
            _engine = engine;
        }

        public SlmRuntimeMode Mode { get; }

        public ISlmSession Session
        {
            get
            {
                ThrowIfDisposed();

                return _session;
            }
        }

        public bool HasKeyValueCache
        {
            get
            {
                ThrowIfDisposed();

                return _session.HasKeyValueCache;
            }
        }

        public int MaxContextLength
        {
            get
            {
                ThrowIfDisposed();

                return _session.MaxContextLength;
            }
        }

        public int VocabularySize
        {
            get
            {
                ThrowIfDisposed();

                return _session.VocabularySize;
            }
        }

        public int Generate(
            ReadOnlySpan<int> promptTokens,
            Span<int> outputTokens,
            in GenerationOptions options)
        {
            ThrowIfDisposed();

            return _session.Generate(
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
                _session.MaxContextLength,
                SamplingOptions.Greedy,
                stopOnEndOfTextToken: false);

            return _session.Generate(
                promptTokens,
                outputTokens,
                in options);
        }

        public void Dispose()
        {
            if (_disposed)
            {
                return;
            }

            _disposed = true;

            _session.Dispose();
            _engine?.Dispose();
        }

        private void ThrowIfDisposed()
        {
            if (_disposed)
            {
                throw new ObjectDisposedException(nameof(SlmRuntimeHandle));
            }
        }
    }
}
