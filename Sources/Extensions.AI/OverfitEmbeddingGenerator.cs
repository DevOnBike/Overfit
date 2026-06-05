// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels.Embeddings;
using Microsoft.Extensions.AI;

namespace DevOnBike.Overfit.Extensions.AI
{
    /// <summary>
    /// Exposes an Overfit <see cref="SentenceEmbedder"/> (MiniLM / BGE / E5, pure-.NET BERT encoder) as a
    /// standard <see cref="IEmbeddingGenerator{TInput,TEmbedding}"/> for <c>string</c> → <c>Embedding&lt;float&gt;</c>,
    /// so it backs any <c>Microsoft.Extensions.AI</c> RAG / vector pipeline. In-process, no Python.
    ///
    /// <code>
    /// using var embedder = SentenceEmbedder.ForMiniLm("C:\\minilm");
    /// IEmbeddingGenerator&lt;string, Embedding&lt;float&gt;&gt; gen = embedder.AsEmbeddingGenerator();
    /// var vectors = await gen.GenerateAsync(["hello world"]);
    /// </code>
    /// </summary>
    public sealed class OverfitEmbeddingGenerator : IEmbeddingGenerator<string, Embedding<float>>
    {
        private readonly SentenceEmbedder _embedder;
        private readonly EmbeddingGeneratorMetadata _metadata;
        private readonly SemaphoreSlim _gate = new(1, 1);
        private bool _disposed;

        /// <param name="embedder">The loaded sentence embedder (borrowed — NOT disposed by this adapter).</param>
        /// <param name="modelId">Optional model id surfaced through <see cref="EmbeddingGeneratorMetadata"/>.</param>
        /// <param name="dimensions">Optional embedding dimension surfaced as the default model dimensions.</param>
        public OverfitEmbeddingGenerator(SentenceEmbedder embedder, string? modelId = null, int? dimensions = null)
        {
            _embedder = embedder ?? throw new ArgumentNullException(nameof(embedder));
            _metadata = new EmbeddingGeneratorMetadata(
                "overfit", providerUri: null, defaultModelId: modelId ?? "overfit-embeddings", defaultModelDimensions: dimensions);
        }

        /// <inheritdoc />
        public async Task<GeneratedEmbeddings<Embedding<float>>> GenerateAsync(
            IEnumerable<string> values,
            EmbeddingGenerationOptions? options = null,
            CancellationToken cancellationToken = default)
        {
            ArgumentNullException.ThrowIfNull(values);
            ObjectDisposedException.ThrowIf(_disposed, this);

            var inputs = values as IReadOnlyList<string> ?? values.ToList();

            await _gate.WaitAsync(cancellationToken).ConfigureAwait(false);
            try
            {
                return await Task.Run(() =>
                {
                    var embeddings = new GeneratedEmbeddings<Embedding<float>>();
                    foreach (var text in inputs)
                    {
                        cancellationToken.ThrowIfCancellationRequested();
                        var vector = _embedder.EmbedQuery(text ?? string.Empty);
                        embeddings.Add(new Embedding<float>(vector) { ModelId = _metadata.DefaultModelId });
                    }
                    return embeddings;
                }, cancellationToken).ConfigureAwait(false);
            }
            finally
            {
                _gate.Release();
            }
        }

        /// <inheritdoc />
        public object? GetService(Type serviceType, object? serviceKey = null)
        {
            ArgumentNullException.ThrowIfNull(serviceType);
            if (serviceKey is null && serviceType == typeof(EmbeddingGeneratorMetadata))
            {
                return _metadata;
            }
            return serviceType.IsInstanceOfType(this) ? this : null;
        }

        /// <summary>Disposes adapter-owned state only — the wrapped <see cref="SentenceEmbedder"/> is borrowed.</summary>
        public void Dispose()
        {
            if (_disposed)
            {
                return;
            }
            _disposed = true;
            _gate.Dispose();
        }
    }
}
