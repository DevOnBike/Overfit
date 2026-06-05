// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels;
using DevOnBike.Overfit.LanguageModels.Chat;
using DevOnBike.Overfit.LanguageModels.Embeddings;
using Microsoft.Extensions.AI;

namespace DevOnBike.Overfit.Extensions.AI
{
    /// <summary>
    /// One-line bridges from the Overfit runtime to the <c>Microsoft.Extensions.AI</c> abstractions.
    /// </summary>
    public static class OverfitAiExtensions
    {
        /// <summary>Wraps an <see cref="OverfitClient"/> as a standard <see cref="IChatClient"/>.</summary>
        public static IChatClient AsChatClient(this OverfitClient client, string? modelId = null)
        {
            return new OverfitChatClient((client ?? throw new ArgumentNullException(nameof(client))).Chat, modelId);
        }

        /// <summary>Wraps a <see cref="ChatSession"/> as a standard <see cref="IChatClient"/>.</summary>
        public static IChatClient AsChatClient(this ChatSession session, string? modelId = null)
        {
            return new OverfitChatClient(session, modelId);
        }

        /// <summary>Wraps a <see cref="SentenceEmbedder"/> as a standard
        /// <see cref="IEmbeddingGenerator{TInput,TEmbedding}"/> (<c>string</c> → <c>Embedding&lt;float&gt;</c>).</summary>
        public static IEmbeddingGenerator<string, Embedding<float>> AsEmbeddingGenerator(this SentenceEmbedder embedder, string? modelId = null, int? dimensions = null)
        {
            return new OverfitEmbeddingGenerator(embedder, modelId, dimensions);
        }
    }
}
