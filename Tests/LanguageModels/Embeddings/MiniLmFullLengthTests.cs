// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Linq;
using DevOnBike.Overfit.LanguageModels.Embeddings;

namespace DevOnBike.Overfit.Tests.LanguageModels.Embeddings
{
    /// <summary>
    /// Regression for the BertEncoder arena at FULL sequence length. A passage long enough to be
    /// truncated to the encoder's MaxSequenceLength (256) drives the worst-case forward tape — which
    /// must hold every layer's activations plus per-node gradient buffers at once. An under-sized arena
    /// threw <c>NativeBuffer exhausted</c> here (surfaced first by Polish RAG chunks, which sub-word
    /// heavily and reach 256 tokens where short English sentences do not). [LongFact] — needs C:\minilm.
    /// </summary>
    public sealed class MiniLmFullLengthTests
    {
        private const string MiniLmDir = @"C:\minilm";

        [LongFact]
        public void Embed_FullLengthPassage_DoesNotExhaustArena()
        {
            if (!Directory.Exists(MiniLmDir))
            {
                return;
            }

            using var embedder = SentenceEmbedder.ForMiniLm(MiniLmDir);

            // Far more than 256 tokens → truncated to the arena's full MaxSequenceLength.
            var longText = string.Concat(Enumerable.Repeat(
                "Rękojmia oraz prawo odstąpienia od umowy w sklepie internetowym dla klientów z Unii Europejskiej. ",
                80));

            var vec = embedder.Embed(longText);

            Assert.Equal(embedder.Dimension, vec.Length);
            Assert.True(vec.Any(v => v != 0f), "embedding is all zeros");
        }
    }
}
