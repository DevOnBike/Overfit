// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.LanguageModels.Embeddings
{
    /// <summary>
    /// Hyper-parameters of a BERT-family encoder (the architecture behind sentence-transformers
    /// embedding models). Maps 1:1 onto a HuggingFace <c>config.json</c> for the encoder family.
    /// </summary>
    public sealed class BertConfig
    {
        public BertConfig(
            int hiddenSize,
            int numLayers,
            int numHeads,
            int intermediateSize,
            int maxPositionEmbeddings,
            int vocabSize,
            int typeVocabSize = 2,
            float layerNormEps = 1e-12f)
        {
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(hiddenSize);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(numLayers);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(numHeads);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(intermediateSize);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(maxPositionEmbeddings);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(vocabSize);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(typeVocabSize);
            if (hiddenSize % numHeads != 0)
            {
                throw new ArgumentException(
                    $"hiddenSize ({hiddenSize}) must be divisible by numHeads ({numHeads}).");
            }

            HiddenSize = hiddenSize;
            NumLayers = numLayers;
            NumHeads = numHeads;
            IntermediateSize = intermediateSize;
            MaxPositionEmbeddings = maxPositionEmbeddings;
            VocabSize = vocabSize;
            TypeVocabSize = typeVocabSize;
            LayerNormEps = layerNormEps;
        }

        /// <summary>Model / embedding dimension (a.k.a. d_model). 384 for all-MiniLM-L6-v2.</summary>
        public int HiddenSize { get; }

        public int NumLayers { get; }

        public int NumHeads { get; }

        /// <summary>FFN inner dimension (typically 4·HiddenSize). 1536 for all-MiniLM-L6-v2.</summary>
        public int IntermediateSize { get; }

        public int MaxPositionEmbeddings { get; }

        public int VocabSize { get; }

        /// <summary>Segment / token-type vocabulary (2 for BERT). Single-sentence embedding uses type 0.</summary>
        public int TypeVocabSize { get; }

        /// <summary>LayerNorm epsilon. BERT uses 1e-12 (not the 1e-5 common elsewhere).</summary>
        public float LayerNormEps { get; }

        /// <summary>
        /// sentence-transformers/all-MiniLM-L6-v2: 6 layers, hidden 384, 12 heads, FFN 1536,
        /// 512 positions, 30522 WordPiece vocab, eps 1e-12.
        /// </summary>
        public static BertConfig AllMiniLmL6V2 => new(
            hiddenSize: 384,
            numLayers: 6,
            numHeads: 12,
            intermediateSize: 1536,
            maxPositionEmbeddings: 512,
            vocabSize: 30522,
            typeVocabSize: 2,
            layerNormEps: 1e-12f);
    }
}
