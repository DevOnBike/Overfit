// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels.Contracts;
using DevOnBike.Overfit.LanguageModels.Retrieval;
using DevOnBike.Overfit.LanguageModels.Tokenizers;

namespace DevOnBike.Overfit.LanguageModels.Embeddings
{
    /// <summary>
    /// Turnkey sentence-embedding facade: a <see cref="WordPieceTokenizer"/> + <see cref="BertEncoder"/>
    /// wired together so callers go straight from text to a pooled, L2-normalized vector — the in-process
    /// .NET equivalent of <c>SentenceTransformer("all-MiniLM-L6-v2").encode(text)</c>, no Python.
    ///
    /// <see cref="FromPretrained"/> loads a raw HuggingFace model directory (<c>config.json</c> +
    /// <c>vocab.txt</c> + <c>model.safetensors</c>) directly. Sequences longer than the model's position
    /// limit are truncated (keeping the trailing <c>[SEP]</c>).
    /// </summary>
    public sealed class SentenceEmbedder : IDisposable
    {
        private readonly WordPieceTokenizer _tokenizer;
        private readonly BertEncoder _encoder;
        private readonly EmbeddingPooling _pooling;
        private readonly int _maxTokens;

        public SentenceEmbedder(
            WordPieceTokenizer tokenizer,
            BertEncoder encoder,
            EmbeddingPooling pooling = EmbeddingPooling.Mean)
        {
            ArgumentNullException.ThrowIfNull(tokenizer);
            ArgumentNullException.ThrowIfNull(encoder);
            _tokenizer = tokenizer;
            _encoder = encoder;
            _pooling = pooling;
            _maxTokens = encoder.Config.MaxPositionEmbeddings;
        }

        /// <summary>Embedding dimensionality (== model hidden size).</summary>
        public int Dimension => _encoder.Config.HiddenSize;

        /// <summary>
        /// Loads a sentence-transformers BERT encoder from a HuggingFace directory containing
        /// <c>config.json</c>, <c>vocab.txt</c> and <c>model.safetensors</c>.
        /// </summary>
        public static SentenceEmbedder FromPretrained(string modelDir, EmbeddingPooling pooling = EmbeddingPooling.Mean)
        {
            ArgumentException.ThrowIfNullOrEmpty(modelDir);

            var config = BertConfigReader.ReadFromDirectory(modelDir);
            var tokenizer = WordPieceTokenizer.FromVocabFile(Path.Combine(modelDir, "vocab.txt"));
            var encoder = BertSafetensorsLoader.Load(Path.Combine(modelDir, "model.safetensors"), config);
            return new SentenceEmbedder(tokenizer, encoder, pooling);
        }

        /// <summary>Encodes <paramref name="text"/> into a new pooled, L2-normalized embedding.</summary>
        public float[] Embed(string text)
        {
            var output = new float[Dimension];
            Embed(text, output);
            return output;
        }

        /// <summary>Encodes into a caller-owned destination (length == <see cref="Dimension"/>).</summary>
        public void Embed(string text, Span<float> destination)
        {
            ArgumentNullException.ThrowIfNull(text);
            var tokens = _tokenizer.Encode(text, addSpecialTokens: true);
            if (tokens.Length > _maxTokens)
            {
                tokens[_maxTokens - 1] = _tokenizer.SeparatorTokenId; // keep a trailing [SEP]
                _encoder.Embed(tokens.AsSpan(0, _maxTokens), destination, _pooling);
                return;
            }

            _encoder.Embed(tokens, destination, _pooling);
        }

        /// <summary>Embeds <paramref name="text"/> and adds it to <paramref name="store"/> under <paramref name="id"/>.</summary>
        public void AddTo(VectorStore store, string id, string text)
        {
            ArgumentNullException.ThrowIfNull(store);
            Span<float> vec = Dimension <= 1024 ? stackalloc float[Dimension] : new float[Dimension];
            Embed(text, vec);
            store.Add(id, vec, text);
        }

        public void Dispose() => _encoder.Dispose();
    }
}
