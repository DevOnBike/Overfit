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
    /// .NET equivalent of <c>SentenceTransformer(...).encode(text)</c>, no Python.
    ///
    /// Supports model-family conventions via optional retrieval prefixes — E5 wants
    /// <c>"query: "</c> / <c>"passage: "</c>, BGE wants a query-side instruction like
    /// <c>"Represent this sentence for searching relevant passages: "</c>. Use
    /// <c>EmbedQuery</c> / <c>EmbedPassage</c> to apply the configured prefix per call;
    /// plain <see cref="Embed(string)"/> stays prefix-free (the MiniLM convention).
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
        private readonly string? _queryPrefix;
        private readonly string? _passagePrefix;
        private readonly int _maxTokens;

        public SentenceEmbedder(
            WordPieceTokenizer tokenizer,
            BertEncoder encoder,
            EmbeddingPooling pooling = EmbeddingPooling.Mean,
            string? queryPrefix = null,
            string? passagePrefix = null)
        {
            ArgumentNullException.ThrowIfNull(tokenizer);
            ArgumentNullException.ThrowIfNull(encoder);
            _tokenizer = tokenizer;
            _encoder = encoder;
            _pooling = pooling;
            _queryPrefix = queryPrefix;
            _passagePrefix = passagePrefix;
            _maxTokens = encoder.MaxSequenceLength;
        }

        /// <summary>Embedding dimensionality (== model hidden size).</summary>
        public int Dimension => _encoder.Config.HiddenSize;

        /// <summary>Pooling mode used to collapse per-token states into one vector.</summary>
        public EmbeddingPooling Pooling => _pooling;

        /// <summary>The configured retrieval-side query prefix, if any (e.g. <c>"query: "</c>).</summary>
        public string? QueryPrefix => _queryPrefix;

        /// <summary>The configured retrieval-side passage prefix, if any (e.g. <c>"passage: "</c>).</summary>
        public string? PassagePrefix => _passagePrefix;

        /// <summary>
        /// Generic loader: pick the pooling / prefixes your model family expects. For known families
        /// prefer <see cref="ForMiniLm"/> / <see cref="ForBgeEnV15"/> / <see cref="ForE5"/>.
        /// </summary>
        public static SentenceEmbedder FromPretrained(
            string modelDir,
            EmbeddingPooling pooling = EmbeddingPooling.Mean,
            string? queryPrefix = null,
            string? passagePrefix = null)
        {
            ArgumentException.ThrowIfNullOrEmpty(modelDir);

            var config = BertConfigReader.ReadFromDirectory(modelDir);
            var tokenizer = WordPieceTokenizer.FromVocabFile(Path.Combine(modelDir, "vocab.txt"));
            var encoder = BertSafetensorsLoader.Load(Path.Combine(modelDir, "model.safetensors"), config);
            return new SentenceEmbedder(tokenizer, encoder, pooling, queryPrefix, passagePrefix);
        }

        /// <summary>
        /// sentence-transformers/all-MiniLM-L6-v2 convention: <see cref="EmbeddingPooling.Mean"/>, no prefixes.
        /// </summary>
        public static SentenceEmbedder ForMiniLm(string modelDir)
            => FromPretrained(modelDir, EmbeddingPooling.Mean);

        /// <summary>
        /// BAAI/bge-*-en-v1.5 convention: <see cref="EmbeddingPooling.Cls"/>, query-side instruction
        /// <c>"Represent this sentence for searching relevant passages: "</c>, no passage prefix.
        /// </summary>
        public static SentenceEmbedder ForBgeEnV15(string modelDir)
            => FromPretrained(
                modelDir,
                pooling: EmbeddingPooling.Cls,
                queryPrefix: "Represent this sentence for searching relevant passages: ",
                passagePrefix: null);

        /// <summary>
        /// intfloat/e5-* convention: <see cref="EmbeddingPooling.Mean"/>, both sides prefixed
        /// (<c>"query: "</c> / <c>"passage: "</c>).
        /// </summary>
        public static SentenceEmbedder ForE5(string modelDir)
            => FromPretrained(
                modelDir,
                pooling: EmbeddingPooling.Mean,
                queryPrefix: "query: ",
                passagePrefix: "passage: ");

        /// <summary>Encodes <paramref name="text"/> into a new pooled, L2-normalized embedding (no prefix).</summary>
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
            EmbedRaw(text, destination);
        }

        /// <summary>Encodes <paramref name="text"/> with the configured retrieval-side query prefix prepended.</summary>
        public float[] EmbedQuery(string text)
        {
            var output = new float[Dimension];
            EmbedQuery(text, output);
            return output;
        }

        /// <summary>Encodes a retrieval query into a caller-owned destination.</summary>
        public void EmbedQuery(string text, Span<float> destination)
        {
            ArgumentNullException.ThrowIfNull(text);
            EmbedRaw(_queryPrefix is null ? text : _queryPrefix + text, destination);
        }

        /// <summary>Encodes <paramref name="text"/> with the configured retrieval-side passage prefix prepended.</summary>
        public float[] EmbedPassage(string text)
        {
            var output = new float[Dimension];
            EmbedPassage(text, output);
            return output;
        }

        /// <summary>Encodes a retrieval passage into a caller-owned destination.</summary>
        public void EmbedPassage(string text, Span<float> destination)
        {
            ArgumentNullException.ThrowIfNull(text);
            EmbedRaw(_passagePrefix is null ? text : _passagePrefix + text, destination);
        }

        private void EmbedRaw(string text, Span<float> destination)
        {
            var tokens = _tokenizer.Encode(text, addSpecialTokens: true);
            if (tokens.Length > _maxTokens)
            {
                tokens[_maxTokens - 1] = _tokenizer.SeparatorTokenId; // keep a trailing [SEP]
                _encoder.Embed(tokens.AsSpan(0, _maxTokens), destination, _pooling);
                return;
            }

            _encoder.Embed(tokens, destination, _pooling);
        }

        /// <summary>Embeds <paramref name="text"/> (no prefix) and adds it to <paramref name="store"/> under <paramref name="id"/>.</summary>
        public void AddTo(VectorStore store, string id, string text)
        {
            ArgumentNullException.ThrowIfNull(store);
            Span<float> vec = Dimension <= 1024 ? stackalloc float[Dimension] : new float[Dimension];
            Embed(text, vec);
            store.Add(id, vec, text);
        }

        /// <summary>Embeds a passage (with passage prefix, if configured) and adds it to <paramref name="store"/>.</summary>
        public void AddPassageTo(VectorStore store, string id, string text)
        {
            ArgumentNullException.ThrowIfNull(store);
            Span<float> vec = Dimension <= 1024 ? stackalloc float[Dimension] : new float[Dimension];
            EmbedPassage(text, vec);
            store.Add(id, vec, text);
        }

        public void Dispose() => _encoder.Dispose();
    }
}
