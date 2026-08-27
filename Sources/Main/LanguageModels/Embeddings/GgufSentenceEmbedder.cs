// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels.Contracts;
using DevOnBike.Overfit.LanguageModels.Loading;
using DevOnBike.Overfit.LanguageModels.Retrieval;
using DevOnBike.Overfit.LanguageModels.Runtime;
using DevOnBike.Overfit.LanguageModels.Tokenizers;

namespace DevOnBike.Overfit.LanguageModels.Embeddings
{
    /// <summary>
    /// Turnkey sentence-embedding facade over a <b>decoder</b> language model in a GGUF file — the
    /// counterpart to <see cref="SentenceEmbedder"/>, which is BERT/WordPiece by construction and cannot
    /// hold one. Loads the model, tokenizes with the file's own embedded vocabulary, pools the per-token
    /// post-final-norm hidden states and L2-normalises, so a caller goes from text to a cosine-ready vector
    /// in one call with no Python and no second process.
    ///
    /// <code>
    /// using var embedder = GgufSentenceEmbedder.ForQwen3Embedding(@"C:\qwen3-embed\Qwen3-Embedding-0.6B-Q8_0.gguf");
    /// var q = embedder.EmbedQuery("What is the capital of France?");
    /// var d = embedder.EmbedPassage("The capital of France is Paris.");
    /// </code>
    ///
    /// <para><b>Three defaults here are a contract, not an implementation detail</b>, because anything a
    /// caller stores depends on them: the pooling mode, the prefixes, and whether an end-of-text token is
    /// appended. Change one and every stored vector is in a different space, with no mechanism able to
    /// notice — see
    /// <see cref="PersistentVectorStore(int, string, string)"/>'s embedding-space id, which exists for
    /// exactly that reason.</para>
    ///
    /// <para><b>Not wired to <c>POST /v1/embeddings</c>.</b> That endpoint takes a concrete
    /// <see cref="SentenceEmbedder"/> and is externally fed, so serving a decoder LM there is a separate
    /// decision with its own security review.</para>
    /// </summary>
    public sealed class GgufSentenceEmbedder : IDisposable
    {
        /// <summary>
        /// Qwen3-Embedding's own retrieval task string, quoted verbatim from the model's
        /// <c>config_sentence_transformers.json</c> (<c>prompts.query</c>) and matching
        /// <c>get_detailed_instruct</c> in the model card's <c>transformers</c> usage block.
        /// </summary>
        public const string Qwen3EmbeddingRetrievalTask =
            "Given a web search query, retrieve relevant passages that answer the query";

        private readonly CachedLlamaInferenceEngine _engine;
        private readonly CachedLlamaSession _session;
        private readonly GgufTokenizer _tokenizer;
        private readonly EmbeddingPooling _pooling;
        private readonly string? _queryPrefix;
        private readonly string? _passagePrefix;
        private readonly bool _appendEndOfText;
        private readonly int _maxTokens;
        private bool _disposed;

        private GgufSentenceEmbedder(
            CachedLlamaInferenceEngine engine,
            CachedLlamaSession session,
            GgufTokenizer tokenizer,
            EmbeddingPooling pooling,
            string? queryPrefix,
            string? passagePrefix,
            bool appendEndOfText,
            int maxTokens)
        {
            _engine = engine;
            _session = session;
            _tokenizer = tokenizer;
            _pooling = pooling;
            _queryPrefix = queryPrefix;
            _passagePrefix = passagePrefix;
            _appendEndOfText = appendEndOfText;
            _maxTokens = maxTokens;
        }

        /// <summary>Embedding dimensionality (== the model's hidden size).</summary>
        public int Dimension => _session.EmbeddingDimension;

        /// <summary>Pooling mode used to collapse per-token states into one vector.</summary>
        public EmbeddingPooling Pooling => _pooling;

        /// <summary>The configured retrieval-side query prefix, if any.</summary>
        public string? QueryPrefix => _queryPrefix;

        /// <summary>The configured retrieval-side passage prefix, if any.</summary>
        public string? PassagePrefix => _passagePrefix;

        /// <summary>
        /// Generic loader: pick the pooling / prefixes your model expects. For Qwen3-Embedding prefer
        /// <see cref="ForQwen3Embedding"/>, which fills them from the model's own published convention.
        /// </summary>
        /// <param name="ggufPath">Path to the <c>*.gguf</c> file. Its embedded tokenizer vocabulary is used;
        /// no sibling <c>tokenizer.json</c> is needed.</param>
        /// <param name="pooling">How per-token states collapse to one vector. Decoder LMs trained as
        /// embedders almost always want <see cref="EmbeddingPooling.LastToken"/>, which is why it is the
        /// default here and why that differs from <see cref="SentenceEmbedder"/>, whose BERT-family models
        /// want <see cref="EmbeddingPooling.Mean"/> or <see cref="EmbeddingPooling.Cls"/>.
        ///
        /// <para><b>The default was <see cref="EmbeddingPooling.Mean"/> until 2026-08-27 (<c>XC-133</c>),
        /// which paired the worse half of this parameter with the worse half of <paramref name="quantize"/>.</b>
        /// Mean pooling reads position 0, the dequantised path disagrees with llama.cpp there, and
        /// <paramref name="quantize"/> defaults to <c>false</c> — so the out-of-the-box combination was the
        /// weakest of the four measured. <see cref="EmbeddingPooling.LastToken"/> never reads position 0, so
        /// the default pair is now the strongest one. The measurements are on
        /// <paramref name="quantize"/>.</para></param>
        /// <param name="queryPrefix">Prepended by <see cref="EmbedQuery(string)"/>.</param>
        /// <param name="passagePrefix">Prepended by <see cref="EmbedPassage(string)"/>.</param>
        /// <param name="appendEndOfText">Append the file's end-of-text token to every input. When null, the
        /// file's own <c>tokenizer.ggml.add_eos_token</c> flag decides
        /// (<see cref="GgufTokenizer.AddEosByDefault"/>).</param>
        /// <param name="quantize">Quantise weights at load. <b>Defaults to false, unlike the chat loaders</b>
        /// — an embedder's entire product is the vector, and re-quantising an already-quantised file is a
        /// correctness cost on it, not just a speed/RAM trade. Measured on Qwen3-Embedding-0.6B Q8_0 against
        /// llama.cpp reading the same bytes: PAIRWISE similarity between two embeddings drifts by 8.5e-3 at
        /// <c>true</c> against 4.6e-4 at <c>false</c>, and the worst deviation from the model card's own
        /// published matrix is 0.002733 against 0.000988. <b>The price is peak RAM, and it is not small:</b>
        /// 3270 MB against 1608 MB peak working set for the 0.6B file at a 1024-token context (two runs
        /// each, agreeing to 1 MB). That scales with the model, so <c>true</c> is the sane choice on the 4B
        /// and 8B siblings unless the box is large — the 8B would need roughly 32 GB dequantised.
        ///
        /// <para><b>An earlier version of this comment told you to pass <c>true</c> with mean pooling. That
        /// advice was BACKWARDS and is withdrawn — see <c>XC-132</c>, settled 2026-08-27.</b> It rested on
        /// "at position 0 the dequantised state diverges from llama.cpp", which is true and does not mean
        /// what it looks like: <b>llama.cpp is not an independent opinion there.</b> It quantises activations
        /// exactly as this loader does at <c>true</c>, so the two share one error and agree with each other.
        /// Measured: <c>cos(ours-true, llama.cpp)</c> at position 0 is 0.999979-0.999997, while
        /// <c>cos(ours-true, ours-false)</c> equals <c>cos(ours-false, llama.cpp)</c> to five decimals.</para>
        ///
        /// <para><b>A float64 reference decided it, and it is not close.</b> The same Q8_0 weights,
        /// dequantised once and pushed through the whole stack in double precision by
        /// <c>Scripts/xc132_float64_reference.py</c>, reproduce the <c>false</c> arm at cosine
        /// <b>1.000000</b> on all four texts, norms agreeing to two decimals. The <c>true</c> arm sits
        /// <b>0.906116 to 0.978504</b> away from that reference. <b>So <c>false</c> is the accurate arm at
        /// position 0 and <c>true</c> carries the error</b>, and the earlier mean-pooled table scored the
        /// correct arm against a reference that shares the incorrect arm's mistake.</para>
        ///
        /// <para><b>The mechanism, measured rather than argued.</b> At position 0 the residual stream is
        /// dominated by a single channel: channel 35 reaches <b>32x the RMS</b> of the row and holds it flat
        /// from layer 2 to layer 26 (5977 down to 5704 on a row whose norm is about 6000), and block 27
        /// cancels it. Q8 activation quantisation carries one scale per 32-element block, so the block
        /// holding channel 35 sets its step from 5704 and every other element inside it is destroyed. That
        /// is the <c>true</c> arm's error, and it is confined to the positions where a massive activation
        /// appears.</para>
        ///
        /// <para><b>So: leave this <c>false</c> for BOTH pooling modes when accuracy is what you want.</b>
        /// The reason to pass <c>true</c> is peak RAM and nothing else — which is a real reason on the 4B
        /// and 8B siblings, and is why the parameter exists. It is a precision trade, not a free one.</para></param>
        /// <param name="maxContextLength">KV-cache size in tokens; longer inputs are truncated. 1024 matches
        /// <c>OverfitClient</c>'s dedicated embed session and keeps the cache well under the weights.</param>
        public static GgufSentenceEmbedder FromGguf(
            string ggufPath,
            EmbeddingPooling pooling = EmbeddingPooling.LastToken,
            string? queryPrefix = null,
            string? passagePrefix = null,
            bool? appendEndOfText = null,
            bool quantize = false,
            int maxContextLength = 1024)
        {
            ArgumentException.ThrowIfNullOrEmpty(ggufPath);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(maxContextLength);
            if (!File.Exists(ggufPath))
            {
                throw new FileNotFoundException($"GGUF model file not found: '{ggufPath}'.", ggufPath);
            }

            var tokenizer = GgufTokenizer.Load(ggufPath);
            var appendEos = appendEndOfText ?? tokenizer.AddEosByDefault;
            var engine = GgufLlamaLoader.Load(ggufPath, quantize: quantize, mmap: true);

            try
            {
                var session = engine.CreateSession(maxContextLength);

                try
                {
                    return new GgufSentenceEmbedder(
                        engine, session, tokenizer, pooling, queryPrefix, passagePrefix,
                        appendEos, maxContextLength);
                }
                catch
                {
                    session.Dispose();
                    throw;
                }
            }
            catch
            {
                engine.Dispose();
                throw;
            }
        }

        /// <summary>
        /// Qwen/Qwen3-Embedding convention, taken from the model's own artefacts rather than from a blog:
        /// <see cref="EmbeddingPooling.LastToken"/> (<c>1_Pooling/config.json</c> sets
        /// <c>pooling_mode_lasttoken</c>), L2 normalisation (<c>modules.json</c> ends with a
        /// <c>Normalize</c> module), a query-side instruction and a bare passage side
        /// (<c>config_sentence_transformers.json</c>: <c>prompts.query</c> is
        /// <c>"Instruct: {task}\nQuery:"</c> and <c>prompts.document</c> is empty), and the file's own
        /// <c>add_eos_token</c> flag, which Qwen3-Embedding sets.
        ///
        /// <para><b>The absent space after <c>Query:</c> is deliberate.</b> The model card is internally
        /// inconsistent here — its <c>get_detailed_instruct</c> helper and
        /// <c>config_sentence_transformers.json</c> both emit <c>Query:</c> with no following space, while
        /// its Text-Embeddings-Inference <c>curl</c> example shows one. Two of three sources, including the
        /// machine-read config that sentence-transformers actually applies, say no space.</para>
        ///
        /// <para>Verified end to end against the model card's own published similarity matrix for its four
        /// example texts. Worst absolute deviation over the four pairs: <b>0.000988</b> at
        /// <c>quantize:false</c> on the 0.6B Q8_0 file.</para>
        /// </summary>
        /// <param name="ggufPath">Path to a Qwen3-Embedding <c>*.gguf</c>.</param>
        /// <param name="instructionTask">The one-sentence task description placed after <c>Instruct: </c>.
        /// Qwen recommend tailoring it per task and writing it in English, and report 1-5% retrieval loss
        /// when the query side carries none. Null uses
        /// <see cref="Qwen3EmbeddingRetrievalTask"/>.</param>
        /// <param name="quantize">See <see cref="FromGguf"/>; false by default for the same reason.</param>
        /// <param name="maxContextLength">KV-cache size in tokens; longer inputs are truncated.</param>
        public static GgufSentenceEmbedder ForQwen3Embedding(
            string ggufPath,
            string? instructionTask = null,
            bool quantize = false,
            int maxContextLength = 1024)
            => FromGguf(
                ggufPath,
                pooling: EmbeddingPooling.LastToken,
                queryPrefix: $"Instruct: {instructionTask ?? Qwen3EmbeddingRetrievalTask}\nQuery:",
                passagePrefix: null,
                appendEndOfText: null,
                quantize: quantize,
                maxContextLength: maxContextLength);

        /// <summary>Encodes <paramref name="text"/> into a new pooled, L2-normalised embedding (no prefix).</summary>
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

        /// <summary>Encodes <paramref name="text"/> with the configured query prefix prepended.</summary>
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
            EmbedRaw(_queryPrefix == null ? text : _queryPrefix + text, destination);
        }

        /// <summary>Encodes <paramref name="text"/> with the configured passage prefix prepended.</summary>
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
            EmbedRaw(_passagePrefix == null ? text : _passagePrefix + text, destination);
        }

        public void Dispose()
        {
            if (_disposed)
            {
                return;
            }

            _disposed = true;
            _session.Dispose();
            _engine.Dispose();
        }

        private void EmbedRaw(string text, Span<float> destination)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);

            var tokens = _tokenizer.Encode(text, addBos: false);
            if (tokens.Length == 0)
            {
                tokens = _tokenizer.Encode(" ", addBos: false);   // empty input, avoid the empty-sequence throw
            }

            if (!_appendEndOfText)
            {
                _session.Embed(Truncate(tokens), destination, _pooling);
                return;
            }

            // The end-of-text token must survive truncation, and under LastToken pooling it IS the pooled
            // position — dropping it does not degrade the vector, it produces a different one. Measured on
            // Qwen3-Embedding-0.6B against the model card's published similarity matrix: worst deviation
            // 0.000988 with the token, 0.106111 without it.
            var kept = Math.Min(tokens.Length, _maxTokens - 1);
            var withEos = new int[kept + 1];
            tokens.AsSpan(0, kept).CopyTo(withEos);
            withEos[kept] = _tokenizer.EosId;
            _session.Embed(withEos, destination, _pooling);
        }

        private ReadOnlySpan<int> Truncate(int[] tokens)
            => tokens.Length <= _maxTokens ? tokens : tokens.AsSpan(0, _maxTokens);
    }
}
