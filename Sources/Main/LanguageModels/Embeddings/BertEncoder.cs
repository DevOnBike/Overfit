// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.LanguageModels.Contracts;
using DevOnBike.Overfit.Ops;
using DevOnBike.Overfit.Tensors;
using DevOnBike.Overfit.Tensors.Core;

namespace DevOnBike.Overfit.LanguageModels.Embeddings
{
    /// <summary>
    /// A bidirectional BERT-family encoder for producing sentence embeddings (the architecture behind
    /// sentence-transformers / all-MiniLM-L6-v2). Built from the existing graph layers in <c>Eval</c>
    /// mode — bidirectional self-attention (no causal mask), learned absolute position + token-type
    /// embeddings, an embeddings LayerNorm, post-LayerNorm transformer blocks, then pooling
    /// (<see cref="EmbeddingPooling.Mean"/> by default) + L2 normalization.
    ///
    /// Forward is inference-only (no gradients): each <see cref="Embed"/> call resets and replays a
    /// short autograd tape, so it allocates graph temporaries (acceptable for an embedding API — this
    /// is not the zero-allocation decode hot path). Weights are loaded by a converter writing directly
    /// into the public layer <c>Parameter</c>s (see <c>BertSafetensorsLoader</c>).
    /// </summary>
    public sealed class BertEncoder : IDisposable
    {
        private readonly BertConfig _config;
        private readonly ComputationGraph _graph;

        /// <summary>
        /// </summary>
        /// <param name="config">Model architecture.</param>
        /// <param name="expectedMaxSequenceLength">Upper bound on the token sequence length you intend to
        /// embed. The autograd arena is sized for this — sequences exceeding it throw at runtime
        /// (sentence embedders truncate first). Defaults to <c>min(MaxPositionEmbeddings, 256)</c> —
        /// covers virtually all real sentence-embedding inputs; raise to 512 for long-passage models.</param>
        public BertEncoder(BertConfig config, int? expectedMaxSequenceLength = null)
        {
            ArgumentNullException.ThrowIfNull(config);
            _config = config;

            MaxSequenceLength = expectedMaxSequenceLength ?? Math.Min(config.MaxPositionEmbeddings, 256);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(MaxSequenceLength);
            if (MaxSequenceLength > config.MaxPositionEmbeddings)
            {
                throw new ArgumentOutOfRangeException(
                    nameof(expectedMaxSequenceLength),
                    $"expectedMaxSequenceLength ({MaxSequenceLength}) exceeds the model's MaxPositionEmbeddings ({config.MaxPositionEmbeddings}).");
            }

            var d = config.HiddenSize;
            WordEmbeddings = new EmbeddingLayer(config.VocabSize, d);
            PositionEmbeddings = new EmbeddingLayer(config.MaxPositionEmbeddings, d);
            TokenTypeEmbeddings = new EmbeddingLayer(config.TypeVocabSize, d);
            EmbeddingLayerNorm = new LayerNormLayer(d, config.LayerNormEps);

            Layers = new TransformerBlock[config.NumLayers];
            for (var i = 0; i < config.NumLayers; i++)
            {
                Layers[i] = new TransformerBlock(
                    dModel: d,
                    nHeads: config.NumHeads,
                    dFF: config.IntermediateSize,
                    causalMask: false,     // bidirectional encoder
                    preLayerNorm: false,   // BERT is post-LayerNorm
                    lnEps: config.LayerNormEps);
            }

            // Arena sizing: per layer the dominant residents are FFN intermediate [T, dFF] plus a
            // handful of [T, d] tensors and per-head projections. A factor of 4·dFF per token per
            // layer comfortably covers BERT-family blocks; 1 << 22 (4M floats) is the floor so the
            // short-sentence MiniLM case stays cheap. Scales linearly with layers — 12L BGE/E5 get
            // double 6L MiniLM, as needed.
            var perLayer = checked(MaxSequenceLength * config.IntermediateSize * 4);
            var arenaFloats = Math.Max(1 << 22, checked(config.NumLayers * perLayer));
            _graph = new ComputationGraph(arenaFloats);
            Eval();
        }

        /// <summary>The maximum token sequence length the arena is sized for; longer inputs throw.</summary>
        public int MaxSequenceLength { get; }

        public BertConfig Config => _config;

        public EmbeddingLayer WordEmbeddings { get; }
        public EmbeddingLayer PositionEmbeddings { get; }
        public EmbeddingLayer TokenTypeEmbeddings { get; }
        public LayerNormLayer EmbeddingLayerNorm { get; }
        public TransformerBlock[] Layers { get; }

        /// <summary>Puts every sub-layer into eval mode (no dropout, no train-time stats).</summary>
        public void Eval()
        {
            WordEmbeddings.Eval();
            PositionEmbeddings.Eval();
            TokenTypeEmbeddings.Eval();
            EmbeddingLayerNorm.Eval();
            foreach (var layer in Layers) { layer.Eval(); }
        }

        /// <summary>
        /// Encodes a token-id sequence (already wrapped in <c>[CLS] … [SEP]</c> by the tokenizer) into a
        /// single pooled, L2-normalized embedding vector of length <see cref="BertConfig.HiddenSize"/>.
        /// </summary>
        public float[] Embed(ReadOnlySpan<int> tokenIds, EmbeddingPooling pooling = EmbeddingPooling.Mean)
        {
            var output = new float[_config.HiddenSize];
            Embed(tokenIds, output, pooling);
            return output;
        }

        /// <summary>
        /// Encodes into a caller-owned <paramref name="destination"/> (length == HiddenSize).
        /// </summary>
        public void Embed(ReadOnlySpan<int> tokenIds, Span<float> destination, EmbeddingPooling pooling = EmbeddingPooling.Mean)
        {
            if (tokenIds.IsEmpty)
            {
                throw new ArgumentException("Token id sequence must be non-empty.", nameof(tokenIds));
            }

            var d = _config.HiddenSize;
            if (destination.Length != d)
            {
                throw new ArgumentException(
                    $"Destination length {destination.Length} must equal hidden size {d}.", nameof(destination));
            }

            var t = tokenIds.Length;
            if (t > MaxSequenceLength)
            {
                throw new ArgumentException(
                    $"Sequence length {t} exceeds the encoder's expectedMaxSequenceLength {MaxSequenceLength} " +
                    "(arena sized for that). Pass a larger value to BertEncoder's ctor, or truncate the input.",
                    nameof(tokenIds));
            }

            var hidden = Forward(tokenIds, t, d);
            Pool(hidden, t, d, pooling, destination);
            Normalize(destination);
        }

        /// <summary>Runs the encoder, returning the final hidden states span [T·d] (row-major).</summary>
        private ReadOnlySpan<float> Forward(ReadOnlySpan<int> tokenIds, int t, int d)
        {
            _graph.Reset();

            var tokens = new int[t];
            tokenIds.CopyTo(tokens);
            var positions = new int[t];
            var types = new int[t]; // all zeros → segment 0
            for (var i = 0; i < t; i++) { positions[i] = i; }

            // emb = LayerNorm(word[ids] + position[0..T] + tokenType[0])
            var wordEmb = WordEmbeddings.Forward(_graph, tokens);          // [T, d]
            var posEmb = PositionEmbeddings.Forward(_graph, positions);    // [T, d]
            var typeEmb = TokenTypeEmbeddings.Forward(_graph, types);      // [T, d]

            var summed = TensorMath.Add(_graph, TensorMath.Add(_graph, wordEmb, posEmb), typeEmb);
            var x3 = _graph.Reshape(summed, 1, t, d);                      // [1, T, d]
            var x = EmbeddingLayerNorm.Forward(_graph, x3);

            foreach (var layer in Layers)
            {
                x = layer.Forward(_graph, x);                             // [1, T, d]
            }

            return x.DataView.AsReadOnlySpan();
        }

        private static void Pool(ReadOnlySpan<float> hidden, int t, int d, EmbeddingPooling pooling, Span<float> dst)
        {
            switch (pooling)
            {
                case EmbeddingPooling.Mean:
                    {
                        dst.Clear();
                        for (var i = 0; i < t; i++)
                        {
                            var row = hidden.Slice(i * d, d);
                            for (var j = 0; j < d; j++) { dst[j] += row[j]; }
                        }

                        var inv = 1f / t;
                        for (var j = 0; j < d; j++) { dst[j] *= inv; }
                        break;
                    }

                case EmbeddingPooling.LastToken:
                    {
                        hidden.Slice((t - 1) * d, d).CopyTo(dst);
                        break;
                    }

                case EmbeddingPooling.Cls:
                    {
                        // BGE / SBERT-CLS: the [CLS] token's hidden state (row 0).
                        hidden.Slice(0, d).CopyTo(dst);
                        break;
                    }

                default:
                    throw new ArgumentOutOfRangeException(nameof(pooling), pooling, "Unsupported pooling mode.");
            }
        }

        private static void Normalize(Span<float> v)
        {
            var norm = 0f;
            for (var j = 0; j < v.Length; j++) { norm += v[j] * v[j]; }
            norm = MathF.Sqrt(norm);
            if (norm > 1e-12f)
            {
                var inv = 1f / norm;
                for (var j = 0; j < v.Length; j++) { v[j] *= inv; }
            }
        }

        public void Dispose()
        {
            _graph.Dispose();
            WordEmbeddings.Dispose();
            PositionEmbeddings.Dispose();
            TokenTypeEmbeddings.Dispose();
            EmbeddingLayerNorm.Dispose();
            foreach (var layer in Layers) { layer.Dispose(); }
        }
    }
}
