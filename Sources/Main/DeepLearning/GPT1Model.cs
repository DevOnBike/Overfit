// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Numerics.Tensors;
using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.DeepLearning.Abstractions;
using DevOnBike.Overfit.Ops;
using DevOnBike.Overfit.Parameters;
using DevOnBike.Overfit.Tensors;
using DevOnBike.Overfit.Tensors.Core;

namespace DevOnBike.Overfit.DeepLearning
{
    /// <summary>
    /// GPT-1 style language model.
    ///
    /// Architecture:
    ///   1. Token embedding:     tokens [B, T] → [B, T, dModel]
    ///   2. Positional embedding: positions [T] → [1, T, dModel]  (learned)
    ///   3. x = tok_emb + pos_emb                                 (broadcast add)
    ///   4. x = TransformerBlock(x) × nLayers                     (causal, Pre-LN)
    ///   5. x = FinalLayerNorm(x)
    ///   6. logits = x @ W_vocab^T                                 (LM head, weight-tied)
    ///
    /// Weight tying: the LM head shares weights with the token embedding.
    /// This is standard for language models (reduces params, improves generalisation).
    ///
    /// Forward returns logits [B, T, vocabSize].
    /// For language modelling loss: use SoftmaxCrossEntropy on logits[:, :-1, :]
    /// against targets[:, 1:, :] (next-token prediction).
    ///
    /// Inference: call <see cref="GenerateLogits"/> for a single step,
    /// then sample or argmax.
    /// </summary>
    public sealed class GPT1Model : IDisposable
    {
        private readonly GPT1Config _config;
        private bool _isTraining = true;
        private AutogradNode? _lmHeadNode;
        private bool _disposed;

        public GPT1Model(GPT1Config config)
        {
            _config = config ?? throw new ArgumentNullException(nameof(config));

            TokenEmbedding = new EmbeddingLayer(config.VocabSize, config.DModel);
            PositionEmbedding = new EmbeddingLayer(config.ContextLength, config.DModel);

            var blocks = new TransformerBlock[config.NLayers];
            for (var i = 0; i < config.NLayers; i++)
            {
                blocks[i] = new TransformerBlock(
                    config.DModel,
                    config.NHeads,
                    config.DFF,
                    causalMask: true,
                    preLayerNorm: config.PreLayerNorm,
                    lnEps: config.LNEps);
            }

            Blocks = blocks;
            FinalNorm = new LayerNormLayer(config.DModel, config.LNEps);

            // LM head: [dModel, vocabSize]
            // When weight-tying: shares Parameter with TokenEmbedding.Weight (transposed).
            // We store as a separate Parameter but copy data post-init (tie via forward).
            LMHead = new Parameter(
                new TensorShape(config.DModel, config.VocabSize),
                requiresGrad: !config.TieWeights,  // no grad needed if tied (tok emb has it)
                clearData: false);

            if (config.TieWeights)
            {
                // Copy transposed token embedding weights as initialisation.
                // During forward we re-tie dynamically.
                TransposeInto(
                    TokenEmbedding.Weight.DataReadOnlySpan,
                    LMHead.DataSpan,
                    config.VocabSize,
                    config.DModel);
            }
            else
            {
                var scale = MathF.Sqrt(2f / config.DModel);
                var s = LMHead.DataSpan;
                for (var i = 0; i < s.Length; i++)
                {
                    s[i] = Maths.MathUtils.NextGaussian() * scale;
                }
            }
        }

        public GPT1Config Config => _config;
        public EmbeddingLayer TokenEmbedding { get; }
        public EmbeddingLayer PositionEmbedding { get; }
        public TransformerBlock[] Blocks { get; }
        public LayerNormLayer FinalNorm { get; }
        public Parameter LMHead { get; }

        public bool IsTraining => _isTraining;

        public void Train()
        {
            _isTraining = true;
            TokenEmbedding.Train();
            PositionEmbedding.Train();
            foreach (var b in Blocks)
            {
                b.Train();
            }
            FinalNorm.Train();
        }

        public void Eval()
        {
            _isTraining = false;
            TokenEmbedding.Eval();
            PositionEmbedding.Eval();
            foreach (var b in Blocks)
            {
                b.Eval();
            }
            FinalNorm.Eval();
        }

        /// <summary>
        /// Forward pass.
        ///
        /// tokenIds: int[batchSize, seqLen] flattened in row-major order.
        /// Returns logits [batchSize, seqLen, vocabSize].
        ///
        /// For single sequence: pass int[1 * seqLen] with batchSize=1.
        /// </summary>
        public AutogradNode Forward(
            ComputationGraph graph,
            int[] tokenIds,
            int batchSize,
            int seqLen)
        {
            if (seqLen > _config.ContextLength)
            {
                throw new ArgumentException($"seqLen={seqLen} exceeds ContextLength={_config.ContextLength}.");
            }

            if (tokenIds.Length != batchSize * seqLen)
            {
                throw new ArgumentException($"tokenIds.Length={tokenIds.Length} must equal batchSize*seqLen={batchSize * seqLen}.");
            }

            // ── 1. Token embeddings ──────────────────────────────────────────
            // Look up each token independently, then reshape to [B, T, dModel].
            var tokEmb = EmbedBatch(graph, tokenIds, batchSize, seqLen, TokenEmbedding);

            // ── 2. Positional embeddings ─────────────────────────────────────
            // Positions 0..seqLen-1, same for all batch items.
            var posIds = CreatePositionIds(seqLen);
            var posEmb = EmbedPositions(graph, posIds, batchSize, seqLen);

            // ── 3. x = tok_emb + pos_emb ────────────────────────────────────
            var x = AddEmbeddings(graph, tokEmb, posEmb, batchSize, seqLen, _config.DModel);

            // ── 4. Transformer blocks ────────────────────────────────────────
            foreach (var block in Blocks)
            {
                x = block.Forward(graph, x);
            }

            // ── 5. Final LayerNorm ───────────────────────────────────────────
            x = FinalNorm.Forward(graph, x);

            // ── 6. LM head: [B, T, dModel] → [B, T, vocabSize] ──────────────
            return LMHeadForward(graph, x, batchSize, seqLen);
        }

        /// <summary>
        /// Generates logits for the LAST token position only.
        /// More efficient for autoregressive generation.
        /// Returns [1, vocabSize].
        /// </summary>
        public float[] GenerateLogits(int[] tokenIds)
        {
            var seqLen = tokenIds.Length;

            using var graph = new ComputationGraph();
            using var logits = Forward(graph, tokenIds, batchSize: 1, seqLen);

            var logitSpan = logits.DataView.AsReadOnlySpan();
            var vocabSize = _config.VocabSize;

            // Extract last position: logits[(seqLen-1)*vocabSize .. seqLen*vocabSize)
            var lastLogits = new float[vocabSize];
            logitSpan.Slice((seqLen - 1) * vocabSize, vocabSize).CopyTo(lastLogits);
            return lastLogits;
        }

        /// <summary>
        /// Greedy argmax token generation for <paramref name="maxNewTokens"/> steps.
        /// Simple greedy decoding — no sampling, no temperature.
        /// </summary>
        public int[] Generate(int[] promptTokenIds, int maxNewTokens)
        {
            var tokens = new List<int>(promptTokenIds);

            for (var step = 0; step < maxNewTokens; step++)
            {
                // Truncate to context window
                var ctx = tokens.Count > _config.ContextLength
                    ? tokens.GetRange(tokens.Count - _config.ContextLength, _config.ContextLength).ToArray()
                    : tokens.ToArray();

                var logits = GenerateLogits(ctx);
                var nextTok = ArgMax(logits);
                tokens.Add(nextTok);
            }

            return tokens.GetRange(promptTokenIds.Length, tokens.Count - promptTokenIds.Length).ToArray();
        }

        /// <summary>All trainable parameters — for optimizers.</summary>
        public IEnumerable<Parameter> TrainableParameters()
        {
            yield return TokenEmbedding.Weight;
            yield return PositionEmbedding.Weight;

            foreach (var block in Blocks)
            {
                foreach (var p in block.TrainableParameters())
                {
                    yield return p;
                }
            }

            foreach (var p in FinalNorm.TrainableParameters())
            {
                yield return p;
            }

            if (!_config.TieWeights)
            {
                yield return LMHead;
            }
        }

        public void Save(BinaryWriter bw)
        {
            TokenEmbedding.Save(bw);
            PositionEmbedding.Save(bw);
            foreach (var b in Blocks)
            {
                b.Save(bw);
            }
            FinalNorm.Save(bw);
            if (!_config.TieWeights)
            {
                LMHead.Save(bw);
            }
        }

        public void Load(BinaryReader br)
        {
            TokenEmbedding.Load(br);
            PositionEmbedding.Load(br);
            foreach (var b in Blocks)
            {
                b.Load(br);
            }
            FinalNorm.Load(br);
            if (!_config.TieWeights)
            {
                LMHead.Load(br);
            }
            else
            {
                // Re-sync LM head with token embedding (weight tying)
                TransposeInto(
                    TokenEmbedding.Weight.DataReadOnlySpan,
                    LMHead.DataSpan,
                    _config.VocabSize,
                    _config.DModel);
            }
        }

        public void InvalidateAllCaches()
        {
            TokenEmbedding.InvalidateParameterCaches();
            PositionEmbedding.InvalidateParameterCaches();
            foreach (var b in Blocks)
            {
                b.InvalidateParameterCaches();
            }
            FinalNorm.InvalidateParameterCaches();
            _lmHeadNode = null;
        }

        public void Dispose()
        {
            if (_disposed)
            {
                return;
            }
            _disposed = true;

            _lmHeadNode?.Dispose();
            TokenEmbedding.Dispose();
            PositionEmbedding.Dispose();
            foreach (var b in Blocks)
            {
                b.Dispose();
            }
            FinalNorm.Dispose();
            LMHead.Dispose();
        }

        // ── Private helpers ───────────────────────────────────────────────────

        /// <summary>
        /// Embeds all tokens in a batch and reshapes to [B, T, dModel].
        /// Processes each batch item sequentially.
        /// </summary>
        private static AutogradNode EmbedBatch(
            ComputationGraph graph,
            int[] tokenIds,
            int batchSize,
            int seqLen,
            EmbeddingLayer embedding)
        {
            var dModel = embedding.EmbeddingDim;

            // Single embedding forward for all B*T tokens — produces [B*T, dModel] ON TAPE.
            // requiresGrad flows naturally: grad accumulates into embedding weights during backward.
            // This is what enables true batch training (B>1).
            var embNode = embedding.Forward(graph, tokenIds);  // [B*T, dModel]

            // Reshape to [B, T, dModel] — also on tape, grad flows through reshape backward.
            return graph.Reshape(embNode, batchSize, seqLen, dModel);
        }

        private AutogradNode EmbedPositions(
            ComputationGraph graph,
            int[] posIds,
            int batchSize,
            int seqLen)
        {
            // Repeat positions [0..T-1] for each batch item → [B*T] ids
            // Then single embedding forward → [B*T, dModel] ON TAPE.
            var allPosIds = new int[batchSize * seqLen];
            for (var b = 0; b < batchSize; b++)
            {
                posIds.AsSpan().CopyTo(allPosIds.AsSpan(b * seqLen, seqLen));
            }

            var posNode = PositionEmbedding.Forward(graph, allPosIds);  // [B*T, dModel]
            return graph.Reshape(posNode, batchSize, seqLen, _config.DModel);
        }

        private static AutogradNode AddEmbeddings(
            ComputationGraph graph,
            AutogradNode tokEmb,
            AutogradNode posEmb,
            int batchSize,
            int seqLen,
            int dModel)
        {
            // ON TAPE: gradient flows to both token and positional embeddings.
            return TensorMath.Add(graph, tokEmb, posEmb);
        }

        private AutogradNode LMHeadForward(
            ComputationGraph graph,
            AutogradNode x,
            int batchSize,
            int seqLen)
        {
            var dModel = _config.DModel;
            var vocabSize = _config.VocabSize;

            if (_config.TieWeights)
            {
                TransposeInto(
                    TokenEmbedding.Weight.DataReadOnlySpan,
                    LMHead.DataSpan,
                    vocabSize,
                    dModel);
            }

            // Flatten [B, T, dModel] -> [B*T, dModel]
            var flat = graph.Reshape(x, batchSize * seqLen, dModel);

            // LM head ON TAPE: gradient flows back through transformer.
            // graph.Linear(input[B*T,d], weight[d,V], bias[V]) -> [B*T,V]
            var biasStorage = new TensorStorage<float>(vocabSize, clearMemory: true);
            var bias = AutogradNode.CreateBorrowed(biasStorage, new TensorShape(vocabSize));

            _lmHeadNode ??= LMHead.AsNode();
            var flatLogits = graph.Linear(flat, _lmHeadNode, bias);

            // Reshape to [B, T, vocabSize]
            return graph.Reshape(flatLogits, batchSize, seqLen, vocabSize);
        }

        private static int[] CreatePositionIds(int seqLen)
        {
            var ids = new int[seqLen];
            for (var i = 0; i < seqLen; i++)
            {
                ids[i] = i;
            }
            return ids;
        }

        private static int ArgMax(ReadOnlySpan<float> logits)
        {
            var maxIdx = 0;
            var maxVal = logits[0];
            for (var i = 1; i < logits.Length; i++)
            {
                if (logits[i] > maxVal)
                {
                    maxVal = logits[i];
                    maxIdx = i;
                }
            }
            return maxIdx;
        }

        /// <summary>
        /// Transposes [vocabSize, dModel] → [dModel, vocabSize] in place.
        /// Used for weight tying: token embedding is [vocabSize, dModel],
        /// LM head needs [dModel, vocabSize].
        /// </summary>
        private static void TransposeInto(
            ReadOnlySpan<float> src,
            Span<float> dst,
            int rows,
            int cols)
        {
            for (var r = 0; r < rows; r++)
            {
                for (var c = 0; c < cols; c++)
                {
                    dst[c * rows + r] = src[r * cols + c];
                }
            }
        }
    }
}