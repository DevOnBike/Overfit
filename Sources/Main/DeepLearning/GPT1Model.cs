// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.Maths;
using DevOnBike.Overfit.Ops;
using DevOnBike.Overfit.Parameters;
using DevOnBike.Overfit.Tensors;

namespace DevOnBike.Overfit.DeepLearning
{
    /// <summary>
    /// GPT-1 style language model.
    ///
    /// Architecture:
    /// 1. Token embedding: tokens [B, T] -> [B, T, dModel]
    /// 2. Positional embedding: positions [T] -> [1, T, dModel] (learned)
    /// 3. x = tok_emb + pos_emb (broadcast add)
    /// 4. x = TransformerBlock(x) x nLayers (causal, Pre-LN)
    /// 5. x = FinalLayerNorm(x)
    /// 6. logits = x @ W_vocab^T (LM head, weight-tied)
    ///
    /// Weight tying:
    /// the LM head shares values with the token embedding. The LM head storage is
    /// re-synchronised from token embeddings before the LM projection when
    /// TieWeights=true.
    ///
    /// Forward returns logits [B, T, vocabSize].
    ///
    /// For language modelling loss:
    /// use SoftmaxCrossEntropy on logits[:, :-1, :] against targets[:, 1:, :]
    /// (next-token prediction).
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

            // LM head: [dModel, vocabSize].
            // When weight-tying: shares values with TokenEmbedding.Weight
            // transposed from [vocabSize, dModel] to [dModel, vocabSize].
            LMHead = new Parameter(new TensorShape(config.DModel, config.VocabSize), requiresGrad: !config.TieWeights, clearData: false);

            if (config.TieWeights)
            {
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
                    s[i] = MathUtils.NextGaussian() * scale;
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

            foreach (var block in Blocks)
            {
                block.Train();
            }

            FinalNorm.Train();
        }

        public void Eval()
        {
            _isTraining = false;

            TokenEmbedding.Eval();
            PositionEmbedding.Eval();

            foreach (var block in Blocks)
            {
                block.Eval();
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
            if (graph is null)
            {
                throw new ArgumentNullException(nameof(graph));
            }

            if (seqLen > _config.ContextLength)
            {
                throw new ArgumentException(
                    $"seqLen={seqLen} exceeds ContextLength={_config.ContextLength}.");
            }

            if (tokenIds.Length != batchSize * seqLen)
            {
                throw new ArgumentException(
                    $"tokenIds.Length={tokenIds.Length} must equal batchSize*seqLen={batchSize * seqLen}.");
            }

            // 1. Token embeddings.
            var tokEmb = EmbedBatch(
                graph,
                tokenIds,
                batchSize,
                seqLen,
                TokenEmbedding);

            // 2. Positional embeddings.
            var posIds = CreatePositionIds(seqLen);
            var posEmb = EmbedPositions(
                graph,
                posIds,
                batchSize,
                seqLen);

            // 3. x = token embedding + positional embedding.
            var x = AddEmbeddings(
                graph,
                tokEmb,
                posEmb,
                batchSize,
                seqLen,
                _config.DModel);

            // 4. Transformer blocks.
            foreach (var block in Blocks)
            {
                x = block.Forward(graph, x);
            }

            // 5. Final LayerNorm.
            x = FinalNorm.Forward(graph, x);

            // 6. LM head.
            return LMHeadForward(
                graph,
                x,
                batchSize,
                seqLen);
        }

        /// <summary>
        /// Generates logits for the last token position only.
        ///
        /// This is still the pre-KV-cache path: it builds a ComputationGraph,
        /// computes logits for the full provided context and copies the final
        /// position to a new float[].
        /// </summary>
        public float[] GenerateLogits(int[] tokenIds)
        {
            if (tokenIds is null)
            {
                throw new ArgumentNullException(nameof(tokenIds));
            }

            if (tokenIds.Length == 0)
            {
                throw new ArgumentException("Token sequence cannot be empty.", nameof(tokenIds));
            }

            var seqLen = tokenIds.Length;

            // 100M floats = 400MB. This is intentionally large enough for the
            // current graph-based GPT path. The stateful KV-cache runtime should
            // not use this path in the future.
            // Arena must fit LM head output [seqLen, vocabSize] which dominates for large vocabs.
            // GPT-2: seqLen=1024, vocabSize=50257 → 1024 × 50257 × 4B ≈ 206MB per token position.
            // Add 3× safety factor for attention intermediates.
            var arenaFloats = Math.Max(100_000_000, (long)tokenIds.Length * _config.VocabSize * 6);
            arenaFloats = Math.Min(arenaFloats, 2_000_000_000); // cap at 8GB

            using var graph = new ComputationGraph((int)arenaFloats);
            using var logits = Forward(graph, tokenIds, batchSize: 1, seqLen);

            var logitSpan = logits.DataView.AsReadOnlySpan();
            var vocabSize = _config.VocabSize;
            var lastLogits = new float[vocabSize];

            logitSpan.Slice((seqLen - 1) * vocabSize, vocabSize).CopyTo(lastLogits);

            return lastLogits;
        }

        /// <summary>
        /// Greedy argmax token generation for maxNewTokens steps.
        ///
        /// This is the legacy pre-KV-cache generation path. It recomputes the full
        /// context for every generated token.
        /// </summary>
        public int[] Generate(int[] promptTokenIds, int maxNewTokens)
        {
            if (promptTokenIds is null)
            {
                throw new ArgumentNullException(nameof(promptTokenIds));
            }

            if (maxNewTokens < 0)
            {
                throw new ArgumentOutOfRangeException(nameof(maxNewTokens));
            }

            var tokens = new List<int>(promptTokenIds);

            for (var step = 0; step < maxNewTokens; step++)
            {
                var ctx = tokens.Count > _config.ContextLength
                    ? tokens.GetRange(tokens.Count - _config.ContextLength, _config.ContextLength).ToArray()
                    : tokens.ToArray();

                var logits = GenerateLogits(ctx);
                var nextTok = ArgMax(logits);

                tokens.Add(nextTok);
            }

            return tokens
                .GetRange(promptTokenIds.Length, tokens.Count - promptTokenIds.Length)
                .ToArray();
        }

        public IEnumerable<Parameter> TrainableParameters()
        {
            yield return TokenEmbedding.Weight;
            yield return PositionEmbedding.Weight;

            foreach (var block in Blocks)
            {
                foreach (var parameter in block.TrainableParameters())
                {
                    yield return parameter;
                }
            }

            foreach (var parameter in FinalNorm.TrainableParameters())
            {
                yield return parameter;
            }

            if (!_config.TieWeights)
            {
                yield return LMHead;
            }
        }

        public void Save(BinaryWriter writer)
        {
            if (writer is null)
            {
                throw new ArgumentNullException(nameof(writer));
            }

            TokenEmbedding.Save(writer);
            PositionEmbedding.Save(writer);

            foreach (var block in Blocks)
            {
                block.Save(writer);
            }

            FinalNorm.Save(writer);

            if (!_config.TieWeights)
            {
                LMHead.Save(writer);
            }
        }

        public void Load(BinaryReader reader)
        {
            if (reader is null)
            {
                throw new ArgumentNullException(nameof(reader));
            }

            TokenEmbedding.Load(reader);
            PositionEmbedding.Load(reader);

            foreach (var block in Blocks)
            {
                block.Load(reader);
            }

            FinalNorm.Load(reader);

            if (!_config.TieWeights)
            {
                LMHead.Load(reader);
            }
            else
            {
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

            foreach (var block in Blocks)
            {
                block.InvalidateParameterCaches();
            }

            FinalNorm.InvalidateParameterCaches();

            _lmHeadNode?.Dispose();
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

            foreach (var block in Blocks)
            {
                block.Dispose();
            }

            FinalNorm.Dispose();
            LMHead.Dispose();
        }

        private static AutogradNode EmbedBatch(
            ComputationGraph graph,
            int[] tokenIds,
            int batchSize,
            int seqLen,
            EmbeddingLayer embedding)
        {
            var dModel = embedding.EmbeddingDim;

            // Single embedding forward for all B*T tokens.
            // Produces [B*T, dModel] on tape.
            var embNode = embedding.Forward(graph, tokenIds);

            return graph.Reshape(
                embNode,
                batchSize,
                seqLen,
                dModel);
        }

        private AutogradNode EmbedPositions(
            ComputationGraph graph,
            int[] posIds,
            int batchSize,
            int seqLen)
        {
            var allPosIds = new int[batchSize * seqLen];

            for (var b = 0; b < batchSize; b++)
            {
                posIds
                    .AsSpan()
                    .CopyTo(allPosIds.AsSpan(b * seqLen, seqLen));
            }

            var posNode = PositionEmbedding.Forward(graph, allPosIds);

            return graph.Reshape(
                posNode,
                batchSize,
                seqLen,
                _config.DModel);
        }

        private static AutogradNode AddEmbeddings(
            ComputationGraph graph,
            AutogradNode tokenEmbedding,
            AutogradNode positionalEmbedding,
            int batchSize,
            int seqLen,
            int dModel)
        {
            _ = batchSize;
            _ = seqLen;
            _ = dModel;

            return TensorMath.Add(
                graph,
                tokenEmbedding,
                positionalEmbedding);
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

            // Flatten [B, T, dModel] -> [B*T, dModel].
            var flat = graph.Reshape(
                x,
                batchSize * seqLen,
                dModel);

            // LM head on tape:
            // graph.Linear(input[B*T,d], weight[d,V], bias[V]) -> [B*T,V].
            //
            // This bias is a graph-owned zero tensor. The previous implementation
            // used new TensorStorage(...) + AutogradNode.CreateBorrowed(...), which
            // marked locally-created storage as externally-owned. That made graph
            // ownership metadata incorrect and prevented Reset() from disposing it.
            var bias = graph.CreateAuxiliary(
                new TensorShape(vocabSize),
                clearMemory: true);

            _lmHeadNode ??= LMHead.AsNode();

            var flatLogits = graph.Linear(
                flat,
                _lmHeadNode,
                bias);

            return graph.Reshape(
                flatLogits,
                batchSize,
                seqLen,
                vocabSize);
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
        /// Transposes [rows, cols] to [cols, rows] in place.
        ///
        /// Used for weight tying:
        /// token embedding is [vocabSize, dModel],
        /// LM head needs [dModel, vocabSize].
        /// </summary>
        private static void TransposeInto(
            ReadOnlySpan<float> source,
            Span<float> destination,
            int rows,
            int cols)
        {
            for (var r = 0; r < rows; r++)
            {
                for (var c = 0; c < cols; c++)
                {
                    destination[c * rows + r] = source[r * cols + c];
                }
            }
        }
    }
}
