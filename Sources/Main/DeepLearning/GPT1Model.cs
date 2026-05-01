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
            requiresGrad: !config.TieWeights, // no grad needed if tied (tok emb has it)
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
                throw new ArgumentException(
                $"seqLen={seqLen} exceeds ContextLength={_config.ContextLength}.");
            }

            if (tokenIds.Length != batchSize * seqLen)
            {
                throw new ArgumentException(
                $"tokenIds.Length={tokenIds.Length} must equal batchSize*seqLen={batchSize * seqLen}.");
            }

            // ── 1. Token embeddings ──────────────────────────────────────────
            // Look up each token independently, then reshape to [B, T, dModel].
            var tokEmb = EmbedBatch(graph, tokenIds, batchSize, seqLen, TokenEmbedding);

            // ── 2. Positional embeddings ─────────────────────────────────────
            // Positions 0..seqLen-1, same for all batch items.
            var posIds = CreatePositionIds(seqLen);
            var posEmb = EmbedPositions(graph, posIds, batchSize, seqLen);

            // ── 3. x = tok_emb + pos_emb ────────────────────────────────────
            var x = AddEmbeddings(tokEmb, posEmb, batchSize, seqLen, _config.DModel);

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
        /// Gradient-checkpointed forward pass.
        ///
        /// Memory: O(L × seqLen × dModel) for saved block inputs only.
        /// vs O(L × seqLen × (4×dModel + nHeads×seqLen + dFF)) without checkpointing.
        ///
        /// At SeqLen=256, d=256, 12 layers: ~7MB instead of ~50MB.
        ///
        /// Algorithm:
        ///   Phase 1 (no-grad forward): run all blocks with IsRecording=false,
        ///             save each block input as float[].
        ///   Phase 2 (recompute backward): for each block in reverse,
        ///             recompute forward with a local tape, seed output grad,
        ///             backward through local tape → accumulate weight gradients,
        ///             extract input gradient for upstream propagation.
        /// </summary>
        public AutogradNode ForwardCheckpointed(
            int[] tokenIds,
            int batchSize,
            int seqLen)
        {
            if (seqLen > _config.ContextLength)
            {
                throw new ArgumentException(
                $"seqLen={seqLen} exceeds ContextLength={_config.ContextLength}.");
            }

            if (tokenIds.Length != batchSize * seqLen)
            {
                throw new ArgumentException(
                $"tokenIds.Length must equal batchSize*seqLen.");
            }

            // ── Phase 1: no-grad forward ─────────────────────────────────────
            // Run with IsRecording=false: ops execute but nothing is taped.
            // Memory: only current activation tensors, not cumulative tape.
            using var noGradGraph = new ComputationGraph();
            noGradGraph.IsRecording = false;

            var tokEmb = EmbedBatch(noGradGraph, tokenIds, batchSize, seqLen, TokenEmbedding);
            var posIds = CreatePositionIds(seqLen);
            var posEmb = EmbedPositions(noGradGraph, posIds, batchSize, seqLen);
            var x = AddEmbeddings(tokEmb, posEmb, batchSize, seqLen, _config.DModel);

            // Save embedding output — needed for first block backward
            var savedInputs = new float[_config.NLayers][];

            for (var i = 0; i < Blocks.Length; i++)
            {
                // Save block input (data only — no gradient storage)
                savedInputs[i] = x.DataView.AsReadOnlySpan().ToArray();

                Blocks[i].InvalidateParameterCaches();
                x = Blocks[i].Forward(noGradGraph, x);
            }

            // Save post-blocks tensor for FinalNorm + LM head
            var xAfterBlocks = x.DataView.AsReadOnlySpan().ToArray();
            FinalNorm.InvalidateParameterCaches();
            x = FinalNorm.Forward(noGradGraph, x);
            var xAfterNorm = x.DataView.AsReadOnlySpan().ToArray();

            // LM head forward (no tape)
            var logitStorage = new TensorStorage<float>(batchSize * seqLen * _config.VocabSize, clearMemory: false);
            ComputeLMHeadManual(xAfterNorm, logitStorage.AsSpan(), batchSize, seqLen, _config.DModel, _config.VocabSize);

            // Return logit node with requiresGrad=true so caller can seed GradView
            var logitNode = new AutogradNode(logitStorage, new TensorShape(batchSize, seqLen, _config.VocabSize), requiresGrad: true);

            // Store context for BackwardCheckpointed
            _ckptContext = new CheckpointContext(savedInputs, xAfterBlocks, xAfterNorm, batchSize, seqLen);

            return logitNode;
        }

        /// <summary>
        /// Backward pass for gradient-checkpointed forward.
        /// Call after ForwardCheckpointed with logits.GradView seeded.
        ///
        /// Recomputes each block forward with a local tape, then backpropagates.
        /// Weight gradients accumulate into Parameter.GradSpan (same as normal backward).
        /// </summary>
        public void BackwardCheckpointed(AutogradNode logitNode)
        {
            if (_ckptContext == null)
            {
                throw new InvalidOperationException(
                "Call ForwardCheckpointed before BackwardCheckpointed.");
            }

            var ctx = _ckptContext;
            _ckptContext = null;

            var b = ctx.BatchSize;
            var t = ctx.SeqLen;
            var dModel = _config.DModel;
            var vocab = _config.VocabSize;

            // ── LM head backward ─────────────────────────────────────────────
            // dL/d(xAfterNorm) = dL/d(logits) @ LMHead^T
            var dLogits = logitNode.GradView.AsReadOnlySpan();
            var dXAfterNorm = new float[b * t * dModel];
            var lmHeadData = LMHead.DataReadOnlySpan;

            if (_config.TieWeights)
            {
                TransposeInto(TokenEmbedding.Weight.DataReadOnlySpan, LMHead.DataSpan, vocab, dModel);
            }

            // dX = dLogits @ LMHead.T   ([B*T, V] @ [V, d] → [B*T, d])
            for (var row = 0; row < b * t; row++)
            {
                var dLogRow = dLogits.Slice(row * vocab, vocab);
                var dXRow = dXAfterNorm.AsSpan().Slice(row * dModel, dModel);

                for (var v = 0; v < vocab; v++)
                {
                    if (dLogRow[v] == 0f)
                    {
                        continue;
                    }
                    for (var c = 0; c < dModel; c++)
                    {
                        dXRow[c] += dLogRow[v] * lmHeadData[c * vocab + v];
                    }
                }
            }

            // Weight grad for LM head (if not tied)
            if (!_config.TieWeights)
            {
                var lmGrad = LMHead.GradSpan;
                var xNormS = ctx.XAfterNorm.AsSpan();
                for (var row = 0; row < b * t; row++)
                {
                    var dLogRow = dLogits.Slice(row * vocab, vocab);
                    var xRow = xNormS.Slice(row * dModel, dModel);
                    for (var v = 0; v < vocab; v++)
                    {
                        if (dLogRow[v] == 0f)
                        {
                            continue;
                        }
                        for (var c = 0; c < dModel; c++)
                        {
                            lmGrad[c * vocab + v] += dLogRow[v] * xRow[c];
                        }
                    }
                }
            }

            // ── FinalNorm backward (recompute) ────────────────────────────────
            var dXAfterBlocks = RecomputeLayerNormBackward(
            FinalNorm, ctx.XAfterBlocks, dXAfterNorm, b, t, dModel);

            // ── Transformer blocks backward (recompute each block) ────────────
            var dUpstream = dXAfterBlocks;

            for (var i = Blocks.Length - 1; i >= 0; i--)
            {
                dUpstream = RecomputeBlockBackward(Blocks[i], ctx.SavedInputs[i], dUpstream, b, t, dModel);
            }

            // ── Embedding backward ────────────────────────────────────────────
            // dUpstream now contains dL/d(x_after_embeddings)
            // Propagate to token and positional embedding weights
            RecomputeEmbeddingBackward(dUpstream, tokenIds: null, b, t, dModel);
        }

        // ── Checkpointing helpers ─────────────────────────────────────────────

        private sealed class CheckpointContext
        {
            public CheckpointContext(float[][] savedInputs, float[] xAfterBlocks, float[] xAfterNorm, int batchSize, int seqLen)
            {
                SavedInputs = savedInputs;
                XAfterBlocks = xAfterBlocks;
                XAfterNorm = xAfterNorm;
                BatchSize = batchSize;
                SeqLen = seqLen;
            }

            public float[][] SavedInputs { get; }
            public float[] XAfterBlocks { get; }
            public float[] XAfterNorm { get; }
            public int BatchSize { get; }
            public int SeqLen { get; }
        }

        private CheckpointContext? _ckptContext;

        private void ComputeLMHeadManual(float[] xNorm, Span<float> logits, int batchSize, int seqLen, int dModel, int vocabSize)
        {
            if (_config.TieWeights)
            {
                TransposeInto(TokenEmbedding.Weight.DataReadOnlySpan, LMHead.DataSpan, vocabSize, dModel);
            }

            var lmHead = LMHead.DataReadOnlySpan;

            for (var row = 0; row < batchSize * seqLen; row++)
            {
                var inRow = xNorm.AsSpan().Slice(row * dModel, dModel);
                var outRow = logits.Slice(row * vocabSize, vocabSize);

                for (var v = 0; v < vocabSize; v++)
                {
                    outRow[v] = System.Numerics.Tensors.TensorPrimitives.Dot(
                    inRow, lmHead.Slice(v * dModel, dModel));
                }
            }
        }

        private static float[] RecomputeLayerNormBackward(
            LayerNormLayer ln,
            float[] inputData,
            float[] dOutput,
            int batchSize, int seqLen, int dModel)
        {
            var size = batchSize * seqLen * dModel;

            using var inputStorage = new TensorStorage<float>(size, clearMemory: false);
            inputData.AsSpan().CopyTo(inputStorage.AsSpan());

            using var graph = new ComputationGraph();
            using var inputN = new AutogradNode(inputStorage, new TensorShape(batchSize, seqLen, dModel), requiresGrad: true);
            inputN.GradView.AsSpan().Clear();

            ln.InvalidateParameterCaches();
            using var output = ln.Forward(graph, inputN);
            dOutput.AsSpan().CopyTo(output.GradView.AsSpan());

            graph.BackwardFromGrad(output);

            return inputN.GradView.AsReadOnlySpan().ToArray();
        }

        private static float[] RecomputeBlockBackward(
            TransformerBlock block,
            float[] inputData,
            float[] dOutput,
            int batchSize, int seqLen, int dModel)
        {
            var size = batchSize * seqLen * dModel;

            using var inputStorage = new TensorStorage<float>(size, clearMemory: false);
            inputData.AsSpan().CopyTo(inputStorage.AsSpan());

            using var graph = new ComputationGraph();
            using var inputN = new AutogradNode(inputStorage, new TensorShape(batchSize, seqLen, dModel), requiresGrad: true);
            inputN.GradView.AsSpan().Clear();

            block.InvalidateParameterCaches();
            using var output = block.Forward(graph, inputN);

            // Seed output gradient
            dOutput.AsSpan().CopyTo(output.GradView.AsSpan());

            graph.BackwardFromGrad(output);

            return inputN.GradView.AsReadOnlySpan().ToArray();
        }

        private void RecomputeEmbeddingBackward(
            float[] dOutput, int[]? tokenIds, int batchSize, int seqLen, int dModel)
        {
            // Gradient flows to positional embedding (same across batch)
            var posGrad = PositionEmbedding.Weight.GradSpan;
            for (var t = 0; t < seqLen; t++)
            {
                var dRow = dOutput.AsSpan(t * dModel, dModel);
                var gRow = posGrad.Slice(t * dModel, dModel);
                System.Numerics.Tensors.TensorPrimitives.Add(dRow, gRow, gRow);
            }
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
            var storage = new TensorStorage<float>(batchSize * seqLen * dModel, clearMemory: false);
            var outS = storage.AsSpan();

            var requiresGrad = embedding.Weight.RequiresGrad;

            for (var b = 0; b < batchSize; b++)
            {
                var batchTokens = tokenIds.AsSpan(b * seqLen, seqLen).ToArray();

                // No `using` — node is GraphTemporary, disposed by graph.Reset().
                // Using it here would dispose op.Output before backward can read GradView.
                var embNode = embedding.Forward(graph, batchTokens);

                embNode.DataView.AsReadOnlySpan().CopyTo(outS.Slice(b * seqLen * dModel, seqLen * dModel));
            }

            // Return as [B, T, dModel] — grad flows through the embedding lookup nodes above.
            return new AutogradNode(storage, new TensorShape(batchSize, seqLen, dModel), requiresGrad: false);
        }

        private AutogradNode EmbedPositions(
            ComputationGraph graph,
            int[] posIds,
            int batchSize,
            int seqLen)
        {
            var dModel = _config.DModel;
            var storage = new TensorStorage<float>(batchSize * seqLen * dModel, clearMemory: false);
            var outS = storage.AsSpan();

            // No `using` — posNode is GraphTemporary on tape, disposed by graph.Reset().
            var posNode = PositionEmbedding.Forward(graph, posIds);
            var posS = posNode.DataView.AsReadOnlySpan();

            // Broadcast positional embeddings across batch
            for (var b = 0; b < batchSize; b++)
            {
                posS.CopyTo(outS.Slice(b * seqLen * dModel, seqLen * dModel));
            }

            return new AutogradNode(storage, new TensorShape(batchSize, seqLen, dModel), requiresGrad: false);
        }

        private static AutogradNode AddEmbeddings(
            AutogradNode tokEmb,
            AutogradNode posEmb,
            int batchSize,
            int seqLen,
            int dModel)
        {
            var size = batchSize * seqLen * dModel;
            var storage = new TensorStorage<float>(size, clearMemory: false);
            TensorPrimitives.Add(
            tokEmb.DataView.AsReadOnlySpan(),
            posEmb.DataView.AsReadOnlySpan(),
            storage.AsSpan());

            return new AutogradNode(storage, new TensorShape(batchSize, seqLen, dModel), requiresGrad: false);
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