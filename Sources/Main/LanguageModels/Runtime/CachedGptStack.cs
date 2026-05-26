// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels.Rope;

namespace DevOnBike.Overfit.LanguageModels.Runtime
{
    /// <summary>
    /// Cached GPT-style transformer stack for single-token autoregressive decode.
    /// </summary>
    public class CachedGptStack
    {
        private readonly CachedTransformerBlock[] _blocks;
        private readonly float[] _currentHidden;
        private readonly float[] _nextHidden;
        private readonly float[] _finalHidden;
        private readonly float[] _lastFinalHidden;  // hidden BEFORE final norm
        private readonly float[] _lastLogits;
        private readonly sbyte[] _lmHeadInputQuants;   // Q8 LM-head activation scratch
        private readonly float[] _lmHeadInputScales;
        private readonly sbyte[] _lmHeadQ8KQuants;     // Q4_K LM-head activation scratch (Q8_K-quantized)
        private readonly float[] _lmHeadQ8KScales;
        private readonly short[] _lmHeadQ8KBsums;

        public CachedGptStack(
            int layerCount,
            int dModel,
            int headCount,
            int dFF,
            int vocabSize,
            int maxSequenceLength,
            float layerNormEpsilon = 1e-5f,
            FeedForwardActivation feedForwardActivation = FeedForwardActivation.GeLU,
            int kvHeadCount = 0,
            int expertCount = 0,
            int expertUsedCount = 0,
            int expertFeedForwardLength = 0,
            bool normalizeExpertWeights = true,
            bool hasSharedExpert = true)
        {
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(layerCount);

            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(dModel);

            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(headCount);

            if (dModel % headCount != 0)
            {
                throw new ArgumentException(
                    $"dModel ({dModel}) must be divisible by headCount ({headCount}).",
                    nameof(dModel));
            }

            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(dFF);

            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(vocabSize);

            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(maxSequenceLength);

            ArgumentOutOfRangeException.ThrowIfLessThanOrEqual(layerNormEpsilon, 0f);

            LayerCount = layerCount;
            DModel = dModel;
            HeadCount = headCount;
            HeadDimension = dModel / headCount;
            DFF = dFF;
            VocabSize = vocabSize;
            MaxSequenceLength = maxSequenceLength;
            LayerNormEpsilon = layerNormEpsilon;
            FeedForwardActivation = feedForwardActivation;

            _blocks = new CachedTransformerBlock[layerCount];

            for (var layer = 0; layer < layerCount; layer++)
            {
                _blocks[layer] = new CachedTransformerBlock(
                    dModel,
                    headCount,
                    dFF,
                    maxSequenceLength,
                    layerNormEpsilon,
                    feedForwardActivation,
                    kvHeadCount > 0 ? kvHeadCount : headCount,
                    expertCount,
                    expertUsedCount,
                    expertFeedForwardLength,
                    normalizeExpertWeights,
                    hasSharedExpert);
            }

            _currentHidden = new float[dModel];
            _nextHidden = new float[dModel];
            _finalHidden = new float[dModel];
            _lastFinalHidden = new float[dModel];
            _lastLogits = new float[vocabSize];
            _lmHeadInputQuants = new sbyte[dModel];
            _lmHeadInputScales = new float[(dModel + Q8DotKernel.BlockSize - 1) / Q8DotKernel.BlockSize];
            _lmHeadQ8KQuants = new sbyte[dModel];
            _lmHeadQ8KScales = new float[(dModel + Q4KDotKernel.SuperBlockElements - 1) / Q4KDotKernel.SuperBlockElements];
            _lmHeadQ8KBsums = new short[(dModel + Q4KDotKernel.GroupSize - 1) / Q4KDotKernel.GroupSize];
        }

        public int LayerCount { get; }

        public int DModel { get; }

        public int HeadCount { get; }

        public int HeadDimension { get; }

        public int DFF { get; }

        public int VocabSize { get; }

        public int MaxSequenceLength { get; }

        public float LayerNormEpsilon { get; }

        public FeedForwardActivation FeedForwardActivation { get; }

        /// <summary>
        /// Decodes one token through all transformer layers + LM head using KV-cache.
        /// Zero allocations — all weights accessed via StackWeights references.
        /// </summary>
        internal void Decode(
            ReadOnlySpan<float> inputHidden,
            StackWeights weights,
            KeyValueCache cache,
            int position,
            Span<float> logits,
            RopeTable? rope = null)
        {
            if (logits.Length < VocabSize)
            {
                throw new ArgumentException($"logits length {logits.Length} < VocabSize {VocabSize}.");
            }

            DecodeWithoutLogits(inputHidden, weights, cache, position, rope);
            ProjectLogits(weights, logits);
        }

        /// <summary>
        /// Decodes one token through all transformer layers + final norm using
        /// KV-cache, but skips the LM-head projection. Used by the prefill phase
        /// for every prompt token except the last — those intermediate logits
        /// would be computed only to be immediately overwritten by the next
        /// decode, so skipping the LM head (~27 % of per-token decode cost on
        /// GPT-2 Small) gives free prefill speedup.
        ///
        /// After this call, <see cref="LastFinalHidden"/> is updated; the
        /// previous <see cref="LastLogits"/> snapshot is left untouched.
        /// </summary>
        internal void DecodeWithoutLogits(
            ReadOnlySpan<float> inputHidden,
            StackWeights weights,
            KeyValueCache cache,
            int position,
            RopeTable? rope = null)
        {
            if (inputHidden.Length < DModel)
            {
                throw new ArgumentException($"inputHidden length {inputHidden.Length} < DModel {DModel}.");
            }

            inputHidden.Slice(0, DModel).CopyTo(_currentHidden);

            var current = _currentHidden;
            var next = _nextHidden;

            for (var layer = 0; layer < LayerCount; layer++)
            {
                _blocks[layer].Decode(
                    current,
                    in weights.Block(layer),
                    cache,
                    layerIndex: layer,
                    position,
                    next,
                    rope);

                (current, next) = (next, current);
            }

            // Save hidden state BEFORE final norm.
            // LastFinalHidden matches Python: x before rms_norm(x, fg2, eps).
            new ReadOnlySpan<float>(current, 0, DModel).CopyTo(_lastFinalHidden);

            if (weights.FinalNormBeta.IsEmpty)
            {
                // RMSNorm (Llama/Qwen/Mistral)
                var sumSq = 0f;
                for (var i = 0; i < DModel; i++)
                {
                    sumSq += current[i] * current[i];
                }
                var scale = 1f / MathF.Sqrt(sumSq / DModel + LayerNormEpsilon);
                if (weights.FinalNormGamma.IsEmpty)
                {
                    for (var i = 0; i < DModel; i++)
                    {
                        _finalHidden[i] = current[i] * scale;
                    }
                }
                else
                {
                    for (var i = 0; i < DModel; i++)
                    {
                        _finalHidden[i] = current[i] * scale * weights.FinalNormGamma[i];
                    }
                }
            }
            else
            {
                SingleTokenLayerNormKernel.Normalize(current, weights.FinalNormGamma, weights.FinalNormBeta, _finalHidden, DModel, LayerNormEpsilon);
            }
        }

        /// <summary>
        /// Batched prefill (Phase 3): runs <paramref name="rows"/> prompt tokens
        /// through all transformer layers + final norm in one pass — the multi-token
        /// counterpart of looping <see cref="DecodeWithoutLogits"/>. Each layer uses
        /// <see cref="CachedTransformerBlock.DecodeBatched"/> (batched MHA + FFN), so
        /// the result is <b>bit-identical</b> to the single-token loop. After the call
        /// <see cref="LastFinalHidden"/> and the final-norm output hold the LAST
        /// token's state, ready for <see cref="ProjectLogits"/> (the only token whose
        /// logits a prefill needs). Scoped to the F32 / GPT-2 path (standard LayerNorm,
        /// GeLU/ReLU FFN, MHA, no RoPE — the blocks throw otherwise). The caller must
        /// advance the cache to length <c>basePosition + rows</c> before calling.
        /// </summary>
        internal void PrefillBatched(
            ReadOnlySpan<float> inputHidden,
            int rows,
            StackWeights weights,
            KeyValueCache cache,
            int basePosition,
            RopeTable? rope = null)
        {
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(rows);

            if (inputHidden.Length < (long)rows * DModel)
            {
                throw new ArgumentException(
                    $"inputHidden length {inputHidden.Length} < rows*DModel {(long)rows * DModel}.", nameof(inputHidden));
            }

            var cur = new float[rows * DModel];
            inputHidden.Slice(0, rows * DModel).CopyTo(cur);
            var next = new float[rows * DModel];

            for (var layer = 0; layer < LayerCount; layer++)
            {
                _blocks[layer].DecodeBatched(
                    cur, rows, in weights.Block(layer), cache, layer, basePosition, next, rope);

                (cur, next) = (next, cur);
            }

            // Last token's hidden BEFORE final norm (matches DecodeWithoutLogits).
            var lastRow = cur.AsSpan((rows - 1) * DModel, DModel);
            lastRow.CopyTo(_lastFinalHidden);

            if (weights.FinalNormBeta.IsEmpty)
            {
                var sumSq = 0f;
                for (var i = 0; i < DModel; i++)
                {
                    sumSq += lastRow[i] * lastRow[i];
                }
                var scale = 1f / MathF.Sqrt(sumSq / DModel + LayerNormEpsilon);
                if (weights.FinalNormGamma.IsEmpty)
                {
                    for (var i = 0; i < DModel; i++)
                    {
                        _finalHidden[i] = lastRow[i] * scale;
                    }
                }
                else
                {
                    for (var i = 0; i < DModel; i++)
                    {
                        _finalHidden[i] = lastRow[i] * scale * weights.FinalNormGamma[i];
                    }
                }
            }
            else
            {
                SingleTokenLayerNormKernel.Normalize(
                    lastRow, weights.FinalNormGamma, weights.FinalNormBeta, _finalHidden, DModel, LayerNormEpsilon);
            }
        }

        /// <summary>
        /// Projects the saved <see cref="LastFinalHidden"/>-norm output into
        /// vocabulary logits. Call this once per token-of-interest after one
        /// or more <see cref="DecodeWithoutLogits"/> calls. The standard
        /// <see cref="Decode"/> entry point does this automatically.
        /// </summary>
        internal void ProjectLogits(StackWeights weights, Span<float> logits)
        {
            // LM head: [DModel] → [VocabSize]. Dispatches on the weight's
            // residency — Q8_0-quantized (the GGUF/Qwen path, step 2.3a) or F32.
            // Both projection paths run on the allocation-free OverfitParallelFor
            // pool, so the 0 B / generated token contract (validated by
            // Gpt2GenerationDemoTests.Demo_Gpt2Small_KvCacheDecode_AllocatesZeroBytesPerToken)
            // still holds.
            var lmHead = weights.LmHeadWeights;
            if (lmHead.IsQ6K)
            {
                // Q6_K-resident LM head (step 3.3c — wires the Q6_K kernel).
                Q6KDotKernel.ProjectParallel(
                    _finalHidden,
                    lmHead.Quantized6K,
                    ReadOnlySpan<float>.Empty,
                    logits,
                    _lmHeadQ8KQuants,
                    _lmHeadQ8KScales,
                    _lmHeadQ8KBsums);
            }
            else if (lmHead.IsQ4K)
            {
                // Q4_K-resident LM head (step 3.2a — wires the Q4_K kernel).
                Q4KDotKernel.ProjectParallel(
                    _finalHidden,
                    lmHead.Quantized4K,
                    ReadOnlySpan<float>.Empty,
                    logits,
                    _lmHeadQ8KQuants,
                    _lmHeadQ8KScales,
                    _lmHeadQ8KBsums);
            }
            else if (lmHead.IsQuantized)
            {
                Q8DotKernel.ProjectParallel(
                    _finalHidden,
                    lmHead.Quantized,
                    ReadOnlySpan<float>.Empty,
                    logits,
                    _lmHeadInputQuants,
                    _lmHeadInputScales);
            }
            else
            {
                SingleTokenProjectionKernel.ProjectParallel(
                    _finalHidden,
                    lmHead.F32,
                    ReadOnlySpan<float>.Empty,
                    logits,
                    DModel,
                    VocabSize);
            }

            logits.Slice(0, VocabSize).CopyTo(_lastLogits);
        }

        // Validation helpers removed — StackWeights guarantees correct dimensions
        // by construction (bound directly to GPT1Model parameters).



        /// <summary>
        /// Hidden state AFTER all transformer layers, BEFORE final RMSNorm.
        /// Matches Python forward_multitoken.py: x before rms_norm(x, fg2, eps).
        /// </summary>
        internal ReadOnlySpan<float> LastFinalHidden => _lastFinalHidden.AsSpan(0, DModel);

        public void GetLastFinalHidden(Span<float> destination)
            => _finalHidden.AsSpan(0, DModel).CopyTo(destination);

        public void GetLastLogits(Span<float> destination)
            => _lastLogits.AsSpan(0, VocabSize).CopyTo(destination);

        /// <summary>Exposes internal blocks for testing.</summary>
        internal CachedTransformerBlock[] Blocks => _blocks;

        /// <summary>Returns the block at <paramref name="layerIndex"/> for testing.</summary>
        public CachedTransformerBlock GetBlock(int layerIndex)
        {
            if ((uint)layerIndex >= (uint)_blocks.Length)
            {
                throw new ArgumentOutOfRangeException(nameof(layerIndex));
            }
            return _blocks[layerIndex];
        }
    }
}


