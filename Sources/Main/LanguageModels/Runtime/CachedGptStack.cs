// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels.Rope;
using DevOnBike.Overfit.Tensors;

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
        private readonly float _finalLogitSoftcap;   // Gemma-2 final logit soft-cap; 0 = off
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
            bool hasSharedExpert = true,
            int headDim = 0,
            float attnLogitSoftcap = 0f,
            float finalLogitSoftcap = 0f)
        {
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(layerCount);

            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(dModel);

            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(headCount);

            // Qwen3 sets head_dim explicitly (head_dim ≠ dModel/headCount); only require divisibility otherwise.
            if (headDim <= 0 && dModel % headCount != 0)
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
            HeadDimension = headDim > 0 ? headDim : dModel / headCount;
            DFF = dFF;
            VocabSize = vocabSize;
            MaxSequenceLength = maxSequenceLength;
            LayerNormEpsilon = layerNormEpsilon;
            FeedForwardActivation = feedForwardActivation;
            _finalLogitSoftcap = finalLogitSoftcap;

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
                    hasSharedExpert,
                    headDim: headDim,
                    attnLogitSoftcap: attnLogitSoftcap);
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

        public int LayerCount
        {
            get;
        }

        public int DModel
        {
            get;
        }

        public int HeadCount
        {
            get;
        }

        public int HeadDimension
        {
            get;
        }

        public int DFF
        {
            get;
        }

        public int VocabSize
        {
            get;
        }

        public int MaxSequenceLength
        {
            get;
        }

        public float LayerNormEpsilon
        {
            get;
        }

        public FeedForwardActivation FeedForwardActivation
        {
            get;
        }

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
            var profLm = DecodeProfiler.Start();
            ProjectLogits(weights, logits);
            DecodeProfiler.Stop(DecodeProfiler.Component.LmHead, profLm);
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
        /// previous <c>LastLogits</c> snapshot is left untouched.
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

            var cur = PooledBuffer<float>.RentArray(rows * DModel);
            inputHidden.Slice(0, rows * DModel).CopyTo(cur);
            var next = PooledBuffer<float>.RentArray(rows * DModel);
            try
            {

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
            finally
            {
                PooledBuffer<float>.ReturnArray(next);
                PooledBuffer<float>.ReturnArray(cur);
            }
        }

        /// <summary>
        /// Batched prefill for the Llama/Qwen quantized path — the multi-token counterpart of looping
        /// <see cref="DecodeWithoutLogits"/>, using <see cref="CachedTransformerBlock.DecodeBatchedQuant"/>
        /// (RMSNorm + RoPE + GQA + SwiGLU + quantized weights). After the call, the final-norm output of
        /// the LAST row is in <see cref="LastFinalHidden"/> / <c>_finalHidden</c>, ready for
        /// <see cref="ProjectLogits"/> (the only token whose logits a prefill needs). Bit-identical to
        /// the single-token loop. The caller must advance the cache to <c>basePosition + rows</c> first.
        /// </summary>
        internal void PrefillBatchedQuant(
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

            var cur = RunBatchedStack(inputHidden, rows, weights, cache, basePosition, rope);
            try
            {
                // Last token's hidden BEFORE final norm (matches DecodeWithoutLogits).
                var lastRow = cur.AsSpan((rows - 1) * DModel, DModel);
                lastRow.CopyTo(_lastFinalHidden);
                FinalNorm(lastRow, weights, _finalHidden);
            }
            finally
            {
                PooledBuffer<float>.ReturnArray(cur);
            }
        }

        /// <summary>
        /// Batched-prefill variant that writes the final-norm output of EVERY row into
        /// <paramref name="finalNormAllRows"/> (<c>rows × DModel</c>) — used by speculative decoding, which
        /// needs per-position logits to verify the draft. Same batched layer pass as
        /// <see cref="PrefillBatchedQuant"/>; the caller LM-heads each row via <see cref="ProjectLogitsFrom"/>.
        /// </summary>
        internal void PrefillBatchedQuantAllRows(
            ReadOnlySpan<float> inputHidden,
            int rows,
            StackWeights weights,
            KeyValueCache cache,
            int basePosition,
            Span<float> finalNormAllRows,
            RopeTable? rope = null)
        {
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(rows);
            if (finalNormAllRows.Length < (long)rows * DModel)
            {
                throw new ArgumentException("finalNormAllRows < rows*DModel.", nameof(finalNormAllRows));
            }

            var cur = RunBatchedStack(inputHidden, rows, weights, cache, basePosition, rope);
            try
            {
                for (var n = 0; n < rows; n++)
                {
                    FinalNorm(cur.AsSpan(n * DModel, DModel), weights, finalNormAllRows.Slice(n * DModel, DModel));
                }
            }
            finally
            {
                PooledBuffer<float>.ReturnArray(cur);
            }
        }

        /// <summary>Runs the batched layer pass, returning the all-rows post-stack hidden (pre-final-norm).</summary>
        /// <remarks>Returns a POOLED array (length >= rows*DModel) — the caller MUST hand it back
        /// via <c>PooledBuffer&lt;float&gt;.ReturnArray</c> (both internal callers do, in finally).</remarks>
        private float[] RunBatchedStack(
            ReadOnlySpan<float> inputHidden, int rows, StackWeights weights, KeyValueCache cache, int basePosition, RopeTable? rope)
        {
            var cur = PooledBuffer<float>.RentArray(rows * DModel);
            inputHidden.Slice(0, rows * DModel).CopyTo(cur);
            var next = PooledBuffer<float>.RentArray(rows * DModel);

            try
            {
                for (var layer = 0; layer < LayerCount; layer++)
                {
                    _blocks[layer].DecodeBatchedQuant(
                        cur, rows, in weights.Block(layer), cache, layer, basePosition, next, rope);
                    (cur, next) = (next, cur);
                }
                return cur;
            }
            catch
            {
                PooledBuffer<float>.ReturnArray(cur);
                throw;
            }
            finally
            {
                PooledBuffer<float>.ReturnArray(next);
            }
        }

        /// <summary>Final norm of one row (RMSNorm when beta is empty, else standard LayerNorm).</summary>
        private void FinalNorm(ReadOnlySpan<float> row, StackWeights weights, Span<float> dst)
        {
            if (weights.FinalNormBeta.IsEmpty)
            {
                var sumSq = 0f;
                for (var i = 0; i < DModel; i++)
                {
                    sumSq += row[i] * row[i];
                }
                var scale = 1f / MathF.Sqrt(sumSq / DModel + LayerNormEpsilon);
                if (weights.FinalNormGamma.IsEmpty)
                {
                    for (var i = 0; i < DModel; i++)
                    {
                        dst[i] = row[i] * scale;
                    }
                }
                else
                {
                    for (var i = 0; i < DModel; i++)
                    {
                        dst[i] = row[i] * scale * weights.FinalNormGamma[i];
                    }
                }
            }
            else
            {
                SingleTokenLayerNormKernel.Normalize(
                    row, weights.FinalNormGamma, weights.FinalNormBeta, dst, DModel, LayerNormEpsilon);
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
            ProjectLogitsFrom(_finalHidden, weights, logits);
            logits.Slice(0, VocabSize).CopyTo(_lastLogits);
        }

        /// <summary>
        /// LM-head projection from an arbitrary final-norm vector <paramref name="finalNorm"/> (does NOT
        /// touch <c>LastLogits</c>) — used by speculative decoding to score each draft position.
        /// Dispatches on the head's residency (Q6_K / Q4_K / Q8_0 / F32); allocation-free.
        /// </summary>
        internal void ProjectLogitsFrom(ReadOnlySpan<float> finalNorm, StackWeights weights, Span<float> logits)
        {
            var lmHead = weights.LmHeadWeights;
            if (lmHead.IsQ6K)
            {
                Q6KDotKernel.ProjectParallel(
                    finalNorm, lmHead.Quantized6K, [], logits,
                    _lmHeadQ8KQuants, _lmHeadQ8KScales, _lmHeadQ8KBsums);
            }
            else if (lmHead.IsQ4K)
            {
                Q4KDotKernel.ProjectParallel(
                    finalNorm, lmHead.Quantized4K, [], logits,
                    _lmHeadQ8KQuants, _lmHeadQ8KScales, _lmHeadQ8KBsums);
            }
            else if (lmHead.IsQuantized)
            {
                Q8DotKernel.ProjectParallel(
                    finalNorm, lmHead.Quantized, [], logits,
                    _lmHeadInputQuants, _lmHeadInputScales);
            }
            else
            {
                SingleTokenProjectionKernel.ProjectParallel(
                    finalNorm, lmHead.F32, [], logits, DModel, VocabSize);
            }

            // Gemma-2 final logit soft-cap: tanh(l/cap)·cap. Monotone, so it doesn't change greedy argmax, but it
            // reshapes the distribution for temperature/top-p sampling.
            if (_finalLogitSoftcap > 0f)
            {
                SoftcapInPlace(logits.Slice(0, VocabSize), _finalLogitSoftcap);
            }
        }

        // x ← tanh(x / cap) · cap, in place.
        internal static void SoftcapInPlace(Span<float> values, float cap)
        {
            var inv = 1f / cap;
            for (var i = 0; i < values.Length; i++)
            {
                values[i] = MathF.Tanh(values[i] * inv) * cap;
            }
        }

        /// <summary>
        /// Batched LM-head projection: <paramref name="rows"/> final-norm vectors → logits in ONE pass
        /// that reads the (large) LM-head weights from DRAM once, reusing them across all rows — used by
        /// speculative decoding so verifying N draft positions doesn't re-read the head N× (which would
        /// cancel the batched stack's weight-bandwidth saving). <paramref name="logitsAllRows"/> is
        /// <c>rows × VocabSize</c>.
        /// </summary>
        internal void ProjectLogitsBatched(
            ReadOnlySpan<float> finalNormAllRows, int rows, StackWeights weights, Span<float> logitsAllRows)
        {
            BatchedQuantProjection.Dispatch(
                finalNormAllRows, rows, weights.LmHeadWeights, [],
                logitsAllRows, DModel, VocabSize);

            if (_finalLogitSoftcap > 0f) // Gemma-2 final logit soft-cap, per row
            {
                for (var r = 0; r < rows; r++)
                {
                    SoftcapInPlace(logitsAllRows.Slice(r * VocabSize, VocabSize), _finalLogitSoftcap);
                }
            }
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


