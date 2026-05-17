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

        public CachedGptStack(
            int layerCount,
            int dModel,
            int headCount,
            int dFF,
            int vocabSize,
            int maxSequenceLength,
            float layerNormEpsilon = 1e-5f,
            FeedForwardActivation feedForwardActivation = FeedForwardActivation.GeLU,
            int kvHeadCount = 0)
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
                    kvHeadCount > 0 ? kvHeadCount : headCount);
            }

            _currentHidden = new float[dModel];
            _nextHidden = new float[dModel];
            _finalHidden = new float[dModel];
            _lastFinalHidden = new float[dModel];
            _lastLogits = new float[vocabSize];
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
        /// Projects the saved <see cref="LastFinalHidden"/>-norm output into
        /// vocabulary logits. Call this once per token-of-interest after one
        /// or more <see cref="DecodeWithoutLogits"/> calls. The standard
        /// <see cref="Decode"/> entry point does this automatically.
        /// </summary>
        private void ProjectLogits(StackWeights weights, Span<float> logits)
        {
            // LM head: [DModel] × [DModel × VocabSize] → [VocabSize].
            //
            // Stays sequential by design — ProjectParallel exists but allocates
            // ~3 KB per call from Parallel.For task scheduling, which breaks the
            // 0 B / generated token contract validated by
            // Gpt2GenerationDemoTests.Demo_Gpt2Small_KvCacheDecode_AllocatesZeroBytesPerToken.
            // Speedup on GPT-2 Small was only ~3 % at the per-token level (Parallel.For
            // overhead dominates vs the ~3.8 ms LM-head matmul), nowhere near the
            // ~10× implied by the steady-state LmHeadParallelBenchmark.
            //
            // A true win here needs an allocation-free worker pool — tracked in
            // ROADMAP under "LM head hot-path: allocation-free parallel matmul".
            SingleTokenProjectionKernel.Project(
                _finalHidden,
                weights.LmHeadWeights,
                ReadOnlySpan<float>.Empty,
                logits,
                DModel,
                VocabSize);

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


