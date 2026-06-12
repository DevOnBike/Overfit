// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.Tensors;

namespace DevOnBike.Overfit.LanguageModels.Runtime
{
    /// <summary>
    /// The routed-or-shared MoE feed-forward block. In the Qwen2-MoE shape it is a routed
    /// Mixture-of-Experts <b>plus</b> a sigmoid-gated shared expert that runs for every token:
    /// <code>FFN(x) = σ(w_shared · x) · shared(x) + Σ_{i∈top-k} weightᵢ · expertᵢ(x)</code>
    /// In the Mixtral shape there is no shared expert (pass <c>sharedFeedForwardLength = 0</c>) and the
    /// block reduces to the routed sum alone:
    /// <code>FFN(x) = Σ_{i∈top-k} weightᵢ · expertᵢ(x)</code>
    /// The routed sum is delegated to <see cref="MoeFeedForwardBlock"/>; the shared expert (when
    /// present) is an ordinary SwiGLU FFN (its own, larger dFF) scaled by the sigmoid of a single
    /// learned dot.
    ///
    /// All weights are <see cref="DecodeWeight"/>s (F32 / Q8_0 / Q4_K / Q5_0 / Q5_K / Q6_K —
    /// mmap-able); the router projections (`ffn_gate_inp`, and `ffn_gate_inp_shexp` when shared) stay
    /// F32. Zero-allocation per call.
    /// </summary>
    public sealed class Qwen2MoeFeedForwardBlock
    {
        private readonly MoeFeedForwardBlock _routed;
        private readonly CachedFeedForwardBlock? _shared;   // null ⇒ Mixtral (routed-only)
        private readonly float[]? _routedOut;
        private readonly float[]? _sharedOut;

        public Qwen2MoeFeedForwardBlock(
            int dModel, int expertFeedForwardLength, int sharedFeedForwardLength,
            int expertCount, int expertUsedCount, bool normalizeExpertWeights = true)
        {
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(dModel);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(expertFeedForwardLength);
            ArgumentOutOfRangeException.ThrowIfNegative(sharedFeedForwardLength);

            DModel = dModel;
            _routed = new MoeFeedForwardBlock(dModel, expertFeedForwardLength, expertCount, expertUsedCount, normalizeExpertWeights);

            // sharedFeedForwardLength == 0 ⇒ Mixtral (no shared expert); skip its buffers entirely.
            if (sharedFeedForwardLength > 0)
            {
                _shared = new CachedFeedForwardBlock(dModel, sharedFeedForwardLength, FeedForwardActivation.SwiGLU);
                _routedOut = new float[dModel];
                _sharedOut = new float[dModel];
            }
        }

        public int DModel { get; }
        public bool HasSharedExpert => _shared is not null;
        public int ExpertCount => _routed.ExpertCount;
        public int ExpertUsedCount => _routed.ExpertUsedCount;

        /// <summary>
        /// Computes the Qwen2-MoE FFN for one token into <paramref name="output"/>.
        /// <paramref name="routerWeight"/> / <paramref name="sharedGateWeight"/> are F32 router
        /// matrices (<c>ffn_gate_inp</c> <c>[dModel × ExpertCount]</c>, <c>ffn_gate_inp_shexp</c>
        /// <c>[dModel]</c>); the expert arrays are the routed experts; <paramref name="sharedGate"/>/
        /// <paramref name="sharedUp"/>/<paramref name="sharedDown"/> are the shared expert's SwiGLU
        /// weights.
        /// </summary>
        public void Decode(
            ReadOnlySpan<float> hidden,
            ReadOnlySpan<float> routerWeight,
            DecodeWeight[] gateExperts,
            DecodeWeight[] upExperts,
            DecodeWeight[] downExperts,
            ReadOnlySpan<float> sharedGateWeight,
            in DecodeWeight sharedGate,
            in DecodeWeight sharedUp,
            in DecodeWeight sharedDown,
            Span<float> output)
        {
            if (hidden.Length < DModel) { throw new ArgumentException("Hidden span smaller than dModel.", nameof(hidden)); }
            if (output.Length < DModel) { throw new ArgumentException("Output span smaller than dModel.", nameof(output)); }

            // Mixtral (no shared expert): the FFN is the routed sum alone — write it straight out.
            if (_shared is null)
            {
                _routed.Decode(hidden, routerWeight, gateExperts, upExperts, downExperts, output);
                return;
            }

            if (sharedGateWeight.Length < DModel)
            {
                throw new ArgumentException("Shared gate weight smaller than dModel.", nameof(sharedGateWeight));
            }

            // Routed experts → _routedOut.
            _routed.Decode(hidden, routerWeight, gateExperts, upExperts, downExperts, _routedOut);

            // Shared expert, gated by sigmoid(w_shared · x).
            var gateLogit = 0f;
            for (var d = 0; d < DModel; d++) { gateLogit += hidden[d] * sharedGateWeight[d]; }
            var gate = 1f / (1f + MathF.Exp(-gateLogit));

            _shared.DecodeSwiGluDispatched(hidden, in sharedGate, in sharedUp, in sharedDown, _sharedOut);

            for (var d = 0; d < DModel; d++)
            {
                output[d] = (gate * _sharedOut![d]) + _routedOut![d];
            }
        }

        /// <summary>
        /// Batched (prefill) MoE FFN over <paramref name="rows"/> token rows — the multi-row counterpart
        /// of <see cref="Decode"/>. The routed sum is delegated to
        /// <see cref="MoeFeedForwardBlock.DecodeBatched"/> (group-by-expert); the shared expert (when
        /// present) is a batched SwiGLU scaled per-row by <c>σ(w_shared · xₙ)</c>. Numerically equal to
        /// N× <see cref="Decode"/> up to FP associativity in the routed sum. Scratch is per-call.
        /// </summary>
        public void DecodeBatched(
            ReadOnlySpan<float> hidden,
            int rows,
            ReadOnlySpan<float> routerWeight,
            DecodeWeight[] gateExperts,
            DecodeWeight[] upExperts,
            DecodeWeight[] downExperts,
            ReadOnlySpan<float> sharedGateWeight,
            in DecodeWeight sharedGate,
            in DecodeWeight sharedUp,
            in DecodeWeight sharedDown,
            Span<float> output)
        {
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(rows);
            if (hidden.Length < (long)rows * DModel) { throw new ArgumentException("Hidden span smaller than rows*dModel.", nameof(hidden)); }
            if (output.Length < (long)rows * DModel) { throw new ArgumentException("Output span smaller than rows*dModel.", nameof(output)); }

            // Mixtral (no shared expert): the FFN is the routed sum alone.
            if (_shared is null)
            {
                _routed.DecodeBatched(hidden, rows, routerWeight, gateExperts, upExperts, downExperts, output);
                return;
            }

            if (sharedGateWeight.Length < DModel)
            {
                throw new ArgumentException("Shared gate weight smaller than dModel.", nameof(sharedGateWeight));
            }

            var routedOut = PooledBuffer<float>.RentArray(rows * DModel);
            var sharedOut = PooledBuffer<float>.RentArray(rows * DModel);
            try
            {

                _routed.DecodeBatched(hidden, rows, routerWeight, gateExperts, upExperts, downExperts, routedOut);
                _shared.DecodeSwiGluBatchedDispatched(hidden, rows, in sharedGate, in sharedUp, in sharedDown, sharedOut);

                for (var n = 0; n < rows; n++)
                {
                    var row = hidden.Slice(n * DModel, DModel);
                    var gateLogit = 0f;
                    for (var d = 0; d < DModel; d++) { gateLogit += row[d] * sharedGateWeight[d]; }
                    var gate = 1f / (1f + MathF.Exp(-gateLogit));

                    var dst = output.Slice(n * DModel, DModel);
                    for (var d = 0; d < DModel; d++)
                    {
                        dst[d] = (gate * sharedOut[n * DModel + d]) + routedOut[n * DModel + d];
                    }
                }
            }
            finally
            {
                PooledBuffer<float>.ReturnArray(sharedOut);
                PooledBuffer<float>.ReturnArray(routedOut);
            }
        }
    }
}
