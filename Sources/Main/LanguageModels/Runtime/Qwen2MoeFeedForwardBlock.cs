// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.DeepLearning;

namespace DevOnBike.Overfit.LanguageModels.Runtime
{
    /// <summary>
    /// The Qwen2-MoE feed-forward block — a routed Mixture-of-Experts <b>plus</b> a sigmoid-gated
    /// shared expert that runs for every token (Mixtral has only the routed part):
    /// <code>FFN(x) = σ(w_shared · x) · shared(x) + Σ_{i∈top-k} weightᵢ · expertᵢ(x)</code>
    /// The routed sum is delegated to <see cref="MoeFeedForwardBlock"/>; the shared expert is an
    /// ordinary SwiGLU FFN (its own, larger dFF) scaled by the sigmoid of a single learned dot.
    ///
    /// All weights are <see cref="DecodeWeight"/>s (F32 / Q8_0 / Q4_K / Q6_K — mmap-able); the two
    /// router projections (`ffn_gate_inp`, `ffn_gate_inp_shexp`) stay F32. Zero-allocation per call.
    /// </summary>
    public sealed class Qwen2MoeFeedForwardBlock
    {
        private readonly MoeFeedForwardBlock _routed;
        private readonly CachedFeedForwardBlock _shared;
        private readonly float[] _routedOut;
        private readonly float[] _sharedOut;

        public Qwen2MoeFeedForwardBlock(
            int dModel, int expertFeedForwardLength, int sharedFeedForwardLength,
            int expertCount, int expertUsedCount)
        {
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(dModel);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(expertFeedForwardLength);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(sharedFeedForwardLength);

            DModel = dModel;
            _routed = new MoeFeedForwardBlock(dModel, expertFeedForwardLength, expertCount, expertUsedCount);
            _shared = new CachedFeedForwardBlock(dModel, sharedFeedForwardLength, FeedForwardActivation.SwiGLU);
            _routedOut = new float[dModel];
            _sharedOut = new float[dModel];
        }

        public int DModel { get; }
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
                output[d] = (gate * _sharedOut[d]) + _routedOut[d];
            }
        }
    }
}
