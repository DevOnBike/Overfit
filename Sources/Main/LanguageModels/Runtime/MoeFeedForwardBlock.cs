// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.DeepLearning;

namespace DevOnBike.Overfit.LanguageModels.Runtime
{
    /// <summary>
    /// A Mixture-of-Experts feed-forward block (Mixtral / Qwen-MoE style): a router scores the token
    /// against <see cref="ExpertCount"/> experts, the top <see cref="ExpertUsedCount"/> are selected
    /// (<see cref="MoeRouter"/>), each runs a SwiGLU FFN, and their outputs are combined by the
    /// router weights. Per-token cost is ≈ <c>ExpertUsedCount</c> dense FFNs (not all experts), which
    /// is the whole point — large total capacity, small active compute.
    ///
    /// Each expert's weights are ordinary <see cref="DecodeWeight"/>s, so they reuse the existing
    /// (F32 / Q8_0 / Q4_K / Q6_K) SwiGLU dispatch — and are mmap-able just like the dense FFN. The
    /// router projection (<c>ffn_gate_inp</c>) is small and stays F32. Zero-allocation per call.
    /// </summary>
    public sealed class MoeFeedForwardBlock
    {
        private readonly CachedFeedForwardBlock _expert;   // reused across the selected experts
        private readonly float[] _routerLogits;
        private readonly float[] _expertOut;
        private readonly int[] _selectedExperts;
        private readonly float[] _selectedWeights;

        public MoeFeedForwardBlock(int dModel, int dFF, int expertCount, int expertUsedCount)
        {
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(dModel);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(dFF);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(expertCount);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(expertUsedCount);
            if (expertUsedCount > expertCount)
            {
                throw new ArgumentOutOfRangeException(nameof(expertUsedCount),
                    $"ExpertUsedCount ({expertUsedCount}) cannot exceed ExpertCount ({expertCount}).");
            }

            DModel = dModel;
            DFF = dFF;
            ExpertCount = expertCount;
            ExpertUsedCount = expertUsedCount;

            _expert = new CachedFeedForwardBlock(dModel, dFF, FeedForwardActivation.SwiGLU);
            _routerLogits = new float[expertCount];
            _expertOut = new float[dModel];
            _selectedExperts = new int[expertUsedCount];
            _selectedWeights = new float[expertUsedCount];
        }

        public int DModel { get; }
        public int DFF { get; }
        public int ExpertCount { get; }
        public int ExpertUsedCount { get; }

        /// <summary>
        /// Routes <paramref name="hidden"/> and writes the combined output of the top-k experts into
        /// <paramref name="output"/>. <paramref name="routerWeight"/> is the F32 gating matrix
        /// (<c>ffn_gate_inp</c>), input-major <c>[dModel × ExpertCount]</c>; the expert arrays hold one
        /// <see cref="DecodeWeight"/> per expert (gate/up: <c>[dModel × dFF]</c>, down: <c>[dFF × dModel]</c>).
        /// </summary>
        public void Decode(
            ReadOnlySpan<float> hidden,
            ReadOnlySpan<float> routerWeight,
            DecodeWeight[] gateExperts,
            DecodeWeight[] upExperts,
            DecodeWeight[] downExperts,
            Span<float> output)
        {
            ArgumentNullException.ThrowIfNull(gateExperts);
            ArgumentNullException.ThrowIfNull(upExperts);
            ArgumentNullException.ThrowIfNull(downExperts);
            if (hidden.Length < DModel) { throw new ArgumentException("Hidden span smaller than dModel.", nameof(hidden)); }
            if (output.Length < DModel) { throw new ArgumentException("Output span smaller than dModel.", nameof(output)); }
            if (routerWeight.Length < DModel * ExpertCount)
            {
                throw new ArgumentException("Router weight smaller than dModel * ExpertCount.", nameof(routerWeight));
            }
            if (gateExperts.Length < ExpertCount || upExperts.Length < ExpertCount || downExperts.Length < ExpertCount)
            {
                throw new ArgumentException("Expert weight arrays smaller than ExpertCount.");
            }

            // Router: logits[e] = hidden · routerWeight[:, e]
            SingleTokenProjectionKernel.ProjectParallel(
                hidden[..DModel], routerWeight, [], _routerLogits, DModel, ExpertCount);

            var k = MoeRouter.SelectTopK(_routerLogits, ExpertUsedCount, _selectedExperts, _selectedWeights);

            // Combine only the selected experts, weighted by the renormalised router probabilities.
            output[..DModel].Clear();
            for (var i = 0; i < k; i++)
            {
                var e = _selectedExperts[i];
                var weight = _selectedWeights[i];

                _expert.DecodeSwiGluDispatched(hidden, in gateExperts[e], in upExperts[e], in downExperts[e], _expertOut);

                for (var d = 0; d < DModel; d++)
                {
                    output[d] += weight * _expertOut[d];
                }
            }
        }
    }
}
