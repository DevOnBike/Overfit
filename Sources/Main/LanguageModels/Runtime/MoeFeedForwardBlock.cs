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
        private readonly bool _normalizeWeights;

        public MoeFeedForwardBlock(int dModel, int dFF, int expertCount, int expertUsedCount, bool normalizeWeights = true)
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
            _normalizeWeights = normalizeWeights;
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

            var k = MoeRouter.SelectTopK(_routerLogits, ExpertUsedCount, _selectedExperts, _selectedWeights, _normalizeWeights);

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

        /// <summary>
        /// Batched (prefill) routed MoE over <paramref name="rows"/> token rows. Routes every row, then
        /// <b>groups rows by expert</b>: the rows that selected expert <c>e</c> are gathered and run
        /// through that expert's SwiGLU in one batched pass (<c>DecodeSwiGluBatchedDispatched</c> — each
        /// expert's weights read once, reused across its rows). This is the prefill win — per-token cost
        /// stays ≈ ExpertUsedCount FFNs but each active expert's weights are read once per prefill instead
        /// of once per routing token. Each expert's output is stashed at the row's <i>top-k slot</i> and
        /// the per-row weighted sum is then accumulated in <b>top-k order</b> — the SAME order as
        /// single-token <see cref="Decode"/>, so (expert outputs being bit-identical) the result is
        /// <b>bit-identical</b> to N× <see cref="Decode"/>, not merely FP-close. Scratch is per-call.
        /// </summary>
        public void DecodeBatched(
            ReadOnlySpan<float> hidden,
            int rows,
            ReadOnlySpan<float> routerWeight,
            DecodeWeight[] gateExperts,
            DecodeWeight[] upExperts,
            DecodeWeight[] downExperts,
            Span<float> output)
        {
            ArgumentNullException.ThrowIfNull(gateExperts);
            ArgumentNullException.ThrowIfNull(upExperts);
            ArgumentNullException.ThrowIfNull(downExperts);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(rows);
            if (hidden.Length < (long)rows * DModel) { throw new ArgumentException("Hidden span smaller than rows*dModel.", nameof(hidden)); }
            if (output.Length < (long)rows * DModel) { throw new ArgumentException("Output span smaller than rows*dModel.", nameof(output)); }

            var k = ExpertUsedCount;

            // 1. Route every row (router projection is small; per-row is fine).
            var selExperts = new int[rows * k];
            var selWeights = new float[rows * k];
            var logits = new float[ExpertCount];
            for (var n = 0; n < rows; n++)
            {
                SingleTokenProjectionKernel.ProjectParallel(
                    hidden.Slice(n * DModel, DModel), routerWeight, [], logits, DModel, ExpertCount);
                MoeRouter.SelectTopK(
                    logits, k,
                    selExperts.AsSpan(n * k, k),
                    selWeights.AsSpan(n * k, k),
                    _normalizeWeights);
            }

            // 2. Group rows by expert: gather → batched SwiGLU → stash each result at the row's top-k
            //    slot (so the final per-row sum can run in top-k order = single-token order).
            var rowBuf = new int[rows];
            var slotBuf = new int[rows];
            var gathered = new float[rows * DModel];
            var expertOut = new float[rows * DModel];
            var slotOut = new float[rows * k * DModel];   // [row, slot] → that expert's output

            for (var e = 0; e < ExpertCount; e++)
            {
                var count = 0;
                for (var n = 0; n < rows; n++)
                {
                    for (var j = 0; j < k; j++)
                    {
                        if (selExperts[n * k + j] == e)
                        {
                            rowBuf[count] = n;
                            slotBuf[count] = j;
                            count++;
                            break;
                        }
                    }
                }
                if (count == 0) { continue; }

                for (var c = 0; c < count; c++)
                {
                    hidden.Slice(rowBuf[c] * DModel, DModel).CopyTo(gathered.AsSpan(c * DModel, DModel));
                }

                _expert.DecodeSwiGluBatchedDispatched(
                    gathered, count, in gateExperts[e], in upExperts[e], in downExperts[e], expertOut);

                for (var c = 0; c < count; c++)
                {
                    expertOut.AsSpan(c * DModel, DModel)
                        .CopyTo(slotOut.AsSpan((rowBuf[c] * k + slotBuf[c]) * DModel, DModel));
                }
            }

            // 3. Per-row weighted sum in top-k slot order — identical accumulation order to Decode.
            for (var n = 0; n < rows; n++)
            {
                var dst = output.Slice(n * DModel, DModel);
                dst.Clear();
                for (var j = 0; j < k; j++)
                {
                    var w = selWeights[n * k + j];
                    var src = slotOut.AsSpan((n * k + j) * DModel, DModel);
                    for (var d = 0; d < DModel; d++) { dst[d] += w * src[d]; }
                }
            }
        }
    }
}
