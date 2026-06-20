// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.LanguageModels.Runtime
{
    /// <summary>
    /// The gating step of a Mixture-of-Experts FFN: given the router's per-expert logits for one
    /// token, selects the <c>topK</c> experts and returns their indices plus the weights to combine
    /// their outputs with. This is the only genuinely new MoE math; the per-expert FFN reuses the
    /// existing feed-forward kernels and the combine is a weighted sum.
    ///
    /// Weighting follows the Mixtral / Qwen-MoE convention — softmax over all experts, take the
    /// top-k, then renormalise those k weights to sum to 1. That is mathematically identical to a
    /// softmax computed over <i>only</i> the top-k logits
    /// (<c>exp(lᵢ-max) / Σ_{j∈topk} exp(lⱼ-max)</c>), which is what this computes — so there's no
    /// need to normalise over the full expert set.
    ///
    /// Zero-allocation: the caller owns the output spans. Ties are broken toward the lower index.
    /// </summary>
    public static class MoeRouter
    {
        /// <summary>
        /// Writes the top-<paramref name="topK"/> expert indices (descending by logit) into
        /// <paramref name="expertIndices"/> and their renormalised softmax weights (summing to 1)
        /// into <paramref name="expertWeights"/>. Returns the number written
        /// (<c>min(topK, logits.Length)</c>).
        /// </summary>
        /// <param name="logits">Router logits, one per expert.</param>
        /// <param name="topK">Number of experts to select (capped at the number of logits).</param>
        /// <param name="expertIndices">Caller-owned destination for the selected expert indices (descending by logit).</param>
        /// <param name="expertWeights">Caller-owned destination for the selected experts' softmax weights.</param>
        /// <param name="normalize">
        /// true (Mixtral / Qwen with <c>norm_topk_prob=true</c>): renormalise so the k weights sum to
        /// 1 — i.e. softmax over the top-k logits. false (Qwen1.5-MoE, <c>norm_topk_prob=false</c>):
        /// the weights are the full softmax probabilities over ALL experts, evaluated at the top-k
        /// (they sum to ≤ 1, not renormalised).
        /// </param>
        public static int SelectTopK(
            ReadOnlySpan<float> logits,
            int topK,
            Span<int> expertIndices,
            Span<float> expertWeights,
            bool normalize = true)
        {
            if (logits.IsEmpty)
            {
                throw new ArgumentException("Router logits cannot be empty.", nameof(logits));
            }
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(topK);

            var k = Math.Min(topK, logits.Length);
            if (expertIndices.Length < k)
            {
                throw new ArgumentException($"expertIndices too small: {expertIndices.Length} < {k}.", nameof(expertIndices));
            }
            if (expertWeights.Length < k)
            {
                throw new ArgumentException($"expertWeights too small: {expertWeights.Length} < {k}.", nameof(expertWeights));
            }

            // ── Top-k selection by logit (softmax is monotonic, so this is top-k by probability).
            // Insertion into a descending [index, logit] list of size k; ties keep the lower index.
            for (var i = 0; i < k; i++)
            {
                expertIndices[i] = -1;
                expertWeights[i] = float.NegativeInfinity;
            }
            var filled = 0;
            for (var e = 0; e < logits.Length; e++)
            {
                var l = logits[e];
                if (filled == k && l <= expertWeights[k - 1])
                {
                    continue;
                }

                var pos = filled < k ? filled : k - 1;
                while (pos > 0 && expertWeights[pos - 1] < l)
                {
                    expertWeights[pos] = expertWeights[pos - 1];
                    expertIndices[pos] = expertIndices[pos - 1];
                    pos--;
                }
                expertWeights[pos] = l;
                expertIndices[pos] = e;
                if (filled < k)
                {
                    filled++;
                }
            }

            var max = expertWeights[0];   // sorted descending ⇒ [0] is the global max logit

            if (normalize)
            {
                // Softmax over the k selected logits (== renormalised full softmax over the top-k).
                var sum = 0f;
                for (var i = 0; i < k; i++)
                {
                    var w = MathF.Exp(expertWeights[i] - max);
                    expertWeights[i] = w;
                    sum += w;
                }
                var inv = sum > 0f ? 1f / sum : 0f;
                for (var i = 0; i < k; i++)
                {
                    expertWeights[i] *= inv;
                }
            }
            else
            {
                // Full softmax over ALL experts; keep the top-k probabilities un-renormalised.
                var total = 0f;
                for (var e = 0; e < logits.Length; e++)
                {
                    total += MathF.Exp(logits[e] - max);
                }
                var inv = total > 0f ? 1f / total : 0f;
                for (var i = 0; i < k; i++)
                {
                    expertWeights[i] = MathF.Exp(expertWeights[i] - max) * inv;
                }
            }

            return k;
        }
    }
}
