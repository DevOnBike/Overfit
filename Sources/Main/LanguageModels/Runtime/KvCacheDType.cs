// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.LanguageModels.Runtime
{
    /// <summary>
    /// Element type of the KV cache. <see cref="F32"/> is the default full-precision store;
    /// <see cref="Q8"/> stores each cached K/V vector as per-vector symmetric int8 plus one F32 scale
    /// (~4× less KV RAM and attention read traffic), for long-context / low-memory decode. Decode reads
    /// int8 directly via <see cref="CachedAttentionKernel.ComputeSingleHeadQ8"/>; the batched prefill /
    /// verify path dequantizes the needed range to an F32 scratch and reuses the F32 attention kernel.
    /// </summary>
    public enum KvCacheDType
    {
        /// <summary>Full-precision F32 K/V (default).</summary>
        F32,

        /// <summary>Per-vector symmetric int8 K/V (~4× smaller; cosine ≈ 1 round-trip).</summary>
        Q8,
    }
}
