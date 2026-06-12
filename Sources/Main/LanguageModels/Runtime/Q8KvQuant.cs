// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using DevOnBike.Overfit.Intrinsics;

namespace DevOnBike.Overfit.LanguageModels.Runtime
{
    /// <summary>
    /// Per-vector symmetric int8 quantization for the Q8 KV-cache — one F32 scale per cached
    /// K or V vector (<c>scale = maxAbs / 127</c>, round-to-nearest). Shared by the cache write
    /// path and <see cref="CachedAttentionKernel.ComputeSingleHeadQ8"/> so the round-trip matches.
    /// </summary>
    public static class Q8KvQuant
    {
        /// <summary>
        /// Quantizes <paramref name="src"/> (one K/V vector) into <paramref name="dst"/> int8 and
        /// returns the F32 scale such that <c>src[d] ≈ scale · dst[d]</c>. A zero vector yields
        /// scale 0 and all-zero bytes.
        /// </summary>
        public static float Quantize(ReadOnlySpan<float> src, Span<sbyte> dst)
        {
            var n = src.Length;
            var maxAbs = 0f;
            var i = 0;
            if (CpuFeatures.HasAvx && n >= 8)
            {
                ref var s = ref MemoryMarshal.GetReference(src);
                var absMask = Vector256.Create(0x7FFFFFFF).AsSingle();
                var m = Vector256<float>.Zero;
                for (; i + 8 <= n; i += 8)
                {
                    m = Avx.Max(m, Avx.And(Vector256.LoadUnsafe(ref s, (nuint)i), absMask));
                }
                maxAbs = Vector256.Max(m, Vector256.Shuffle(m, Vector256.Create(4, 5, 6, 7, 0, 1, 2, 3))).ToScalar();
                for (var j = 1; j < 8; j++)
                {
                    maxAbs = MathF.Max(maxAbs, m.GetElement(j));
                }
            }
            for (; i < n; i++)
            {
                var a = MathF.Abs(src[i]);
                if (a > maxAbs)
                {
                    maxAbs = a;
                }
            }

            if (maxAbs <= 0f)
            {
                dst.Slice(0, n).Clear();
                return 0f;
            }

            var scale = maxAbs / 127f;
            var inv = 127f / maxAbs;
            for (var d = 0; d < n; d++)
            {
                var q = MathF.Round(src[d] * inv);
                if (q > 127f) { q = 127f; }
                else if (q < -127f) { q = -127f; }
                dst[d] = (sbyte)q;
            }
            return scale;
        }
    }
}
