// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace DevOnBike.Overfit.LanguageModels.Rope
{
    /// <summary>
    /// Applies Rotary Position Embedding (RoPE) to a single attention head vector.
    ///
    /// Convention: Adjacent-pair / GPT-NeoX (llama.cpp LLAMA_ROPE_TYPE_NEOX).
    ///   Pairs: (x[2i], x[2i+1]) for i in [0, headDim/2)
    ///
    /// This matches GGUF weight ordering used by llama.cpp for Qwen2/Llama models.
    /// DO NOT use split-half rotation (x[i], x[i+headDim/2]) — that would require
    /// weights stored in HuggingFace interleaved format, not GGUF adjacent format.
    ///
    /// Applied in-place to Q and K before KV cache write.
    /// Cached K vectors already contain their rotated values —
    /// RoPE only needs to be applied once, at write time.
    ///
    /// Zero allocations.
    /// </summary>
    public static class RopeKernel
    {
        /// <summary>
        /// Rotates a head vector in-place using precomputed cos/sin at a given position.
        /// Adjacent-pair rotation: pairs (x[2i], x[2i+1]) share the i-th frequency.
        /// </summary>
        /// <summary>Adjacent-pair rotation (x[2i], x[2i+1]). Default convention.</summary>
        public static void Apply(
            Span<float> headVector,
            ReadOnlySpan<float> cos,
            ReadOnlySpan<float> sin)
            => Apply(headVector, cos, sin, splitHalf: false);

        /// <summary>
        /// Rotates a head vector in-place. <paramref name="splitHalf"/> false = adjacent pairs
        /// (x[2i], x[2i+1]); true = split-half pairs (x[i], x[i+headDim/2]) (HF rotate_half / NEOX).
        /// </summary>
        public static void Apply(
            Span<float> headVector,
            ReadOnlySpan<float> cos,
            ReadOnlySpan<float> sin,
            bool splitHalf)
        {
            var halfDim = headVector.Length / 2;

            if (cos.Length < halfDim || sin.Length < halfDim)
            {
                throw new ArgumentException("cos/sin spans shorter than headDim/2.");
            }

            if (splitHalf)
            {
                // Split-half (HF rotate_half / NEOX): pairs (x[i], x[i+halfDim]) share frequency i.
                var i = 0;
                if (Avx2.IsSupported)
                {
                    i = ApplySplitHalfAvx2(headVector, cos, sin, halfDim);
                }

                for (; i < halfDim; i++)
                {
                    var x0 = headVector[i];
                    var x1 = headVector[i + halfDim];
                    headVector[i] = x0 * cos[i] - x1 * sin[i];
                    headVector[i + halfDim] = x0 * sin[i] + x1 * cos[i];
                }
                return;
            }

            // Adjacent-pair: pairs (x[2i], x[2i+1]) share frequency i.
            var p = 0;
            if (Avx2.IsSupported)
            {
                p = ApplyAdjacentAvx2(headVector, cos, sin, halfDim);
            }

            for (; p < halfDim; p++)
            {
                var x0 = headVector[2 * p];
                var x1 = headVector[2 * p + 1];

                headVector[2 * p] = x0 * cos[p] - x1 * sin[p];
                headVector[2 * p + 1] = x0 * sin[p] + x1 * cos[p];
            }
        }

        /// <summary>
        /// AVX2 split-half rotation over whole 8-lane chunks; returns the count of frequencies
        /// processed (the caller's scalar loop finishes any <c>halfDim % 8</c> tail). The two
        /// halves are contiguous, so this is a clean vector load/store. Bit-identical to the
        /// scalar path: same two FP multiplies then sub/add (NOT fused — FMA would change rounding).
        /// </summary>
        private static int ApplySplitHalfAvx2(
            Span<float> headVector, ReadOnlySpan<float> cos, ReadOnlySpan<float> sin, int halfDim)
        {
            ref var v = ref MemoryMarshal.GetReference(headVector);
            ref var c = ref MemoryMarshal.GetReference(cos);
            ref var s = ref MemoryMarshal.GetReference(sin);

            var i = 0;
            for (; i + 8 <= halfDim; i += 8)
            {
                var x0 = Vector256.LoadUnsafe(ref v, (nuint)i);
                var x1 = Vector256.LoadUnsafe(ref v, (nuint)(i + halfDim));
                var cv = Vector256.LoadUnsafe(ref c, (nuint)i);
                var sv = Vector256.LoadUnsafe(ref s, (nuint)i);

                var out0 = Avx.Subtract(Avx.Multiply(x0, cv), Avx.Multiply(x1, sv));
                var out1 = Avx.Add(Avx.Multiply(x0, sv), Avx.Multiply(x1, cv));

                out0.StoreUnsafe(ref v, (nuint)i);
                out1.StoreUnsafe(ref v, (nuint)(i + halfDim));
            }

            return i;
        }

        /// <summary>
        /// AVX2 adjacent-pair rotation, 4 pairs (8 floats) per chunk; returns the count of pairs
        /// processed (caller finishes any <c>halfDim % 4</c> tail). Treats each pair as a complex
        /// number: <c>result = v·cosExp + swap(v)·sinSigned</c> where <c>cosExp = [c,c,...]</c>,
        /// <c>sinSigned = [-s,+s,...]</c>, and <c>swap</c> exchanges the two lanes of each pair —
        /// which expands to <c>(x0·c − x1·s, x1·c + x0·s)</c>, bit-identical to the scalar (separate
        /// multiplies + add; <c>x1·(−s) == −(x1·s)</c> and FP add is commutative, so no drift).
        /// </summary>
        private static int ApplyAdjacentAvx2(
            Span<float> headVector, ReadOnlySpan<float> cos, ReadOnlySpan<float> sin, int halfDim)
        {
            ref var v = ref MemoryMarshal.GetReference(headVector);
            ref var c = ref MemoryMarshal.GetReference(cos);
            ref var s = ref MemoryMarshal.GetReference(sin);

            // Duplicate-expand control: [c0,c1,c2,c3] -> [c0,c0,c1,c1,c2,c2,c3,c3].
            var dup = Vector256.Create(0, 0, 1, 1, 2, 2, 3, 3);
            var signs = Vector256.Create(-1f, 1f, -1f, 1f, -1f, 1f, -1f, 1f);

            var p = 0;
            for (; p + 4 <= halfDim; p += 4)
            {
                var baseF = (nuint)(2 * p);
                var vv = Vector256.LoadUnsafe(ref v, baseF);

                var cos4 = Vector128.LoadUnsafe(ref c, (nuint)p);
                var sin4 = Vector128.LoadUnsafe(ref s, (nuint)p);
                var cosExp = Avx2.PermuteVar8x32(Vector256.Create(cos4, cos4), dup);
                var sinExp = Avx2.PermuteVar8x32(Vector256.Create(sin4, sin4), dup);
                var sinSigned = Avx.Multiply(sinExp, signs);

                // Swap the two lanes within each pair: [a,b,c,d] -> [b,a,d,c] (per 128-bit lane).
                var swapped = Avx.Shuffle(vv, vv, 0b_10_11_00_01);

                var result = Avx.Add(Avx.Multiply(vv, cosExp), Avx.Multiply(swapped, sinSigned));
                result.StoreUnsafe(ref v, baseF);
            }

            return p;
        }

        /// <summary>
        /// Rotates a head vector in-place using a <see cref="RopeTable"/> at a given position.
        /// </summary>
        public static void Apply(Span<float> headVector, RopeTable table, int position)
        {
            Apply(headVector, table.CosAt(position), table.SinAt(position), table.SplitHalf);
        }
    }
}