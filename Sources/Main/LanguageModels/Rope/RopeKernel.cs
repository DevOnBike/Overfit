// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.LanguageModels.Rope
{
    /// <summary>
    /// Applies Rotary Position Embedding (RoPE) to a single attention head vector.
    ///
    /// Convention: GPT-NeoX / Llama (half-rotation).
    ///   Pairs: (x[i], x[i + headDim/2]) for i in [0, headDim/2)
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
        /// </summary>
        public static void Apply(
            Span<float> headVector,
            ReadOnlySpan<float> cos,
            ReadOnlySpan<float> sin)
        {
            var halfDim = headVector.Length / 2;

            if (cos.Length < halfDim || sin.Length < halfDim)
            {
                throw new ArgumentException("cos/sin spans shorter than headDim/2.");
            }

            for (var i = 0; i < halfDim; i++)
            {
                var x0 = headVector[i];
                var x1 = headVector[i + halfDim];

                headVector[i]         = x0 * cos[i] - x1 * sin[i];
                headVector[i + halfDim] = x0 * sin[i] + x1 * cos[i];
            }
        }

        /// <summary>
        /// Rotates a head vector in-place using a <see cref="RopeTable"/> at a given position.
        /// </summary>
        public static void Apply(Span<float> headVector, RopeTable table, int position)
            => Apply(headVector, table.CosAt(position), table.SinAt(position));
    }
}
