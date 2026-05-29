// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

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
                for (var i = 0; i < halfDim; i++)
                {
                    var x0 = headVector[i];
                    var x1 = headVector[i + halfDim];
                    headVector[i] = x0 * cos[i] - x1 * sin[i];
                    headVector[i + halfDim] = x0 * sin[i] + x1 * cos[i];
                }
                return;
            }

            // Adjacent-pair: pairs (x[2i], x[2i+1]) share frequency i.
            for (var i = 0; i < halfDim; i++)
            {
                var x0 = headVector[2 * i];
                var x1 = headVector[2 * i + 1];

                headVector[2 * i] = x0 * cos[i] - x1 * sin[i];
                headVector[2 * i + 1] = x0 * sin[i] + x1 * cos[i];
            }
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