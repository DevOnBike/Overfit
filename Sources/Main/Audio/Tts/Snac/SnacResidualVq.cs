// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Audio.Tts.Snac
{
    /// <summary>
    /// The decode side of SNAC's residual vector quantizer — codes back to a continuous latent. Grounded in the
    /// SNAC source: per level, look the integer codes up in the codebook (<c>decode_code</c> → channel-major
    /// <c>[dim × T]</c>), project with a 1×1 conv (<c>out_proj</c>, an existing conv op), upsample by the level's
    /// stride (<c>repeat_interleave</c>), then sum the per-level latents (<c>z_q += z_q_i</c>). This type owns the
    /// two model-free, exactly-testable primitives — the codebook gather and the time repeat-interleave; the 1×1
    /// projection and the cross-level sum reuse plain conv / addition.
    /// </summary>
    internal static class SnacResidualVq
    {
        /// <summary>
        /// Codebook lookup (<c>decode_code</c>): maps each integer code to its embedding row and lays the result
        /// out channel-major. <paramref name="codes"/> is <c>[time]</c>; <paramref name="table"/> is the codebook
        /// <c>[codebookSize × dim]</c> (row-major); <paramref name="dst"/> receives <c>[dim × time]</c> (the
        /// <c>(B,T,D) → (B,D,T)</c> transpose). Throws if a code is out of range.
        /// </summary>
        public static void DecodeCodebook(
            ReadOnlySpan<int> codes, ReadOnlySpan<float> table, Span<float> dst, int codebookSize, int dim, int time)
        {
            for (var t = 0; t < time; t++)
            {
                var id = codes[t];
                if ((uint)id >= (uint)codebookSize)
                {
                    throw new OverfitRuntimeException(
                        $"SNAC code {id} at frame {t} is out of range for a codebook of size {codebookSize}.");
                }

                var rowBase = id * dim;
                for (var d = 0; d < dim; d++)
                {
                    dst[(d * time) + t] = table[rowBase + d];
                }
            }
        }

        /// <summary>
        /// Time-axis <c>repeat_interleave(stride)</c>: replicates each frame <paramref name="stride"/> times so a
        /// level decoded at a lower rate lines up with the others before summation. Channel-major
        /// <paramref name="src"/> <c>[channels × time]</c> → <paramref name="dst"/> <c>[channels × time·stride]</c>,
        /// where <c>dst[c, t·stride + s] = src[c, t]</c>.
        /// </summary>
        public static void RepeatInterleaveTime(
            ReadOnlySpan<float> src, Span<float> dst, int channels, int time, int stride)
        {
            var outTime = time * stride;
            for (var c = 0; c < channels; c++)
            {
                var srcBase = c * time;
                var dstBase = c * outTime;
                for (var t = 0; t < time; t++)
                {
                    var v = src[srcBase + t];
                    var at = dstBase + (t * stride);
                    for (var s = 0; s < stride; s++)
                    {
                        dst[at + s] = v;
                    }
                }
            }
        }
    }
}
