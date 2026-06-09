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
        /// The encode-side VQ step (<c>decode_latents</c>): assigns each latent frame to its nearest codebook
        /// entry. SNAC L2-normalizes both sides and picks the minimum-distance code — equivalently the maximum
        /// cosine, and since the query norm is constant per frame, the <b>argmax of <c>z·(codebook_row/‖row‖)</c></b>.
        /// <paramref name="zE"/> is channel-major <c>[dim × time]</c>; <paramref name="codebook"/> is
        /// <c>[codebookSize × dim]</c>; <paramref name="indices"/> receives the chosen code per frame.
        /// </summary>
        public static void EncodeCodebook(
            ReadOnlySpan<float> zE, int dim, int time, ReadOnlySpan<float> codebook, int codebookSize, Span<int> indices)
        {
            // Pre-normalize codebook rows once (matches torch F.normalize, eps 1e-12).
            var normalized = new float[codebookSize * dim];
            for (var c = 0; c < codebookSize; c++)
            {
                double ss = 0.0;
                for (var d = 0; d < dim; d++)
                {
                    double x = codebook[(c * dim) + d];
                    ss += x * x;
                }
                var inv = (float)(1.0 / Math.Max(Math.Sqrt(ss), 1e-12));
                for (var d = 0; d < dim; d++)
                {
                    normalized[(c * dim) + d] = codebook[(c * dim) + d] * inv;
                }
            }

            for (var t = 0; t < time; t++)
            {
                var best = float.NegativeInfinity;
                var bestIdx = 0;
                for (var c = 0; c < codebookSize; c++)
                {
                    var dot = 0f;
                    var cbase = c * dim;
                    for (var d = 0; d < dim; d++)
                    {
                        dot += zE[(d * time) + t] * normalized[cbase + d];
                    }
                    if (dot > best)
                    {
                        best = dot;
                        bestIdx = c;
                    }
                }
                indices[t] = bestIdx;
            }
        }

        /// <summary>
        /// Time-axis average pooling (<c>avg_pool1d(stride, stride)</c>): the encode-side downsample a higher-stride
        /// VQ level applies before quantizing. Channel-major <c>[channels × time]</c> → <c>[channels × time/stride]</c>,
        /// each output the mean of a non-overlapping window of <paramref name="stride"/> samples.
        /// </summary>
        public static void AveragePoolTime(
            ReadOnlySpan<float> src, Span<float> dst, int channels, int time, int stride)
        {
            var outTime = time / stride;
            for (var c = 0; c < channels; c++)
            {
                var srcBase = c * time;
                var dstBase = c * outTime;
                for (var ot = 0; ot < outTime; ot++)
                {
                    var sum = 0f;
                    var at = srcBase + (ot * stride);
                    for (var k = 0; k < stride; k++)
                    {
                        sum += src[at + k];
                    }
                    dst[dstBase + ot] = sum / stride;
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
