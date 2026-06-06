// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Audio.Tts.Snac
{
    /// <summary>
    /// Building blocks shared by the SNAC encoder and decoder. The <see cref="ResidualUnit"/> (Snake → depthwise
    /// k7 conv → Snake → 1×1 conv → skip) is identical on both sides, so it lives here once.
    /// </summary>
    internal static class SnacBlocks
    {
        /// <summary>
        /// One SNAC residual unit at weight-prefix <paramref name="prefix"/> (e.g. <c>dec.block.0.res.1.</c>):
        /// <c>x + conv1x1(Snake(depthwiseConv7(Snake(x))))</c>. The k7 conv uses padding 3·dilation so length is
        /// preserved (no skip-trim needed). Returns a new <c>[dim × t]</c> buffer.
        /// </summary>
        public static float[] ResidualUnit(SnacWeights w, string prefix, float[] x, int dim, int t, int dilation)
        {
            var h = new float[x.Length];
            x.AsSpan().CopyTo(h);
            SnacActivations.Snake1dInPlace(h, w[prefix + "snake1.alpha"], dim, t);

            var pad = 3 * dilation; // (kernel-1)*dilation/2 with kernel=7
            var c1 = new float[dim * t];
            SnacConv.Conv1d(h, w[prefix + "conv1.weight"], w[prefix + "conv1.bias"], c1,
                inC: dim, tIn: t, outC: dim, kSize: 7, stride: 1, pad: pad, dilation: dilation, groups: dim, tOut: t);

            SnacActivations.Snake1dInPlace(c1, w[prefix + "snake2.alpha"], dim, t);

            var c2 = new float[dim * t];
            SnacConv.Conv1d(c1, w[prefix + "conv2.weight"], w[prefix + "conv2.bias"], c2,
                inC: dim, tIn: t, outC: dim, kSize: 1, stride: 1, pad: 0, dilation: 1, groups: 1, tOut: t);

            for (var i = 0; i < c2.Length; i++)
            {
                c2[i] += x[i];
            }
            return c2;
        }
    }
}
