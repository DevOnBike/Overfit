// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Tensors;

namespace DevOnBike.Overfit.Kernels
{
    /// <summary>
    /// Winograd F(2×2, 3×3) convolution for 3×3 stride-1 convs (VGG/ResNet's bulk). Computes a 2×2 output tile
    /// from a 4×4 input tile with 16 element-wise products instead of the 36 (4·9) a direct/im2col conv needs —
    /// ~2.25× fewer multiplies. The conv reduces to: transform every 3×3 filter (U = G·g·Gᵀ, once) and every 4×4
    /// input tile (V = Bᵀ·d·B), then for each of the 16 transform positions a dense GEMM contracts over input
    /// channels (reusing <see cref="Conv2DGemmKernels.Gemm"/>), then the inverse transform (Y = Aᵀ·M·A) yields the
    /// 2×2 outputs. The three constant transforms are sparse, so they are applied as separable 1-D passes.
    ///
    /// NOT WIRED INTO THE HOT PATH. Kept as a parity-correct (cos 1.0 vs scalar/ORT) reference implementation.
    /// As written it REGRESSED deepcnn 119.7→214.4 ms (+79%) vs the tuned im2col+GEMM path: the transforms here
    /// are SEQUENTIAL SCALAR, the 16 GEMMs are small (poor per-GEMM parallelism), and the 16× U/V/M buffer
    /// blow-up costs bandwidth — together they outweigh the FLOP cut at these channel/tile sizes. Making it win
    /// would need (a) the input/output transforms parallelized over channels and SIMD-vectorised across
    /// tiles/channels, and (b) the 16 GEMMs batched and parallelised over the 16·outC dimension rather than each
    /// one parallelising over a small T. Until that is built and MEASURED to beat im2col, leave it unused.
    /// </summary>
    internal static class Conv2DWinogradKernels
    {
        public static bool IsSupported => Conv2DGemmKernels.IsSupported;

        public static void Forward(
            ReadOnlySpan<float> input,   // [batch, inChannels, H, W]
            ReadOnlySpan<float> kernels, // [outChannels, inChannels, 3, 3]
            Span<float> output,          // [batch, outChannels, outH, outW]
            int batchSize,
            int inChannels,
            int outChannels,
            int inputH,
            int inputW,
            int padding)
        {
            var outH = inputH + 2 * padding - 2; // k=3, stride=1
            var outW = inputW + 2 * padding - 2;
            var tilesH = (outH + 1) / 2;
            var tilesW = (outW + 1) / 2;
            var t = tilesH * tilesW;

            var inPlane = inChannels * inputH * inputW;
            var outPlane = outChannels * outH * outW;

            // U[ξ][outC][inC] — filter transform, once (filters don't depend on the batch).
            using var uBuf = new PooledBuffer<float>(checked(16 * outChannels * inChannels), clearMemory: false);
            var u = uBuf.Span;
            FilterTransform(kernels, u, outChannels, inChannels);

            using var vBuf = new PooledBuffer<float>(checked(16 * inChannels * t), clearMemory: false);
            using var mBuf = new PooledBuffer<float>(checked(16 * outChannels * t), clearMemory: false);
            var v = vBuf.Span;
            var m = mBuf.Span;

            for (var b = 0; b < batchSize; b++)
            {
                InputTransform(input.Slice(b * inPlane, inPlane), v, inChannels, inputH, inputW, padding, tilesH, tilesW, t);

                // 16 element-wise GEMMs: M_ξ[outC × T] = U_ξ[outC × inC] @ V_ξ[inC × T].
                for (var xi = 0; xi < 16; xi++)
                {
                    Conv2DGemmKernels.Gemm(
                        u.Slice(xi * outChannels * inChannels, outChannels * inChannels),
                        v.Slice(xi * inChannels * t, inChannels * t),
                        m.Slice(xi * outChannels * t, outChannels * t),
                        outChannels, t, inChannels);
                }

                OutputTransform(m, output.Slice(b * outPlane, outPlane), outChannels, outH, outW, tilesH, tilesW, t);
            }
        }

        // U[ξ·outC·inC + oc·inC + ic] = (G g Gᵀ)[ξ] for each 3×3 filter g.
        private static void FilterTransform(ReadOnlySpan<float> kernels, Span<float> u, int outC, int inC)
        {
            Span<float> tmp = stackalloc float[12]; // [4 rows × 3 cols]
            var ocInC = outC * inC;

            for (var oc = 0; oc < outC; oc++)
            {
                for (var ic = 0; ic < inC; ic++)
                {
                    var g = (oc * inC + ic) * 9;

                    // Column pass: filter-1D over each of the 3 columns of g (3 → 4).
                    for (var c = 0; c < 3; c++)
                    {
                        Filter1D(kernels[g + 0 * 3 + c], kernels[g + 1 * 3 + c], kernels[g + 2 * 3 + c],
                            out var f0, out var f1, out var f2, out var f3);
                        tmp[0 * 3 + c] = f0;
                        tmp[1 * 3 + c] = f1;
                        tmp[2 * 3 + c] = f2;
                        tmp[3 * 3 + c] = f3;
                    }

                    // Row pass: filter-1D over each of the 4 rows of tmp (3 → 4) → the 4×4 U.
                    var slot = oc * inC + ic;
                    for (var r = 0; r < 4; r++)
                    {
                        Filter1D(tmp[r * 3 + 0], tmp[r * 3 + 1], tmp[r * 3 + 2],
                            out var u0, out var u1, out var u2, out var u3);
                        u[(r * 4 + 0) * ocInC + slot] = u0;
                        u[(r * 4 + 1) * ocInC + slot] = u1;
                        u[(r * 4 + 2) * ocInC + slot] = u2;
                        u[(r * 4 + 3) * ocInC + slot] = u3;
                    }
                }
            }
        }

        // V[ξ·inC·T + ic·T + tile] = (Bᵀ d B)[ξ] for each 4×4 input tile d (with zero padding).
        private static void InputTransform(
            ReadOnlySpan<float> input, Span<float> v, int inC, int h, int w, int pad, int tilesH, int tilesW, int t)
        {
            Span<float> d = stackalloc float[16];
            Span<float> tmp = stackalloc float[16];
            var inCT = inC * t;

            for (var ic = 0; ic < inC; ic++)
            {
                var chan = ic * h * w;
                for (var th = 0; th < tilesH; th++)
                {
                    for (var tw = 0; tw < tilesW; tw++)
                    {
                        var tile = th * tilesW + tw;

                        // Gather the 4×4 input tile (origin (th·2-pad, tw·2-pad)); out-of-image reads are zero.
                        for (var i = 0; i < 4; i++)
                        {
                            var ih = th * 2 - pad + i;
                            for (var j = 0; j < 4; j++)
                            {
                                var iw = tw * 2 - pad + j;
                                d[i * 4 + j] = (uint)ih < (uint)h && (uint)iw < (uint)w ? input[chan + ih * w + iw] : 0f;
                            }
                        }

                        // Column pass then row pass (Bᵀ d B is separable with the same 1-D transform both ways).
                        for (var j = 0; j < 4; j++)
                        {
                            Input1D(d[0 * 4 + j], d[1 * 4 + j], d[2 * 4 + j], d[3 * 4 + j],
                                out var a0, out var a1, out var a2, out var a3);
                            tmp[0 * 4 + j] = a0;
                            tmp[1 * 4 + j] = a1;
                            tmp[2 * 4 + j] = a2;
                            tmp[3 * 4 + j] = a3;
                        }

                        var slot = ic * t + tile;
                        for (var i = 0; i < 4; i++)
                        {
                            Input1D(tmp[i * 4 + 0], tmp[i * 4 + 1], tmp[i * 4 + 2], tmp[i * 4 + 3],
                                out var v0, out var v1, out var v2, out var v3);
                            v[(i * 4 + 0) * inCT + slot] = v0;
                            v[(i * 4 + 1) * inCT + slot] = v1;
                            v[(i * 4 + 2) * inCT + slot] = v2;
                            v[(i * 4 + 3) * inCT + slot] = v3;
                        }
                    }
                }
            }
        }

        // output[oc] tile = (Aᵀ M A) — the 2×2 inverse transform of the 4×4 gathered from M[ξ][oc][tile].
        private static void OutputTransform(
            ReadOnlySpan<float> m, Span<float> output, int outC, int outH, int outW, int tilesH, int tilesW, int t)
        {
            Span<float> mm = stackalloc float[16];
            Span<float> tmp = stackalloc float[8]; // [2 rows × 4 cols]
            var outCT = outC * t;

            for (var oc = 0; oc < outC; oc++)
            {
                var outBase = oc * outH * outW;
                for (var th = 0; th < tilesH; th++)
                {
                    for (var tw = 0; tw < tilesW; tw++)
                    {
                        var tile = th * tilesW + tw;
                        var slot = oc * t + tile;

                        for (var i = 0; i < 4; i++)
                        {
                            for (var j = 0; j < 4; j++)
                            {
                                mm[i * 4 + j] = m[(i * 4 + j) * outCT + slot];
                            }
                        }

                        // Column pass (4 → 2) then row pass (4 → 2) → the 2×2 output tile.
                        for (var j = 0; j < 4; j++)
                        {
                            Output1D(mm[0 * 4 + j], mm[1 * 4 + j], mm[2 * 4 + j], mm[3 * 4 + j], out var a0, out var a1);
                            tmp[0 * 4 + j] = a0;
                            tmp[1 * 4 + j] = a1;
                        }

                        for (var i = 0; i < 2; i++)
                        {
                            Output1D(tmp[i * 4 + 0], tmp[i * 4 + 1], tmp[i * 4 + 2], tmp[i * 4 + 3], out var y0, out var y1);
                            var oy = th * 2 + i;
                            if (oy >= outH)
                            {
                                continue;
                            }
                            var ox = tw * 2;
                            if (ox < outW)
                            {
                                output[outBase + oy * outW + ox] = y0;
                            }
                            if (ox + 1 < outW)
                            {
                                output[outBase + oy * outW + ox + 1] = y1;
                            }
                        }
                    }
                }
            }
        }

        // Constant 1-D transforms (rows of G / Bᵀ / Aᵀ for F(2,3)).
        private static void Filter1D(float g0, float g1, float g2, out float o0, out float o1, out float o2, out float o3)
        {
            o0 = g0;
            o1 = 0.5f * (g0 + g1 + g2);
            o2 = 0.5f * (g0 - g1 + g2);
            o3 = g2;
        }

        private static void Input1D(float d0, float d1, float d2, float d3, out float o0, out float o1, out float o2, out float o3)
        {
            o0 = d0 - d2;
            o1 = d1 + d2;
            o2 = d2 - d1;
            o3 = d1 - d3;
        }

        private static void Output1D(float m0, float m1, float m2, float m3, out float o0, out float o1)
        {
            o0 = m0 + m1 + m2;
            o1 = m1 - m2 - m3;
        }
    }
}
