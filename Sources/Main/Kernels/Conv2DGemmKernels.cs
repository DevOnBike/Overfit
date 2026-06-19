// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using DevOnBike.Overfit.Intrinsics;
using DevOnBike.Overfit.Runtime;
using DevOnBike.Overfit.Tensors;

namespace DevOnBike.Overfit.Kernels
{
    /// <summary>
    /// Conv2D as im2col + a register-blocked SIMD GEMM — the structural path that closes most of the gap to a
    /// native conv (MLAS). The convolution becomes <c>C[outC, outHW] = kernel[outC, K] @ cols[K, outHW]</c> with
    /// <c>K = inChannels·kernel²</c>: im2col gathers the (stride/pad-aware) input patches into a contiguous
    /// <c>cols[K, N]</c> matrix once, then a blocked GEMM keeps an 8×8 output tile resident in AVX2/FMA registers
    /// across the whole K contraction (no per-(k) accumulator round-trip), with the B panel packed contiguous for
    /// sequential streaming. Parallelised over N-panels. Requires AVX2+FMA; callers fall back to the direct-conv
    /// kernels otherwise. Bias is applied by the caller (this computes the bare matmul).
    /// </summary>
    internal static class Conv2DGemmKernels
    {
        private const int Mr = 8; // micro-kernel rows (output channels) — one AVX register accumulator per row
        private const int Nr = 8; // micro-kernel cols (spatial positions) — one Vector256<float> wide

        public static bool IsSupported => CpuFeatures.HasFma;

        public static void Forward(
            ReadOnlySpan<float> input,   // [batch, inChannels, H, W]
            ReadOnlySpan<float> kernels, // [outChannels, inChannels, k, k] == [outChannels, K]
            Span<float> output,          // [batch, outChannels, outH, outW]
            int batchSize,
            int inChannels,
            int outChannels,
            int inputH,
            int inputW,
            int kernelSize,
            int padding,
            int stride)
        {
            var outH = (inputH + 2 * padding - kernelSize) / stride + 1;
            var outW = (inputW + 2 * padding - kernelSize) / stride + 1;
            var n = outH * outW;
            var k = inChannels * kernelSize * kernelSize;
            var m = outChannels;

            var inputPlane = inChannels * inputH * inputW;
            var outputPlane = outChannels * outH * outW;

            // 1×1 stride-1 unpadded conv: im2col is the identity (cols[k, pos] = input[ic, pos], k == ic,
            // N == H·W). Skip the copy and GEMM straight on the input — a real win for ResNet's many 1×1 bottleneck
            // layers (K = inChannels here, so no patch gathering at all).
            if (kernelSize == 1 && padding == 0 && stride == 1)
            {
                for (var b = 0; b < batchSize; b++)
                {
                    Gemm(kernels, input.Slice(b * inputPlane, inputPlane), output.Slice(b * outputPlane, outputPlane), m, n, k);
                }
                return;
            }

            using var colsBuf = new PooledBuffer<float>(checked(k * n), clearMemory: false);
            var cols = colsBuf.Span;

            for (var b = 0; b < batchSize; b++)
            {
                Im2Col(
                    input.Slice(b * inputPlane, inputPlane), cols,
                    inChannels, inputH, inputW, kernelSize, padding, stride, outH, outW);

                Gemm(kernels, cols, output.Slice(b * outputPlane, outputPlane), m, n, k);
            }
        }

        // cols[krow, pos] = input[ic, oy·stride-pad+ky, ox·stride-pad+kx] (0 outside the image), where
        // krow = (ic·k + ky)·k + kx (matches the [outC, inC, k, k] kernel's flattened K), pos = oy·outW + ox.
        private static void Im2Col(
            ReadOnlySpan<float> input, Span<float> cols,
            int inChannels, int inputH, int inputW, int kernelSize, int padding, int stride, int outH, int outW)
        {
            var n = outH * outW;
            for (var ic = 0; ic < inChannels; ic++)
            {
                var inChanBase = ic * inputH * inputW;
                for (var ky = 0; ky < kernelSize; ky++)
                {
                    for (var kx = 0; kx < kernelSize; kx++)
                    {
                        var krow = ((ic * kernelSize) + ky) * kernelSize + kx;
                        var dst = cols.Slice(krow * n, n);

                        for (var oy = 0; oy < outH; oy++)
                        {
                            var iy = oy * stride - padding + ky;
                            var rowDst = dst.Slice(oy * outW, outW);
                            if ((uint)iy >= (uint)inputH)
                            {
                                rowDst.Clear();
                                continue;
                            }

                            var inRowBase = inChanBase + iy * inputW;
                            for (var ox = 0; ox < outW; ox++)
                            {
                                var ix = ox * stride - padding + kx;
                                rowDst[ox] = (uint)ix < (uint)inputW ? input[inRowBase + ix] : 0f;
                            }
                        }
                    }
                }
            }
        }

        // C[M,N] = A[M,K] @ B[K,N], parallelised over N-panels (each worker packs its 8-col B panel and sweeps M
        // with the full-K register-blocked micro-kernel). NOTE: a BLIS-style K-blocked + A-packed variant was
        // tried and MEASURED to regress on these CNN dims (deepcnn 101→125, vgg 140→189, resnet 45→118 ms) —
        // most im2col K values are ≤ a few hundred (single K-block → no blocking benefit) while the one-time A
        // pack adds single-threaded O(M·K) overhead. Cache-blocking pays on large dense GEMM, not CNN-shaped im2col.
        // Internal so the Winograd path can reuse the same tuned micro-kernel for its 16 element-wise GEMMs.
        internal static unsafe void Gemm(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> c, int m, int n, int k)
        {
            var nPanels = (n + Nr - 1) / Nr;
            fixed (float* pa = a, pb = b, pc = c)
            {
                var ctx = new GemmCtx(pa, pb, pc, m, n, k);
                OverfitParallel.For(0, nPanels, 1, &GemmNPanelWorker, &ctx);
            }
        }

        private readonly unsafe struct GemmCtx
        {
            public readonly float* A;
            public readonly float* B;
            public readonly float* C;
            public readonly int M;
            public readonly int N;
            public readonly int K;

            public GemmCtx(float* a, float* b, float* c, int m, int n, int k)
            {
                A = a;
                B = b;
                C = c;
                M = m;
                N = n;
                K = k;
            }
        }

        private static unsafe void GemmNPanelWorker(int npStart, int npEnd, void* ctxPtr)
        {
            ref readonly var c = ref Unsafe.AsRef<GemmCtx>(ctxPtr);
            var k = c.K;
            var n = c.N;
            var m = c.M;

            using var packBuf = new PooledBuffer<float>(checked(k * Nr), clearMemory: false);
            var packB = packBuf.Span;

            for (var np = npStart; np < npEnd; np++)
            {
                var n0 = np * Nr;
                var nrEff = Math.Min(Nr, n - n0);

                for (var kk = 0; kk < k; kk++)
                {
                    var srcBase = kk * n + n0;
                    var dstBase = kk * Nr;
                    for (var j = 0; j < Nr; j++)
                    {
                        packB[dstBase + j] = j < nrEff ? c.B[srcBase + j] : 0f;
                    }
                }

                fixed (float* pPackB = packB)
                {
                    for (var m0 = 0; m0 < m; m0 += Mr)
                    {
                        var mrEff = Math.Min(Mr, m - m0);
                        if (mrEff == Mr)
                        {
                            MicroKernel8x8(c.A, m0, k, pPackB, c.C, n, n0, nrEff);
                        }
                        else
                        {
                            MicroKernelTail(c.A, m0, mrEff, k, pPackB, c.C, n, n0, nrEff);
                        }
                    }
                }
            }
        }

        // Full 8×8 micro-kernel: C[m0..m0+8, n0..n0+nrEff) = A[m0..m0+8, :K] @ packB[:K, :8]. Eight accumulators
        // (one per output row) live in registers across the entire K loop; one FMA per (row,k) against the shared
        // 8-wide B vector. Stores nrEff lanes of each row.
        private static unsafe void MicroKernel8x8(float* a, int m0, int k, float* packB, float* c, int n, int n0, int nrEff)
        {
            var a0 = a + (long)(m0 + 0) * k;
            var a1 = a + (long)(m0 + 1) * k;
            var a2 = a + (long)(m0 + 2) * k;
            var a3 = a + (long)(m0 + 3) * k;
            var a4 = a + (long)(m0 + 4) * k;
            var a5 = a + (long)(m0 + 5) * k;
            var a6 = a + (long)(m0 + 6) * k;
            var a7 = a + (long)(m0 + 7) * k;

            var acc0 = Vector256<float>.Zero;
            var acc1 = Vector256<float>.Zero;
            var acc2 = Vector256<float>.Zero;
            var acc3 = Vector256<float>.Zero;
            var acc4 = Vector256<float>.Zero;
            var acc5 = Vector256<float>.Zero;
            var acc6 = Vector256<float>.Zero;
            var acc7 = Vector256<float>.Zero;

            for (var kk = 0; kk < k; kk++)
            {
                var bVec = Avx.LoadVector256(packB + kk * Nr);
                acc0 = Fma.MultiplyAdd(Vector256.Create(a0[kk]), bVec, acc0);
                acc1 = Fma.MultiplyAdd(Vector256.Create(a1[kk]), bVec, acc1);
                acc2 = Fma.MultiplyAdd(Vector256.Create(a2[kk]), bVec, acc2);
                acc3 = Fma.MultiplyAdd(Vector256.Create(a3[kk]), bVec, acc3);
                acc4 = Fma.MultiplyAdd(Vector256.Create(a4[kk]), bVec, acc4);
                acc5 = Fma.MultiplyAdd(Vector256.Create(a5[kk]), bVec, acc5);
                acc6 = Fma.MultiplyAdd(Vector256.Create(a6[kk]), bVec, acc6);
                acc7 = Fma.MultiplyAdd(Vector256.Create(a7[kk]), bVec, acc7);
            }

            StoreRow(c, (long)(m0 + 0) * n + n0, acc0, nrEff);
            StoreRow(c, (long)(m0 + 1) * n + n0, acc1, nrEff);
            StoreRow(c, (long)(m0 + 2) * n + n0, acc2, nrEff);
            StoreRow(c, (long)(m0 + 3) * n + n0, acc3, nrEff);
            StoreRow(c, (long)(m0 + 4) * n + n0, acc4, nrEff);
            StoreRow(c, (long)(m0 + 5) * n + n0, acc5, nrEff);
            StoreRow(c, (long)(m0 + 6) * n + n0, acc6, nrEff);
            StoreRow(c, (long)(m0 + 7) * n + n0, acc7, nrEff);
        }

        // Edge micro-kernel for the last M-panel (1..7 rows). Generic loop; small relative to the bulk.
        private static unsafe void MicroKernelTail(float* a, int m0, int mrEff, int k, float* packB, float* c, int n, int n0, int nrEff)
        {
            for (var mi = 0; mi < mrEff; mi++)
            {
                var aRow = a + (long)(m0 + mi) * k;
                var acc = Vector256<float>.Zero;
                for (var kk = 0; kk < k; kk++)
                {
                    acc = Fma.MultiplyAdd(Vector256.Create(aRow[kk]), Avx.LoadVector256(packB + kk * Nr), acc);
                }
                StoreRow(c, (long)(m0 + mi) * n + n0, acc, nrEff);
            }
        }

        private static unsafe void StoreRow(float* c, long cBase, Vector256<float> acc, int nrEff)
        {
            if (nrEff == Nr)
            {
                Avx.Store(c + cBase, acc);
                return;
            }

            var tmp = stackalloc float[Nr];
            Avx.Store(tmp, acc);
            for (var j = 0; j < nrEff; j++)
            {
                c[cBase + j] = tmp[j];
            }
        }
    }
}
