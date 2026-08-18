// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Diagnostics;
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

        /// <summary>
        /// Opt-in split of conv time into the im2col patch gather versus the GEMM. Off by default and checked
        /// before any timestamp, so the inference path is unchanged when it is off.
        ///
        /// <para>Needed because "Conv is 90.7% of VGG-16" does not say <i>which half</i>. The gather moves
        /// O(K·N) floats and the GEMM does O(M·N·K) FLOPs, so their ratio varies enormously across layers —
        /// and the answer decides whether the next lever is the micro-kernel or the gather. Guessing which
        /// would be guessing about mechanism.</para>
        /// </summary>
        public static bool ProfileParts;

        private static long _im2colTicks;
        private static long _gemmTicks;

        /// <summary>
        /// Measurement-only: skip the B-panel pack, leaving the micro-kernel to run over whatever the previous
        /// panel left in the buffer. Produces WRONG results by construction.
        ///
        /// <para>Splits the GEMM's 54 ms into packing versus arithmetic without putting timestamps inside the
        /// parallel region (where per-thread accumulation and Interlocked would distort what is being measured).
        /// The pack is the prime suspect: it reads <c>B[kk·n + n0]</c> with stride <c>n</c> — 200 KB apart on
        /// VGG's early layers — one scalar element at a time with a bounds branch each, and across a whole GEMM
        /// it moves the entire im2col matrix once more. <b>MEASURED 2026-08-17 and REFUTED: the micro-kernel
        /// costs ~38 ms against the pack's ~11.5 ms on VGG-16.</b> The suspicion above is kept as the record
        /// of what was believed before the ablation could run.</para>
        ///
        /// <para><b>Honoured by BOTH GEMM workers since 2026-08-17, and it was honoured by neither on an
        /// AVX-512 box before that.</b> The checks existed only in <see cref="GemmNPanelWorker"/>, the AVX2
        /// path, while every machine with AVX-512 runs <see cref="GemmNPanelWorker512"/> and ignored them —
        /// so <c>ConvGemm_PackVersusMicroKernel_Split</c> printed three numbers that were pure run-to-run
        /// noise, and printed them in a table that looks like a result. Measured on VGG-16 before the fix:
        /// baseline 84.90 ms, "pack only" 86.50 ms (i.e. removing the arithmetic made it SLOWER) and
        /// "micro only" 81.68 ms. A diagnostic whose arms cannot move is worse than no diagnostic, because
        /// somebody acts on it.</para>
        /// </summary>
        internal static bool AblatePackB;

        /// <summary>
        /// Measurement-only: skip the micro-kernel, leaving only the pack. Wrong results by construction.
        ///
        /// <para>See <see cref="AblatePackB"/> for why this is checked in both workers rather than one.</para>
        /// </summary>
        internal static bool AblateMicroKernel;

        /// <summary>Clears the part accumulators (call before the measured segment).</summary>
        public static void ResetPartProfile()
        {
            _im2colTicks = 0;
            _gemmTicks = 0;
        }

        /// <summary>im2col versus GEMM, in milliseconds and as a share of the two combined.</summary>
        public static string PartProfileReport()
        {
            var toMs = 1000.0 / Stopwatch.Frequency;
            var total = _im2colTicks + _gemmTicks;
            var share = total == 0 ? 1.0 : total;

#pragma warning disable OVERFIT047 // Wall-clock milliseconds and their share. The only consumer in this repository
            // is Tests/Diagnostics/ConvGemmPartProfileTests.cs:90, which writes it to test output for a person to
            // read; the numbers differ between two runs on the same machine, so nothing downstream can match on
            // them and there is no determinism to protect.
            return $"im2col {_im2colTicks * toMs,8:F2} ms {100.0 * _im2colTicks / share,5:F1}%   "
                + $"gemm {_gemmTicks * toMs,8:F2} ms {100.0 * _gemmTicks / share,5:F1}%";
#pragma warning restore OVERFIT047
        }

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
                var t0 = ProfileParts ? Stopwatch.GetTimestamp() : 0L;

                Im2Col(
                    input.Slice(b * inputPlane, inputPlane), cols,
                    inChannels, inputH, inputW, kernelSize, padding, stride, outH, outW);

                var t1 = ProfileParts ? Stopwatch.GetTimestamp() : 0L;

                Gemm(kernels, cols, output.Slice(b * outputPlane, outputPlane), m, n, k);

                if (ProfileParts)
                {
                    _im2colTicks += t1 - t0;
                    _gemmTicks += Stopwatch.GetTimestamp() - t1;
                }
            }
        }

        /// <summary>
        /// Minimum <c>K·N</c> before im2col fans out. Below this the patch gather is a fraction of a
        /// millisecond and the dispatch would cost more than the work.
        /// </summary>
        private const int ParallelIm2ColMinElements = 1 << 16;

        /// <summary>Set <c>OVERFIT_PARALLEL_IM2COL=0</c> to force the original serial gather (A/B switch).</summary>
        internal static bool UseParallelIm2Col =
            Environment.GetEnvironmentVariable(OverfitEnvironment.ParallelIm2Col) != "0";

        // cols[krow, pos] = input[ic, oy·stride-pad+ky, ox·stride-pad+kx] (0 outside the image), where
        // krow = (ic·k + ky)·k + kx (matches the [outC, inC, k, k] kernel's flattened K), pos = oy·outW + ox.
        //
        // Parallelised over krow. Each krow owns a disjoint n-element row of `cols` and only ever READS the
        // input, so the fan-out needs no synchronisation and is bit-identical to the serial gather.
        //
        // This matters more than it looks. The GEMM below was already parallel but the gather was not, and on
        // VGG-16 the gather is enormous — conv1_2 alone materialises a [576 × 50176] matrix (115 MB) one
        // scalar element at a time. An Amdahl fit over the measured worker sweep (663 ms at 1 worker, 138 ms
        // at 16) put the serial fraction at ~15.5%, i.e. **~103 of those 138 ms were serial**, which is why 16
        // workers bought only 4.79× instead of ~14×.
        private static unsafe void Im2Col(
            ReadOnlySpan<float> input, Span<float> cols,
            int inChannels, int inputH, int inputW, int kernelSize, int padding, int stride, int outH, int outW)
        {
            var n = outH * outW;
            var kRows = inChannels * kernelSize * kernelSize;

            if (!UseParallelIm2Col || (long)kRows * n < ParallelIm2ColMinElements)
            {
                for (var krow = 0; krow < kRows; krow++)
                {
                    Im2ColRow(input, cols, krow, inputH, inputW, kernelSize, padding, stride, outH, outW, n);
                }

                return;
            }

            fixed (float* pIn = input, pCols = cols)
            {
                var ctx = new Im2ColCtx(
                    pIn, pCols, inChannels, inputH, inputW, kernelSize, padding, stride, outH, outW, n);

                OverfitParallel.For(0, kRows, 1, &Im2ColRowRange, &ctx);
            }
        }

        private static unsafe void Im2ColRowRange(int start, int end, void* context)
        {
            ref var ctx = ref Unsafe.AsRef<Im2ColCtx>(context);

            var input = new ReadOnlySpan<float>(ctx.Input, ctx.InChannels * ctx.InputH * ctx.InputW);
            var cols = new Span<float>(ctx.Cols, ctx.KRows * ctx.N);

            for (var krow = start; krow < end; krow++)
            {
                Im2ColRow(
                    input, cols, krow,
                    ctx.InputH, ctx.InputW, ctx.KernelSize, ctx.Padding, ctx.Stride, ctx.OutH, ctx.OutW, ctx.N);
            }
        }

        /// <summary>One row of the im2col matrix — the unit of both the serial and the parallel path, so the
        /// two cannot drift apart.</summary>
        private static void Im2ColRow(
            ReadOnlySpan<float> input, Span<float> cols, int krow,
            int inputH, int inputW, int kernelSize, int padding, int stride, int outH, int outW, int n)
        {
            var kx = krow % kernelSize;
            var ky = (krow / kernelSize) % kernelSize;
            var ic = krow / (kernelSize * kernelSize);

            var inChanBase = ic * inputH * inputW;
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

        private readonly unsafe struct Im2ColCtx
        {
            public Im2ColCtx(
                float* input, float* cols,
                int inChannels, int inputH, int inputW, int kernelSize,
                int padding, int stride, int outH, int outW, int n)
            {
                Input = input;
                Cols = cols;
                InChannels = inChannels;
                InputH = inputH;
                InputW = inputW;
                KernelSize = kernelSize;
                Padding = padding;
                Stride = stride;
                OutH = outH;
                OutW = outW;
                N = n;
                KRows = inChannels * kernelSize * kernelSize;
            }

            public readonly float* Input;
            public readonly float* Cols;
            public readonly int InChannels;
            public readonly int InputH;
            public readonly int InputW;
            public readonly int KernelSize;
            public readonly int Padding;
            public readonly int Stride;
            public readonly int OutH;
            public readonly int OutW;
            public readonly int N;
            public readonly int KRows;
        }

        // C[M,N] = A[M,K] @ B[K,N], parallelised over N-panels (each worker packs its 8-col B panel and sweeps M
        // with the full-K register-blocked micro-kernel). NOTE: a BLIS-style K-blocked + A-packed variant was
        // tried and MEASURED to regress on these CNN dims (deepcnn 101→125, vgg 140→189, resnet 45→118 ms) —
        // most im2col K values are ≤ a few hundred (single K-block → no blocking benefit) while the one-time A
        // pack adds single-threaded O(M·K) overhead. Cache-blocking pays on large dense GEMM, not CNN-shaped im2col.
        // Internal so the Winograd path can reuse the same tuned micro-kernel for its 16 element-wise GEMMs.
        /// <summary>Micro-kernel columns for the AVX-512 path: 8 rows × 32 columns = 16 zmm accumulators.</summary>
        private const int Nr512 = 32;

        /// <summary>
        /// Route the conv GEMM through the AVX-512 8×32 micro-kernel; <c>OVERFIT_CONV_AVX512=0</c> forces the
        /// AVX2 8×8 path.
        ///
        /// <para><b>Why a wider tile and not a faster one.</b> Shape benchmarks showed the 8×8 kernel already
        /// runs at 83% of this machine's single-core FMA peak, so there is no instruction-level headroom. What
        /// a wider tile changes is <b>arithmetic intensity</b>: per k-step an <c>Mr×Nr</c> tile loads
        /// <c>Mr + Nr</c> floats and performs <c>2·Mr·Nr</c> FLOPs, giving <c>Mr·Nr / (2(Mr+Nr))</c> FLOP per
        /// byte — <b>2.0 at 8×8, 3.2 at 8×32</b>. That is 38% less memory traffic for the same arithmetic.</para>
        ///
        /// <para><b>Why that is the lever here.</b> The machine probe found bandwidth scales 10–13× across
        /// cores only while the per-core working set stays under ~2 MB, and collapses to 1–2× beyond it — so a
        /// parallel kernel that outruns its cache cannot be fixed by adding cores or by blocking, only by
        /// needing fewer bytes per FLOP.</para>
        /// </summary>
        internal static bool UseAvx512Conv =
            CpuFeatures.HasAvx512
            && Environment.GetEnvironmentVariable(OverfitEnvironment.ConvAvx512) != "0";

        // C[M,N] = A[M,K] @ B[K,N], parallelised over N-panels (each worker packs its B panel and sweeps M
        // with the full-K register-blocked micro-kernel). NOTE: a BLIS-style K-blocked + A-packed variant was
        // tried and MEASURED to regress on these CNN dims (deepcnn 101→125, vgg 140→189, resnet 45→118 ms) —
        // most im2col K values are ≤ a few hundred (single K-block → no blocking benefit) while the one-time A
        // pack adds single-threaded O(M·K) overhead. Cache-blocking pays on large dense GEMM, not CNN-shaped im2col.
        // Internal so the Winograd path can reuse the same tuned micro-kernel for its 16 element-wise GEMMs.
        internal static unsafe void Gemm(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> c, int m, int n, int k)
        {
            var nr = UseAvx512Conv ? Nr512 : Nr;
            var nPanels = (n + nr - 1) / nr;
            var mBlocks = UseAvx512Conv ? ResolveMBlocks(m, nPanels) : 1;

            fixed (float* pa = a, pb = b, pc = c)
            {
                var ctx = new GemmCtx(pa, pb, pc, m, n, k, mBlocks);

                if (UseAvx512Conv)
                {
                    OverfitParallel.For(0, nPanels * mBlocks, 1, &GemmNPanelWorker512, &ctx);
                    return;
                }

                OverfitParallel.For(0, nPanels, 1, &GemmNPanelWorker, &ctx);
            }
        }

        /// <summary>Set <c>OVERFIT_CONV_M_SPLIT=0</c> to dispatch one work item per N-panel, as before.</summary>
        internal static readonly bool MSplitEnabled =
            Environment.GetEnvironmentVariable(OverfitEnvironment.ConvMSplit) != "0";

        /// <summary>
        /// How many ways to split the M sweep, so a GEMM with few N-panels can still fill the machine.
        ///
        /// <para><b>Measured on VGG-16, 2026-08-18 (XC-78).</b> Work is dispatched one N-panel per item and
        /// an AVX-512 panel is 32 columns wide, so VGG's last three convolutions — <c>N = 196</c> — produce
        /// <b>seven</b> work items. Seven cannot occupy 32 workers whatever the micro-kernel does, and the
        /// seventh panel is 4 columns wide against the others' 32, so even those seven are imbalanced. Those
        /// layers measured <b>2.84x scaling against this machine's own 14.30x ceiling, i.e. 20% efficiency,
        /// while the well-shaped layers reached 7.3x</b> — and they are 20% of all-core convolution time.</para>
        ///
        /// <para><b>The split costs a duplicated B pack</b>, once per M-block rather than once per panel, so
        /// it is a trade and not a free win. It is therefore applied only where there are fewer panels than
        /// workers, and never widens the domain past the number of <c>Mr</c> row blocks that exist.</para>
        /// </summary>
        private static int ResolveMBlocks(int m, int nPanels)
        {
            if (!MSplitEnabled)
            {
                return 1;
            }

            var workers = OverfitParallel.MaxDegreeOfParallelism;

            // MEASURED 2026-08-18: the shortfall has to be large before the duplicated pack pays for
            // itself. VGG's conv8-10 produce 25 panels against 32 workers, and splitting them cost 3-5%
            // (668.8 -> 633.7 GFLOP/s on conv8) because the extra B pack outweighed a occupancy already at
            // 78%. Its conv11-13 produce 7, and splitting them paid 1.79x (228 -> 405 GFLOP/s). Requiring
            // fewer than half the workers to be reachable keeps the second case and drops the first.
            if (nPanels * 2 > workers || nPanels <= 0)
            {
                return 1;
            }

            var rowBlocks = (m + Mr - 1) / Mr;

            if (rowBlocks <= 1)
            {
                return 1;
            }

            var wanted = (workers + nPanels - 1) / nPanels;

            return Math.Min(rowBlocks, wanted);
        }

        /// <summary>
        /// One 32-column panel per work item: pack it, then sweep M with the 8×32 AVX-512 micro-kernel.
        /// Structurally identical to <see cref="GemmNPanelWorker"/>, only wider.
        /// </summary>
        private static unsafe void GemmNPanelWorker512(int npStart, int npEnd, void* ctxPtr)
        {
            ref readonly var c = ref Unsafe.AsRef<GemmCtx>(ctxPtr);
            var k = c.K;
            var n = c.N;
            var m = c.M;
            var mBlocks = c.MBlocks;
            var rowBlocks = (m + Mr - 1) / Mr;

            using var packBuf = new PooledBuffer<float>(checked(k * Nr512), clearMemory: false);
            var packB = packBuf.Span;

            for (var item = npStart; item < npEnd; item++)
            {
                // One item is one (N-panel, M-block) pair. At MBlocks == 1 the row range below is the whole
                // of M, so the dispatch is byte-for-byte the original loop wherever the split does not apply.
                var np = item / mBlocks;
                var mb = item - (np * mBlocks);

                var rowBlockStart = (int)((long)rowBlocks * mb / mBlocks);
                var rowBlockEnd = (int)((long)rowBlocks * (mb + 1) / mBlocks);

                if (rowBlockStart >= rowBlockEnd)
                {
                    continue;
                }

                var n0 = np * Nr512;
                var nrEff = Math.Min(Nr512, n - n0);

                // Both ablation switches are checked HERE as well as in the AVX2 worker. They used to exist
                // only there, which made the pack-versus-micro-kernel diagnostic inert on every AVX-512
                // machine — see AblatePackB.
                if (!AblatePackB)
                {
                    for (var kk = 0; kk < k; kk++)
                    {
                        var srcBase = (kk * n) + n0;
                        var dstBase = kk * Nr512;

                        for (var j = 0; j < Nr512; j++)
                        {
                            packB[dstBase + j] = j < nrEff ? c.B[srcBase + j] : 0f;
                        }
                    }
                }

                if (AblateMicroKernel)
                {
                    continue;
                }

                fixed (float* pPackB = packB)
                {
                    for (var rb = rowBlockStart; rb < rowBlockEnd; rb++)
                    {
                        var m0 = rb * Mr;
                        var mrEff = Math.Min(Mr, m - m0);

                        MicroKernel8x32Avx512(c.A, m0, mrEff, k, pPackB, c.C, n, n0, nrEff);
                    }
                }
            }
        }

        /// <summary>
        /// 8 rows × 32 columns in sixteen <see cref="Vector512{T}"/> accumulators, held in registers across the
        /// whole K contraction. Per k-step: two 512-bit B loads and up to eight A broadcasts feed sixteen FMAs
        /// — 3.2 FLOP per byte loaded, against 2.0 for the 8×8 AVX2 kernel.
        ///
        /// <para>Handles a partial row block (<paramref name="mrEff"/> &lt; 8) and a partial column tail
        /// (<paramref name="nrEff"/> &lt; 32) with scalar stores rather than a separate kernel; both are edge
        /// cases of the last block, not the steady state.</para>
        /// </summary>
        private static unsafe void MicroKernel8x32Avx512(
            float* a, int m0, int mrEff, int k, float* packB, float* c, int n, int n0, int nrEff)
        {
            // Row pointers, clamped to the last valid row so a short block reads in-bounds; the extra rows'
            // results are simply not stored below.
            var rows = stackalloc float*[Mr];
            for (var r = 0; r < Mr; r++)
            {
                rows[r] = a + ((long)(m0 + Math.Min(r, mrEff - 1)) * k);
            }

            Vector512<float> c00 = default, c01 = default, c10 = default, c11 = default;
            Vector512<float> c20 = default, c21 = default, c30 = default, c31 = default;
            Vector512<float> c40 = default, c41 = default, c50 = default, c51 = default;
            Vector512<float> c60 = default, c61 = default, c70 = default, c71 = default;

            for (var kk = 0; kk < k; kk++)
            {
                var b0 = Vector512.Load(packB + (kk * Nr512));
                var b1 = Vector512.Load(packB + (kk * Nr512) + 16);

                var r = Vector512.Create(rows[0][kk]);
                c00 = Avx512F.FusedMultiplyAdd(r, b0, c00);
                c01 = Avx512F.FusedMultiplyAdd(r, b1, c01);
                r = Vector512.Create(rows[1][kk]);
                c10 = Avx512F.FusedMultiplyAdd(r, b0, c10);
                c11 = Avx512F.FusedMultiplyAdd(r, b1, c11);
                r = Vector512.Create(rows[2][kk]);
                c20 = Avx512F.FusedMultiplyAdd(r, b0, c20);
                c21 = Avx512F.FusedMultiplyAdd(r, b1, c21);
                r = Vector512.Create(rows[3][kk]);
                c30 = Avx512F.FusedMultiplyAdd(r, b0, c30);
                c31 = Avx512F.FusedMultiplyAdd(r, b1, c31);
                r = Vector512.Create(rows[4][kk]);
                c40 = Avx512F.FusedMultiplyAdd(r, b0, c40);
                c41 = Avx512F.FusedMultiplyAdd(r, b1, c41);
                r = Vector512.Create(rows[5][kk]);
                c50 = Avx512F.FusedMultiplyAdd(r, b0, c50);
                c51 = Avx512F.FusedMultiplyAdd(r, b1, c51);
                r = Vector512.Create(rows[6][kk]);
                c60 = Avx512F.FusedMultiplyAdd(r, b0, c60);
                c61 = Avx512F.FusedMultiplyAdd(r, b1, c61);
                r = Vector512.Create(rows[7][kk]);
                c70 = Avx512F.FusedMultiplyAdd(r, b0, c70);
                c71 = Avx512F.FusedMultiplyAdd(r, b1, c71);
            }

            var tile = stackalloc float[Nr512];

            StoreTile(c, n, n0, m0, 0, mrEff, nrEff, c00, c01, tile);
            StoreTile(c, n, n0, m0, 1, mrEff, nrEff, c10, c11, tile);
            StoreTile(c, n, n0, m0, 2, mrEff, nrEff, c20, c21, tile);
            StoreTile(c, n, n0, m0, 3, mrEff, nrEff, c30, c31, tile);
            StoreTile(c, n, n0, m0, 4, mrEff, nrEff, c40, c41, tile);
            StoreTile(c, n, n0, m0, 5, mrEff, nrEff, c50, c51, tile);
            StoreTile(c, n, n0, m0, 6, mrEff, nrEff, c60, c61, tile);
            StoreTile(c, n, n0, m0, 7, mrEff, nrEff, c70, c71, tile);
        }

        private static unsafe void StoreTile(
            float* c, int n, int n0, int m0, int row, int mrEff, int nrEff,
            Vector512<float> lo, Vector512<float> hi, float* scratch)
        {
            if (row >= mrEff)
            {
                return;
            }

            var dst = c + ((long)(m0 + row) * n) + n0;

            if (nrEff == Nr512)
            {
                lo.Store(dst);
                hi.Store(dst + 16);
                return;
            }

            lo.Store(scratch);
            hi.Store(scratch + 16);

            for (var j = 0; j < nrEff; j++)
            {
                dst[j] = scratch[j];
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

            /// <summary>How many ways the M sweep is split, so one N-panel can occupy more than one
            /// worker. 1 reproduces the original one-item-per-panel dispatch exactly.</summary>
            public readonly int MBlocks;

            public GemmCtx(float* a, float* b, float* c, int m, int n, int k, int mBlocks)
            {
                A = a;
                B = b;
                C = c;
                M = m;
                N = n;
                K = k;
                MBlocks = mBlocks;
            }
        }

        /// <summary>
        /// N-panels packed and swept together, so the A row-block is read once per <i>group</i> rather than
        /// once per panel. 1 reproduces the original loop exactly.
        ///
        /// <para><b>Measured motivation.</b> The original loop sweeps M inside the panel loop, so the whole A
        /// matrix is re-read for every N-panel: on VGG-16 that is <b>~7.5 GB of A traffic for 30.7 GFLOP of
        /// arithmetic — 0.24 bytes/FLOP where a blocked GEMM runs at ~0.01</b>, and 7.5 GB in 73 ms is
        /// ≈103 GB/s against a 90 GB/s DRAM read ceiling. It is why one thread reaches 132 GFLOP/s on this
        /// kernel while sixteen reach only ~30 each: alone, a core has the cache to itself.</para>
        ///
        /// <para>Grouping trades A traffic for a larger packed-B working set (<c>K · Nr · group</c> floats), so
        /// the useful group size is bounded by cache, not by the arithmetic — hence a flag rather than a
        /// constant, and a measurement rather than a guess.</para>
        /// </summary>
        internal static int NPanelGroup = ResolveNPanelGroup();

        private static int ResolveNPanelGroup()
        {
            var raw = Environment.GetEnvironmentVariable(OverfitEnvironment.ConvPanelGroup);

            // Default 1 — the original per-panel loop. Grouping was built to cut A re-reads and MEASURED to do
            // nothing: interleaved with an ORT canary (4% spread), groups 1/2/4 came out 72.7/73.0/73.3 ms.
            // The traffic argument that motivated it (7.5 GB of A re-reads against a 90 GB/s DRAM ceiling) was
            // wrong about WHERE the traffic goes: this CPU has 128 MB of L3, so every layer's A (≤9 MB) is
            // re-read from L3, not DRAM. Kept behind the flag because the negative is worth preserving.
            return int.TryParse(raw, out var parsed) && parsed >= 1 ? parsed : 1;
        }

        private static unsafe void GemmNPanelWorker(int npStart, int npEnd, void* ctxPtr)
        {
            ref readonly var c = ref Unsafe.AsRef<GemmCtx>(ctxPtr);
            var k = c.K;
            var n = c.N;
            var m = c.M;
            var group = NPanelGroup;

            using var packBuf = new PooledBuffer<float>(checked(k * Nr * group), clearMemory: false);
            var packB = packBuf.Span;

            for (var gStart = npStart; gStart < npEnd; gStart += group)
            {
                var gCount = Math.Min(group, npEnd - gStart);

                if (!AblatePackB)
                {
                    for (var g = 0; g < gCount; g++)
                    {
                        var n0 = (gStart + g) * Nr;
                        var nrEff = Math.Min(Nr, n - n0);
                        var panelBase = g * k * Nr;

                        for (var kk = 0; kk < k; kk++)
                        {
                            var srcBase = (kk * n) + n0;
                            var dstBase = panelBase + (kk * Nr);
                            for (var j = 0; j < Nr; j++)
                            {
                                packB[dstBase + j] = j < nrEff ? c.B[srcBase + j] : 0f;
                            }
                        }
                    }
                }

                if (AblateMicroKernel)
                {
                    continue;
                }

                fixed (float* pPackB = packB)
                {
                    // M outermost: each 8-row block of A is loaded once and reused across every panel in the
                    // group, which is the whole point of grouping.
                    for (var m0 = 0; m0 < m; m0 += Mr)
                    {
                        var mrEff = Math.Min(Mr, m - m0);

                        for (var g = 0; g < gCount; g++)
                        {
                            var n0 = (gStart + g) * Nr;
                            var nrEff = Math.Min(Nr, n - n0);
                            var panel = pPackB + ((long)g * k * Nr);

                            if (mrEff == Mr)
                            {
                                MicroKernel8x8(c.A, m0, k, panel, c.C, n, n0, nrEff);
                                continue;
                            }

                            MicroKernelTail(c.A, m0, mrEff, k, panel, c.C, n, n0, nrEff);
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
