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
            ReadOnlySpan<float> packedKernels, // MR-major repack of `kernels`, or empty to build one here
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

            // Fused: the patch gather happens inside the GEMM's own B pack, so the k*n column matrix is
            // never built. On VGG-16's conv2 that matrix alone is 115.6 MB of scratch, written once and
            // read once, for arithmetic that could have read the 3.2 MB input directly.
            if (UseFusedIm2Col && UseAvx512Conv)
            {
                for (var b = 0; b < batchSize; b++)
                {
                    var tf = ProfileParts ? Stopwatch.GetTimestamp() : 0L;

                    GemmFusedIm2Col(
                        kernels,
                        packedKernels,
                        input.Slice(b * inputPlane, inputPlane),
                        output.Slice(b * outputPlane, outputPlane),
                        m,
                        n,
                        k,
                        inputH,
                        inputW,
                        kernelSize,
                        padding,
                        stride,
                        outW);

                    if (ProfileParts)
                    {
                        // There is no separate gather to time any more. The split reports 0% im2col here,
                        // and that is the point rather than a broken instrument.
                        _gemmTicks += Stopwatch.GetTimestamp() - tf;
                    }
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

        /// <summary>
        /// Set <c>OVERFIT_CONV_PACK_A=1</c> to repack the kernel matrix into MR-major micro-panels before
        /// the sweep, so the eight A values a k-step needs are contiguous.
        ///
        /// <para><b>Read from BLIS, whose kernel contract states it outright</b>
        /// (<c>docs/KernelsHowTo.md</c>): the micropanel of A is <i>"stored by columns with leading
        /// dimension PACKMR"</i>, so the MR values consumed per k-step sit side by side. This kernel does
        /// the opposite — <c>rows[r] = a + (m0 + r) * k</c>, then <c>rows[r][kk]</c> — so one k-step reads
        /// eight floats from eight addresses <b>k * 4 bytes apart, which is 18 KB at VGG-16's K = 4608</b>.
        /// Same bytes, eight streams instead of one.</para>
        ///
        /// <para><b>Why this is a better bet here than it was for BLIS.</b> BLIS packs A on every call
        /// because it is a general GEMM and A is caller data. In inference A is the convolution's own
        /// weights, which never change — so the pack can happen once at model load and never again. That
        /// also explains the negative already recorded here for BLIS-style blocking-plus-packing: <b>it
        /// paid the packing cost per call</b>. This switch still packs per call, deliberately, because a
        /// per-call pack is the cheap way to find out whether the layout is worth moving at all; if it
        /// pays, the pack belongs at load time and this cost comes back too.</para>
        ///
        /// <para><b>What would refute it:</b> if the packed layout does not raise the single-threaded rate,
        /// then A's access pattern is not the limit and this whole line of reasoning is wrong.</para>
        /// </summary>
        internal static readonly bool UsePackedA =
            Environment.GetEnvironmentVariable(OverfitEnvironment.ConvPackA) == "1";

        /// <summary>
        /// Contraction length per K-block, or 0 to contract the whole of K in one pass (the original).
        ///
        /// <para><b>The hypothesis, taken from MLAS.</b> <c>MlasSgemmOperation</c> blocks BOTH dimensions —
        /// <c>MLAS_SGEMM_STRIDEN = MLAS_SGEMM_STRIDEK = 128</c> — so its packed B panel is a constant
        /// 128 x 128 floats = <b>64 KB</b> whatever K is, and the A slice it sweeps is <c>M x 128</c>.
        /// Both sit in L2 together. This kernel contracts the whole of K in one pass, so its packed panel
        /// is <c>K x 32</c> — <b>589 KB at VGG-16's K = 4608</b> — and the A it sweeps is the full
        /// <c>M x K</c>, 9.4 MB, which is L3 rather than L2.</para>
        ///
        /// <para><b>The cost, which is why this is a switch and not a constant.</b> Contracting the whole
        /// of K keeps the C tile in registers from start to finish, so C is written exactly once. Blocking
        /// K forces C to be read back and re-accumulated once per block: at VGG-16's conv6 that turns
        /// 3.2 MB of C traffic into about 115 MB. Whether the L1/L2 residency is worth that is a
        /// measurement, not an argument.</para>
        ///
        /// <para><b>Do not read the earlier negative as covering this.</b> A BLIS-style K-blocked AND
        /// A-packed variant was measured here and regressed (vgg 140 -> 189 ms), and its recorded reason
        /// was that <i>most im2col K values are at most a few hundred, so a single K-block means no
        /// blocking benefit</i>. <b>That premise is false for VGG-16</b>, whose K runs 576 to 4608 — at
        /// Kc = 128 the late layers get 36 blocks, not one. The earlier result refutes the pair, not this.</para>
        /// </summary>
        internal static readonly int ConvKBlock = ResolveConvKBlock();

        private static int ResolveConvKBlock()
        {
            var raw = Environment.GetEnvironmentVariable(OverfitEnvironment.ConvKBlock);

            if (!int.TryParse(raw, out var parsed) || parsed <= 0)
            {
                return 0;
            }

            return parsed;
        }

        /// <summary>
        /// Set <c>OVERFIT_CONV_FUSED_IM2COL=0</c> to build the column matrix first, as before the fusion.
        /// </summary>
        internal static readonly bool UseFusedIm2Col =
            Environment.GetEnvironmentVariable(OverfitEnvironment.ConvFusedIm2Col) != "0";

        /// <summary>
        /// The conv GEMM with the patch gather folded into its own B pack, so no column matrix exists.
        ///
        /// <para><b>What this removes.</b> The unfused path materialises a <c>K x N</c> float matrix, then
        /// each worker copies a 32-column slice of it into <c>packB</c>. On VGG-16's conv2 that matrix is
        /// <b>115.6 MB</b>, written once and read once, to hold values that could have been read from the
        /// 3.2 MB input. Measured before the change, the gather was <b>19.8% of convolution time</b>.</para>
        ///
        /// <para><b>The same idea as MLAS, taken one level finer.</b> ONNX Runtime's
        /// <c>MlasConvExpandThenGemmSegmented</c> expands a <c>CountK x CountN</c> block into a column
        /// buffer and GEMMs that, so it never holds the whole matrix either. Here the destination is the
        /// micro-kernel's own packed panel, so there is no intermediate buffer at all — a panel is
        /// <c>K x 32</c> floats, 589 KB at VGG's largest K, which sits inside this core's 1 MB L2.</para>
        ///
        /// <para><b>The cost, so it is not sold as free.</b> The unfused pack reads 32 contiguous floats;
        /// this one computes an input address per element. The row and column bases are hoisted per panel
        /// — 32 divisions for the whole panel rather than one per element — leaving two adds and two
        /// bounds checks in the inner loop. Whether that is cheaper than moving the matrix twice is a
        /// measurement, not an argument, and <c>OVERFIT_CONV_FUSED_IM2COL=0</c> is how it is taken.</para>
        /// </summary>
        private static unsafe void GemmFusedIm2Col(
            ReadOnlySpan<float> kernels,
            ReadOnlySpan<float> packedKernels,
            ReadOnlySpan<float> input,
            Span<float> output,
            int m,
            int n,
            int k,
            int inputH,
            int inputW,
            int kernelSize,
            int padding,
            int stride,
            int outW)
        {
            var nPanels = (n + Nr512 - 1) / Nr512;
            var mBlocks = ResolveMBlocks(m, nPanels);

            // Padded to whole MR row blocks so the last block needs no special case; the padding rows
            // contribute zeros and are discarded at the store, exactly as the clamped rows were.
            var rowBlocksTotal = (m + Mr - 1) / Mr;

            // A caller that owns the weights repacks them once and hands the result in; that is the
            // whole point, because a convolution's A matrix never changes between inferences. Packing
            // here is only the fallback for callers with nowhere to keep it, and it is measurably a
            // loss on layers with small N — see UsePackedA.
            var suppliedPack = UsePackedA && packedKernels.Length >= rowBlocksTotal * Mr * k;
            var packedLength = UsePackedA && !suppliedPack ? checked(rowBlocksTotal * Mr * k) : 1;

            using var packedABuf = new PooledBuffer<float>(packedLength, clearMemory: false);

            if (UsePackedA && !suppliedPack)
            {
                PackKernelsMrMajor(kernels, packedABuf.Span, m, k, rowBlocksTotal);
            }

            var packedSource = suppliedPack ? packedKernels : (ReadOnlySpan<float>)packedABuf.Span;

            fixed (float* pa = kernels, pin = input, pc = output, pPackedA = packedSource)
            {
                var ctx = new FusedGemmCtx(
                    UsePackedA ? pPackedA : pa,
                    pin,
                    pc,
                    m,
                    n,
                    k,
                    mBlocks,
                    inputH,
                    inputW,
                    kernelSize,
                    padding,
                    stride,
                    outW);

                OverfitParallel.For(0, nPanels * mBlocks, 1, &GemmFusedPanelWorker512, &ctx);
            }
        }

        private static unsafe void GemmFusedPanelWorker512(int itemStart, int itemEnd, void* ctxPtr)
        {
            ref readonly var c = ref Unsafe.AsRef<FusedGemmCtx>(ctxPtr);

            var k = c.K;
            var n = c.N;
            var m = c.M;
            var mBlocks = c.MBlocks;
            var rowBlocks = (m + Mr - 1) / Mr;

            var inputW = c.InputW;
            var inputH = c.InputH;
            var kernelSize = c.KernelSize;
            var padding = c.Padding;
            var stride = c.Stride;
            var outW = c.OutW;
            var inputPlane = inputH * inputW;
            var window = kernelSize * kernelSize;

            var packRows = ConvKBlock > 0 ? Math.Min(ConvKBlock, k) : k;

            using var packBuf = new PooledBuffer<float>(checked(packRows * Nr512), clearMemory: false);
            var packB = packBuf.Span;

            // Input row and column origins for this panel's 32 output positions, hoisted out of the K loop.
            // Computing them per element would put a division on every one of K*32 gathers.
#pragma warning disable OVERFIT026 // BOUND: exactly Nr512 (32) ints each = 128 B per array, fixed at compile time.
            var rowOrigin = stackalloc int[Nr512];
            var colOrigin = stackalloc int[Nr512];
#pragma warning restore OVERFIT026

            for (var item = itemStart; item < itemEnd; item++)
            {
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

                for (var j = 0; j < nrEff; j++)
                {
                    var position = n0 + j;
                    var oy = position / outW;

                    rowOrigin[j] = (oy * stride) - padding;
                    colOrigin[j] = ((position - (oy * outW)) * stride) - padding;
                }

                if (ConvKBlock > 0)
                {
                    GatherAndSweepKBlocked(
                        in c,
                        packB,
                        rowOrigin,
                        colOrigin,
                        n0,
                        nrEff,
                        rowBlockStart,
                        rowBlockEnd);

                    continue;
                }

                if (!AblatePackB)
                {
                    for (var kk = 0; kk < k; kk++)
                    {
                        var kx = kk % kernelSize;
                        var ky = (kk / kernelSize) % kernelSize;
                        var channelBase = (kk / window) * inputPlane;
                        var dstBase = kk * Nr512;

                        for (var j = 0; j < nrEff; j++)
                        {
                            var iy = rowOrigin[j] + ky;
                            var ix = colOrigin[j] + kx;

                            packB[dstBase + j] =
                                (uint)iy < (uint)inputH && (uint)ix < (uint)inputW
                                    ? c.Input[channelBase + (iy * inputW) + ix]
                                    : 0f;
                        }

                        // Lanes past nrEff are never stored — StoreTile writes exactly nrEff columns — so
                        // this fill is not load-bearing, and a mutation that writes 1f here leaves the whole
                        // suite green. It stays because uninitialised pool memory can hold NaN, and feeding
                        // NaN through the FMA chain costs on some parts even when the lane is discarded. It
                        // is now outside the gather loop rather than a branch on every one of K*32 elements.
                        for (var j = nrEff; j < Nr512; j++)
                        {
                            packB[dstBase + j] = 0f;
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

                        if (UsePackedA)
                        {
                            MicroKernel8x32Avx512PackedA(
                                c.A + ((long)rb * Mr * k), mrEff, k, pPackB, c.C, n, n0, m0, nrEff);

                            continue;
                        }

                        MicroKernel8x32Avx512(c.A, m0, mrEff, k, pPackB, c.C, n, n0, nrEff);
                    }
                }
            }
        }

        /// <summary>
        /// One N-panel swept in K-blocks: gather a <c>Kc x 32</c> B block, sweep this worker's M rows
        /// against it, then move to the next block and accumulate. See <see cref="ConvKBlock"/>.
        /// </summary>
        private static unsafe void GatherAndSweepKBlocked(
            ref readonly FusedGemmCtx c,
            Span<float> packB,
            int* rowOrigin,
            int* colOrigin,
            int n0,
            int nrEff,
            int rowBlockStart,
            int rowBlockEnd)
        {
            var k = c.K;
            var n = c.N;
            var m = c.M;
            var inputW = c.InputW;
            var inputH = c.InputH;
            var kernelSize = c.KernelSize;
            var inputPlane = inputH * inputW;
            var window = kernelSize * kernelSize;
            var kc = ConvKBlock;

            fixed (float* pPackB = packB)
            {
                // BOUND: kb advances by kc >= 1 each pass, so at most ceil(k / kc) passes.
                for (var kb = 0; kb < k; kb += kc)
                {
                    var kcEff = Math.Min(kc, k - kb);

                    for (var kk = 0; kk < kcEff; kk++)
                    {
                        var kRow = kb + kk;
                        var kx = kRow % kernelSize;
                        var ky = (kRow / kernelSize) % kernelSize;
                        var channelBase = (kRow / window) * inputPlane;
                        var dstBase = kk * Nr512;

                        for (var j = 0; j < nrEff; j++)
                        {
                            var iy = rowOrigin[j] + ky;
                            var ix = colOrigin[j] + kx;

                            packB[dstBase + j] =
                                (uint)iy < (uint)inputH && (uint)ix < (uint)inputW
                                    ? c.Input[channelBase + (iy * inputW) + ix]
                                    : 0f;
                        }

                        for (var j = nrEff; j < Nr512; j++)
                        {
                            packB[dstBase + j] = 0f;
                        }
                    }

                    for (var rb = rowBlockStart; rb < rowBlockEnd; rb++)
                    {
                        var m0 = rb * Mr;
                        var mrEff = Math.Min(Mr, m - m0);

                        MicroKernel8x32Avx512Accumulating(
                            c.A, m0, mrEff, k, kb, kcEff, pPackB, c.C, n, n0, nrEff, kb > 0);
                    }
                }
            }
        }

        /// <summary>
        /// The 8x32 micro-kernel over a K SLICE, seeding its accumulators from C rather than from zero when
        /// this is not the first slice. Kept separate from <see cref="MicroKernel8x32Avx512"/> so the
        /// full-K path measured before this experiment is not perturbed by it.
        /// </summary>
        private static unsafe void MicroKernel8x32Avx512Accumulating(
            float* a,
            int m0,
            int mrEff,
            int kStride,
            int kStart,
            int kCount,
            float* packB,
            float* c,
            int n,
            int n0,
            int nrEff,
            bool accumulate)
        {
            var rows = stackalloc float*[Mr];

            for (var r = 0; r < Mr; r++)
            {
                rows[r] = a + ((long)(m0 + Math.Min(r, mrEff - 1)) * kStride) + kStart;
            }

            var tile = stackalloc float[Nr512];

            Vector512<float> c00 = default, c01 = default, c10 = default, c11 = default;
            Vector512<float> c20 = default, c21 = default, c30 = default, c31 = default;
            Vector512<float> c40 = default, c41 = default, c50 = default, c51 = default;
            Vector512<float> c60 = default, c61 = default, c70 = default, c71 = default;

            if (accumulate)
            {
                LoadTile(c, n, n0, m0, 0, mrEff, nrEff, tile, ref c00, ref c01);
                LoadTile(c, n, n0, m0, 1, mrEff, nrEff, tile, ref c10, ref c11);
                LoadTile(c, n, n0, m0, 2, mrEff, nrEff, tile, ref c20, ref c21);
                LoadTile(c, n, n0, m0, 3, mrEff, nrEff, tile, ref c30, ref c31);
                LoadTile(c, n, n0, m0, 4, mrEff, nrEff, tile, ref c40, ref c41);
                LoadTile(c, n, n0, m0, 5, mrEff, nrEff, tile, ref c50, ref c51);
                LoadTile(c, n, n0, m0, 6, mrEff, nrEff, tile, ref c60, ref c61);
                LoadTile(c, n, n0, m0, 7, mrEff, nrEff, tile, ref c70, ref c71);
            }

            for (var kk = 0; kk < kCount; kk++)
            {
                var b0 = Vector512.Load(packB + (kk * Nr512));
                var b1 = Vector512.Load(packB + (kk * Nr512) + 16);

                var a0 = Vector512.Create(rows[0][kk]);
                c00 = Avx512F.FusedMultiplyAdd(a0, b0, c00);
                c01 = Avx512F.FusedMultiplyAdd(a0, b1, c01);

                var a1 = Vector512.Create(rows[1][kk]);
                c10 = Avx512F.FusedMultiplyAdd(a1, b0, c10);
                c11 = Avx512F.FusedMultiplyAdd(a1, b1, c11);

                var a2 = Vector512.Create(rows[2][kk]);
                c20 = Avx512F.FusedMultiplyAdd(a2, b0, c20);
                c21 = Avx512F.FusedMultiplyAdd(a2, b1, c21);

                var a3 = Vector512.Create(rows[3][kk]);
                c30 = Avx512F.FusedMultiplyAdd(a3, b0, c30);
                c31 = Avx512F.FusedMultiplyAdd(a3, b1, c31);

                var a4 = Vector512.Create(rows[4][kk]);
                c40 = Avx512F.FusedMultiplyAdd(a4, b0, c40);
                c41 = Avx512F.FusedMultiplyAdd(a4, b1, c41);

                var a5 = Vector512.Create(rows[5][kk]);
                c50 = Avx512F.FusedMultiplyAdd(a5, b0, c50);
                c51 = Avx512F.FusedMultiplyAdd(a5, b1, c51);

                var a6 = Vector512.Create(rows[6][kk]);
                c60 = Avx512F.FusedMultiplyAdd(a6, b0, c60);
                c61 = Avx512F.FusedMultiplyAdd(a6, b1, c61);

                var a7 = Vector512.Create(rows[7][kk]);
                c70 = Avx512F.FusedMultiplyAdd(a7, b0, c70);
                c71 = Avx512F.FusedMultiplyAdd(a7, b1, c71);
            }

            StoreTile(c, n, n0, m0, 0, mrEff, nrEff, c00, c01, tile);
            StoreTile(c, n, n0, m0, 1, mrEff, nrEff, c10, c11, tile);
            StoreTile(c, n, n0, m0, 2, mrEff, nrEff, c20, c21, tile);
            StoreTile(c, n, n0, m0, 3, mrEff, nrEff, c30, c31, tile);
            StoreTile(c, n, n0, m0, 4, mrEff, nrEff, c40, c41, tile);
            StoreTile(c, n, n0, m0, 5, mrEff, nrEff, c50, c51, tile);
            StoreTile(c, n, n0, m0, 6, mrEff, nrEff, c60, c61, tile);
            StoreTile(c, n, n0, m0, 7, mrEff, nrEff, c70, c71, tile);
        }

        /// <summary>Seeds one row of the register tile from C, for a K-block after the first.</summary>
        private static unsafe void LoadTile(
            float* c, int n, int n0, int m0, int row, int mrEff, int nrEff, float* scratch,
            ref Vector512<float> lo, ref Vector512<float> hi)
        {
            if (row >= mrEff)
            {
                return;
            }

            var src = c + ((long)(m0 + row) * n) + n0;

            if (nrEff == Nr512)
            {
                lo = Vector512.Load(src);
                hi = Vector512.Load(src + 16);
                return;
            }

            // Lanes past nrEff are never stored, but they must not be NaN on the way through the FMA chain.
            for (var j = 0; j < Nr512; j++)
            {
                scratch[j] = j < nrEff ? src[j] : 0f;
            }

            lo = Vector512.Load(scratch);
            hi = Vector512.Load(scratch + 16);
        }

        /// <summary>
        /// Floats needed to hold the MR-major repack of an <c>[M, K]</c> kernel matrix. M is rounded
        /// up to a whole micro-panel, so the last row block needs no special case.
        /// </summary>
        internal static int PackedKernelLength(int m, int k)
        {
            return checked(((m + Mr - 1) / Mr) * Mr * k);
        }

        /// <summary>The repack, for a caller that keeps the result across inferences.</summary>
        internal static void PackKernels(ReadOnlySpan<float> kernels, Span<float> packed, int m, int k)
        {
            PackKernelsMrMajor(kernels, packed, m, k, (m + Mr - 1) / Mr);
        }

        /// <summary>
        /// Rewrites the kernel matrix from <c>[M, K]</c> row-major into MR-major micro-panels,
        /// <c>[M/MR][K][MR]</c>, so the MR values one k-step needs are contiguous. Rows past M are zero,
        /// which the store discards. See <see cref="UsePackedA"/>.
        /// </summary>
        private static unsafe void PackKernelsMrMajor(
            ReadOnlySpan<float> kernels,
            Span<float> packed,
            int m,
            int k,
            int rowBlocksTotal)
        {
            fixed (float* src = kernels, dst = packed)
            {
                var ctx = new PackACtx(src, dst, m, k);

                OverfitParallel.For(0, rowBlocksTotal, 1, &PackKernelsRowBlockWorker, &ctx);
            }
        }

        private static unsafe void PackKernelsRowBlockWorker(int blockStart, int blockEnd, void* ctxPtr)
        {
            ref readonly var ctx = ref Unsafe.AsRef<PackACtx>(ctxPtr);

            var m = ctx.M;
            var k = ctx.K;

            for (var rb = blockStart; rb < blockEnd; rb++)
            {
                var blockBase = ctx.Packed + ((long)rb * Mr * k);

                for (var r = 0; r < Mr; r++)
                {
                    var row = (rb * Mr) + r;

                    // Read the source row sequentially and write with a 32-byte stride: one of the two has
                    // to be strided, and a strided write is the cheaper half.
                    if (row >= m)
                    {
                        for (var kk = 0; kk < k; kk++)
                        {
                            blockBase[((long)kk * Mr) + r] = 0f;
                        }

                        continue;
                    }

                    var srcRow = ctx.Source + ((long)row * k);

                    for (var kk = 0; kk < k; kk++)
                    {
                        blockBase[((long)kk * Mr) + r] = srcRow[kk];
                    }
                }
            }
        }

        /// <summary>
        /// The 8x32 micro-kernel reading A from an MR-major packed block, so the eight values a k-step
        /// needs are one contiguous 32-byte group rather than eight addresses <c>k * 4</c> bytes apart.
        /// </summary>
        internal static unsafe void MicroKernel8x32Avx512PackedA(
            float* packedA,
            int mrEff,
            int k,
            float* packB,
            float* c,
            int n,
            int n0,
            int m0,
            int nrEff)
        {
            var tile = stackalloc float[Nr512];

            Vector512<float> c00 = default, c01 = default, c10 = default, c11 = default;
            Vector512<float> c20 = default, c21 = default, c30 = default, c31 = default;
            Vector512<float> c40 = default, c41 = default, c50 = default, c51 = default;
            Vector512<float> c60 = default, c61 = default, c70 = default, c71 = default;

            for (var kk = 0; kk < k; kk++)
            {
                var b0 = Vector512.Load(packB + (kk * Nr512));
                var b1 = Vector512.Load(packB + (kk * Nr512) + 16);
                var aSlot = packedA + ((long)kk * Mr);

                var a0 = Vector512.Create(aSlot[0]);
                c00 = Avx512F.FusedMultiplyAdd(a0, b0, c00);
                c01 = Avx512F.FusedMultiplyAdd(a0, b1, c01);

                var a1 = Vector512.Create(aSlot[1]);
                c10 = Avx512F.FusedMultiplyAdd(a1, b0, c10);
                c11 = Avx512F.FusedMultiplyAdd(a1, b1, c11);

                var a2 = Vector512.Create(aSlot[2]);
                c20 = Avx512F.FusedMultiplyAdd(a2, b0, c20);
                c21 = Avx512F.FusedMultiplyAdd(a2, b1, c21);

                var a3 = Vector512.Create(aSlot[3]);
                c30 = Avx512F.FusedMultiplyAdd(a3, b0, c30);
                c31 = Avx512F.FusedMultiplyAdd(a3, b1, c31);

                var a4 = Vector512.Create(aSlot[4]);
                c40 = Avx512F.FusedMultiplyAdd(a4, b0, c40);
                c41 = Avx512F.FusedMultiplyAdd(a4, b1, c41);

                var a5 = Vector512.Create(aSlot[5]);
                c50 = Avx512F.FusedMultiplyAdd(a5, b0, c50);
                c51 = Avx512F.FusedMultiplyAdd(a5, b1, c51);

                var a6 = Vector512.Create(aSlot[6]);
                c60 = Avx512F.FusedMultiplyAdd(a6, b0, c60);
                c61 = Avx512F.FusedMultiplyAdd(a6, b1, c61);

                var a7 = Vector512.Create(aSlot[7]);
                c70 = Avx512F.FusedMultiplyAdd(a7, b0, c70);
                c71 = Avx512F.FusedMultiplyAdd(a7, b1, c71);
            }

            StoreTile(c, n, n0, m0, 0, mrEff, nrEff, c00, c01, tile);
            StoreTile(c, n, n0, m0, 1, mrEff, nrEff, c10, c11, tile);
            StoreTile(c, n, n0, m0, 2, mrEff, nrEff, c20, c21, tile);
            StoreTile(c, n, n0, m0, 3, mrEff, nrEff, c30, c31, tile);
            StoreTile(c, n, n0, m0, 4, mrEff, nrEff, c40, c41, tile);
            StoreTile(c, n, n0, m0, 5, mrEff, nrEff, c50, c51, tile);
            StoreTile(c, n, n0, m0, 6, mrEff, nrEff, c60, c61, tile);
            StoreTile(c, n, n0, m0, 7, mrEff, nrEff, c70, c71, tile);
        }

        /// <summary>Source, destination and shape for the MR-major kernel repack.</summary>
        private readonly unsafe struct PackACtx
        {
            public readonly float* Source;
            public readonly float* Packed;
            public readonly int M;
            public readonly int K;

            public PackACtx(float* source, float* packed, int m, int k)
            {
                Source = source;
                Packed = packed;
                M = m;
                K = k;
            }
        }

        /// <summary>Pointers, shape and convolution geometry for one fused GEMM.</summary>
        private readonly unsafe struct FusedGemmCtx
        {
            public readonly float* A;
            public readonly float* Input;
            public readonly float* C;
            public readonly int M;
            public readonly int N;
            public readonly int K;
            public readonly int MBlocks;
            public readonly int InputH;
            public readonly int InputW;
            public readonly int KernelSize;
            public readonly int Padding;
            public readonly int Stride;
            public readonly int OutW;

            public FusedGemmCtx(
                float* a,
                float* input,
                float* c,
                int m,
                int n,
                int k,
                int mBlocks,
                int inputH,
                int inputW,
                int kernelSize,
                int padding,
                int stride,
                int outW)
            {
                A = a;
                Input = input;
                C = c;
                M = m;
                N = n;
                K = k;
                MBlocks = mBlocks;
                InputH = inputH;
                InputW = inputW;
                KernelSize = kernelSize;
                Padding = padding;
                Stride = stride;
                OutW = outW;
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
