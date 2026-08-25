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
        /// Set <c>OVERFIT_CONV_PACK_A=0</c> to skip the MR-major repack of the kernel matrix before
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
        ///
        /// <para><b>This shipped opt-IN by mistake, and a null result is what caught it.</b> The switch
        /// originally read <c>== "1"</c> while the four other conv switches read <c>!= "0"</c>, so the
        /// measured 8.4% never reached a default build — and `README.md` was publishing a VGG-16 figure the
        /// library did not produce unless an environment variable was set. It surfaced when a prefetch sweep
        /// came back perfectly flat: the prefetch path is gated on this flag, so with the flag off the lever
        /// under test was never connected. <b>A flat result is the signature of a dead lever at least as
        /// often as of a real null</b>, and the baseline is what gives it away — 55.3 ms in that sweep where
        /// the packed path measures 50.7.</para>
        /// </summary>
        internal static readonly bool UsePackedA =
            Environment.GetEnvironmentVariable(OverfitEnvironment.ConvPackA) != "0";

        /// <summary>Cap on <see cref="ConvNBlock"/>, and the size the origin scratch is fixed at.</summary>
        private const int MaxNBlock = 4;

        /// <summary>
        /// How many 32-column sub-panels one work item covers. 1 is the original; 4 makes the N-block 128
        /// columns wide, which is <c>MLAS_SGEMM_STRIDEN</c>.
        ///
        /// <para><b>Read from MLAS.</b> `MlasSgemmOperation` takes <c>CountN = min(N - n, 128)</c>, packs a
        /// panel that wide, and sweeps every row of M through it in one pass. This kernel did the same for
        /// <b>32</b>.</para>
        ///
        /// <para><b>What it buys, corrected from a first reading that was wrong.</b> It does NOT reduce how
        /// often A is read: with a 128-wide block the micro-kernel is still called four times per row
        /// block, so A is read <c>N / 32</c> times either way. What changes is <b>where those reads come
        /// from</b> - the four calls hit the same 147 KB of A back to back, so three of them find it in L1
        /// rather than fetching it again for a separately scheduled panel. <b>A locality change, not a
        /// traffic change</b>, and correspondingly smaller than a naive count of A reads suggests.</para>
        ///
        /// <para><b>An earlier grouping experiment measured this neutral</b> (groups of 1/2/4 at 72.7 / 73.0
        /// / 73.3 ms) - but on the AVX2 path, against a micro-kernel that was spilling its accumulators, and
        /// with A unpacked. <b>A null measured against a broken baseline is not a null</b>, which is the only
        /// reason this is worth re-running now that the kernel reaches 93% of single-core peak.</para>
        /// </summary>
        internal static readonly int ConvNBlock = ResolveConvNBlock();

        private static int ResolveConvNBlock()
        {
            var raw = Environment.GetEnvironmentVariable(OverfitEnvironment.ConvNBlock);

            if (!int.TryParse(raw, out var parsed) || parsed < 1)
            {
                return 1;
            }

            return Math.Min(parsed, MaxNBlock);
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
            var mBlocks = ResolveFusedMBlocks(m, nPanels);

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

            if (UsePackedA && ShouldExpandPanels(m, n, k, nPanels))
            {
                GemmExpandedIm2Col(
                    packedSource, input, output, m, n, k, nPanels,
                    inputH, inputW, kernelSize, padding, stride, outW);

                return;
            }

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

                // Opts in to a finer split. Its body claims nothing by chunk ordinal — every work item is
                // derived from the item index and rents its own scratch — so more chunks than workers is
                // safe here in a way it is not for a body that treats its chunk as a worker slot.
                OverfitParallel.For(
                    0, nPanels * mBlocks, 1, OverfitParallel.MaxDegreeOfParallelism,
                    &GemmFusedPanelWorker512, &ctx, OverfitParallel.ChunkFactor);
            }
        }

        /// <summary>
        /// The convolution GEMM for a layer with too few panels to fill the workers: expand every panel once
        /// into a shared buffer, then split M over it.
        /// </summary>
        private static unsafe void GemmExpandedIm2Col(
            ReadOnlySpan<float> packedA,
            ReadOnlySpan<float> input,
            Span<float> output,
            int m,
            int n,
            int k,
            int nPanels,
            int inputH,
            int inputW,
            int kernelSize,
            int padding,
            int stride,
            int outW)
        {
            // Flat (row block, panel) pairs rather than a panel x M-block grid. The grid is what produced
            // the defect this path exists to remove: seven panels and ceil(32/7) = 5 blocks is 35 items on
            // 32 workers, so three workers run two items and the critical path is twice the ideal. With the
            // panels already expanded, an item costs one micro-kernel call and 64 x 7 = 448 of them split
            // into 32 ranges of 14 — balanced to one item.
            var rowBlocks = (m + Mr - 1) / Mr;

            using var expandedBuf = new PooledBuffer<float>(
                checked(nPanels * k * Nr512), clearMemory: false);

            fixed (float* pin = input, pc = output, pPackedA = packedA, pExpanded = expandedBuf.Span)
            {
                var ctx = new FusedGemmCtx(
                    pPackedA, pin, pc, m, n, k, 1,
                    inputH, inputW, kernelSize, padding, stride, outW, pExpanded, nPanels);

                OverfitParallel.For(0, nPanels, 1, &ExpandPanelWorker, &ctx);
                OverfitParallel.For(0, nPanels * rowBlocks, 1, &ExpandedGemmWorker, &ctx);
            }
        }

        /// <summary>
        /// Set <c>OVERFIT_CONV_VECTOR_GATHER=0</c> to gather im2col one element at a time behind a bounds
        /// test, as every version before 2026-08-19 did.
        ///
        /// <para><b>Why this is the item that matters, and how that was established.</b> Both engines were
        /// run through one loop at 1, 2, 4, 8 and 16 physical cores with ONNX Runtime's thread count set to
        /// match. <b>Overfit scales 6.74x across 16 cores and ONNX Runtime scales 6.68x</b> - ours is the
        /// better of the two - and the ratio between them is flat at <b>2.00x on a single core</b>. So the
        /// gap is per-core work. A per-layer cost model, fitted on two layers and checked on seven held-out
        /// ones (eight of nine inside 13%), splits that work in two: the GEMM term is <b>301 GFLOP/s, 84% of
        /// this machine's single-core FMA ceiling</b>, and the gather term is <b>0.964 ns per element, about
        /// 4.8 cycles</b>. Across VGG-16 the gather is <b>81.7 M elements, 78.8 ms of 204.4, or 39%</b>.</para>
        ///
        /// <para><b>And it does not have to be a gather.</b> At <c>stride == 1</c> and a fixed
        /// <c>(ky, kx)</c>, consecutive output positions read <b>consecutive input addresses</b>. A run of
        /// output positions sharing an output row is therefore a contiguous copy, not a scatter of indices,
        /// so the interior is <c>Vector512</c> loads and stores and only the run's two ends need the bounds
        /// test at all.</para>
        ///
        /// <para><b>The run structure depends on the panel, not on <c>kk</c></b>, which is what makes this
        /// cheap: a 32-column panel of a 224-wide output spans one or two output rows, and that table is
        /// built once and reused for every K row. Deriving it per element is what the old code did, and it
        /// put an integer division on every one of 81.7 M gathers.</para>
        /// </summary>
        internal static readonly bool VectorGatherEnabled =
            Environment.GetEnvironmentVariable(OverfitEnvironment.ConvVectorGather) != "0";

        /// <summary>Copies a contiguous run of floats, widest registers first.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void CopyRun(float* src, float* dst, int length)
        {
            var i = 0;

            if (CpuFeatures.HasVector512)
            {
                for (; i + 16 <= length; i += 16)
                {
                    Vector512.Store(Vector512.Load(src + i), dst + i);
                }
            }

            for (; i + 8 <= length; i += 8)
            {
                Vector256.Store(Vector256.Load(src + i), dst + i);
            }

            for (; i < length; i++)
            {
                dst[i] = src[i];
            }
        }

        /// <summary>
        /// Gathers one 32-column panel of the im2col matrix into <c>dst</c> as a contiguous <c>[k][32]</c>
        /// region, which is the layout the micro-kernel consumes, so nothing downstream changes.
        /// </summary>
        private static unsafe void GatherSubPanel(
            float* input,
            float* dst,
            int n0,
            int nrEff,
            int k,
            int inputH,
            int inputW,
            int kernelSize,
            int padding,
            int stride,
            int outW)
        {
            if (VectorGatherEnabled && stride == 1)
            {
                GatherSubPanelUnitStride(
                    input, dst, n0, nrEff, k, inputH, inputW, kernelSize, padding, outW);

                return;
            }

            GatherSubPanelScalar(
                input, dst, n0, nrEff, k, inputH, inputW, kernelSize, padding, stride, outW);
        }

        /// <summary>
        /// The unit-stride gather: runs of output positions sharing an output row read contiguous input, so
        /// each run is a leading zero fill, a vector copy and a trailing zero fill.
        /// </summary>
        private static unsafe void GatherSubPanelUnitStride(
            float* input,
            float* dst,
            int n0,
            int nrEff,
            int k,
            int inputH,
            int inputW,
            int kernelSize,
            int padding,
            int outW)
        {
            var window = kernelSize * kernelSize;
            var inputPlane = inputH * inputW;

            // At most one run per column, which is the degenerate outW == 1 case; normally one or two.
#pragma warning disable OVERFIT026 // BOUND: Nr512 (32) ints each = 128 B per array, fixed at compile time.
            var runStart = stackalloc int[Nr512];
            var runLength = stackalloc int[Nr512];
            var runRow = stackalloc int[Nr512];
            var runCol = stackalloc int[Nr512];
#pragma warning restore OVERFIT026

            var runs = 0;
            var j = 0;

            while (j < nrEff)
            {
                var position = n0 + j;
                var oy = position / outW;
                var ox = position - (oy * outW);
                var length = Math.Min(nrEff - j, outW - ox);

                runStart[runs] = j;
                runLength[runs] = length;
                runRow[runs] = oy;
                runCol[runs] = ox;
                runs++;

                j += length;
            }

            for (var kk = 0; kk < k; kk++)
            {
                var kx = kk % kernelSize;
                var ky = (kk / kernelSize) % kernelSize;
                var channelBase = (kk / window) * inputPlane;
                var row = dst + (kk * Nr512);

                for (var r = 0; r < runs; r++)
                {
                    var at = row + runStart[r];
                    var length = runLength[r];
                    var iy = runRow[r] - padding + ky;

                    if ((uint)iy >= (uint)inputH)
                    {
                        new Span<float>(at, length).Clear();

                        continue;
                    }

                    var ixStart = runCol[r] - padding + kx;

                    // The run splits into three pieces: columns left of the input, the part inside it, and
                    // columns past its right edge. Only the middle one reads memory.
                    var lead = ixStart < 0 ? Math.Min(-ixStart, length) : 0;
                    var copyFrom = ixStart + lead;
                    var available = inputW - copyFrom;
                    var copyLength = Math.Min(length - lead, available);

                    if (copyLength < 0)
                    {
                        copyLength = 0;
                    }

                    if (lead > 0)
                    {
                        new Span<float>(at, lead).Clear();
                    }

                    if (copyLength > 0)
                    {
                        CopyRun(input + channelBase + (iy * inputW) + copyFrom, at + lead, copyLength);
                    }

                    var trail = length - lead - copyLength;

                    if (trail > 0)
                    {
                        new Span<float>(at + lead + copyLength, trail).Clear();
                    }
                }

                // Lanes past nrEff are never stored, so this fill is not load-bearing. It stays because
                // uninitialised pool memory can hold NaN, and NaN through an FMA chain costs on some parts
                // even in a discarded lane.
                if (nrEff < Nr512)
                {
                    new Span<float>(row + nrEff, Nr512 - nrEff).Clear();
                }
            }
        }

        /// <summary>The element-at-a-time gather, kept for strides other than 1 and as the A/B arm.</summary>
        private static unsafe void GatherSubPanelScalar(
            float* input,
            float* dst,
            int n0,
            int nrEff,
            int k,
            int inputH,
            int inputW,
            int kernelSize,
            int padding,
            int stride,
            int outW)
        {
            var window = kernelSize * kernelSize;
            var inputPlane = inputH * inputW;

#pragma warning disable OVERFIT026 // BOUND: Nr512 (32) ints each = 128 B per array, fixed at compile time.
            var rowOrigin = stackalloc int[Nr512];
            var colOrigin = stackalloc int[Nr512];
#pragma warning restore OVERFIT026

            for (var j = 0; j < nrEff; j++)
            {
                var position = n0 + j;
                var oy = position / outW;

                rowOrigin[j] = (oy * stride) - padding;
                colOrigin[j] = ((position - (oy * outW)) * stride) - padding;
            }

            for (var kk = 0; kk < k; kk++)
            {
                var kx = kk % kernelSize;
                var ky = (kk / kernelSize) % kernelSize;
                var channelBase = (kk / window) * inputPlane;
                var row = dst + (kk * Nr512);

                for (var j = 0; j < nrEff; j++)
                {
                    var iy = rowOrigin[j] + ky;
                    var ix = colOrigin[j] + kx;

                    row[j] = (uint)iy < (uint)inputH && (uint)ix < (uint)inputW
                        ? input[channelBase + (iy * inputW) + ix]
                        : 0f;
                }

                for (var j = nrEff; j < Nr512; j++)
                {
                    row[j] = 0f;
                }
            }
        }

        /// <summary>
        /// Expands one 32-column panel of the im2col matrix into <c>dst</c> as a contiguous <c>[k][32]</c>
        /// region — the layout the micro-kernel already consumes, so nothing downstream changes.
        /// </summary>
        private static unsafe void GatherPanel(in FusedGemmCtx c, int panel, float* dst)
        {
            var n0 = panel * Nr512;

            GatherSubPanel(
                c.Input, dst, n0, Math.Min(Nr512, c.N - n0), c.K,
                c.InputH, c.InputW, c.KernelSize, c.Padding, c.Stride, c.OutW);
        }

        /// <summary>
        /// Set <c>OVERFIT_CONV_EXPAND_PANELS=1</c> to expand every panel once into a shared buffer instead of
        /// gathering each panel inside its own work item. <b>Off, because it is measured to lose — and it
        /// refuted the hypothesis it was built on.</b>
        ///
        /// <para><b>What it fixes, in the measurement that produced it.</b> Work is one item per 32-column
        /// panel, so VGG-16's last three convolutions — <c>N = 196</c> — produce <b>seven</b> items. With the
        /// M-split off they run at 1007 GFLOP/s against this box's 5136 all-core ceiling: <b>19.6%, where
        /// 7/32 workers is 21.9%</b>. They are near-perfectly efficient on seven cores and idle on the other
        /// twenty-five. With the M-split on they get more items and run <i>slower</i>, because five row
        /// blocks sharing a panel gather that panel five times.</para>
        ///
        /// <para><b>So the split is not the problem; the redundant gather is.</b> Expanding every panel once
        /// into a shared buffer separates them: phase one gathers each panel exactly once, phase two splits
        /// M over the result and reads it. This is the shape <c>MlasConvExpandThenGemmSegmented</c> has, and
        /// the reason it is affordable here is the same reason the layer needed it — a small N. Seven panels
        /// of <c>K = 4608</c> is <b>4.1 MB</b>; node 2's 1568 panels of <c>K = 576</c> would be 115 MB, which
        /// is why the path is gated on panel count rather than applied everywhere.</para>
        /// </summary>
        internal static readonly bool ExpandPanelsEnabled =
            Environment.GetEnvironmentVariable(OverfitEnvironment.ConvExpandPanels) == "1";

        /// <summary>
        /// Whether a layer's shape is one the shared expansion helps: fewer panels than workers, more than
        /// one row block to split, and a buffer small enough that the expansion is not itself the cost.
        /// </summary>
        private static bool ShouldExpandPanels(int m, int n, int k, int nPanels)
        {
            if (!ExpandPanelsEnabled || ConvNBlock != 1 || nPanels <= 0)
            {
                return false;
            }

            if (nPanels >= OverfitParallel.MaxDegreeOfParallelism)
            {
                return false;
            }

            if ((m + Mr - 1) / Mr <= 1)
            {
                return false;
            }

            // 32 MiB, chosen so the buffer stays a small multiple of L3 rather than a second copy of the
            // activation. VGG-16's small-N layers need 4.1 MB; a layer that needed more than this has enough
            // panels to fill the workers anyway, so the bound never turns away a shape that wanted it.
            return (long)nPanels * k * Nr512 * sizeof(float) <= 32L * 1024 * 1024;
        }

        /// <summary>Phase one of the expand-then-GEMM path: every panel gathered exactly once.</summary>
        private static unsafe void ExpandPanelWorker(int panelStart, int panelEnd, void* ctxPtr)
        {
            ref readonly var c = ref Unsafe.AsRef<FusedGemmCtx>(ctxPtr);

            for (var panel = panelStart; panel < panelEnd; panel++)
            {
                GatherPanel(in c, panel, c.Expanded + ((long)panel * c.K * Nr512));
            }
        }

        /// <summary>
        /// Phase two: the GEMM over the shared expansion, split on both M and N.
        ///
        /// <para>The split that was a loss in the fused worker is a win here for one reason — the panel is
        /// already expanded, so the row blocks sharing it read it instead of gathering it again.</para>
        /// </summary>
        private static unsafe void ExpandedGemmWorker(int itemStart, int itemEnd, void* ctxPtr)
        {
            ref readonly var c = ref Unsafe.AsRef<FusedGemmCtx>(ctxPtr);

            var k = c.K;
            var n = c.N;
            var m = c.M;

#pragma warning disable OVERFIT026 // BOUND: Nr512 (32) floats = 128 B, fixed at compile time.
            var tileScratch = stackalloc float[Nr512];
#pragma warning restore OVERFIT026

            // Panel OUTERMOST, so a range of items walks consecutive row blocks inside one panel and that
            // panel stays in L2 while every row of A streams past it once.
            //
            // Measured 2026-08-19, and the wrong way round is expensive: with the panel innermost, B changes
            // on every item and a layer with 25 panels re-reads 25 x 590 KB per row block. VGG-16's
            // `out=401408` convolutions ran +48.9%, +22.2% and +39.3% that way, against +6.9% on the whole
            // model. Same items, same arithmetic, same balance — only the traversal order.
            var rowBlocks = (m + Mr - 1) / Mr;

            for (var item = itemStart; item < itemEnd; item++)
            {
                var np = item / rowBlocks;
                var rb = item - (np * rowBlocks);

                var m0 = rb * Mr;
                var mrEff = Math.Min(Mr, m - m0);
                var n0 = np * Nr512;
                var nrEff = Math.Min(Nr512, n - n0);
                var packB = c.Expanded + ((long)np * k * Nr512);

                if (!UsePackedA)
                {
                    MicroKernel8x32Avx512(c.A, m0, mrEff, k, packB, c.C, n, n0, nrEff);

                    continue;
                }

                var cTile = c.C + ((long)m0 * n) + n0;

                if (mrEff == Mr && nrEff == Nr512)
                {
                    MicroKernel8x32Avx512PackedAFull(c.A + ((long)rb * Mr * k), k, packB, cTile, n);

                    continue;
                }

                MicroKernel8x32Avx512PackedAPartial(
                    c.A + ((long)rb * Mr * k), mrEff, k, packB, cTile, n, nrEff, tileScratch);
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

            var group = ConvNBlock;
            var blockWidth = Nr512 * group;

            // Rented per invocation. `XC-94` replaced this with a per-thread buffer and MEASURED the
            // replacement to be worth nothing: at one chunk per worker the two arms are 18.91 against
            // 18.81 ms, and at eight the per-thread arm still pays +6.1% worker time against the pooled
            // arm's +7.4% — so the rent was 1.3 points of a 7.4-point cost and not the mechanism. It also
            // broke `ResNetBlock_DAG_InferenceAllocatesZeroBytes`, because a thread-held array grows when a
            // larger layer arrives and that growth lands inside a measured window.
            using var packBuf = new PooledBuffer<float>(checked(k * blockWidth), clearMemory: false);
            var packB = packBuf.Span;

            // One scratch tile per worker, not per micro-kernel call: a stackalloc inside the kernel is
            // what put its frame back and spilled the accumulators in the first place.
#pragma warning disable OVERFIT026 // BOUND: Nr512 (32) floats = 128 B, fixed at compile time.
            var tileScratch = stackalloc float[Nr512];
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

                var n0 = np * blockWidth;
                var blockEff = Math.Min(blockWidth, n - n0);

                if (!AblatePackB)
                {
                    // Each sub-panel keeps its own contiguous [k][32] region, so the micro-kernel is
                    // unchanged whatever the block width is.
                    fixed (float* pGather = packB)
                    {
                        for (var s = 0; s < group; s++)
                        {
                            var subFirst = s * Nr512;
                            var subEff = Math.Min(Nr512, blockEff - subFirst);

                            if (subEff <= 0)
                            {
                                continue;
                            }

                            GatherSubPanel(
                                c.Input,
                                pGather + ((long)s * k * Nr512),
                                n0 + subFirst,
                                subEff,
                                k,
                                inputH,
                                inputW,
                                kernelSize,
                                padding,
                                stride,
                                outW);
                        }
                    }
                }

                if (AblateMicroKernel)
                {
                    continue;
                }

                fixed (float* pPackB = packB)
                {
                    // M outer, sub-panel inner: the whole point of a wider block. The same eight rows of A
                    // serve every sub-panel back to back, so only the first call has to fetch them.
                    for (var rb = rowBlockStart; rb < rowBlockEnd; rb++)
                    {
                        var m0 = rb * Mr;
                        var mrEff = Math.Min(Mr, m - m0);

                        for (var s = 0; s < group; s++)
                        {
                            var subFirst = s * Nr512;
                            var nrEff = Math.Min(Nr512, blockEff - subFirst);

                            if (nrEff <= 0)
                            {
                                continue;
                            }

                            var subPackB = pPackB + ((long)s * k * Nr512);
                            var subN0 = n0 + subFirst;

                            if (UsePackedA)
                            {
                                var cTile = c.C + ((long)m0 * n) + subN0;

                                if (mrEff == Mr && nrEff == Nr512)
                                {
                                    MicroKernel8x32Avx512PackedAFull(
                                        c.A + ((long)rb * Mr * k), k, subPackB, cTile, n);

                                    continue;
                                }

                                MicroKernel8x32Avx512PackedAPartial(
                                    c.A + ((long)rb * Mr * k), mrEff, k, subPackB, cTile, n, nrEff, tileScratch);

                                continue;
                            }

                            MicroKernel8x32Avx512(c.A, m0, mrEff, k, subPackB, c.C, n, subN0, nrEff);
                        }
                    }
                }
            }
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
            // The steady state - a full 8x32 tile - goes to a body that holds nothing but the sixteen
            // accumulators. Everything else here is edge-case machinery, and its mere presence was costing
            // 3.46x. See MicroKernel8x32Avx512PackedAFull.
            if (mrEff == Mr && nrEff == Nr512)
            {
                MicroKernel8x32Avx512PackedAFull(packedA, k, packB, c + ((long)m0 * n) + n0, n);
                return;
            }

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

        /// <summary>
        /// The 8x32 micro-kernel for a FULL tile: no partial row block, no column tail, and therefore no
        /// scratch buffer and no store helper - only the sixteen accumulators.
        ///
        /// <para><b>Measured 2026-08-18, and it is the largest single factor found in `XC-78`.</b> An arm
        /// of `ConvGemmCostLadderBenchmark` that inlines this exact FMA sequence into the benchmark method,
        /// against the same buffers and the same addresses, ran <b>11.28 ms against 39.03 ms</b> for the
        /// same work calling the general kernel - <b>3.46x, from removing the call alone</b>. Sixteen
        /// <c>Vector512</c> accumulators need sixteen zmm registers, and in a method that also holds nine
        /// parameters, a <c>stackalloc</c> and eight store calls, the register allocator does not keep
        /// them there.</para>
        ///
        /// <para><b>This is why every memory hypothesis measured null.</b> Cache blocking, software
        /// prefetch, tile shape and working-set capacity were each tested and each moved nothing, because
        /// none of them touches register allocation. The gap was never in the memory hierarchy.</para>
        /// </summary>
        private static unsafe void MicroKernel8x32Avx512PackedAFull(
            float* packedA,
            int k,
            float* packB,
            float* c,
            int n)
        {
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

            c00.Store(c + ((long)0 * n));
            c01.Store(c + ((long)0 * n) + 16);
            c10.Store(c + ((long)1 * n));
            c11.Store(c + ((long)1 * n) + 16);
            c20.Store(c + ((long)2 * n));
            c21.Store(c + ((long)2 * n) + 16);
            c30.Store(c + ((long)3 * n));
            c31.Store(c + ((long)3 * n) + 16);
            c40.Store(c + ((long)4 * n));
            c41.Store(c + ((long)4 * n) + 16);
            c50.Store(c + ((long)5 * n));
            c51.Store(c + ((long)5 * n) + 16);
            c60.Store(c + ((long)6 * n));
            c61.Store(c + ((long)6 * n) + 16);
            c70.Store(c + ((long)7 * n));
            c71.Store(c + ((long)7 * n) + 16);
        }

        /// <summary>
        /// The 8x32 micro-kernel for a PARTIAL tile — fewer than eight rows, or fewer than thirty-two
        /// columns, or both — with the same register-only K loop as the full-tile body.
        ///
        /// <para><b>Why a third body rather than a branch in the second.</b> The full-tile body sits
        /// exactly at the register limit: adding three prefetch instructions to it, and nothing else,
        /// measured <b>40% slower</b> and landed back at the spilling kernel's speed. Anything added to
        /// that method costs the whole win, so the edge case gets its own body instead.</para>
        ///
        /// <para><b>The scratch buffer belongs to the caller</b>, allocated once per worker rather than
        /// per call. A <c>stackalloc</c> inside this method would put the frame back and reintroduce the
        /// spill it exists to avoid — which is exactly what the general kernel does.</para>
        ///
        /// <para><b>What it is worth, computed from the shapes rather than assumed.</b> On VGG-16 every
        /// output-channel count divides by eight, so there are no partial ROW blocks at all, and the only
        /// partial tiles are one column panel each in conv8-13 — about <b>0.55% of the model</b>. The case
        /// this is really for is a network whose channel count is not a multiple of eight, where every
        /// panel ends in a partial row block instead of one panel in seven ending short.</para>
        /// </summary>
        private static unsafe void MicroKernel8x32Avx512PackedAPartial(
            float* packedA,
            int mrEff,
            int k,
            float* packB,
            float* c,
            int n,
            int nrEff,
            float* scratch)
        {
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

            StorePartialRow(c, n, 0, mrEff, nrEff, c00, c01, scratch);
            StorePartialRow(c, n, 1, mrEff, nrEff, c10, c11, scratch);
            StorePartialRow(c, n, 2, mrEff, nrEff, c20, c21, scratch);
            StorePartialRow(c, n, 3, mrEff, nrEff, c30, c31, scratch);
            StorePartialRow(c, n, 4, mrEff, nrEff, c40, c41, scratch);
            StorePartialRow(c, n, 5, mrEff, nrEff, c50, c51, scratch);
            StorePartialRow(c, n, 6, mrEff, nrEff, c60, c61, scratch);
            StorePartialRow(c, n, 7, mrEff, nrEff, c70, c71, scratch);
        }

        /// <summary>One row of a partial tile: a whole row goes straight out, a short one through the
        /// caller's scratch. Rows past <paramref name="mrEff"/> are not written at all.</summary>
        private static unsafe void StorePartialRow(
            float* c, int n, int row, int mrEff, int nrEff,
            Vector512<float> lo, Vector512<float> hi, float* scratch)
        {
            if (row >= mrEff)
            {
                return;
            }

            var dst = c + ((long)row * n);

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

            /// <summary>
            /// The shared im2col expansion, one contiguous <c>[k][32]</c> region per panel, or
            /// <see langword="null"/> when each work item gathers its own panel.
            /// </summary>
            public readonly float* Expanded;

            /// <summary>
            /// Panel count, used only by the expand-then-GEMM path, whose work items are flat
            /// <c>(row block, panel)</c> pairs rather than the <c>panel x M-block</c> grid <see cref="MBlocks"/>
            /// describes. Kept as its own field because reusing <see cref="MBlocks"/> for a second meaning is
            /// the kind of saving that reads correctly and is wrong.
            /// </summary>
            public readonly int Panels;
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
                int outW,
                float* expanded = null,
                int panels = 0)
            {
                A = a;
                Input = input;
                C = c;
                Expanded = expanded;
                Panels = panels;
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
        /// <summary>
        /// Whether the fused im2col path splits M as well as N. <b>On since 2026-08-25</b>, when the split
        /// was capped at two blocks; <c>OVERFIT_CONV_FUSED_M_SPLIT=0</c> turns it off for an A/B.
        ///
        /// <para><b>It was off before that, and the reason it lost is worth keeping.</b> The M-split was
        /// measured as a gain when the B pack it duplicated was a pack. In the fused path the same
        /// duplication is an <b>im2col gather</b>: work items are <c>item / mBlocks</c>, so five row-blocks
        /// sharing a panel gather that panel five times. Under the old
        /// <c>ceil(workers / nPanels)</c> rule that is what the shipping 32-worker pool asked for, and on
        /// VGG-16's last three convolutions it gathered 16.5 MB where 4.1 MB is needed. <b>Measured
        /// 2026-08-19, ABAB, three interleaved passes:</b> VGG-16 ran 35.47/34.73/34.92 ms with the split
        /// against <b>33.68/33.62/33.49 without</b>, <b>-4.1% mean</b>.</para>
        ///
        /// <para><b>What changed is the number of blocks, not the mechanism</b> — see
        /// <see cref="MaxFusedMBlocks"/> for the arm that separates the two.</para>
        /// </summary>
        internal static readonly bool FusedMSplitEnabled =
            Environment.GetEnvironmentVariable(OverfitEnvironment.ConvFusedMSplit) != "0";

        /// <summary>
        /// How many ways one panel's M sweep is split, when it is split at all.
        ///
        /// <para><b>Two, and the number is measured rather than reasoned.</b> Each extra block re-gathers
        /// the whole panel, so the added work is <c>(mBlocks - 1) x gather</c> while the added width is
        /// only useful up to the number of PHYSICAL cores — which the pool, sized from
        /// <c>Environment.ProcessorCount</c>, does not know. From the published per-term costs the ratio
        /// GEMM/gather for one panel is <c>2 x m x 32 x k / 301e9</c> over <c>32 x k x 0.964e-9</c>, i.e.
        /// <c>m / 145</c>: for VGG's <c>m = 512</c> a two-way split adds 22% work and a four-way adds
        /// 66%.</para>
        ///
        /// <para><b>Measured 2026-08-25 on the shipped binary, VGG-16, 9950X3D, Release, per-node
        /// profiler, three sittings, arm order rotated, box quiet, canary 1.211-1.228 ms throughout.</b>
        /// Nodes 14/15/16, median ms, <c>OVERFIT_CONV_FUSED_M_SPLIT=0</c> against the default:
        /// <b>0.84/0.83/0.82 -> 0.63/0.65/0.64</b> at 16 workers pinned one per physical core, and
        /// <b>0.85/0.89/0.90 -> 0.58/0.67/0.66</b> at the shipping 32-logical pool. Convolution total
        /// 17.46 -> 16.58 and 17.08 -> 16.52 ms. <b>The one-core arm is the control and does not move</b>
        /// (3.96/3.94/3.94 against 3.96/3.96/3.95): a single worker never splits, so the change adds no
        /// work of its own.</para>
        ///
        /// <para><b>Why two and not four, measured on an exploratory build carrying a cap lever.</b> Nodes
        /// 14/15/16 at the 32-logical pool: no split 0.85/0.88/0.91, cap 4 — which is what
        /// <c>floor(32/7)</c> asks for — 0.82/0.89/0.88, cap 3 0.71/0.82/0.77, cap 2 0.59/0.67/0.65. The
        /// cost is monotone in the cap, which is the duplication term and not the width term.</para>
        ///
        /// <para><b>The one-round argument is not the whole story, and that is why this is a constant.</b>
        /// <c>ceil(16/7) = 3</c> puts 21 items on 16 workers — two rounds — and measured
        /// <b>0.81/0.81/0.80</b>, no better than no split at all. But <c>floor(32/7) = 4</c> fits in one
        /// round of the 32-worker pool and is <i>also</i> no better, because sixteen of those workers are
        /// SMT siblings that add no gather throughput. Rounds explain the first arm; duplicated work
        /// explains the second.</para>
        /// </summary>
        private const int MaxFusedMBlocks = 2;

        /// <summary>
        /// Measurement only: <c>OVERFIT_CONV_FUSED_M_BLOCKS=n</c> forces the fused path to exactly
        /// <c>n</c> M-blocks on every layer, bypassing the panel-count gate in
        /// <see cref="FusedMBlocksFor"/>. Unset it is 0 and the gate decides, which is what ships.
        ///
        /// <para><b>It is kept because the loss it measured is the useful part</b>, in the same way as
        /// <see cref="ExpandPanelsEnabled"/>. See <see cref="FusedMBlocksFor"/> for the numbers.</para>
        ///
        /// <para><b>A field rather than a readonly, so a test can drive it</b> — a lever that changes the
        /// row decomposition needs a correctness arm, and the shipping rule never asks for more than two
        /// blocks, so nothing else in the suite ever executes three or four. Mirrors
        /// <see cref="AblatePackB"/>. Anything that writes it belongs in
        /// <c>ExclusiveProcessMeasurementCollection</c>: it is process-wide.</para>
        /// </summary>
        internal static int FusedMBlocksOverride = ResolveFusedMBlocksOverride();

        private static int ResolveFusedMBlocksOverride()
        {
            var raw = Environment.GetEnvironmentVariable(OverfitEnvironment.ConvFusedMBlocks);

            return int.TryParse(raw, out var blocks) && blocks > 0 ? blocks : 0;
        }

        /// <summary>The M-split decision for the fused path.</summary>
        private static int ResolveFusedMBlocks(int m, int nPanels)
        {
            if (!FusedMSplitEnabled)
            {
                return 1;
            }

            var rowBlocks = (m + Mr - 1) / Mr;

            if (FusedMBlocksOverride > 0)
            {
                return Math.Min(rowBlocks, FusedMBlocksOverride);
            }

            return FusedMBlocksFor(rowBlocks, nPanels, OverfitParallel.MaxDegreeOfParallelism);
        }

        /// <summary>
        /// How many ways the fused path splits M, from the row blocks that exist, the panels the layer
        /// produces and the workers available. Pure arithmetic, so a test can drive every case with exact
        /// integers instead of depending on the box it runs on.
        ///
        /// <para><b>Two blocks where the panels cannot fill the pool, one everywhere else.</b> VGG-16's
        /// 14x14 convolutions produce <c>ceil(196/32) = 7</c> panels, so the dispatch at
        /// <see cref="GemmFusedIm2Col"/> offers seven work items and at most seven workers can ever be
        /// busy. Measured 2026-08-25 on this box: those layers take <b>3.93 ms at one core, 0.87 at seven
        /// and 0.83 at sixteen</b> — seven to sixteen cores buys nothing, while a many-panel layer in the
        /// same run keeps improving (node 1, 1568 panels: 3.85 -> 2.40 ms). Nine of sixteen cores were
        /// idle, which is what seven items predicts.</para>
        ///
        /// <para>How many blocks, and why not more, is <see cref="MaxFusedMBlocks"/>.</para>
        /// </summary>
        internal static int FusedMBlocksFor(int rowBlocks, int nPanels, int workers)
        {
            if (nPanels <= 0 || rowBlocks <= 1)
            {
                return 1;
            }

            // Integer division rather than `nPanels * 2 > workers`: it cannot overflow, and it IS the
            // guard. A layer with more than half the workers' worth of panels gets 1 here.
            //
            // MEASURED DIRECTLY 2026-08-25 on the layers this excludes, through the override above, which
            // replaces the 2026-08-18 inference from GFLOP/s at 32 workers on the unfused path. VGG-16's
            // 28x28 convolutions (nodes 10/11/12, 25 panels, m = 512), 16 workers pinned one per physical
            // core, three sittings with the arm order rotated, median ms against the shipping rule:
            //
            //     blocks   node 10   node 11   node 12   the three together
            //     1 (ship)   0.970     1.690     1.610     4.27 ms
            //     2          1.010     1.770     1.690     4.47 ms   +4.7%
            //     3          1.080     1.860     1.760     4.70 ms  +10.1%
            //     4          1.130     1.980     1.890     5.00 ms  +17.1%
            //
            // EVERY split is a loss and the loss is monotone in the block count, so the gate is not a
            // threshold to tune — there is no better value on the other side of it.
            //
            // WHY, given that 25 items on 16 workers looks like it wants more parallelism. It does not:
            // `OverfitParallel.For` slices 25 items into ceil(25/16) = 2 per chunk, so the critical path
            // is 2 items and the DECOMPOSITION alone permits 25/2 = 12.5x. These layers measure 6.79x,
            // 7.78x and 8.12x at 16 cores, i.e. 54-65% of that, so the binding constraint is not the
            // decomposition and adding items cannot reach it. Doubling to 50 items gives ceil(50/16) = 4
            // half-items, which is the SAME 78% occupancy for one extra gather per panel.
            //
            // The duplicated gather was measured too, at one core where the override still runs
            // nPanels * mBlocks items: three blocks add +14.3/+15.0/+14.6% of single-core work on these
            // three layers and +19.5% on the 14x14 ones. So a gather is ~7% of a panel here, far cheaper
            // than the m/145 cost model in `MaxFusedMBlocks` predicts — and the split still loses.
            if (workers / nPanels < 2)
            {
                return 1;
            }

            return Math.Min(rowBlocks, MaxFusedMBlocks);
        }

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
            // The steady state gets a body holding nothing but the sixteen accumulators; see
            // MicroKernel8x32Avx512Full for why the edge-case machinery cannot share a method with it.
            if (mrEff == Mr && nrEff == Nr512)
            {
                MicroKernel8x32Avx512Full(a, m0, k, packB, c, n, n0);
                return;
            }

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

        /// <summary>
        /// The row-major 8x32 micro-kernel for a FULL tile: eight whole rows and thirty-two whole columns,
        /// so no clamped row pointers, no scratch buffer and no store helper.
        ///
        /// <para><b>Why this path matters even though production does not take it.</b> It is reached only
        /// with <c>OVERFIT_CONV_PACK_A=0</c> or <c>OVERFIT_CONV_FUSED_IM2COL=0</c> - the A/B switches. Left
        /// unfixed, <c>PACK_A=0</c> measures "no packing AND a spilling kernel" and so overstates what the
        /// packing itself is worth. <b>A switch whose off-arm is broken makes its own measurement
        /// dishonest</b>, which is the trap the prefetch experiment fell into earlier the same day.</para>
        ///
        /// <para>The AVX2 path already had this split - <c>GemmNPanelWorker</c> chooses between
        /// <c>MicroKernel8x8</c> and <c>MicroKernelTail</c> on <c>mrEff == Mr</c>. It was the AVX-512
        /// kernel, added later, that lost it.</para>
        /// </summary>
        private static unsafe void MicroKernel8x32Avx512Full(
            float* a, int m0, int k, float* packB, float* c, int n, int n0)
        {
            var r0 = a + ((long)m0 * k);
            var r1 = r0 + k;
            var r2 = r1 + k;
            var r3 = r2 + k;
            var r4 = r3 + k;
            var r5 = r4 + k;
            var r6 = r5 + k;
            var r7 = r6 + k;

            Vector512<float> c00 = default, c01 = default, c10 = default, c11 = default;
            Vector512<float> c20 = default, c21 = default, c30 = default, c31 = default;
            Vector512<float> c40 = default, c41 = default, c50 = default, c51 = default;
            Vector512<float> c60 = default, c61 = default, c70 = default, c71 = default;

            for (var kk = 0; kk < k; kk++)
            {
                var b0 = Vector512.Load(packB + (kk * Nr512));
                var b1 = Vector512.Load(packB + (kk * Nr512) + 16);

                var r = Vector512.Create(r0[kk]);
                c00 = Avx512F.FusedMultiplyAdd(r, b0, c00);
                c01 = Avx512F.FusedMultiplyAdd(r, b1, c01);
                r = Vector512.Create(r1[kk]);
                c10 = Avx512F.FusedMultiplyAdd(r, b0, c10);
                c11 = Avx512F.FusedMultiplyAdd(r, b1, c11);
                r = Vector512.Create(r2[kk]);
                c20 = Avx512F.FusedMultiplyAdd(r, b0, c20);
                c21 = Avx512F.FusedMultiplyAdd(r, b1, c21);
                r = Vector512.Create(r3[kk]);
                c30 = Avx512F.FusedMultiplyAdd(r, b0, c30);
                c31 = Avx512F.FusedMultiplyAdd(r, b1, c31);
                r = Vector512.Create(r4[kk]);
                c40 = Avx512F.FusedMultiplyAdd(r, b0, c40);
                c41 = Avx512F.FusedMultiplyAdd(r, b1, c41);
                r = Vector512.Create(r5[kk]);
                c50 = Avx512F.FusedMultiplyAdd(r, b0, c50);
                c51 = Avx512F.FusedMultiplyAdd(r, b1, c51);
                r = Vector512.Create(r6[kk]);
                c60 = Avx512F.FusedMultiplyAdd(r, b0, c60);
                c61 = Avx512F.FusedMultiplyAdd(r, b1, c61);
                r = Vector512.Create(r7[kk]);
                c70 = Avx512F.FusedMultiplyAdd(r, b0, c70);
                c71 = Avx512F.FusedMultiplyAdd(r, b1, c71);
            }

            var tile = stackalloc float[Nr512];

            c00.Store(c + ((long)(m0 + 0) * n) + n0);
            c01.Store(c + ((long)(m0 + 0) * n) + n0 + 16);
            c10.Store(c + ((long)(m0 + 1) * n) + n0);
            c11.Store(c + ((long)(m0 + 1) * n) + n0 + 16);
            c20.Store(c + ((long)(m0 + 2) * n) + n0);
            c21.Store(c + ((long)(m0 + 2) * n) + n0 + 16);
            c30.Store(c + ((long)(m0 + 3) * n) + n0);
            c31.Store(c + ((long)(m0 + 3) * n) + n0 + 16);
            c40.Store(c + ((long)(m0 + 4) * n) + n0);
            c41.Store(c + ((long)(m0 + 4) * n) + n0 + 16);
            c50.Store(c + ((long)(m0 + 5) * n) + n0);
            c51.Store(c + ((long)(m0 + 5) * n) + n0 + 16);
            c60.Store(c + ((long)(m0 + 6) * n) + n0);
            c61.Store(c + ((long)(m0 + 6) * n) + n0 + 16);
            c70.Store(c + ((long)(m0 + 7) * n) + n0);
            c71.Store(c + ((long)(m0 + 7) * n) + n0 + 16);
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
