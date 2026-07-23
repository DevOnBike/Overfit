// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Runtime.CompilerServices;
using DevOnBike.Overfit.Intrinsics;
using DevOnBike.Overfit.Runtime;
using DevOnBike.Overfit.Tensors;

namespace DevOnBike.Overfit.LanguageModels.Runtime
{
    /// <summary>
    /// Batched (prefill) projection dispatch over a <see cref="DecodeWeight"/>: picks the
    /// <c>ProjectBatched</c> kernel matching the weight's resident format (Q6_K / Q4_K / Q8_0 / F32)
    /// and runs <c>rows</c> activation rows × the weight matrix in one pass — each weight
    /// row read from DRAM once, reused across all rows (the prefill weight-bandwidth amortisation).
    /// Activation-quantization scratch is POOLED per call and handed to the kernels as exact-length
    /// slices (so any kernel-side <c>.Length</c> arithmetic is unchanged). This dispatcher runs once
    /// per projection per layer per prefill — it was the single largest allocator of the prefill path.
    /// </summary>
    internal static class BatchedQuantProjection
    {
        /// <summary>Test-only A/B toggle for the Q4_K batched kernel (weight-stationary vs the original
        /// re-decode-per-row <see cref="Q4KDotKernel.ProjectBatched"/>). Default true (the validated faster path);
        /// flipped by perf benches to measure the end-to-end delta. Not a runtime knob.</summary>
        internal static bool UseWeightStationaryQ4K = true;

        /// <summary>Gates the register-tiled Q4_K prefill GEMM (<see cref="Q4KGemvKernel.GemmTiled"/>). Defaults to
        /// the <c>OVERFIT_TILED_PREFILL</c> env flag; mutable so perf/coherence benches can A/B it in one process.</summary>
        internal static bool UseTiledPrefillQ4K = Q4KGemvKernel.TiledPrefillEnabled;

        /// <summary>Gates the register-tiled Q6_K prefill GEMM (<see cref="Q6KGemvKernel.GemmTiled"/>).
        /// Mutable so perf tests can A/B it in one process.
        ///
        /// <para><b>Costs RAM:</b> unlike Q4_K, <see cref="Q6KWeight"/> has no prepacked-sidecar path, so
        /// <c>EnsureRepacked</c> always allocates a heap copy (~the size of the Q6_K tensors) on first
        /// use.</para></summary>
        internal static bool UseTiledPrefillQ6K = true;

        /// <summary>
        /// Test hook: forces the NON-repacked batched kernels for both Q4_K and Q6_K, overriding even
        /// <c>IsPrepacked</c>. Mirrors <c>CachedLlamaSession.DisableBatchedPrefillForParity</c>.
        ///
        /// <para>Needed because the repacked <c>block_q*_Kx8</c> GEMMs associate their reduction differently
        /// from the per-row kernels, so they are <b>not</b> bit-identical to the single-token path — measured
        /// at <c>maxAbsLogitDiff ≈ 0.44</c> on Qwen-3B, enough to flip an argmax. That is the accepted trade
        /// (the same standard <c>OVERFIT_REPACK_ATTN</c> is held to: validated by end-to-end coherence, not
        /// byte-parity), but it means a test asserting batched == single-token has to hold the kernel layout
        /// constant, or it silently stops testing the thing it claims to.</para>
        ///
        /// <para>A <c>*.gguf.repack</c> sidecar sets <c>IsPrepacked</c> and therefore turns the repacked path
        /// on regardless of the env flag — which is exactly how <c>BatchedPrefillParityTests</c> came to be
        /// failing unnoticed for two days, being <c>[LongFact]</c>.</para>
        /// </summary>
        internal static bool DisableRepackedKernelsForParity;

        /// <summary>Forces a specific prefill column-tile width; 0 leaves <see cref="ResolveTileCols"/> to choose.</summary>
        internal static int TileColsOverride;

        /// <summary>
        /// Parallelise the tiled prefill GEMM over <b>bands of output rows</b> instead of over column tiles.
        ///
        /// <para><b>The problem it addresses.</b> Today one work item is one column tile, and a column tile walks
        /// the <i>entire</i> weight matrix — so every worker streams all 12.68 MB of `ffn_gate_up`, and the matrix
        /// is re-read <c>rows/NR</c> times per projection. Widening NR halves that traffic but also halves the
        /// number of work items, and the measured sweep showed the two cancelling: NR=16 lost 8% at 672 rows to
        /// scheduling imbalance despite halving the traffic.</para>
        ///
        /// <para><b>The change.</b> Give each worker a contiguous band of output groups and let it loop over all
        /// column tiles inside that band. Its weight working set is then one band — sized below to fit L2 —
        /// which it reads once from L3 and re-reads from its own cache for every remaining column tile. The
        /// memory probe measured exactly this distinction: the same instruction mix ran at 4.60 TFLOP/s against
        /// an L2-resident window and 3.04 against an L3-resident one, a <b>1.50×</b> difference.</para>
        ///
        /// <para>It also decouples the two knobs: work-item count no longer depends on NR, so a wide column tile
        /// stops costing parallelism. This is the L2 blocking level of the standard GEMM structure
        /// (Goto &amp; van de Geijn), which this kernel has never had.</para>
        /// </summary>
        internal static bool UseOutputBlocking;

        /// <summary>
        /// Decode every weight block's F16 scale/min pair to <see cref="float"/> once per projection instead of
        /// once per column tile.
        ///
        /// <para><c>GemmTiled</c> decodes them inline, which reads as amortised — but the kernel runs once per
        /// column tile, 84 times for a 672-token prompt at NR=8, so each F16 pair is widened 84 times over.
        /// Ablation put that decode at <b>12%</b> of the kernel, the largest non-arithmetic item measured, and
        /// it is fixed work per block, so it is exactly the term the tile-width sweep showed being amortised
        /// across columns. Hoisting it divides the work by the tile count.</para>
        /// </summary>
        internal static bool UsePrecomputedScales = true;

        /// <summary>
        /// Route the tiled Q4_K prefill GEMM through <see cref="Q4KGemvKernel.GemmTiled512"/>, which processes
        /// two activation columns per instruction. Defaults to on wherever the silicon supports it.
        ///
        /// <para>Measured ceilings on this machine for the kernel's own instruction mix: 4.63 TFLOP/s at 256
        /// bits against 7.71–9.08 at 512. The port is bit-identical, so the existing parity tests apply to it
        /// unchanged; <c>Avx512PrefillParityTests</c> pins the two kernels against each other directly.</para>
        /// </summary>
        internal static bool UseAvx512PrefillQ4K =
            CpuFeatures.HasAvx512
            && CpuFeatures.HasAvx512Bw
            && Environment.GetEnvironmentVariable(OverfitEnvironment.Avx512PrefillQ4K) != "0";

        /// <summary>
        /// The same port for Q6_K — <b>measured slower and therefore off</b>. Kept behind the flag with
        /// <c>Avx512Q6KPrefillParityTests</c> guarding it, because the negative is the useful part.
        ///
        /// <para>On the identical machine and prompt where the Q4_K port took <c>ffn_gateup</c> down 13.8%,
        /// the Q6_K port took <c>ffn_down</c> from 658 ms to <b>794–900 ms</b> and prefill from 280 back to
        /// 256–265 tok/s. The Q4_K components stayed flat across the same runs, and the run-to-run spread was
        /// concentrated entirely on <c>ffn_down</c>, so it is the change and not the box.</para>
        ///
        /// <para><b>Why the same technique inverts.</b> Column pairing pays for the <c>vinserti64x4</c> that
        /// builds each broadcast out of arithmetic done on it. Q4_K broadcasts eight weight vectors per
        /// sub-block and then issues sixteen paired statements against them. Q6_K broadcasts six per <c>k</c>,
        /// sixteen times per block, for far less arithmetic each — and its <c>ReduceRows</c> cannot widen at all
        /// (no <c>vphaddd</c> for zmm), adding three more cross-half moves per call, 32 calls per block. The
        /// lane-crossing traffic outruns the arithmetic saved.</para>
        /// </summary>
        internal static bool UseAvx512PrefillQ6K =
            CpuFeatures.HasAvx512
            && CpuFeatures.HasAvx512Bw
            && Environment.GetEnvironmentVariable(OverfitEnvironment.Avx512PrefillQ6K) != "0";

        /// <summary>
        /// Weight bytes one worker's band may occupy. Half of a 1 MB Zen-5 L2, leaving the rest for the
        /// activation tile and the output band; the point is residency, not filling the cache exactly.
        /// </summary>
        private const int BandWeightBudgetBytes = 512 * 1024;

        /// <summary>
        /// Output groups per band: small enough that the band's weights sit in L2, and numerous enough that
        /// every core still gets several bands so the tail worker does not set the region's duration.
        /// </summary>
        private static int ResolveGroupsPerBand(int totalGroups, int superBlocksPerRow, int blockBytes, int cores)
        {
            var bytesPerGroup = Math.Max(1, superBlocksPerRow * blockBytes);
            var byCache = Math.Max(1, BandWeightBudgetBytes / bytesPerGroup);
            var byParallelism = Math.Max(1, totalGroups / (cores * 2));

            return Math.Min(byCache, byParallelism);
        }

        /// <summary>
        /// Picks the column-tile width (NR) for one prefill projection: the <b>widest</b> tile that still leaves
        /// at least one tile per core.
        ///
        /// <para><b>Why width matters more than it looks.</b> A tile of NR columns walks the <i>entire</i> weight
        /// matrix, so the matrix is streamed <c>rows/NR</c> times per projection. At NR=8 and 672 rows that is 84
        /// passes over `ffn_gate_up`'s 12.68 MB — 1.07 GB of traffic in 15.3 ms, about 70 GB/s against a measured
        /// 90 GB/s read ceiling. Doubling NR halves that traffic outright. This is the only blocking level the
        /// kernel has: the tile lives in registers, and there is nothing sized to L2 or L3 between it and memory.
        /// </para>
        ///
        /// <para><b>And why it cannot simply be maximised.</b> Fewer, fatter tiles are fewer independent work
        /// items, and the parallel region costs its longest worker. Measured on 672 rows across 32 cores
        /// (`ffn_gate_up`, two runs, agreeing):</para>
        ///
        /// <list type="table">
        ///   <item><term>NR=4</term><description>168 passes, 17.8 ms, 1.70 TFLOP/s</description></item>
        ///   <item><term>NR=8</term><description>84 passes, 15.2–15.9 ms, <b>1.95 TFLOP/s</b></description></item>
        ///   <item><term>NR=16</term><description>42 passes, 16.5–17.0 ms, 1.80 TFLOP/s</description></item>
        /// </list>
        ///
        /// <para>Halving the traffic 4→8 buys +17%, exactly as the bandwidth argument predicts. Halving it again
        /// 8→16 <i>loses</i> 8%, because 42 tiles over 32 cores leaves ten workers with two tiles and twenty-two
        /// with one — a 1.52× imbalance against 1.14× at NR=8. So the rule is not "widest that fits a core" but
        /// "widest that still gives every core a couple of tiles"; below that the granularity loss outruns the
        /// traffic saving. A longer prompt moves the balance back toward the wider tile.</para>
        ///
        /// <para>Escaping the trade-off entirely needs the missing blocking level: block over output rows as
        /// well, so a wide column tile and a large number of independent work items stop being alternatives.</para>
        /// </summary>
        private static int ResolveTileCols(int rows, int cores, int maxTileCols)
        {
            if (TileColsOverride > 0)
            {
                return Math.Min(TileColsOverride, maxTileCols);
            }

            // Require ~2 tiles per core, not 1: at exactly one the tail worker doubles the region's duration.
            const int TilesPerCore = 2;

            for (var nr = 16; nr > 4; nr >>= 1)
            {
                if (nr <= maxTileCols && rows / nr >= cores * TilesPerCore)
                {
                    return nr;
                }
            }

            return 4;
        }

        /// <summary>
        /// <paramref name="preQuants"/> / <paramref name="preScales"/> / <paramref name="preBsums"/> let the
        /// caller supply activations ALREADY quantized to Q8_K, skipping the internal quantization pass.
        /// Empty (the default) keeps the original behaviour: pool the scratch and quantize here.
        ///
        /// <para>Attention needs this because it dispatches Q once <b>per head</b> and K/V once per group,
        /// every one of them over the same loop-invariant <c>hidden</c> — a benchmark measured the Q8_K
        /// quantization of a 672×2048 activation block at ~1.0 ms against a 1.079 ms Q-head dispatch, i.e.
        /// ~93% of the call. Quantizing once per layer is bit-identical, since the quantization is
        /// deterministic.</para>
        ///
        /// <para>Honoured for the Q6_K and Q4_K paths (everything attention uses); the Q8_0 and F32 paths
        /// ignore it and quantize as before.</para>
        /// </summary>
        public static void Dispatch(
            ReadOnlySpan<float> input,
            int rows,
            in DecodeWeight weight,
            ReadOnlySpan<float> bias,
            Span<float> output,
            int inputSize,
            int outputSize,
            Span<sbyte> preQuants = default,
            Span<float> preScales = default,
            Span<short> preBsums = default)
        {
            // Resident-format dispatch, classified once so the original first-match order is explicit.
            var kind = weight.IsQ6K ? 0 : weight.IsQ4K ? 1 : weight.IsQuantized ? 2 : 3;
            var pre = !preQuants.IsEmpty;

            if (kind == 0 && pre)
            {
                var wp = weight.Quantized6K;
                DispatchQ6K(
                    input, rows, wp, bias, output, inputSize,
                    preQuants, preScales, preBsums, preQuantized: true);
                return;
            }

            if (kind == 1 && pre)
            {
                var wp = weight.Quantized4K;
                DispatchQ4K(
                    input, rows, wp, bias, output, inputSize,
                    preQuants, preScales, preBsums, preQuantized: true);
                return;
            }

            if (kind == 0)
            {
                var w = weight.Quantized6K;
                var spr = w.SuperBlocksPerRow;
                var groups = rows * spr * Q6KDotKernel.GroupsPerSuperBlock;
                using var qBytes = new PooledBuffer<sbyte>(rows * inputSize, clearMemory: false);
                using var scales = new PooledBuffer<float>(rows * spr, clearMemory: false);
                using var sums = new PooledBuffer<short>(groups, clearMemory: false);
                DispatchQ6K(
                    input, rows, w, bias, output, inputSize,
                    qBytes.Span.Slice(0, rows * inputSize), scales.Span.Slice(0, rows * spr),
                    sums.Span.Slice(0, groups), preQuantized: false);
            }
            if (kind == 1)
            {
                var w = weight.Quantized4K;
                var spr = w.SuperBlocksPerRow;
                var groups = rows * spr * Q4KDotKernel.GroupsPerSuperBlock;
                using var qBytes = new PooledBuffer<sbyte>(rows * inputSize, clearMemory: false);
                using var scales = new PooledBuffer<float>(rows * spr, clearMemory: false);
                using var sums = new PooledBuffer<short>(groups, clearMemory: false);
                DispatchQ4K(
                    input, rows, w, bias, output, inputSize,
                    qBytes.Span.Slice(0, rows * inputSize), scales.Span.Slice(0, rows * spr),
                    sums.Span.Slice(0, groups), preQuantized: false);
            }
            if (kind == 2)
            {
                var w = weight.Quantized;
                var bpr = inputSize / Q8DotKernel.BlockSize;
                using var qBytes = new PooledBuffer<sbyte>(rows * inputSize, clearMemory: false);
                using var scales = new PooledBuffer<float>(rows * bpr, clearMemory: false);
                Q8DotKernel.ProjectBatched(
                    input, rows, w, bias, output,
                    qBytes.Span.Slice(0, rows * inputSize), scales.Span.Slice(0, rows * bpr));
            }
            if (kind == 3)
            {
                BatchedProjectionKernel.Project(input, rows, weight.F32, bias, output, inputSize, outputSize);
            }
        }

        // Q6_K format path, shared by the pooled and pre-quantized entries so the kernel-selection gates
        // exist in exactly one place.
        private static void DispatchQ6K(
            ReadOnlySpan<float> input, int rows, Q6KWeight w, ReadOnlySpan<float> bias, Span<float> output,
            int inputSize, Span<sbyte> quants, Span<float> scales, Span<short> sums, bool preQuantized)
        {
            // Register-tiled Q6_K GEMM over the repacked block_q6_Kx8 layout. Under Q4_K_M half of ffn_down
            // is Q6_K, and a prefill profile put ffn_down at 37.9% of prefill running at 0.61 TFLOP/s -
            // against ffn_gate_up's 1.78 - precisely because Q6_K had only the re-decode-per-row kernel.
            // No-bias only (GemmTiled applies none); AVX2/FMA required.
            var tiled6 = UseTiledPrefillQ6K && !DisableRepackedKernelsForParity
                && bias.IsEmpty && w.CanRepack
                && CpuFeatures.HasAvx2 && CpuFeatures.HasFma;

            if (tiled6)
            {
                DispatchTiledQ6K(input, rows, w, output, quants, scales, sums, preQuantized);
            }

            if (!tiled6)
            {
                Q6KDotKernel.ProjectBatched(
                    input, rows, w, bias, output, quants, scales, sums, preQuantized);
            }
        }

        // Q4_K format path, shared by the pooled and pre-quantized entries.
        private static void DispatchQ4K(
            ReadOnlySpan<float> input, int rows, Q4KWeight w, ReadOnlySpan<float> bias, Span<float> output,
            int inputSize, Span<sbyte> quants, Span<float> scales, Span<short> sums, bool preQuantized)
        {
            // Register-tiled GEMM: repacked block_q4_Kx8, decode each super-block once and reuse across a
            // tile of NR columns - measured ~3x vs weight-stationary under parallelism, 1.61x end-to-end
            // prefill. Default-on when the weight is already prepacked (an offline sidecar mmap'd it -> zero
            // extra RAM); otherwise opt-in via OVERFIT_TILED_PREFILL since repacking copies the weight.
            // No-bias only (GemmTiled applies none). AVX2/FMA required - the kernel is x86-only, so on ARM
            // (e.g. the Android app) this falls through to the weight-stationary path even if a sidecar
            // mmap'd a prepacked layout (IsPrepacked would otherwise bypass the env flag's AVX2 gate).
            // `bias.IsEmpty` used to sit here because GemmTiled applied none. With attention Q/K/V biased,
            // that kept attn_q (15% of prefill) on the weight-stationary kernel even though its shape
            // [2048 -> 128] repacks fine. GemmTiled now folds the bias into its final store.
            var tiled = (w.IsPrepacked || UseTiledPrefillQ4K) && !DisableRepackedKernelsForParity
                && w.CanRepack
                && CpuFeatures.HasAvx2 && CpuFeatures.HasFma;

            if (tiled)
            {
                DispatchTiledQ4K(input, rows, w, bias, output, quants, scales, sums, preQuantized);
            }

            // Weight-stationary: decode each Q4_K super-block once and reuse across the row tile
            // (bit-identical to ProjectBatched, measured ~1.3-1.7x on the batched matmul).
            if (!tiled && UseWeightStationaryQ4K)
            {
                Q4KDotKernel.ProjectBatchedWeightStationary(
                    input, rows, w, bias, output, quants, scales, sums, preQuantized);
            }

            if (!tiled && !UseWeightStationaryQ4K)
            {
                Q4KDotKernel.ProjectBatched(
                    input, rows, w, bias, output, quants, scales, sums, preQuantized);
            }
        }

        // Register-tiled Q4_K prefill GEMM: quantize all rows to Q8_K, then run GemmTiled over row-tiles of NR
        // columns in parallel. NR is chosen so the tile count stays >= cores (an under-filled pool regressed
        // hard in the Phase-3 bench). No-bias only (checked at the call site) — GemmTiled applies no bias.
        private static unsafe void DispatchTiledQ4K(
            ReadOnlySpan<float> input,
            int rows,
            Q4KWeight w,
            ReadOnlySpan<float> bias,
            Span<float> output,
            Span<sbyte> quants,
            Span<float> scales,
            Span<short> bsums,
            bool preQuantized)
        {
            var inputSize = w.InputSize;
            var outputSize = w.OutputSize;
            var spr = w.SuperBlocksPerRow;
            var bsumsPerRow = spr * Q4KDotKernel.GroupsPerSuperBlock;

            // Q8_K activation quantization — column-contiguous (column c == row c owns inputSize quants).
            if (!preQuantized)
            {
                for (var n = 0; n < rows; n++)
                {
                    Q4KDotKernel.QuantizeActivationQ8K(
                        input.Slice(n * inputSize, inputSize),
                        quants.Slice(n * inputSize, inputSize),
                        scales.Slice(n * spr, spr),
                        bsums.Slice(n * bsumsPerRow, bsumsPerRow));
                }
            }

            var repacked = w.EnsureRepacked();

            var cores = Environment.ProcessorCount;
            var nr = ResolveTileCols(rows, cores, Q4KGemvKernel.MaxTileCols);
            var tiles = (rows + nr - 1) / nr;

            // One decode of the F16 scales for the whole projection, reused by every column tile. Skipped when
            // there is only one tile, where hoisting would just move the same work.
            var scaleCount = UsePrecomputedScales && tiles > 1
                ? (outputSize / 8) * spr * Q4KGemvKernel.DecodedScalesPerBlock
                : 0;

            using var decodedScales = new PooledBuffer<float>(scaleCount, clearMemory: false);

            if (scaleCount > 0)
            {
                Q4KGemvKernel.DecodeBlockScales(
                    repacked, outputSize, inputSize, decodedScales.Span.Slice(0, scaleCount));
            }

            fixed (byte* rp = repacked)
            fixed (float* dsc = decodedScales.Span.Slice(0, scaleCount))
            fixed (sbyte* q = quants)
            fixed (float* sc = scales)
            fixed (short* bs = bsums)
            fixed (float* o = output)
            fixed (float* bi = bias) // null when the projection has no bias
            {
                var ctx = new TiledContext
                {
                    Repacked = rp,
                    RepackedLength = repacked.Length,
                    Quants = q,
                    Scales = sc,
                    Bsums = bs,
                    Output = o,
                    Bias = bi,
                    BiasLength = bias.Length,
                    InputSize = inputSize,
                    OutputSize = outputSize,
                    Spr = spr,
                    BsumsPerRow = bsumsPerRow,
                    Nr = nr,
                    Rows = rows,
                    Tiles = tiles,
                    DecodedScales = dsc,
                    DecodedScalesLength = scaleCount,
                    Avx512 = UseAvx512PrefillQ4K && !DisableRepackedKernelsForParity,
                };

                if (!UseOutputBlocking)
                {
                    OverfitParallel.For(0, tiles, &TiledChunk, &ctx);
                    return;
                }

                var totalGroups = outputSize / 8;
                ctx.GroupsPerBand = ResolveGroupsPerBand(
                    totalGroups, spr, Q4KRepack.BlockKx8Bytes, cores);

                var bands = (totalGroups + ctx.GroupsPerBand - 1) / ctx.GroupsPerBand;

                OverfitParallel.For(0, bands, &TiledBandChunk, &ctx);
            }
        }

        // Register-tiled Q6_K prefill GEMM: quantize all rows to Q8_K, then run GemmTiled over row-tiles of
        // NR columns in parallel. Mirrors DispatchTiledQ4K, minus the bsums — the Q6_K kernel folds the −32
        // bias correction into the maddubs instead of using the activation group sums.
        private static unsafe void DispatchTiledQ6K(
            ReadOnlySpan<float> input,
            int rows,
            Q6KWeight w,
            Span<float> output,
            Span<sbyte> quants,
            Span<float> scales,
            Span<short> bsums,
            bool preQuantized)
        {
            var inputSize = w.InputSize;
            var outputSize = w.OutputSize;
            var spr = w.SuperBlocksPerRow;
            var bsumsPerRow = spr * Q6KDotKernel.GroupsPerSuperBlock;

            if (!preQuantized)
            {
                for (var n = 0; n < rows; n++)
                {
                    Q6KDotKernel.QuantizeActivationQ8K(
                        input.Slice(n * inputSize, inputSize),
                        quants.Slice(n * inputSize, inputSize),
                        scales.Slice(n * spr, spr),
                        bsums.Slice(n * bsumsPerRow, bsumsPerRow));
                }
            }

            var repacked = w.EnsureRepacked();

            var cores = Environment.ProcessorCount;
            var nr = ResolveTileCols(rows, cores, Q6KGemvKernel.MaxTileCols);
            var tiles = (rows + nr - 1) / nr;

            // Same hoist as the Q4_K path: widen the F16 row scales once per projection rather than once per
            // column tile. Q6_K has no dmin, so this is half the scratch.
            var scaleCount = UsePrecomputedScales && tiles > 1
                ? (outputSize / 8) * spr * Q6KGemvKernel.DecodedScalesPerBlock
                : 0;

            using var decodedScales = new PooledBuffer<float>(scaleCount, clearMemory: false);

            if (scaleCount > 0)
            {
                Q6KGemvKernel.DecodeBlockScales(
                    repacked, outputSize, inputSize, decodedScales.Span.Slice(0, scaleCount));
            }

            fixed (byte* rp = repacked)
            fixed (float* dsc = decodedScales.Span.Slice(0, scaleCount))
            fixed (sbyte* q = quants)
            fixed (float* sc = scales)
            fixed (float* o = output)
            {
                var ctx = new TiledQ6KContext
                {
                    Repacked = rp,
                    RepackedLength = repacked.Length,
                    Quants = q,
                    Scales = sc,
                    Output = o,
                    InputSize = inputSize,
                    OutputSize = outputSize,
                    Spr = spr,
                    Nr = nr,
                    Rows = rows,
                    DecodedScales = dsc,
                    DecodedScalesLength = scaleCount,
                    Avx512 = UseAvx512PrefillQ6K && !DisableRepackedKernelsForParity,
                };
                OverfitParallel.For(0, tiles, &TiledQ6KChunk, &ctx);
            }
        }

        private unsafe struct TiledQ6KContext
        {
            public byte* Repacked;
            public int RepackedLength;
            public sbyte* Quants;
            public float* Scales;
            public float* Output;
            public int InputSize;
            public int OutputSize;
            public int Spr;
            public int Nr;
            public int Rows;

            /// <summary>F16 row scales widened once for the whole projection; null when decoded inline.</summary>
            public float* DecodedScales;

            /// <summary>Length of <see cref="DecodedScales"/>; 0 when decoded inline.</summary>
            public int DecodedScalesLength;

            /// <summary>Route through the two-columns-per-instruction AVX-512 kernel.</summary>
            public bool Avx512;
        }

        private static unsafe void TiledQ6KChunk(int start, int end, void* context)
        {
            ref var c = ref Unsafe.AsRef<TiledQ6KContext>(context);
            for (var t = start; t < end; t++)
            {
                var s = t * c.Nr;
                var cols = Math.Min(c.Nr, c.Rows - s);
                var weights = new ReadOnlySpan<byte>(c.Repacked, c.RepackedLength);
                var quants = new ReadOnlySpan<sbyte>(c.Quants + (long)s * c.InputSize, cols * c.InputSize);
                var scales = new ReadOnlySpan<float>(c.Scales + (long)s * c.Spr, cols * c.Spr);
                var dst = new Span<float>(c.Output + (long)s * c.OutputSize, cols * c.OutputSize);
                var decoded = new ReadOnlySpan<float>(c.DecodedScales, c.DecodedScalesLength);

                if (c.Avx512)
                {
                    Q6KGemvKernel.GemmTiled512(
                        weights, c.OutputSize, c.InputSize, cols, quants, scales, dst, decoded);
                    continue;
                }

                Q6KGemvKernel.GemmTiled(
                    weights, c.OutputSize, c.InputSize, cols, quants, scales, dst, decoded);
            }
        }

        private unsafe struct TiledContext
        {
            public byte* Repacked;
            public int RepackedLength;
            public sbyte* Quants;
            public float* Scales;
            public short* Bsums;
            public float* Output;
            public float* Bias;
            public int BiasLength;
            public int InputSize;
            public int OutputSize;
            public int Spr;
            public int BsumsPerRow;
            public int Nr;
            public int Rows;

            /// <summary>Column tiles per projection — the inner loop when banding over output rows.</summary>
            public int Tiles;

            /// <summary>Output groups per band; see <see cref="ResolveGroupsPerBand"/>.</summary>
            public int GroupsPerBand;

            /// <summary>F16 scales widened once for the whole projection; null when decoded inline.</summary>
            public float* DecodedScales;

            /// <summary>Length of <see cref="DecodedScales"/>; 0 when decoded inline.</summary>
            public int DecodedScalesLength;

            /// <summary>Route through the two-columns-per-instruction AVX-512 kernel.</summary>
            public bool Avx512;
        }

        private static unsafe void TiledChunk(int start, int end, void* context)
        {
            ref var c = ref Unsafe.AsRef<TiledContext>(context);
            for (var t = start; t < end; t++)
            {
                var s = t * c.Nr;
                var cols = Math.Min(c.Nr, c.Rows - s);
                var weights = new ReadOnlySpan<byte>(c.Repacked, c.RepackedLength);
                var quants = new ReadOnlySpan<sbyte>(c.Quants + (long)s * c.InputSize, cols * c.InputSize);
                var scales = new ReadOnlySpan<float>(c.Scales + (long)s * c.Spr, cols * c.Spr);
                var sums = new ReadOnlySpan<short>(c.Bsums + (long)s * c.BsumsPerRow, cols * c.BsumsPerRow);
                var dst = new Span<float>(c.Output + (long)s * c.OutputSize, cols * c.OutputSize);
                var bias = new ReadOnlySpan<float>(c.Bias, c.BiasLength);
                var decoded = new ReadOnlySpan<float>(c.DecodedScales, c.DecodedScalesLength);

                if (c.Avx512)
                {
                    Q4KGemvKernel.GemmTiled512(
                        weights, c.OutputSize, c.InputSize, cols, quants, scales, sums, dst, bias, decoded);
                    continue;
                }

                Q4KGemvKernel.GemmTiled(
                    weights, c.OutputSize, c.InputSize, cols, quants, scales, sums, dst, bias, 0, 0, decoded);
            }
        }

        // One band of output groups, swept by every column tile in turn. The band's weights are read from L3
        // once and then re-read from this worker's own L2 for each remaining tile — the whole point of the
        // structure. Bands are disjoint in both weights (read-only) and output rows, so no worker writes where
        // another reads.
        private static unsafe void TiledBandChunk(int start, int end, void* context)
        {
            ref var c = ref Unsafe.AsRef<TiledContext>(context);
            var totalGroups = c.OutputSize / 8;

            for (var band = start; band < end; band++)
            {
                var groupStart = band * c.GroupsPerBand;
                var groupCount = Math.Min(c.GroupsPerBand, totalGroups - groupStart);

                for (var t = 0; t < c.Tiles; t++)
                {
                    var s = t * c.Nr;
                    var cols = Math.Min(c.Nr, c.Rows - s);

                    Q4KGemvKernel.GemmTiled(
                        new ReadOnlySpan<byte>(c.Repacked, c.RepackedLength),
                        c.OutputSize,
                        c.InputSize,
                        cols,
                        new ReadOnlySpan<sbyte>(c.Quants + (long)s * c.InputSize, cols * c.InputSize),
                        new ReadOnlySpan<float>(c.Scales + (long)s * c.Spr, cols * c.Spr),
                        new ReadOnlySpan<short>(c.Bsums + (long)s * c.BsumsPerRow, cols * c.BsumsPerRow),
                        new Span<float>(c.Output + (long)s * c.OutputSize, cols * c.OutputSize),
                        new ReadOnlySpan<float>(c.Bias, c.BiasLength),
                        groupStart,
                        groupCount,
                        new ReadOnlySpan<float>(c.DecodedScales, c.DecodedScalesLength));
                }
            }
        }
    }
}
