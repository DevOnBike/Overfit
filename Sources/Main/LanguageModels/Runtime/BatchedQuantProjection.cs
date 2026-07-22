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

        public static void Dispatch(
            ReadOnlySpan<float> input,
            int rows,
            in DecodeWeight weight,
            ReadOnlySpan<float> bias,
            Span<float> output,
            int inputSize,
            int outputSize)
        {
            // Resident-format dispatch, classified once so the original first-match order is explicit.
            var kind = weight.IsQ6K ? 0 : weight.IsQ4K ? 1 : weight.IsQuantized ? 2 : 3;

            if (kind == 0)
            {
                var w = weight.Quantized6K;
                var spr = w.SuperBlocksPerRow;
                var groups = rows * spr * Q6KDotKernel.GroupsPerSuperBlock;
                using var qBytes = new PooledBuffer<sbyte>(rows * inputSize, clearMemory: false);
                using var scales = new PooledBuffer<float>(rows * spr, clearMemory: false);
                using var sums = new PooledBuffer<short>(groups, clearMemory: false);
                // Register-tiled Q6_K GEMM over the repacked block_q6_Kx8 layout. Under Q4_K_M half of
                // ffn_down is Q6_K, and a prefill profile put ffn_down at 37.9% of prefill running at
                // 0.61 TFLOP/s — against ffn_gate_up's 1.78 — precisely because Q6_K had only the
                // re-decode-per-row kernel below. No-bias only (GemmTiled applies none); AVX2/FMA required.
                var tiled6 = UseTiledPrefillQ6K && bias.IsEmpty && w.CanRepack
                    && CpuFeatures.HasAvx2 && CpuFeatures.HasFma;

                if (tiled6)
                {
                    DispatchTiledQ6K(
                        input, rows, w, output,
                        qBytes.Span.Slice(0, rows * inputSize), scales.Span.Slice(0, rows * spr),
                        sums.Span.Slice(0, groups));
                }

                if (!tiled6)
                {
                    Q6KDotKernel.ProjectBatched(
                        input, rows, w, bias, output,
                        qBytes.Span.Slice(0, rows * inputSize), scales.Span.Slice(0, rows * spr),
                        sums.Span.Slice(0, groups));
                }
            }
            if (kind == 1)
            {
                var w = weight.Quantized4K;
                var spr = w.SuperBlocksPerRow;
                var groups = rows * spr * Q4KDotKernel.GroupsPerSuperBlock;
                using var qBytes = new PooledBuffer<sbyte>(rows * inputSize, clearMemory: false);
                using var scales = new PooledBuffer<float>(rows * spr, clearMemory: false);
                using var sums = new PooledBuffer<short>(groups, clearMemory: false);

                // Register-tiled GEMM: repacked block_q4_Kx8, decode each super-block once and reuse across a
                // tile of NR columns — measured ~3× vs weight-stationary under parallelism, 1.61× end-to-end
                // prefill. Default-on when the weight is already prepacked (an offline sidecar mmap'd it → zero
                // extra RAM); otherwise opt-in via OVERFIT_TILED_PREFILL since repacking copies the weight.
                // No-bias only (GemmTiled applies none). AVX2/FMA required — the kernel is x86-only, so on ARM
                // (e.g. the Android app) this falls through to the weight-stationary path even if a sidecar
                // mmap'd a prepacked layout (IsPrepacked would otherwise bypass the env flag's AVX2 gate).
                var tiled = (w.IsPrepacked || UseTiledPrefillQ4K) && bias.IsEmpty && w.CanRepack
                    && CpuFeatures.HasAvx2 && CpuFeatures.HasFma;

                if (tiled)
                {
                    DispatchTiledQ4K(
                        input, rows, w, output,
                        qBytes.Span.Slice(0, rows * inputSize), scales.Span.Slice(0, rows * spr),
                        sums.Span.Slice(0, groups));
                }
                // Weight-stationary: decode each Q4_K super-block once and reuse across the row tile (bit-identical
                // to ProjectBatched, measured ~1.3–1.7× on the prefill / speculative-verify batched matmul).
                if (!tiled && UseWeightStationaryQ4K)
                {
                    Q4KDotKernel.ProjectBatchedWeightStationary(
                        input, rows, w, bias, output,
                        qBytes.Span.Slice(0, rows * inputSize), scales.Span.Slice(0, rows * spr),
                        sums.Span.Slice(0, groups));
                }
                if (!tiled && !UseWeightStationaryQ4K)
                {
                    Q4KDotKernel.ProjectBatched(
                        input, rows, w, bias, output,
                        qBytes.Span.Slice(0, rows * inputSize), scales.Span.Slice(0, rows * spr),
                        sums.Span.Slice(0, groups));
                }
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

        // Register-tiled Q4_K prefill GEMM: quantize all rows to Q8_K, then run GemmTiled over row-tiles of NR
        // columns in parallel. NR is chosen so the tile count stays >= cores (an under-filled pool regressed
        // hard in the Phase-3 bench). No-bias only (checked at the call site) — GemmTiled applies no bias.
        private static unsafe void DispatchTiledQ4K(
            ReadOnlySpan<float> input,
            int rows,
            Q4KWeight w,
            Span<float> output,
            Span<sbyte> quants,
            Span<float> scales,
            Span<short> bsums)
        {
            var inputSize = w.InputSize;
            var outputSize = w.OutputSize;
            var spr = w.SuperBlocksPerRow;
            var bsumsPerRow = spr * Q4KDotKernel.GroupsPerSuperBlock;

            // Q8_K activation quantization — column-contiguous (column c == row c owns inputSize quants).
            for (var n = 0; n < rows; n++)
            {
                Q4KDotKernel.QuantizeActivationQ8K(
                    input.Slice(n * inputSize, inputSize),
                    quants.Slice(n * inputSize, inputSize),
                    scales.Slice(n * spr, spr),
                    bsums.Slice(n * bsumsPerRow, bsumsPerRow));
            }

            var repacked = w.EnsureRepacked();

            var cores = Environment.ProcessorCount;
            var nr = rows / 8 >= cores ? 8 : 4;
            if (nr > Q4KGemvKernel.MaxTileCols)
            {
                nr = Q4KGemvKernel.MaxTileCols;
            }
            var tiles = (rows + nr - 1) / nr;

            fixed (byte* rp = repacked)
            fixed (sbyte* q = quants)
            fixed (float* sc = scales)
            fixed (short* bs = bsums)
            fixed (float* o = output)
            {
                var ctx = new TiledContext
                {
                    Repacked = rp,
                    RepackedLength = repacked.Length,
                    Quants = q,
                    Scales = sc,
                    Bsums = bs,
                    Output = o,
                    InputSize = inputSize,
                    OutputSize = outputSize,
                    Spr = spr,
                    BsumsPerRow = bsumsPerRow,
                    Nr = nr,
                    Rows = rows,
                };
                OverfitParallel.For(0, tiles, &TiledChunk, &ctx);
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
            Span<short> bsums)
        {
            var inputSize = w.InputSize;
            var outputSize = w.OutputSize;
            var spr = w.SuperBlocksPerRow;
            var bsumsPerRow = spr * Q6KDotKernel.GroupsPerSuperBlock;

            for (var n = 0; n < rows; n++)
            {
                Q6KDotKernel.QuantizeActivationQ8K(
                    input.Slice(n * inputSize, inputSize),
                    quants.Slice(n * inputSize, inputSize),
                    scales.Slice(n * spr, spr),
                    bsums.Slice(n * bsumsPerRow, bsumsPerRow));
            }

            var repacked = w.EnsureRepacked();

            var cores = Environment.ProcessorCount;
            var nr = rows / 8 >= cores ? 8 : 4;
            if (nr > Q6KGemvKernel.MaxTileCols)
            {
                nr = Q6KGemvKernel.MaxTileCols;
            }
            var tiles = (rows + nr - 1) / nr;

            fixed (byte* rp = repacked)
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
        }

        private static unsafe void TiledQ6KChunk(int start, int end, void* context)
        {
            ref var c = ref Unsafe.AsRef<TiledQ6KContext>(context);
            for (var t = start; t < end; t++)
            {
                var s = t * c.Nr;
                var cols = Math.Min(c.Nr, c.Rows - s);
                Q6KGemvKernel.GemmTiled(
                    new ReadOnlySpan<byte>(c.Repacked, c.RepackedLength),
                    c.OutputSize,
                    c.InputSize,
                    cols,
                    new ReadOnlySpan<sbyte>(c.Quants + (long)s * c.InputSize, cols * c.InputSize),
                    new ReadOnlySpan<float>(c.Scales + (long)s * c.Spr, cols * c.Spr),
                    new Span<float>(c.Output + (long)s * c.OutputSize, cols * c.OutputSize));
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
            public int InputSize;
            public int OutputSize;
            public int Spr;
            public int BsumsPerRow;
            public int Nr;
            public int Rows;
        }

        private static unsafe void TiledChunk(int start, int end, void* context)
        {
            ref var c = ref Unsafe.AsRef<TiledContext>(context);
            for (var t = start; t < end; t++)
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
                    new Span<float>(c.Output + (long)s * c.OutputSize, cols * c.OutputSize));
            }
        }
    }
}
