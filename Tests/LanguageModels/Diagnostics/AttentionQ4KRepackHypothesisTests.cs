// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Diagnostics;
using DevOnBike.Overfit.Intrinsics;
using DevOnBike.Overfit.LanguageModels.Loading;
using DevOnBike.Overfit.LanguageModels.Runtime;
using DevOnBike.Overfit.Runtime;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests.LanguageModels.Diagnostics
{
    /// <summary>
    /// HYPOTHESIS PROBE for the "whole-matrix Q4_K attention" perf lever (ROADMAP). Before committing to
    /// the big loader+MHA refactor, this isolates the core claim: that projecting the attention Q (or O)
    /// matrix as <b>one whole-matrix Q4_K repacked GEMV</b> is faster than the current
    /// <b>per-head Q8</b> projection — because Q4_K reads ~half the bytes (0.56 vs 1 B/weight) and the
    /// 8×8 repacked kernel avoids the per-row horizontal reduction.
    ///
    /// Both paths quantize the activation once and compute the same 2048 output dot-products for a
    /// 2048×2048 projection (Qwen-3B Q/O dims); the only differences are the weight format and the
    /// kernel. Single-threaded, best-of-N — isolates per-projection cost from dispatch noise. Prints
    /// ns / projection, effective GB/s and the speedup, the numbers that decide go/no-go on the refactor.
    /// </summary>
    public sealed class AttentionQ4KRepackHypothesisTests
    {
        private readonly ITestOutputHelper _out;

        public AttentionQ4KRepackHypothesisTests(ITestOutputHelper output) => _out = output;

        [LongFact]
        public unsafe void Whole_Q4K_Gemv_vs_PerHead_Q8_Projection()
        {
            if (!CpuFeatures.HasAvx2)
            {
                _out.WriteLine("AVX2 not available — the Q4_K repacked GEMV is AVX2-only; skipping the measurement.");
                return;
            }

            // Qwen-3B attention projection dims: Q and O are 2048×2048 (the heavy ones; K/V are narrow under GQA).
            const int inputSize = 2048;
            const int outputSize = 2048;

            // Deterministic pseudo-random F32 weight + activation (seeded; no Random ban issues — test code).
            var weight = new float[outputSize * inputSize];
            FillDeterministic(weight, seed: 1234);
            var hidden = new float[inputSize];
            FillDeterministic(hidden, seed: 5678);

            // ── Current path: per-head Q8 (one Q8 weight; Project quantizes the activation once, then 2048 row-dots) ──
            var q8 = Q8Weight.QuantizeRows(weight, outputSize, inputSize);
            var q8InputQuants = new sbyte[inputSize];
            var q8InputScales = new float[inputSize / Q8DotKernel.BlockSize];
            var outQ8 = new float[outputSize];

            // ── Proposed path: whole-matrix Q4_K, repacked to block_q4_Kx8 for the 8×8 GEMV ──
            var q4kBytes = GgmlQuant.QuantizeQ4_K(weight, inputSize, outputSize);
            var q4k = new Q4KWeight(q4kBytes, inputSize, outputSize);
            var repacked = q4k.EnsureRepacked().ToArray();
            var actQuants = new sbyte[inputSize];
            var actScales = new float[(inputSize + Q4KDotKernel.SuperBlockElements - 1) / Q4KDotKernel.SuperBlockElements];
            var actBsums = new short[(inputSize + Q4KDotKernel.GroupSize - 1) / Q4KDotKernel.GroupSize];
            var outQ4k = new float[outputSize];

            // The Q4_K GEMV consumes a Q8_K-quantized activation, prepared once (the decode does the same).
            Q4KDotKernel.QuantizeActivationQ8K(hidden, actQuants, actScales, actBsums);

            const int warmup = 50;
            const int iters = 600;

            var q8Ns = BestOf(warmup, iters, () =>
                Q8DotKernel.Project(hidden, q8, [], outQ8, q8InputQuants, q8InputScales));

            var q4kNs = BestOf(warmup, iters, () =>
                Q4KGemvKernel.Gemv(repacked, outputSize, inputSize, actQuants, actScales, actBsums, outQ4k));

            // Bytes read per projection (the bandwidth that decode, being memory-bound, pays each token).
            var q8Bytes = (long)q8.Quants.Length + (long)q8.Scales.Length * sizeof(float);
            var q4kBytesRead = (long)repacked.Length;

            var q8GbS = q8Bytes / (q8Ns * 1e-9) / 1e9;
            var q4kGbS = q4kBytesRead / (q4kNs * 1e-9) / 1e9;

            _out.WriteLine($"projection {outputSize}×{inputSize} (Qwen-3B Q/O), single-thread, best of {iters}");
            _out.WriteLine(string.Empty);
            _out.WriteLine($"  per-head Q8   : {q8Ns,8:F0} ns   weight {q8Bytes / 1024.0 / 1024.0,5:F2} MB   {q8GbS,5:F1} GB/s");
            _out.WriteLine($"  whole  Q4_K   : {q4kNs,8:F0} ns   weight {q4kBytesRead / 1024.0 / 1024.0,5:F2} MB   {q4kGbS,5:F1} GB/s");
            _out.WriteLine(string.Empty);
            _out.WriteLine($"  speedup (Q8/Q4_K time)  : {q8Ns / q4kNs:F2}×");
            _out.WriteLine($"  bytes ratio (Q4_K/Q8)   : {(double)q4kBytesRead / q8Bytes:F2}");
            _out.WriteLine($"  est. saving / projection: {(q8Ns - q4kNs) / 1000.0:F2} µs");

            // ── Parallel: the decode-relevant case. The lever's premise is that PARALLEL decode is
            //    memory-bandwidth-bound (shared DRAM), so reading half the bytes should win even though
            //    the Q4_K kernel is ~2× compute-heavier per byte single-thread. Warm the spin pool first. ──
            for (var i = 0; i < 200; i++)
            {
                Q8DotKernel.ProjectParallel(hidden, q8, [], outQ8, q8InputQuants, q8InputScales);
                Q4KGemvKernel.GemvParallel(repacked, outputSize, inputSize, actQuants, actScales, actBsums, outQ4k);
            }

            var q8ParNs = BestOf(warmup, iters, () =>
                Q8DotKernel.ProjectParallel(hidden, q8, [], outQ8, q8InputQuants, q8InputScales));
            var q4kParNs = BestOf(warmup, iters, () =>
                Q4KGemvKernel.GemvParallel(repacked, outputSize, inputSize, actQuants, actScales, actBsums, outQ4k));

            _out.WriteLine(string.Empty);
            _out.WriteLine($"  PARALLEL ({OverfitParallel.WorkerCount} workers) — the decode-relevant case:");
            _out.WriteLine($"  per-head Q8 ‖ : {q8ParNs,8:F0} ns   {q8Bytes / (q8ParNs * 1e-9) / 1e9,5:F1} GB/s");
            _out.WriteLine($"  whole  Q4_K ‖ : {q4kParNs,8:F0} ns   {q4kBytesRead / (q4kParNs * 1e-9) / 1e9,5:F1} GB/s");
            _out.WriteLine($"  speedup ‖     : {q8ParNs / q4kParNs:F2}×   ← vs Q8 (NOT today's path)");

            // ── PER-HEAD Q4_K — TODAY'S ACTUAL decode path. The whole matrix is split into nHeads head-row
            //    groups (the loader keeps per-head Q4KWeight for a Q4_K_M file), each projected separately via
            //    ProjectPreQuantized over the SHARED Q8_K activation, fanned across heads by ForDecode exactly
            //    like CachedMultiHeadAttention. This is the REAL baseline the refactor must beat: same bytes as
            //    whole-Q4_K, so the only levers are the repacked 8×8 kernel (no per-row horizontal reduction) +
            //    finer parallel granularity (outputSize/8 = 256 groups vs nHeads = 16 chunks). ──
            const int nHeads = 16, headDim = 128; // Qwen-3B Q/O: 2048 = 16 × 128
            var headBytes = q4kBytes.Length / nHeads;
            var heads = new Q4KWeight[nHeads];
            for (var h = 0; h < nHeads; h++)
            {
                var slice = new byte[headBytes];
                Array.Copy(q4kBytes, (long)h * headBytes, slice, 0, headBytes);
                heads[h] = new Q4KWeight(slice, inputSize, headDim);
            }
            var perHeadOut = new float[outputSize];

            var perHeadNs = BestOf(warmup, iters, () =>
            {
                for (var h = 0; h < nHeads; h++)
                {
                    Q4KDotKernel.ProjectPreQuantized(
                        heads[h], [], perHeadOut.AsSpan(h * headDim, headDim), actQuants, actScales, actBsums);
                }
            });

            // Parallel per-head — mirror decode's head fan-out. A function pointer cannot be captured in a
            // lambda, so this times ForDecode directly rather than through BestOf.
            _phHeads = heads;
            _phActQuants = actQuants;
            _phActScales = actScales;
            _phActBsums = actBsums;
            _phOut = perHeadOut;
            _phHeadDim = headDim;
            for (var i = 0; i < warmup + 200; i++)
            {
                OverfitParallel.ForDecode(0, nHeads, &PerHeadChunk, null);
            }
            var bestParTicks = long.MaxValue;
            for (var i = 0; i < iters; i++)
            {
                var t = Stopwatch.GetTimestamp();
                OverfitParallel.ForDecode(0, nHeads, &PerHeadChunk, null);
                var dt = Stopwatch.GetTimestamp() - t;
                if (dt < bestParTicks)
                {
                    bestParTicks = dt;
                }
            }
            var perHeadParNs = bestParTicks * (1_000_000_000.0 / Stopwatch.Frequency);

            _out.WriteLine(string.Empty);
            _out.WriteLine("  ── vs TODAY'S per-head Q4_K (the real refactor baseline) ──");
            _out.WriteLine($"  per-head Q4_K : single {perHeadNs,8:F0} ns   ‖ {perHeadParNs,8:F0} ns");
            _out.WriteLine($"  whole    Q4_K : single {q4kNs,8:F0} ns   ‖ {q4kParNs,8:F0} ns");
            _out.WriteLine($"  REAL speedup single     : {perHeadNs / q4kNs:F2}×");
            _out.WriteLine($"  REAL speedup ‖ (perHd/whole): {perHeadParNs / q4kParNs:F2}×   ← go/no-go vs CURRENT path");

            // Sanity: both produce a finite projection of the right size (not a correctness/parity check —
            // the kernels are parity-tested elsewhere; this probe is purely about timing).
            Assert.Equal(outputSize, outQ8.Length);
            Assert.All(outQ8, v => Assert.True(float.IsFinite(v)));
            Assert.All(outQ4k, v => Assert.True(float.IsFinite(v)));
        }

        // Per-head parallel worker state (test runs one instance, so static is safe) — ForDecode takes a
        // function pointer, which cannot close over locals.
        private static Q4KWeight[] _phHeads = null!;
        private static sbyte[] _phActQuants = null!;
        private static float[] _phActScales = null!;
        private static short[] _phActBsums = null!;
        private static float[] _phOut = null!;
        private static int _phHeadDim;

        private static unsafe void PerHeadChunk(int from, int to, void* _)
        {
            for (var h = from; h < to; h++)
            {
                Q4KDotKernel.ProjectPreQuantized(
                    _phHeads[h], [], _phOut.AsSpan(h * _phHeadDim, _phHeadDim),
                    _phActQuants, _phActScales, _phActBsums);
            }
        }

        private static double BestOf(int warmup, int iters, Action body)
        {
            for (var i = 0; i < warmup; i++)
            {
                body();
            }

            var best = long.MaxValue;
            for (var i = 0; i < iters; i++)
            {
                var t = Stopwatch.GetTimestamp();
                body();
                var dt = Stopwatch.GetTimestamp() - t;
                if (dt < best)
                {
                    best = dt;
                }
            }

            return best * (1_000_000_000.0 / Stopwatch.Frequency);
        }

        private static void FillDeterministic(float[] a, int seed)
        {
            // Cheap xorshift → [-1, 1); deterministic, no allocation, no Random dependency.
            var state = (uint)seed | 1u;
            for (var i = 0; i < a.Length; i++)
            {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                a[i] = (state / (float)uint.MaxValue) * 2f - 1f;
            }
        }
    }
}
