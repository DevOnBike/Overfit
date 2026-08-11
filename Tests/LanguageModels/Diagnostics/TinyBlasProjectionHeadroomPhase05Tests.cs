// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Diagnostics;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using DevOnBike.Overfit.LanguageModels.Loading;
using DevOnBike.Overfit.LanguageModels.Runtime;

namespace DevOnBike.Overfit.Tests.LanguageModels.Diagnostics
{
    /// <summary>
    /// Phase 0.5 for the tinyBLAS register-tiling investigation — the direct GO/STOP decider Phase 0 deferred to.
    /// Isolates ONE FFN projection (dModel=2048 → dFF=11008, Q4_K) and measures the achieved GFLOPS of the current
    /// weight-stationary batched kernel (<see cref="Q4KDotKernel.ProjectBatchedWeightStationary"/>) at rows ∈
    /// {8,32,128,512}, expressed as a fraction of the box's measured VPMADDUBSW ceiling (a tight in-L1 int8-madd
    /// loop across all cores — the contraction's arithmetic peak with unpack/scale overhead removed).
    ///
    ///   kernel at &lt;40% of ceiling ⇒ compute-INEFFICIENT (unpack/accumulator overhead dominates) ⇒ a register-
    ///                                  tiled MR×NR kernel has real room ⇒ GO on Phases 1–4.
    ///   kernel at &gt;70% of ceiling ⇒ already near arithmetic peak ⇒ tiling adds little ⇒ STOP (document negative).
    ///
    /// [LongFact] — synthetic weights (no model file needed), AVX2-only kernel. Best-of-N (min time). Uses a random
    /// but fixed weight; the kernel is data-independent in cost, so synthetic is representative.
    /// </summary>
    public sealed class TinyBlasProjectionHeadroomPhase05Tests
    {
        private const int InputSize = 2048;    // dModel
        private const int OutputSize = 11008;  // dFF (Qwen-3B FFN)
        private const int Repeats = 5;

        private static readonly int[] RowCounts = [8, 32, 128, 512];

        private readonly ITestOutputHelper _out;

        public TinyBlasProjectionHeadroomPhase05Tests(ITestOutputHelper output) => _out = output;

        [FixtureFact(TestFixture.Avx2AndFma, "1s")]
        public void Phase05_ProjectionGflops_VsCeiling()
        {

            // ── build a real Q4_K weight [OutputSize × InputSize] from random F32 ── (22.5M elems, fits int)
            var f32 = new float[OutputSize * InputSize];
            FillDeterministic(f32);
            var q4kBytes = GgmlQuant.QuantizeQ4_K(f32, InputSize, OutputSize);
            var weight = new Q4KWeight(q4kBytes, InputSize, OutputSize);
            var spr = weight.SuperBlocksPerRow;
            var bsumsPerRow = spr * Q4KDotKernel.GroupsPerSuperBlock;

            _out.WriteLine($"projection {InputSize} → {OutputSize}  Q4_K  ({weight.ByteCount / (1024 * 1024)} MB)   cores={Environment.ProcessorCount}");
            _out.WriteLine("");

            // ── measured arithmetic ceiling: tight VPMADDUBSW across all cores (unpack-free) ──
            var ceilingGflops = MeasureMaddCeiling();
            _out.WriteLine($"VPMADDUBSW ceiling (in-L1, all cores): {ceilingGflops:F0} GFLOP-equiv/s");
            _out.WriteLine("");

            _out.WriteLine("rows |   ms  |   GFLOPS |  % ceiling");
            _out.WriteLine("-----+-------+----------+-----------");

            var best = 0.0;
            foreach (var rows in RowCounts)
            {
                var input = new float[(long)rows * InputSize];
                FillDeterministic(input);
                var output = new float[(long)rows * OutputSize];
                var aq = new sbyte[(long)rows * InputSize];
                var asc = new float[(long)rows * spr];
                var ab = new short[(long)rows * bsumsPerRow];

                // warm (JIT + page-in)
                Q4KDotKernel.ProjectBatchedWeightStationary(input, rows, weight, [], output, aq, asc, ab);

                var bestMs = double.MaxValue;
                for (var r = 0; r < Repeats; r++)
                {
                    var sw = Stopwatch.StartNew();
                    Q4KDotKernel.ProjectBatchedWeightStationary(input, rows, weight, [], output, aq, asc, ab);
                    sw.Stop();
                    bestMs = Math.Min(bestMs, sw.Elapsed.TotalMilliseconds);
                }

                var flops = 2.0 * InputSize * OutputSize * rows;
                var gflops = flops / (bestMs / 1000.0) / 1e9;
                var pct = gflops / ceilingGflops * 100.0;
                best = Math.Max(best, gflops);
                _out.WriteLine($"{rows,4} | {bestMs,5:F2} | {gflops,8:F1} | {pct,9:F1}%");
            }

            var bestPct = best / ceilingGflops * 100.0;
            _out.WriteLine("");
            var verdict = bestPct switch
            {
                < 40 => $"GO — best {best:F0} GFLOPS = {bestPct:F0}% of ceiling; compute-inefficient, a register-tiled MR×NR kernel has real room.",
                > 70 => $"STOP — best {best:F0} GFLOPS = {bestPct:F0}% of ceiling; already near arithmetic peak, tiling adds little. Document negative.",
                _ => $"MARGINAL — best {best:F0} GFLOPS = {bestPct:F0}% of ceiling; borderline. Weigh effort vs the modest upside.",
            };
            _out.WriteLine($"VERDICT: {verdict}");
        }

        /// <summary>
        /// Tight VPMADDUBSW loop over in-L1 buffers on every core — the box's int8-contraction arithmetic ceiling
        /// with the 4-bit unpack / scale-decode overhead removed. Eight independent accumulators to saturate the
        /// integer SIMD ports; a data dependency chain so the JIT can't elide it. Counts each 256-bit madd as
        /// 32 MACs (= 64 FLOP-equiv), matching how the projection's GFLOPS is counted (2·in·out·rows).
        /// </summary>
        private static double MeasureMaddCeiling()
        {
            var cores = Environment.ProcessorCount;
            const long iterationsPerThread = 40_000_000L; // ~8 madds each → tune for ~1s
            const int maddsPerIteration = 8;

            var bestSeconds = double.MaxValue;
            for (var rep = 0; rep < 3; rep++)
            {
                var sw = Stopwatch.StartNew();
                Parallel.For(0, cores, _ => MaddChain(iterationsPerThread));
                sw.Stop();
                bestSeconds = Math.Min(bestSeconds, sw.Elapsed.TotalSeconds);
            }

            var totalMadds = (double)cores * iterationsPerThread * maddsPerIteration;
            var flops = totalMadds * 32.0 * 2.0; // 32 MAC/madd × 2 FLOP/MAC
            return flops / bestSeconds / 1e9;
        }

        private static int MaddChain(long iterations)
        {
            var a = Vector256.Create((byte)3);
            var b = Vector256.Create((sbyte)5);
            var c0 = Vector256<short>.Zero;
            var c1 = Vector256<short>.Zero;
            var c2 = Vector256<short>.Zero;
            var c3 = Vector256<short>.Zero;
            var c4 = Vector256<short>.Zero;
            var c5 = Vector256<short>.Zero;
            var c6 = Vector256<short>.Zero;
            var c7 = Vector256<short>.Zero;

            for (long i = 0; i < iterations; i++)
            {
                c0 = Avx2.Add(c0, Avx2.MultiplyAddAdjacent(a, b));
                c1 = Avx2.Add(c1, Avx2.MultiplyAddAdjacent(a, b));
                c2 = Avx2.Add(c2, Avx2.MultiplyAddAdjacent(a, b));
                c3 = Avx2.Add(c3, Avx2.MultiplyAddAdjacent(a, b));
                c4 = Avx2.Add(c4, Avx2.MultiplyAddAdjacent(a, b));
                c5 = Avx2.Add(c5, Avx2.MultiplyAddAdjacent(a, b));
                c6 = Avx2.Add(c6, Avx2.MultiplyAddAdjacent(a, b));
                c7 = Avx2.Add(c7, Avx2.MultiplyAddAdjacent(a, b));
                // keep the operands live / dependent so the loop isn't hoisted
                a = Avx2.Xor(a, Avx2.ShiftRightLogical(c0.AsByte().AsUInt16(), 15).AsByte());
            }

            var s = Avx2.Add(Avx2.Add(Avx2.Add(c0, c1), Avx2.Add(c2, c3)), Avx2.Add(Avx2.Add(c4, c5), Avx2.Add(c6, c7)));
            return Vector256.Sum(s);
        }

        private static void FillDeterministic(float[] data)
        {
            var seed = 0x12345678u;
            for (var i = 0; i < data.Length; i++)
            {
                seed = seed * 1664525u + 1013904223u;
                data[i] = ((seed & 0x00FFFFFF) / 16777216f) * 2f - 1f;
            }
        }
    }
}
