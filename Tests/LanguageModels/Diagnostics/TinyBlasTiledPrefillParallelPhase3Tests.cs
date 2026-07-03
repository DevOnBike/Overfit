// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Diagnostics;
using System.Runtime.Intrinsics.X86;
using DevOnBike.Overfit.LanguageModels.Loading;
using DevOnBike.Overfit.LanguageModels.Runtime;
using DevOnBike.Overfit.Runtime;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests.LanguageModels.Diagnostics
{
    /// <summary>
    /// Phase 3 ship/no-ship micro-bench for the tinyBLAS lever: the REAL comparison the single-thread Phase 2
    /// deferred — parallel register-tiled <see cref="Q4KGemvKernel.GemmTiled"/> (over row-tiles of NR columns,
    /// via <see cref="OverfitParallel"/>) vs the shipped incumbent <see cref="Q4KDotKernel.ProjectBatchedWeightStationary"/>
    /// (parallel over output columns). Same FFN projection (2048 → 11008 Q4_K), same rows, both multi-threaded.
    /// This isolates the repacked-layout edge under real prefill parallelism — a win here is the go signal to
    /// wire the kernel into the dispatcher behind OVERFIT_TILED_PREFILL. Also checks the two paths agree
    /// (cos ≈ 1; not bit-identical — the repacked GEMV reassociates the reduction). [LongFact], best-of-N.
    /// </summary>
    public sealed class TinyBlasTiledPrefillParallelPhase3Tests
    {
        private const int InputSize = 2048;
        private const int OutputSize = 11008;
        private const int Repeats = 7;

        private static readonly int[] RowCounts = [128, 512];
        private static readonly int[] TileWidths = [4, 8];

        private readonly ITestOutputHelper _out;

        public TinyBlasTiledPrefillParallelPhase3Tests(ITestOutputHelper output) => _out = output;

        [Fact]
        public void Phase3_ParallelTiled_VsIncumbent()
        {
            if (!Avx2.IsSupported || !Fma.IsSupported)
            {
                _out.WriteLine("AVX2/FMA not supported — skipping");
                return;
            }

            var spr = InputSize / 256;
            var bsumsPerRow = spr * Q4KDotKernel.GroupsPerSuperBlock;

            var wF32 = new float[OutputSize * InputSize];
            Fill(wF32, 1u);
            var weight = new Q4KWeight(GgmlQuant.QuantizeQ4_K(wF32, InputSize, OutputSize), InputSize, OutputSize);
            var repacked = weight.EnsureRepacked().ToArray();

            _out.WriteLine($"projection {InputSize} → {OutputSize}  Q4_K  parallel  cores={Environment.ProcessorCount}");
            _out.WriteLine("");
            _out.WriteLine("rows | NR | tiled ms | incumb ms | tiled GF | incumb GF | speedup | max|Δ|");
            _out.WriteLine("-----+----+----------+-----------+----------+-----------+---------+-------");

            var best = 0.0;
            foreach (var rows in RowCounts)
            {
                var input = new float[rows * InputSize];
                Fill(input, (uint)(7 + rows));

                // Pre-quantize once for the tiled path (the incumbent quantizes internally; quant is O(rows·in),
                // negligible vs the O(rows·in·out) matmul).
                var aq = new sbyte[rows * InputSize];
                var asc = new float[rows * spr];
                var ab = new short[rows * bsumsPerRow];
                for (var n = 0; n < rows; n++)
                {
                    Q4KDotKernel.QuantizeActivationQ8K(
                        input.AsSpan(n * InputSize, InputSize),
                        aq.AsSpan(n * InputSize, InputSize),
                        asc.AsSpan(n * spr, spr),
                        ab.AsSpan(n * bsumsPerRow, bsumsPerRow));
                }

                // Incumbent: its own quant scratch.
                var incOut = new float[rows * OutputSize];
                var incAq = new sbyte[rows * InputSize];
                var incAsc = new float[rows * spr];
                var incAb = new short[rows * bsumsPerRow];
                void Incumbent() => Q4KDotKernel.ProjectBatchedWeightStationary(
                    input, rows, weight, [], incOut, incAq, incAsc, incAb);

                foreach (var nr in TileWidths)
                {
                    var tiledOut = new float[rows * OutputSize];
                    var tiles = (rows + nr - 1) / nr;
                    void Tiled() => OverfitParallel.For(0, tiles, t =>
                    {
                        var start = t * nr;
                        var cols = Math.Min(nr, rows - start);
                        Q4KGemvKernel.GemmTiled(
                            repacked, OutputSize, InputSize, cols,
                            aq.AsSpan(start * InputSize, cols * InputSize),
                            asc.AsSpan(start * spr, cols * spr),
                            ab.AsSpan(start * bsumsPerRow, cols * bsumsPerRow),
                            tiledOut.AsSpan(start * OutputSize, cols * OutputSize));
                    });

                    // warm both
                    Tiled();
                    Incumbent();

                    var tiledMs = BestMs(Tiled);
                    var incMs = BestMs(Incumbent);

                    // agreement (repacked GEMV reassociates → cos≈1, not bit-exact)
                    var maxAbs = 0.0;
                    for (var i = 0; i < tiledOut.Length; i++)
                    {
                        maxAbs = Math.Max(maxAbs, Math.Abs(tiledOut[i] - incOut[i]));
                    }

                    var flops = 2.0 * InputSize * OutputSize * rows;
                    var tiledGf = flops / (tiledMs / 1000.0) / 1e9;
                    var incGf = flops / (incMs / 1000.0) / 1e9;
                    var speedup = incMs / tiledMs;
                    best = Math.Max(best, speedup);
                    _out.WriteLine(
                        $"{rows,4} | {nr,2} | {tiledMs,8:F3} | {incMs,9:F3} | {tiledGf,8:F0} | {incGf,9:F0} | {speedup,6:F2}× | {maxAbs,6:F3}");
                }
            }

            _out.WriteLine("");
            var verdict = best switch
            {
                >= 1.3 => $"GO — best {best:F2}× over incumbent under real parallelism; wire behind OVERFIT_TILED_PREFILL + measure TTFT.",
                >= 1.05 => $"MARGINAL — best {best:F2}×; register-residency (Phase 2b) may be needed to justify the integration + RAM cost of repack.",
                _ => $"STOP — best {best:F2}×; the repacked-tiled kernel does not beat the incumbent in parallel. Document negative; incumbent stays.",
            };
            _out.WriteLine($"VERDICT: {verdict}");
        }

        private static double BestMs(Action run)
        {
            var best = double.MaxValue;
            for (var r = 0; r < Repeats; r++)
            {
                var sw = Stopwatch.StartNew();
                run();
                sw.Stop();
                best = Math.Min(best, sw.Elapsed.TotalMilliseconds);
            }
            return best;
        }

        private static void Fill(float[] data, uint seed)
        {
            for (var i = 0; i < data.Length; i++)
            {
                seed = seed * 1664525u + 1013904223u;
                data[i] = ((seed & 0x00FFFFFF) / 16777216f) * 2f - 1f;
            }
        }
    }
}
