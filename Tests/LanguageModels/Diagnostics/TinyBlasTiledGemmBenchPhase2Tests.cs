// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Diagnostics;
using System.Runtime.Intrinsics.X86;
using DevOnBike.Overfit.LanguageModels.Loading;
using DevOnBike.Overfit.LanguageModels.Runtime;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests.LanguageModels.Diagnostics
{
    /// <summary>
    /// Phase 2 go/no-go micro-bench for the tinyBLAS register-tiling lever. Isolates the pure tiling thesis —
    /// does decoding each Q4_K weight super-block ONCE across <c>cols</c> (<see cref="Q4KGemvKernel.GemmTiled"/>)
    /// beat decoding it per-column (<c>cols ×</c> <see cref="Q4KGemvKernel.Gemv"/>)? — both SINGLE-THREADED, same
    /// repacked weight, same work. A per-core win here carries into the parallel prefill path × cores.
    ///
    /// Gate: GemmTiled ≥ 1.3× vs cols×Gemv on the Qwen-3B FFN projection ⇒ proceed to register-residency +
    /// integration. If the current (stack-spill, correctness-first) kernel already wins, residency only helps
    /// more; if it loses, that's the signal to lift the accumulators into registers before re-judging.
    /// [LongFact] — synthetic weights, AVX2-only, best-of-N (min).
    /// </summary>
    public sealed class TinyBlasTiledGemmBenchPhase2Tests
    {
        private const int InputSize = 2048;    // dModel
        private const int OutputSize = 11008;  // dFF (Qwen-3B FFN)
        private const int Repeats = 7;

        private static readonly int[] ColCounts = [2, 4, 8];

        private readonly ITestOutputHelper _out;

        public TinyBlasTiledGemmBenchPhase2Tests(ITestOutputHelper output) => _out = output;

        [LongFact("236ms")]
        public void Phase2_TiledVsPerColumnGemv_SingleThread()
        {
            if (!Avx2.IsSupported || !Fma.IsSupported)
            {
                _out.WriteLine("AVX2/FMA not supported — skipping");
                return;
            }

            var nb = InputSize / 256;
            var bsumsPerRow = nb * Q4KDotKernel.GroupsPerSuperBlock;

            var wF32 = new float[OutputSize * InputSize];
            FillDeterministic(wF32, 1u);
            var weight = new Q4KWeight(GgmlQuant.QuantizeQ4_K(wF32, InputSize, OutputSize), InputSize, OutputSize);
            var repacked = weight.EnsureRepacked().ToArray();

            _out.WriteLine($"projection {InputSize} → {OutputSize}  Q4_K  single-thread  (repacked {repacked.Length / (1024 * 1024)} MB)");
            _out.WriteLine("");
            _out.WriteLine("cols |  tiled ms | perCol ms |  tiled GF | perCol GF | speedup");
            _out.WriteLine("-----+-----------+-----------+-----------+-----------+--------");

            var bestSpeedup = 0.0;
            foreach (var cols in ColCounts)
            {
                var inputs = new float[cols * InputSize];
                FillDeterministic(inputs, (uint)(7 + cols));

                var aq = new sbyte[cols * InputSize];
                var asc = new float[cols * nb];
                var ab = new short[cols * bsumsPerRow];
                for (var c = 0; c < cols; c++)
                {
                    Q4KDotKernel.QuantizeActivationQ8K(
                        inputs.AsSpan(c * InputSize, InputSize),
                        aq.AsSpan(c * InputSize, InputSize),
                        asc.AsSpan(c * nb, nb),
                        ab.AsSpan(c * bsumsPerRow, bsumsPerRow));
                }

                var tiledOut = new float[cols * OutputSize];
                var perColOut = new float[OutputSize];

                // warm both
                Q4KGemvKernel.GemmTiled(repacked, OutputSize, InputSize, cols, aq, asc, ab, tiledOut);
                for (var c = 0; c < cols; c++)
                {
                    Q4KGemvKernel.Gemv(repacked, OutputSize, InputSize,
                        aq.AsSpan(c * InputSize, InputSize), asc.AsSpan(c * nb, nb),
                        ab.AsSpan(c * bsumsPerRow, bsumsPerRow), perColOut);
                }

                var tiledMs = double.MaxValue;
                for (var r = 0; r < Repeats; r++)
                {
                    var sw = Stopwatch.StartNew();
                    Q4KGemvKernel.GemmTiled(repacked, OutputSize, InputSize, cols, aq, asc, ab, tiledOut);
                    sw.Stop();
                    tiledMs = Math.Min(tiledMs, sw.Elapsed.TotalMilliseconds);
                }

                var perColMs = double.MaxValue;
                for (var r = 0; r < Repeats; r++)
                {
                    var sw = Stopwatch.StartNew();
                    for (var c = 0; c < cols; c++)
                    {
                        Q4KGemvKernel.Gemv(repacked, OutputSize, InputSize,
                            aq.AsSpan(c * InputSize, InputSize), asc.AsSpan(c * nb, nb),
                            ab.AsSpan(c * bsumsPerRow, bsumsPerRow), perColOut);
                    }
                    sw.Stop();
                    perColMs = Math.Min(perColMs, sw.Elapsed.TotalMilliseconds);
                }

                var flops = 2.0 * InputSize * OutputSize * cols;
                var tiledGf = flops / (tiledMs / 1000.0) / 1e9;
                var perColGf = flops / (perColMs / 1000.0) / 1e9;
                var speedup = perColMs / tiledMs;
                bestSpeedup = Math.Max(bestSpeedup, speedup);
                _out.WriteLine($"{cols,4} | {tiledMs,9:F3} | {perColMs,9:F3} | {tiledGf,9:F1} | {perColGf,9:F1} | {speedup,6:F2}×");
            }

            _out.WriteLine("");
            var verdict = bestSpeedup switch
            {
                >= 1.3 => $"GO — best {bestSpeedup:F2}× per-core over per-column; weight-decode-once tiling pays. Proceed to register-residency + integration.",
                >= 1.05 => $"MARGINAL — best {bestSpeedup:F2}×; the tiling helps but the stack-spill eats most of it. Lift accumulators into registers, then re-judge.",
                _ => $"NEGATIVE (so far) — best {bestSpeedup:F2}×; the correctness-first spill kernel does not win. Register-residency is the deciding experiment before any STOP.",
            };
            _out.WriteLine($"VERDICT: {verdict}");
        }

        private static void FillDeterministic(float[] data, uint seed)
        {
            for (var i = 0; i < data.Length; i++)
            {
                seed = seed * 1664525u + 1013904223u;
                data[i] = ((seed & 0x00FFFFFF) / 16777216f) * 2f - 1f;
            }
        }
    }
}
