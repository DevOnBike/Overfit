// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Diagnostics;
using System.Numerics.Tensors;

namespace DevOnBike.Overfit.AndroidBench
{
    /// <summary>
    /// On-device (ARM) resolution of the FP16-resident spike: is keeping weights as <see cref="Half"/> (½ the
    /// RAM of F32) faster or slower than F32 for a decode GEMV ON THE PHONE? Desktop x86 said F16 is 1.8–5.3×
    /// slower (the widen is pure overhead when F32 is compute-fast and cache-resident), but mobile decode is
    /// memory-bandwidth-bound — Half is half the DRAM bytes, so it could WIN here. This best-of-N micro-bench
    /// settles it; no model needed (pure array math). Logs go to logcat tag "OverfitBench".
    /// </summary>
    public static class Fp16GemvBench
    {
        public static void Run(Action<string> log)
        {
            const int rows = 2048;
            const int cols = 2048;

            var rng = new Random(1);
            var wF32 = new float[rows * cols];
            var wF16 = new Half[rows * cols];
            for (var i = 0; i < wF32.Length; i++)
            {
                var v = (float)(rng.NextDouble() - 0.5);
                wF32[i] = v;
                wF16[i] = (Half)v;
            }

            var x = new float[cols];
            for (var i = 0; i < cols; i++)
            {
                x[i] = (float)(rng.NextDouble() - 0.5);
            }

            var outp = new float[rows];
            var scratch = new float[cols];

            log($"── FP16 GEMV micro-bench {rows}x{cols} (ARM on-device), cores={Environment.ProcessorCount} ──");

            var f32 = Bench("F32", log, () =>
            {
                for (var r = 0; r < rows; r++)
                {
                    outp[r] = TensorPrimitives.Dot(wF32.AsSpan(r * cols, cols), x);
                }
            });

            var f16Scalar = Bench("F16_ScalarWiden", log, () =>
            {
                for (var r = 0; r < rows; r++)
                {
                    var row = wF16.AsSpan(r * cols, cols);
                    var sum = 0f;
                    for (var c = 0; c < cols; c++)
                    {
                        sum += (float)row[c] * x[c];
                    }
                    outp[r] = sum;
                }
            });

            var f16Bulk = Bench("F16_BulkWiden", log, () =>
            {
                for (var r = 0; r < rows; r++)
                {
                    var row = wF16.AsSpan(r * cols, cols);
                    TensorPrimitives.ConvertToSingle(row, scratch);
                    outp[r] = TensorPrimitives.Dot(scratch, x);
                }
            });

            log($"RESULT: F32={f32:F2}ms  F16scalar={f16Scalar:F2}ms ({f16Scalar / f32:F2}x)  F16bulk={f16Bulk:F2}ms ({f16Bulk / f32:F2}x)");
            var best16 = Math.Min(f16Scalar, f16Bulk);
            log(best16 < f32
                ? $"=> FP16 WINS on ARM ({f32 / best16:F2}x faster) — memory-bound; engine FP16 path worth building"
                : $"=> FP16 SLOWER on ARM too ({best16 / f32:F2}x) — widen cost > DRAM saving; negative like desktop");
        }

        // best-of-N (the throttling-resistant metric this project uses); AndroidBench is not Main, so Stopwatch is fine.
        private static double Bench(string name, Action<string> log, Action work, int iters = 15)
        {
            for (var w = 0; w < 3; w++)
            {
                work(); // warmup
            }

            var best = double.MaxValue;
            for (var i = 0; i < iters; i++)
            {
                var t = Stopwatch.GetTimestamp();
                work();
                var ms = (Stopwatch.GetTimestamp() - t) * 1000.0 / Stopwatch.Frequency;
                if (ms < best)
                {
                    best = ms;
                }
            }

            log($"  {name}: {best:F2} ms (best of {iters})");
            return best;
        }
    }
}
