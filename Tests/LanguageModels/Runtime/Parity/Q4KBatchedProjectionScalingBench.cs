// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels.Loading;
using DevOnBike.Overfit.LanguageModels.Runtime;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests.LanguageModels.Runtime.Parity
{
    /// <summary>
    /// Isolates the cost scaling of the batched Q4_K projection (prefill / speculative-verify primitive) as a
    /// function of row count. If batching amortised perfectly (DRAM-bound), time(rows=R) ≈ time(rows=1) and per-row
    /// cost falls ~R×; if compute-bound per row, time grows ~linearly. Diagnoses whether the speculative verify's
    /// per-row cost is a fixable weight-DECODE-per-row inefficiency: <see cref="Q4KDotKernel.ProjectBatched"/> reads
    /// each weight super-block from DRAM once (cache-hot across rows) but RE-DECODES it (unpack scales/mins/nibbles +
    /// the F32 tail in <see cref="Q4KDotKernel.Dot"/>) once PER ROW — a weight-stationary "decode-once, dot-many"
    /// kernel would amortise that. Best-of-N (min) per config + a rows=1 canary at the end (if it drifted, the box
    /// was loaded → numbers untrustworthy). [LongFact] — flip to [Fact] and run on a COLD, idle box only.
    /// </summary>
    public sealed class Q4KBatchedProjectionScalingBench
    {
        private readonly ITestOutputHelper _out;
        public Q4KBatchedProjectionScalingBench(ITestOutputHelper output) => _out = output;

        [LongFact]
        public void BatchedProjection_CostScaling_ByRows()
        {
            const int inputSize = 2048;    // 8 super-blocks / row
            const int outputSize = 8192;   // ~9.4 MB Q4_K matrix (> L2 → real DRAM behaviour)
            var spr = inputSize / Q4KDotKernel.SuperBlockElements;

            var rnd = new Random(1234);
            var blockBytes = new byte[(long)outputSize * spr * Q4KWeight.SuperBlockBytes];
            rnd.NextBytes(blockBytes);
            var weight = new Q4KWeight(blockBytes, inputSize, outputSize);

            const int maxRows = 128;
            var input = new float[maxRows * inputSize];
            for (var i = 0; i < input.Length; i++)
            {
                input[i] = (float)(rnd.NextDouble() * 2 - 1);
            }

            var output = new float[maxRows * outputSize];
            var qBytes = new sbyte[maxRows * inputSize];
            var scales = new float[maxRows * spr];
            var bsums = new short[maxRows * spr * Q4KDotKernel.GroupsPerSuperBlock];


            void Run(int rows, bool ws)
            {
                if (ws)
                {
                    Q4KDotKernel.ProjectBatchedWeightStationary(input, rows, weight, ReadOnlySpan<float>.Empty, output,
                        qBytes.AsSpan(0, rows * inputSize), scales.AsSpan(0, rows * spr),
                        bsums.AsSpan(0, rows * spr * Q4KDotKernel.GroupsPerSuperBlock));
                }
                else
                {
                    Q4KDotKernel.ProjectBatched(input, rows, weight, ReadOnlySpan<float>.Empty, output,
                        qBytes.AsSpan(0, rows * inputSize), scales.AsSpan(0, rows * spr),
                        bsums.AsSpan(0, rows * spr * Q4KDotKernel.GroupsPerSuperBlock));
                }
            }

            // Best-of-N (min) per config — the minimum is the least-contaminated sample (rejects positive spikes
            // from scheduling/GC/turbo). Each "shot" is a short inner loop so dispatch warmup is paid up front.
            double Time(int rows, bool ws)
            {
                for (var w = 0; w < 20; w++) // warm the parallel pool + caches for THIS row count
                {
                    Run(rows, ws);
                }
                var best = double.MaxValue;
                for (var shot = 0; shot < 8; shot++)
                {
                    var sw = System.Diagnostics.Stopwatch.StartNew();
                    for (var it = 0; it < 50; it++)
                    {
                        Run(rows, ws);
                    }
                    sw.Stop();
                    best = Math.Min(best, sw.Elapsed.TotalMilliseconds / 50);
                }
                return best;
            }

            // Global warmup (JIT + parallel pool + caches) so the opening baseline is not measured cold.
            Time(8, false);
            Time(8, true);

            var t1 = Time(1, false);
            _out.WriteLine($"rows=1: current {t1:F3} ms  [baseline]");
            foreach (var rows in new[] { 2, 4, 8, 16, 64, 128 })
            {
                var cur = Time(rows, false);
                var ws = Time(rows, true);
                _out.WriteLine($"rows={rows}: current {cur:F3} (per-row {cur / rows:F3})  |  " +
                               $"weight-stationary {ws:F3} (per-row {ws / rows:F3})  |  WS speedup {cur / ws:F2}×");
            }

            // CANARY: re-measure rows=1. If it drifted from the opening baseline, the box was loaded → distrust all.
            var t1End = Time(1, false);
            var drift = (t1End - t1) / t1 * 100;
            _out.WriteLine($"CANARY rows=1 re-measure: {t1End:F3} ms (drift {drift:+0.0;-0.0}% vs baseline — " +
                           $"{(Math.Abs(drift) < 8 ? "STABLE, numbers trustworthy" : "UNSTABLE BOX, distrust the A/B")})");
        }
    }
}
