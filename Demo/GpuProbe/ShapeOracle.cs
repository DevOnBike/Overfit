// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using ILGPU.Runtime;

namespace DevOnBike.Overfit.GpuProbe
{
    /// <summary>
    /// The first gate: every device kernel against a hand-written triple loop at a deliberately
    /// NON-SQUARE small shape (n=3, k=5, m=7), exact to 1e-6.
    /// <para>
    /// Non-square is the point. A transposed m/n index is invisible on a square shape, and on random data
    /// it survives a cosine check. The shape is also smaller than one tile in every dimension, so it
    /// exercises the tiled kernels' bounds handling rather than their happy path.
    /// </para>
    /// </summary>
    internal static class ShapeOracle
    {
        public const int N = 3;
        public const int K = 5;
        public const int M = 7;
        private const double Tolerance = 1e-6;

        public static IReadOnlyList<string> Run(Accelerator accelerator, TextWriter log)
        {
            var failures = new List<string>();

            var rnd = new Random(20260821);
            var input = new float[N * K];
            var weight = new float[M * K];
            var dy = new float[N * M];
            for (var i = 0; i < input.Length; i++)
            {
                input[i] = (float)(rnd.NextDouble() * 2 - 1);
            }

            for (var i = 0; i < weight.Length; i++)
            {
                weight[i] = (float)(rnd.NextDouble() * 2 - 1);
            }

            for (var i = 0; i < dy.Length; i++)
            {
                dy[i] = (float)(rnd.NextDouble() * 2 - 1);
            }

            // Hand loops. out[b,o] = sum_i in[b,i]*W[o,i]; dx[b,i] = sum_o dy[b,o]*W[o,i].
            var expectedOut = new float[N * M];
            for (var b = 0; b < N; b++)
            {
                for (var o = 0; o < M; o++)
                {
                    var acc = 0f;
                    for (var i = 0; i < K; i++)
                    {
                        acc += input[b * K + i] * weight[o * K + i];
                    }

                    expectedOut[b * M + o] = acc;
                }
            }

            var expectedDx = new float[N * K];
            for (var b = 0; b < N; b++)
            {
                for (var i = 0; i < K; i++)
                {
                    var acc = 0f;
                    for (var o = 0; o < M; o++)
                    {
                        acc += dy[b * M + o] * weight[o * K + i];
                    }

                    expectedDx[b * K + i] = acc;
                }
            }

            using var arms = new GpuArms(accelerator, N, K, M);
            arms.UploadWeight(weight);
            arms.UploadInput(input);
            arms.UploadOutputGrad(dy);

            Check("G1 naive forward", arms.RunNaiveForward, arms.ReadOutput, expectedOut);
            Check("G2 tiled forward", arms.RunTiledForward, arms.ReadOutput, expectedOut);
            Check("G3 tiled backward", arms.RunBackward, arms.ReadInputGrad, expectedDx);

            return failures;

            void Check(string name, Action run, Func<float[]> read, float[] expected)
            {
                run();
                accelerator.Synchronize();
                var actual = read();
                double worst = 0;
                var worstAt = -1;
                for (var i = 0; i < expected.Length; i++)
                {
                    var d = Math.Abs(expected[i] - actual[i]);
                    if (d > worst)
                    {
                        worst = d;
                        worstAt = i;
                    }
                }

                var ok = worst <= Tolerance;
                var where = worstAt < 0 ? "bit-identical to the hand loop" : $"worst at element {worstAt}";
                log.WriteLine($"  shape oracle {name,-20} maxAbs {worst:E2}, {where}  {(ok ? "PASS" : "FAIL")}");
                if (!ok)
                {
                    failures.Add($"{name}: maxAbs {worst:E2} at element {worstAt} (tolerance {Tolerance:E0})");
                }
            }
        }
    }
}
