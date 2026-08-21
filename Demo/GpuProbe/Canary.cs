// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Numerics.Tensors;

namespace DevOnBike.Overfit.GpuProbe
{
    /// <summary>
    /// A fixed host workload — a 512-cubed F32 GEMM — timed at the start of the run and again at the end.
    /// Nothing in the probe changes it, so a move between the two readings is a fact about the machine:
    /// thermal throttling, a background build, a browser. A stranger's machine has a browser open on it.
    /// <para>
    /// Both readings go through the SAME procedure — warm until <see cref="WarmupPolicy"/> says the
    /// timings have stopped moving, then take the median of five. Symmetry is the whole point: two
    /// numbers produced by different procedures cannot be subtracted, and before 2026-08-21 they were,
    /// which made the start reading nineteen times the end reading on an idle machine.
    /// </para>
    /// </summary>
    internal sealed class Canary
    {
        private const int Size = 512;
        private const int Reps = 5;

        /// <summary>A move of more than this fraction makes the whole sitting suspect.</summary>
        public const double MoveThreshold = 0.05;

        private readonly float[] _a;
        private readonly float[] _b;
        private readonly float[] _c;

        public Canary()
        {
            var rnd = new Random(512512);
            _a = new float[Size * Size];
            _b = new float[Size * Size];
            _c = new float[Size * Size];
            for (var i = 0; i < _a.Length; i++)
            {
                _a[i] = (float)(rnd.NextDouble() * 2 - 1);
                _b[i] = (float)(rnd.NextDouble() * 2 - 1);
            }
        }

        /// <summary>Warms to the stopping rule, then returns the median of <see cref="Reps"/> readings.</summary>
        public CanaryReading Measure(WarmupPolicy policy)
        {
            var arm = Arm.Cpu("canary", Multiply);
            var run = ArmRunner.Interleave([arm], policy, Reps);
            return new CanaryReading(run.Timings[arm.Name].MedianMs, run.Warmups[arm.Name]);
        }

        /// <summary>c = a * b, both row-major. b is walked by rows, so this is the same shape every time.</summary>
        private void Multiply()
        {
            for (var i = 0; i < Size; i++)
            {
                var row = _c.AsSpan(i * Size, Size);
                row.Clear();
                for (var p = 0; p < Size; p++)
                {
                    var scale = _a[i * Size + p];
                    TensorPrimitives.MultiplyAdd(_b.AsSpan(p * Size, Size), scale, row, row);
                }
            }
        }
    }
}
