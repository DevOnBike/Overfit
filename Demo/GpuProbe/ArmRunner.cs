// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.GpuProbe
{
    /// <summary>
    /// Runs a set of arms ABAB — one repetition of every arm, then the next repetition — rather than all
    /// of A and then all of B. Drift over the sitting then falls on every arm equally instead of wearing
    /// the costume of the variable (plan, section 3.5 rule 5).
    /// </summary>
    internal static class ArmRunner
    {
        public static IReadOnlyDictionary<string, Measurement> Interleave(
            IReadOnlyList<Arm> arms,
            int warmups,
            int reps)
        {
            ArgumentOutOfRangeException.ThrowIfLessThan(warmups, 2);
            ArgumentOutOfRangeException.ThrowIfLessThan(reps, 5);

            // Untimed. The first ILGPU launch of a kernel includes PTX or OpenCL C generation and the
            // driver's compile of it; timing that measures a compiler.
            for (var w = 0; w < warmups; w++)
            {
                foreach (var arm in arms)
                {
                    arm.TimeOnce();
                }
            }

            var samples = new Dictionary<string, List<double>>(arms.Count);
            foreach (var arm in arms)
            {
                samples[arm.Name] = new List<double>(reps);
            }

            for (var r = 0; r < reps; r++)
            {
                foreach (var arm in arms)
                {
                    samples[arm.Name].Add(arm.TimeOnce());
                }
            }

            var result = new Dictionary<string, Measurement>(arms.Count);
            foreach (var arm in arms)
            {
                result[arm.Name] = new Measurement(arm.Name, samples[arm.Name]);
            }

            return result;
        }
    }
}
