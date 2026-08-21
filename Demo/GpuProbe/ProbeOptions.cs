// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.GpuProbe
{
    /// <summary>Command line of the probe. Every default is the one a stranger should get.</summary>
    internal sealed class ProbeOptions
    {
        /// <summary>Timed repetitions per arm per cell. The plan's floor is 5.</summary>
        public int Reps { get; private set; } = 5;

        /// <summary>
        /// The FLOOR on untimed warm-up rounds, not the count. The count is decided by the stopping rule
        /// in <see cref="WarmupPolicy"/>, because a fixed count is a number that worked once on one box.
        /// The plan's floor is 2; this floor is 10, which is the smallest value at which the rule can be
        /// evaluated at all (two windows of five).
        /// </summary>
        public int WarmupMinimum { get; private set; } = 10;

        /// <summary>Hard cap on warm-up rounds, so the loop always terminates.</summary>
        public int WarmupMaximum { get; private set; } = 100;

        /// <summary>Relative move between two window medians that still counts as settled.</summary>
        public double WarmupTolerance { get; private set; } = 0.05;

        /// <summary>Wall-clock budget for one cell's warm-up phase, after the minimum rounds have run.</summary>
        public double WarmupBudgetMs { get; private set; } = 30_000;

        /// <summary>Token counts to sweep. Provenance for all three is section 3.3 of the plan.</summary>
        public IReadOnlyList<int> Batches { get; private set; } = [16, 128, 256];

        /// <summary>Substring filter over cell names. Empty means every cell.</summary>
        public string CellFilter { get; private set; } = string.Empty;

        /// <summary>Run the oracles and print no timing at all.</summary>
        public bool ParityOnly { get; private set; }

        /// <summary>Reduced shapes that exercise every path in seconds. NOT the QLoRA shapes.</summary>
        public bool Quick { get; private set; }

        /// <summary>
        /// Measure how much accuracy FP16 costs at each real shape, on the host, and print nothing else.
        /// This is what sets the parity ceiling for the FP16 arm, and it needs no GPU at all.
        /// </summary>
        public bool Fp16Bound { get; private set; }

        /// <summary>Arm X1, the cuBLAS upper bound. Off by default; needs the CUDA toolkit.</summary>
        public bool EnableCuBlas { get; private set; }

        /// <summary>
        /// Measure even when the selected accelerator is ILGPU's CPU emulator. Off by default, because a
        /// "GPU" column produced by a CPU emulator is the worst output this probe could produce.
        /// </summary>
        public bool AllowCpuAccelerator { get; private set; }

        /// <summary>cuda / opencl / cpu, or empty for the best available.</summary>
        public string Device { get; private set; } = string.Empty;

        public int Seed { get; private set; } = 42;

        public string? Error { get; private set; }

        public static ProbeOptions Parse(string[] args)
        {
            var o = new ProbeOptions();

            foreach (var raw in args)
            {
                var arg = raw.Trim();
                var eq = arg.IndexOf('=');
                var key = eq < 0 ? arg : arg[..eq];
                var value = eq < 0 ? string.Empty : arg[(eq + 1)..];

                switch (key)
                {
                    case "--parity-only":
                        o.ParityOnly = true;
                        continue;
                    case "--quick":
                        o.Quick = true;
                        continue;
                    case "--fp16-bound":
                        o.Fp16Bound = true;
                        continue;
                    case "--x1":
                        o.EnableCuBlas = true;
                        continue;
                    case "--allow-cpu-accelerator":
                        o.AllowCpuAccelerator = true;
                        continue;
                    case "--reps":
                        o.Reps = ParseInt(value, o, key, minimum: 5);
                        continue;
                    case "--warmups":
                        o.WarmupMinimum = ParseInt(value, o, key, minimum: 10);
                        continue;
                    case "--warmup-max":
                        o.WarmupMaximum = ParseInt(value, o, key, minimum: 10);
                        continue;
                    case "--warmup-budget-ms":
                        o.WarmupBudgetMs = ParseInt(value, o, key, minimum: 1000);
                        continue;
                    case "--warmup-tolerance":
                        o.WarmupTolerance = ParseFraction(value, o, key);
                        continue;
                    case "--seed":
                        o.Seed = ParseInt(value, o, key, minimum: 0);
                        continue;
                    case "--device":
                        o.Device = value.ToLowerInvariant();
                        continue;
                    case "--cells":
                        o.CellFilter = value;
                        continue;
                    case "--batches":
                        o.Batches = ParseBatches(value, o);
                        continue;
                    case "--help":
                    case "-h":
                        o.Error = "help";
                        continue;
                    default:
                        o.Error = $"unknown argument '{arg}'";
                        continue;
                }
            }

            return o;
        }

        /// <summary>
        /// The stopping rule this run will use. Built here so the one place that owns the defaults is
        /// the one place the command line writes to.
        /// </summary>
        public WarmupPolicy WarmupPolicy => new(
            Math.Max(WarmupMinimum, 10),
            Math.Max(WarmupMaximum, Math.Max(WarmupMinimum, 10)),
            WarmupTolerance,
            windowSize: 5,
            WarmupBudgetMs);

        public string Describe() =>
            $"reps={Reps} warmupMin={WarmupMinimum} warmupMax={WarmupMaximum} " +
            $"warmupTolerance={WarmupTolerance} warmupBudgetMs={WarmupBudgetMs} " +
            $"batches=[{string.Join(",", Batches)}] " +
            $"cells='{(CellFilter.Length == 0 ? "all" : CellFilter)}' seed={Seed} " +
            $"quick={Quick} parityOnly={ParityOnly} fp16Bound={Fp16Bound} x1={EnableCuBlas} " +
            $"device='{(Device.Length == 0 ? "auto" : Device)}' allowCpuAccelerator={AllowCpuAccelerator}";

        public static string Usage =>
            """
            GpuProbe - measures ONE operation (FrozenQuantizedLinear, the QLoRA frozen base matmul)
                       on the CPU and on the GPU, in one sitting, on this machine.

              --parity-only             run the correctness oracles only, print no timing
              --quick                   small shapes, seconds not minutes; NOT the QLoRA shapes
              --fp16-bound              measure what FP16 costs in accuracy at each shape, then stop
              --x1                      also measure cuBLAS (needs the CUDA TOOLKIT, not just the driver)
              --allow-cpu-accelerator   measure even if the only accelerator is ILGPU's CPU emulator
              --device=cuda|opencl|cpu  force a backend instead of taking the best available
              --reps=N                  timed repetitions per arm (minimum 5)
              --warmups=N               MINIMUM untimed warm-up rounds (minimum 10). The actual count is
                                        decided by the stopping rule, which warms until the median of the
                                        last 5 readings of every arm is within a tolerance of the median
                                        of the 5 before it. The report prints how many rounds it took.
              --warmup-max=N            hard cap on warm-up rounds (default 100)
              --warmup-budget-ms=N      wall-clock budget for one cell's warm-up (default 30000)
              --warmup-tolerance=F      settled when two window medians are within F, e.g. 0.05
              --cells=SUBSTRING         only cells whose name contains SUBSTRING
              --batches=16,128,256      token counts to sweep
              --seed=N                  seed of the synthetic weights and activations
            """;

        private static int ParseInt(string value, ProbeOptions o, string key, int minimum)
        {
            if (!int.TryParse(value, out var parsed))
            {
                o.Error = $"{key} needs an integer, got '{value}'";
                return minimum;
            }

            if (parsed < minimum)
            {
                o.Error = $"{key} must be at least {minimum} (the plan's floor), got {parsed}";
                return minimum;
            }

            return parsed;
        }

        private static double ParseFraction(string value, ProbeOptions o, string key)
        {
            if (!double.TryParse(value, System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out var parsed))
            {
                o.Error = $"{key} needs a number, got '{value}'";
                return 0.05;
            }

            if (parsed <= 0 || parsed >= 1)
            {
                o.Error = $"{key} must be a fraction strictly between 0 and 1, got {parsed}";
                return 0.05;
            }

            return parsed;
        }

        private static IReadOnlyList<int> ParseBatches(string value, ProbeOptions o)
        {
            var parts = value.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
            var result = new List<int>(parts.Length);
            foreach (var part in parts)
            {
                if (!int.TryParse(part, out var n) || n <= 0)
                {
                    o.Error = $"--batches needs positive integers, got '{part}'";
                    return [16];
                }

                result.Add(n);
            }

            return result.Count == 0 ? [16] : result;
        }
    }
}
