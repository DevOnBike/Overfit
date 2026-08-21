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

        /// <summary>Untimed repetitions before the clock starts. The plan's floor is 2.</summary>
        public int Warmups { get; private set; } = 2;

        /// <summary>Token counts to sweep. Provenance for all three is section 3.3 of the plan.</summary>
        public IReadOnlyList<int> Batches { get; private set; } = [16, 128, 256];

        /// <summary>Substring filter over cell names. Empty means every cell.</summary>
        public string CellFilter { get; private set; } = string.Empty;

        /// <summary>Run the oracles and print no timing at all.</summary>
        public bool ParityOnly { get; private set; }

        /// <summary>Reduced shapes that exercise every path in seconds. NOT the QLoRA shapes.</summary>
        public bool Quick { get; private set; }

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
                        o.Warmups = ParseInt(value, o, key, minimum: 2);
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

        public string Describe() =>
            $"reps={Reps} warmups={Warmups} batches=[{string.Join(",", Batches)}] " +
            $"cells='{(CellFilter.Length == 0 ? "all" : CellFilter)}' seed={Seed} " +
            $"quick={Quick} parityOnly={ParityOnly} x1={EnableCuBlas} " +
            $"device='{(Device.Length == 0 ? "auto" : Device)}' allowCpuAccelerator={AllowCpuAccelerator}";

        public static string Usage =>
            """
            GpuProbe - measures ONE operation (FrozenQuantizedLinear, the QLoRA frozen base matmul)
                       on the CPU and on the GPU, in one sitting, on this machine.

              --parity-only             run the correctness oracles only, print no timing
              --quick                   small shapes, seconds not minutes; NOT the QLoRA shapes
              --x1                      also measure cuBLAS (needs the CUDA TOOLKIT, not just the driver)
              --allow-cpu-accelerator   measure even if the only accelerator is ILGPU's CPU emulator
              --device=cuda|opencl|cpu  force a backend instead of taking the best available
              --reps=N                  timed repetitions per arm (minimum 5)
              --warmups=N               untimed repetitions per arm (minimum 2)
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
