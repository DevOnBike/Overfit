// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.GpuProbe
{
    /// <summary>
    /// One measured shape: a <c>FrozenQuantizedLinear</c> of <c>K</c> input features to <c>M</c> output
    /// features. Shapes and weights come from section 3.2 of
    /// <c>docs/specs/gpu-probe-route-and-design-plan.md</c> (Qwen2.5-3B-Instruct Q4_K_M).
    /// </summary>
    internal sealed class Cell
    {
        public Cell(string name, int k, int m, int callsPerStep, double macShare)
        {
            Name = name;
            K = k;
            M = m;
            CallsPerStep = callsPerStep;
            MacShare = macShare;
        }

        public string Name { get; }

        /// <summary>Contraction dimension (input features). Must be a multiple of 256 for Q4_K.</summary>
        public int K { get; }

        /// <summary>Output features.</summary>
        public int M { get; }

        /// <summary>Calls this shape makes per Qwen-3B training step (36 layers).</summary>
        public int CallsPerStep { get; }

        /// <summary>
        /// Share of the forward MACs of one step. Arithmetic derived from the shapes, NOT a measurement:
        /// it says where the FLOPs are, not where the time is.
        /// </summary>
        public double MacShare { get; }

        /// <summary>Bytes of F32 weight this cell uploads to the device.</summary>
        public long WeightBytesF32 => (long)K * M * sizeof(float);

        public static IReadOnlyList<Cell> Production =>
        [
            new("ffn_gate_up", 2048, 11008, 72, 0.526),
            new("ffn_down", 11008, 2048, 36, 0.263),
            new("attn_qo", 2048, 2048, 72, 0.098),
            new("lm_head", 2048, 151936, 1, 0.101),
            new("attn_kv", 2048, 256, 72, 0.012),
        ];

        /// <summary>
        /// Reduced shapes for <c>--quick</c>. They exercise every code path end to end in seconds so the
        /// plumbing can be checked on a machine without a discrete GPU. They are NOT the QLoRA shapes and
        /// a number taken from them says nothing about the product.
        /// </summary>
        public static IReadOnlyList<Cell> Quick =>
        [
            new("quick_wide", 512, 1024, 0, 0),
            new("quick_tall", 1024, 512, 0, 0),
        ];
    }
}
