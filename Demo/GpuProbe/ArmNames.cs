// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.GpuProbe
{
    /// <summary>
    /// The arm labels, in one place, so the runner and the report cannot drift apart. C1-C3 and G1-G3 and
    /// X1 are section 3.4 of the plan; C4 is an addition and its label says so in the report.
    /// </summary>
    internal static class ArmNames
    {
        public const string C1 = "C1 cpu q4k forward";
        public const string C2 = "C2 cpu q4k backward";
        public const string C3 = "C3 cpu f32 forward";
        public const string C4 = "C4 cpu f32 backward";
        public const string G1 = "G1 gpu naive forward";
        public const string G2 = "G2 gpu tiled forward";
        public const string G3 = "G3 gpu tiled backward";
        public const string X1 = "X1 cublas fp16 fwd";
        public const string X2 = "X2 cublas fp32 fwd";

        /// <summary>
        /// <c>cublasGemmEx</c> with <c>CUBLAS_COMPUTE_32F</c> - FP16 storage, FP32 accumulate. Reached
        /// through this project's own P/Invoke because ILGPU.Algorithms exports no <c>GemmEx</c>, and
        /// named separately from X1 because the two differ in accumulate precision, which is both a
        /// speed and an accuracy difference.
        /// </summary>
        public const string X3 = "X3 cublas gemmex tc";
    }
}
