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
        public const string X1 = "X1 cublas forward";
    }
}
