// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Runtime.Intrinsics.X86;

namespace DevOnBike.Overfit.Intrinsics
{
    internal static class CpuFeatures
    {
        public static readonly bool HasAvx2Fma = Avx2.IsSupported && Fma.IsSupported;

        public static readonly bool HasAvx = Avx.IsSupported;

        public static readonly bool HasSse = Sse.IsSupported;
    }
}
