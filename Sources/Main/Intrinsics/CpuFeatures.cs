// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Runtime.Intrinsics.X86;

namespace DevOnBike.Overfit.Intrinsics
{
    internal static class CpuFeatures
    {
        public static readonly bool HasFma = Fma.IsSupported;

        public static readonly bool HasAvx = Avx.IsSupported;

        public static readonly bool HasAvx2 = Avx2.IsSupported;

        public static readonly bool HasAvx512 = Avx512F.IsSupported;

        public static readonly bool HasAvxVnni = AvxVnni.IsSupported;

        public static readonly bool HasSse = Sse.IsSupported;

        public static readonly bool HasSse3 = Sse3.IsSupported;

        // to zawsze na koncu - bo zalezy od HasFma i HasAvx2 (pola sa inicjowane od gory do dolu)
        public static readonly bool HasAvx2Fma = HasAvx2 && HasFma;
    }
}
