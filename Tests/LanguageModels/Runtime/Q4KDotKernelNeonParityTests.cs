// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System;
using System.Runtime.Intrinsics.Arm;
using DevOnBike.Overfit.LanguageModels.Runtime;

namespace DevOnBike.Overfit.Tests.LanguageModels.Runtime
{
    /// <summary>
    /// Pins the NEON (SDOT) Q4_K main-dot against the scalar oracle. The NEON path only exists on
    /// arm64 with the dot-product extension, and ARM intrinsics throw if invoked off-arm — so this
    /// test is a no-op pass on x86 (where <see cref="Dp.IsSupported"/> is false) and does the real
    /// bit-identity check when run on the phone / an arm64 runner. The two paths are pure INT32, so
    /// the assertion is exact equality, not a tolerance.
    /// </summary>
    public class Q4KDotKernelNeonParityTests
    {
        [Fact]
        public void MainDotNeon_IsBitIdenticalToScalar_OnArm()
        {
            if (!Dp.IsSupported)
            {
                // Off-arm (x86 dev box / CI): the kernel uses AVX2/scalar; the NEON path is unreachable
                // and cannot be JIT-executed here. Validated when this test runs on arm64.
                return;
            }

            var rng = new Random(20260625);
            for (var trial = 0; trial < 256; trial++)
            {
                var qs = new byte[128];
                rng.NextBytes(qs);

                var q8 = new sbyte[256];
                for (var i = 0; i < q8.Length; i++)
                {
                    q8[i] = (sbyte)rng.Next(-128, 128);
                }

                var scales = new byte[8];
                for (var i = 0; i < scales.Length; i++)
                {
                    scales[i] = (byte)rng.Next(0, 64);
                }

                var scalar = Q4KDotKernel.MainDotScalar(qs, q8, scales);
                var neon = Q4KDotKernel.MainDotNeon(qs, q8, scales);

                Assert.Equal(scalar, neon);
            }
        }
    }
}
