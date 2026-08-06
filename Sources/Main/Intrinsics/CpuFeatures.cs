// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.Arm;
using System.Runtime.Intrinsics.X86;

namespace DevOnBike.Overfit.Intrinsics
{
    internal static class CpuFeatures
    {
        public static readonly bool HasFma = Fma.IsSupported;

        // arm64 dot-product extension (SDOT/UDOT, ARMv8.2 FEAT_DotProd) — drives the NEON quant kernels.
        public static readonly bool HasDp = Dp.IsSupported;

        public static readonly bool HasAvx = Avx.IsSupported;

        public static readonly bool HasAvx2 = Avx2.IsSupported;

        public static readonly bool HasAvx512 = Avx512F.IsSupported;

        // AVX-512 byte/word ops (vpmaddubsw/vpmaddwd on zmm) — required by the quantized prefill kernels,
        // which reach their MACs through those rather than through floating-point FMA.
        public static readonly bool HasAvx512Bw = Avx512BW.IsSupported;

        public static readonly bool HasAvxVnni = AvxVnni.IsSupported;

        public static readonly bool HasSse = Sse.IsSupported;

        public static readonly bool HasSse3 = Sse3.IsSupported;

        // ── Portable vector widths ────────────────────────────────────────────────
        //
        // `Vector512.IsHardwareAccelerated` asks "is this width fast here?" rather than "is this x86 ISA
        // present?", so the same branch covers AVX-512 on x86 and SVE on arm64 — which matters because this
        // library also ships to Android. The flags above stay for kernels that genuinely need an x86-specific
        // instruction (`vpmaddubsw`, `vpdpbusd`); these are for kernels whose only question is vector width.
        //
        // The three-step cascade (512 → 256 → 128 → scalar) is what both the BCL's own TensorPrimitives and
        // ImageSharp converged on independently — two production codebases, same shape, so it is adopted here
        // rather than reinvented. Note the property is IsHardwareAccelerated, not IsSupported: a CPU can
        // *support* a width while executing it at half rate (Zen 4 double-pumps 512-bit through a 256-bit
        // datapath), and the runtime reports the useful answer rather than the nominal one.

        public static readonly bool HasVector128 = Vector128.IsHardwareAccelerated;

        public static readonly bool HasVector256 = Vector256.IsHardwareAccelerated;

        public static readonly bool HasVector512 = Vector512.IsHardwareAccelerated;

        // This one stays last: it depends on HasFma and HasAvx2, and fields initialise top to bottom.
        public static readonly bool HasAvx2Fma = HasAvx2 && HasFma;
    }
}
