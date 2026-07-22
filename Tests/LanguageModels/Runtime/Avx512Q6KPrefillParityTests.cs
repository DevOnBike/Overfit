// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Buffers.Binary;
using System.Runtime.Intrinsics.X86;
using DevOnBike.Overfit.LanguageModels.Runtime;

namespace DevOnBike.Overfit.Tests.LanguageModels.Runtime
{
    /// <summary>
    /// Pins <see cref="Q6KGemvKernel.GemmTiled512"/> against <see cref="Q6KGemvKernel.GemmTiled"/>.
    ///
    /// <para>Q6_K needs its own parity coverage rather than leaning on the Q4_K tests, because its port is not
    /// a pure widening: <c>ReduceRows</c> ends in <c>vphaddd</c>, which AVX-512 does not offer for zmm, so the
    /// reduction drops to 256-bit halves while the rest of the loop runs at 512. That split is exactly the kind
    /// of seam where a lane ends up in the wrong half, and only an exact comparison catches it.</para>
    /// </summary>
    public sealed class Avx512Q6KPrefillParityTests
    {
        private const int InputSize = 512;
        private const int OutputSize = 64;

        [Theory]
        [InlineData(1)]
        [InlineData(2)]
        [InlineData(3)]
        [InlineData(7)]
        [InlineData(8)]
        [InlineData(16)]
        public void GemmTiled512_IsBitIdenticalTo_GemmTiled(int cols)
        {
            if (!Avx512BW.IsSupported || !Avx512F.IsSupported)
            {
                return;
            }

            var (weight, quants, scales) = BuildInputs(cols);
            var repacked = weight.EnsureRepacked();

            var reference = new float[cols * OutputSize];
            var wide = new float[cols * OutputSize];

            Q6KGemvKernel.GemmTiled(repacked, OutputSize, InputSize, cols, quants, scales, reference);
            Q6KGemvKernel.GemmTiled512(repacked, OutputSize, InputSize, cols, quants, scales, wide);

            Assert.Equal(reference, wide);
        }

        [Fact]
        public void GemmTiled512_IsBitIdenticalTo_GemmTiled_WithPrecomputedScales()
        {
            if (!Avx512BW.IsSupported || !Avx512F.IsSupported)
            {
                return;
            }

            const int Cols = 8;
            var (weight, quants, scales) = BuildInputs(Cols);
            var repacked = weight.EnsureRepacked();

            var decoded = new float[(OutputSize / 8) * (InputSize / 256) * Q6KGemvKernel.DecodedScalesPerBlock];
            Q6KGemvKernel.DecodeBlockScales(repacked, OutputSize, InputSize, decoded);

            var reference = new float[Cols * OutputSize];
            var wide = new float[Cols * OutputSize];

            Q6KGemvKernel.GemmTiled(repacked, OutputSize, InputSize, Cols, quants, scales, reference);
            Q6KGemvKernel.GemmTiled512(repacked, OutputSize, InputSize, Cols, quants, scales, wide, decoded);

            Assert.Equal(reference, wide);
        }

        private static (Q6KWeight Weight, sbyte[] Quants, float[] Scales) BuildInputs(int cols)
        {
            var rng = new Random(20260722);

            // Synthetic Q6_K blocks, the same construction Q6KDotKernelTests uses: random ql/qh/scales with a
            // sane small positive FP16 d. Parity between two kernels does not need realistic weights, only
            // identical ones.
            var superBlocksPerRow = InputSize / Q6KWeight.SuperBlockElements;
            var blocks = new byte[OutputSize * superBlocksPerRow * Q6KWeight.SuperBlockBytes];

            for (var b = 0; b < OutputSize * superBlocksPerRow; b++)
            {
                var block = blocks.AsSpan(b * Q6KWeight.SuperBlockBytes, Q6KWeight.SuperBlockBytes);

                for (var i = 0; i < block.Length; i++)
                {
                    block[i] = (byte)rng.Next(256);
                }

                var d = (Half)((rng.NextDouble() * 0.05) + 0.001);
                BinaryPrimitives.WriteUInt16LittleEndian(block.Slice(208, 2), BitConverter.HalfToUInt16Bits(d));
            }

            var weight = new Q6KWeight(blocks, InputSize, OutputSize);

            var spr = weight.SuperBlocksPerRow;
            var bsumsPerRow = spr * Q6KDotKernel.GroupsPerSuperBlock;

            var input = new float[cols * InputSize];

            for (var i = 0; i < input.Length; i++)
            {
                input[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
            }

            var quants = new sbyte[cols * InputSize];
            var scales = new float[cols * spr];
            var bsums = new short[cols * bsumsPerRow];

            for (var c = 0; c < cols; c++)
            {
                Q6KDotKernel.QuantizeActivationQ8K(
                    input.AsSpan(c * InputSize, InputSize),
                    quants.AsSpan(c * InputSize, InputSize),
                    scales.AsSpan(c * spr, spr),
                    bsums.AsSpan(c * bsumsPerRow, bsumsPerRow));
            }

            return (weight, quants, scales);
        }
    }
}
