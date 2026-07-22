// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Runtime.Intrinsics.X86;
using DevOnBike.Overfit.LanguageModels.Loading;
using DevOnBike.Overfit.LanguageModels.Runtime;

namespace DevOnBike.Overfit.Tests.LanguageModels.Runtime
{
    /// <summary>
    /// Pins <see cref="Q6KGemvKernel.GemmTiled"/> — the register-tiled Q6_K prefill GEMM over the repacked
    /// <c>block_q6_Kx8</c> layout — as <b>bit-identical</b> to calling the validated decode
    /// <see cref="Q6KGemvKernel.GemvAvx2"/> once per activation column. Same weights, same Q8_K activations,
    /// same per-(row, column) reduction order; only the weight unpack moves out of the column loop.
    ///
    /// <para>This exists because Q6_K carries half of <c>ffn_down</c> under Q4_K_M, which a prefill profile
    /// measured at 37.9% of prefill running at 0.61 TFLOP/s. The first attempt at closing that gap — a
    /// weight-stationary kernel — was bit-identical but 13.5% <i>slower</i>, so correctness alone is not the
    /// bar here; this test is the gate that lets the performance question be asked honestly.</para>
    /// </summary>
    public sealed class Q6KTiledGemmParityTests
    {
        [Theory]
        [InlineData(1)]
        [InlineData(2)]
        [InlineData(3)]
        [InlineData(8)]
        [InlineData(16)]
        public void GemmTiled_MatchesGemvPerColumn_BitIdentical(int cols)
        {
            if (!Avx2.IsSupported || !Fma.IsSupported)
            {
                return; // AVX2/FMA kernel — nothing to verify on this CPU
            }

            const int inputSize = 512;   // 2 Q6_K super-blocks
            const int outputSize = 64;   // 8 row-groups
            var nb = inputSize / 256;
            var bsumsPerRow = nb * Q6KDotKernel.GroupsPerSuperBlock;

            var rng = new Random(6060 + cols);

            // Any byte pattern is a valid Q6_K super-block as long as the fp16 scale is sane — same
            // construction the existing Q6KDotKernelTests use, so no quantizer is needed.
            var blocks = new byte[outputSize * nb * Q6KWeight.SuperBlockBytes];
            for (var b = 0; b < outputSize * nb; b++)
            {
                var block = blocks.AsSpan(b * Q6KWeight.SuperBlockBytes, Q6KWeight.SuperBlockBytes);
                for (var i = 0; i < block.Length; i++)
                {
                    block[i] = (byte)rng.Next(256);
                }

                // Random bits at offset 208 could decode to NaN/Inf — overwrite with a small positive fp16.
                var d = (Half)(rng.NextDouble() * 0.05 + 0.001);
                BitConverter.GetBytes(BitConverter.HalfToUInt16Bits(d)).CopyTo(block.Slice(208, 2));
            }

            var weight = new Q6KWeight(blocks, inputSize, outputSize);
            Assert.True(weight.CanRepack);
            var repacked = weight.EnsureRepacked().ToArray();

            var inputs = new float[cols * inputSize];
            for (var i = 0; i < inputs.Length; i++)
            {
                inputs[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
            }

            var aqAll = new sbyte[cols * inputSize];
            var ascAll = new float[cols * nb];
            var abAll = new short[cols * bsumsPerRow];
            for (var c = 0; c < cols; c++)
            {
                Q6KDotKernel.QuantizeActivationQ8K(
                    inputs.AsSpan(c * inputSize, inputSize),
                    aqAll.AsSpan(c * inputSize, inputSize),
                    ascAll.AsSpan(c * nb, nb),
                    abAll.AsSpan(c * bsumsPerRow, bsumsPerRow));
            }

            var tiled = new float[cols * outputSize];
            Q6KGemvKernel.GemmTiled(repacked, outputSize, inputSize, cols, aqAll, ascAll, tiled);

            var reference = new float[outputSize];
            for (var c = 0; c < cols; c++)
            {
                Q6KGemvKernel.GemvAvx2(
                    repacked, outputSize, inputSize,
                    aqAll.AsSpan(c * inputSize, inputSize),
                    ascAll.AsSpan(c * nb, nb),
                    reference);

                for (var o = 0; o < outputSize; o++)
                {
                    Assert.Equal(reference[o], tiled[c * outputSize + o]);
                }
            }
        }

        [Fact]
        public void GemmTiled_RejectsOutOfRangeColumnCount()
        {
            if (!Avx2.IsSupported || !Fma.IsSupported)
            {
                return;
            }

            const int inputSize = 256;
            const int outputSize = 8;
            var repacked = new byte[Q6KRepack.BlockKx8Bytes];
            var aq = new sbyte[inputSize];
            var asc = new float[1];
            var output = new float[outputSize];

            var threwZero = false;
            try
            {
                Q6KGemvKernel.GemmTiled(repacked, outputSize, inputSize, 0, aq, asc, output);
            }
            catch (ArgumentOutOfRangeException)
            {
                threwZero = true;
            }

            var threwTooMany = false;
            try
            {
                Q6KGemvKernel.GemmTiled(
                    repacked, outputSize, inputSize, Q6KGemvKernel.MaxTileCols + 1, aq, asc, output);
            }
            catch (ArgumentOutOfRangeException)
            {
                threwTooMany = true;
            }

            Assert.True(threwZero, "expected ArgumentOutOfRangeException for cols = 0");
            Assert.True(threwTooMany, $"expected ArgumentOutOfRangeException for cols > {Q6KGemvKernel.MaxTileCols}");
        }
    }
}
