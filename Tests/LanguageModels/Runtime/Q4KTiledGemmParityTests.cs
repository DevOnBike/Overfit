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
    /// Pins Phase 1 of the tinyBLAS register-tiling lever: <see cref="Q4KGemvKernel.GemmTiled"/> (the
    /// multi-column register-tiled Q4_K GEMM) must be <b>bit-identical</b> to calling the validated decode
    /// <see cref="Q4KGemvKernel.Gemv"/> once per column — same weight, same Q8_K activations, same
    /// per-(row,col) reduction order. This is the correctness baseline the Phase 2 perf iteration A/B's against.
    /// </summary>
    public sealed class Q4KTiledGemmParityTests
    {
        [Theory]
        [InlineData(1)]
        [InlineData(2)]
        [InlineData(3)]
        [InlineData(4)]
        [InlineData(8)]
        public void GemmTiled_MatchesGemvPerColumn_BitIdentical(int cols)
        {
            if (!Avx2.IsSupported || !Fma.IsSupported)
            {
                return; // AVX2/FMA kernel — nothing to verify on this CPU
            }

            const int inputSize = 512;   // 2 Q4_K super-blocks
            const int outputSize = 64;   // 8 row-groups
            var nb = inputSize / 256;
            var bsumsPerRow = nb * Q4KDotKernel.GroupsPerSuperBlock;

            var rng = new Random(1234 + cols);

            // Random Q4_K weight [outputSize × inputSize], output-major.
            var wF32 = new float[outputSize * inputSize];
            for (var i = 0; i < wF32.Length; i++)
            {
                wF32[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
            }
            var weight = new Q4KWeight(GgmlQuant.QuantizeQ4_K(wF32, inputSize, outputSize), inputSize, outputSize);
            Assert.True(weight.CanRepack);
            var repacked = weight.EnsureRepacked().ToArray();

            // cols activation vectors, quantized to Q8_K, laid out column-contiguous for GemmTiled.
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
                Q4KDotKernel.QuantizeActivationQ8K(
                    inputs.AsSpan(c * inputSize, inputSize),
                    aqAll.AsSpan(c * inputSize, inputSize),
                    ascAll.AsSpan(c * nb, nb),
                    abAll.AsSpan(c * bsumsPerRow, bsumsPerRow));
            }

            var tiled = new float[cols * outputSize];
            Q4KGemvKernel.GemmTiled(repacked, outputSize, inputSize, cols, aqAll, ascAll, abAll, tiled);

            var reference = new float[outputSize];
            for (var c = 0; c < cols; c++)
            {
                Q4KGemvKernel.Gemv(
                    repacked, outputSize, inputSize,
                    aqAll.AsSpan(c * inputSize, inputSize),
                    ascAll.AsSpan(c * nb, nb),
                    abAll.AsSpan(c * bsumsPerRow, bsumsPerRow),
                    reference);

                for (var r = 0; r < outputSize; r++)
                {
                    Assert.Equal(reference[r], tiled[c * outputSize + r]);
                }
            }
        }
    }
}
