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
    /// Pins increment 2c: a prepacked Q4_K weight (its <c>block_q4_Kx8</c> layout already resident, e.g. mmap'd
    /// from an offline sidecar) routes prefill through the register-tiled GEMM <b>by default</b> — no
    /// <c>OVERFIT_TILED_PREFILL</c> flag needed — because there is no extra RAM cost. Proven by output
    /// discrimination: with the flag OFF, <see cref="BatchedQuantProjection.Dispatch"/> on a prepacked weight
    /// must match a direct <see cref="Q4KGemvKernel.GemmTiled"/> bit-for-bit (a non-tiled path would not).
    /// </summary>
    public sealed class BatchedQuantProjectionTiledDispatchTests
    {
        [Fact]
        public void Dispatch_PrepackedWeight_UsesTiled_EvenWithFlagOff()
        {
            if (!Avx2.IsSupported || !Fma.IsSupported)
            {
                return;
            }

            const int inputSize = 512;
            const int outputSize = 64;
            const int rows = 8;
            var spr = inputSize / 256;
            var bsumsPerRow = spr * Q4KDotKernel.GroupsPerSuperBlock;

            var rng = new Random(7);
            var wF32 = new float[outputSize * inputSize];
            for (var i = 0; i < wF32.Length; i++)
            {
                wF32[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
            }
            var q4k = GgmlQuant.QuantizeQ4_K(wF32, inputSize, outputSize);
            var repacked = Q4KRepack.RepackMatrix(q4k, outputSize, inputSize);

            var weight = new Q4KWeight(q4k, inputSize, outputSize);
            weight.SetPrepacked(repacked); // as the loader does from a sidecar mmap slice
            Assert.True(weight.IsPrepacked);

            var input = new float[rows * inputSize];
            for (var i = 0; i < input.Length; i++)
            {
                input[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
            }

            var prev = BatchedQuantProjection.UseTiledPrefillQ4K;
            try
            {
                BatchedQuantProjection.UseTiledPrefillQ4K = false; // flag OFF — only IsPrepacked can select tiled

                DecodeWeight dw = weight;
                var got = new float[rows * outputSize];
                BatchedQuantProjection.Dispatch(input, rows, in dw, ReadOnlySpan<float>.Empty, got, inputSize, outputSize);

                // Reference: the tiled kernel directly.
                var aq = new sbyte[rows * inputSize];
                var asc = new float[rows * spr];
                var ab = new short[rows * bsumsPerRow];
                for (var n = 0; n < rows; n++)
                {
                    Q4KDotKernel.QuantizeActivationQ8K(
                        input.AsSpan(n * inputSize, inputSize),
                        aq.AsSpan(n * inputSize, inputSize),
                        asc.AsSpan(n * spr, spr),
                        ab.AsSpan(n * bsumsPerRow, bsumsPerRow));
                }
                var expected = new float[rows * outputSize];
                Q4KGemvKernel.GemmTiled(repacked, outputSize, inputSize, rows, aq, asc, ab, expected);

                for (var i = 0; i < expected.Length; i++)
                {
                    Assert.Equal(expected[i], got[i]); // tiled path ran despite the flag being off
                }
            }
            finally
            {
                BatchedQuantProjection.UseTiledPrefillQ4K = prev;
            }
        }
    }
}
