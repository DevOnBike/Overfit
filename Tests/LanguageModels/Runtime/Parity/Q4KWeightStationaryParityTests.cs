// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Buffers.Binary;
using DevOnBike.Overfit.LanguageModels.Loading;
using DevOnBike.Overfit.LanguageModels.Runtime;

namespace DevOnBike.Overfit.Tests.LanguageModels.Runtime.Parity
{
    /// <summary>
    /// The weight-stationary batched Q4_K projection must be BIT-IDENTICAL to <see cref="Q4KDotKernel.ProjectBatched"/>
    /// — it only hoists the per-block weight decode out of the rows loop; the per-row accumulation order over
    /// super-blocks is unchanged, so every output float must match exactly (including across the row-tile boundary).
    /// </summary>
    public sealed class Q4KWeightStationaryParityTests
    {
        [Fact]
        public void WeightStationary_IsBitIdentical_ToProjectBatched()
        {
            const int inputSize = 512;   // 2 super-blocks per row
            const int outputSize = 96;
            var spr = inputSize / Q4KDotKernel.SuperBlockElements;

            var rnd = new Random(7);
            var bytes = new byte[outputSize * spr * Q4KWeight.SuperBlockBytes];
            rnd.NextBytes(bytes);

            // Sanitize each block's d / dmin to finite Halfs (random Half bytes can be NaN/Inf).
            var dBits = BitConverter.HalfToUInt16Bits((Half)0.05f);
            var dminBits = BitConverter.HalfToUInt16Bits((Half)0.012f);
            for (var blk = 0; blk < outputSize * spr; blk++)
            {
                var off = blk * Q4KWeight.SuperBlockBytes;
                BinaryPrimitives.WriteUInt16LittleEndian(bytes.AsSpan(off, 2), dBits);
                BinaryPrimitives.WriteUInt16LittleEndian(bytes.AsSpan(off + 2, 2), dminBits);
            }

            var weight = new Q4KWeight(bytes, inputSize, outputSize);

            // 70 > the 64-row tile → exercises the tiling boundary.
            foreach (var rows in new[] { 1, 3, 8, 64, 70 })
            {
                var input = new float[rows * inputSize];
                for (var i = 0; i < input.Length; i++)
                {
                    input[i] = (float)(rnd.NextDouble() * 2 - 1);
                }

                var bias = new float[outputSize];
                for (var i = 0; i < outputSize; i++)
                {
                    bias[i] = (float)(rnd.NextDouble() - 0.5);
                }

                var expected = new float[rows * outputSize];
                var actual = new float[rows * outputSize];
                var q = new sbyte[rows * inputSize];
                var sc = new float[rows * spr];
                var bs = new short[rows * spr * Q4KDotKernel.GroupsPerSuperBlock];

                Q4KDotKernel.ProjectBatched(input, rows, weight, bias, expected, q, sc, bs);
                Q4KDotKernel.ProjectBatchedWeightStationary(input, rows, weight, bias, actual, q, sc, bs);

                for (var i = 0; i < expected.Length; i++)
                {
                    Assert.Equal(expected[i], actual[i]);
                }
            }
        }
    }
}
