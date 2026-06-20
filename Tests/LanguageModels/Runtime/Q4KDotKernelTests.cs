// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Buffers.Binary;
using DevOnBike.Overfit.LanguageModels.Loading;
using DevOnBike.Overfit.LanguageModels.Runtime;

namespace DevOnBike.Overfit.Tests.LanguageModels.Runtime
{
    /// <summary>
    /// Synthetic correctness tests for <see cref="Q4KDotKernel"/> (step 3.1 —
    /// the Q4_K × Q8_K decode kernel).
    ///
    /// Reference: the same Q4_K bytes decoded to F32 by the proven
    /// <see cref="GgmlDequant.DecodeQ4_KBlock"/>, then a plain F32 dot. The
    /// kernel's <c>Dot</c> is exact given the Q8_K activation, so the only
    /// divergence is the activation's 8-bit quantization — bounded here by an
    /// L2-relative tolerance.
    ///
    /// No model file needed — synthetic Q4_K blocks (sane FP16 d/dmin, random
    /// scale/min/quant bytes).
    /// </summary>
    [Trait("Category", "Quantization")]
    public sealed class Q4KDotKernelTests
    {
        private const int SuperBlockBytes = Q4KWeight.SuperBlockBytes;
        private const int SuperBlockElements = Q4KWeight.SuperBlockElements;

        [Fact]
        public void Project_MatchesF32Reference_WithinQ8ActivationNoise()
        {
            const int inputSize = 512;    // 2 super-blocks per row
            const int outputSize = 96;
            var rng = new Random(20260519);

            var blocks = BuildRandomBlocks(rng, outputSize, inputSize);
            var input = RandomVector(rng, inputSize);

            var weight = new Q4KWeight(blocks, inputSize, outputSize);
            var output = new float[outputSize];
            var superBlocksPerRow = inputSize / SuperBlockElements;
            var actQuants = new sbyte[inputSize];
            var actScales = new float[superBlocksPerRow];
            var actBsums = new short[superBlocksPerRow * Q4KDotKernel.GroupsPerSuperBlock];

            Q4KDotKernel.Project(input, weight, [], output, actQuants, actScales, actBsums);

            var reference = ReferenceProject(blocks, inputSize, outputSize, input);

            // L2-relative error: the weight side is exact, only the activation
            // is Q8-quantized — comparable to the Q8_0 projection's noise floor.
            double errSq = 0, refSq = 0;
            for (var o = 0; o < outputSize; o++)
            {
                var diff = (double)output[o] - reference[o];
                errSq += diff * diff;
                refSq += (double)reference[o] * reference[o];
            }
            var l2Relative = Math.Sqrt(errSq / refSq);

            Assert.True(l2Relative < 0.03,
                $"L2-relative error {l2Relative:F5} exceeds 3% — kernel diverges from the F32 decode.");
        }

        [Fact]
        public void ProjectParallel_IsBitIdenticalToProject()
        {
            const int inputSize = 512;
            const int outputSize = 96;
            var rng = new Random(424242);

            var blocks = BuildRandomBlocks(rng, outputSize, inputSize);
            var input = RandomVector(rng, inputSize);
            var bias = RandomVector(rng, outputSize);
            var weight = new Q4KWeight(blocks, inputSize, outputSize);
            var superBlocksPerRow = inputSize / SuperBlockElements;

            var sequential = new float[outputSize];
            var parallel = new float[outputSize];
            var actQuants = new sbyte[inputSize];
            var actScales = new float[superBlocksPerRow];
            var actBsums = new short[superBlocksPerRow * Q4KDotKernel.GroupsPerSuperBlock];

            Q4KDotKernel.Project(input, weight, bias, sequential, actQuants, actScales, actBsums);
            Q4KDotKernel.ProjectParallel(input, weight, bias, parallel, actQuants, actScales, actBsums);

            // Each output is an independent Dot — parallel must match exactly.
            for (var o = 0; o < outputSize; o++)
            {
                Assert.Equal(sequential[o], parallel[o]);
            }
        }

        [Theory]
        [InlineData(1)]
        [InlineData(3)]
        [InlineData(16)]
        public void ProjectBatched_IsBitIdenticalToPerRowProject(int rows)
        {
            const int inputSize = 512;
            const int outputSize = 96;
            var rng = new Random(rows * 17 + 5);

            var blocks = BuildRandomBlocks(rng, outputSize, inputSize);
            var weight = new Q4KWeight(blocks, inputSize, outputSize);
            var superBlocksPerRow = inputSize / SuperBlockElements;
            var bsumsPerRow = superBlocksPerRow * Q4KDotKernel.GroupsPerSuperBlock;
            var bias = RandomVector(rng, outputSize);

            var input = new float[rows * inputSize];
            for (var i = 0; i < input.Length; i++)
            {
                input[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
            }

            var reference = new float[rows * outputSize];
            for (var n = 0; n < rows; n++)
            {
                Q4KDotKernel.Project(
                    input.AsSpan(n * inputSize, inputSize), weight, bias,
                    reference.AsSpan(n * outputSize, outputSize),
                    new sbyte[inputSize], new float[superBlocksPerRow], new short[bsumsPerRow]);
            }

            var batched = new float[rows * outputSize];
            Q4KDotKernel.ProjectBatched(
                input, rows, weight, bias, batched,
                new sbyte[rows * inputSize], new float[rows * superBlocksPerRow], new short[rows * bsumsPerRow]);

            for (var k = 0; k < reference.Length; k++)
            {
                Assert.Equal(reference[k], batched[k]);
            }
        }

        [Fact]
        public void Project_AppliesBias()
        {
            const int inputSize = 256;
            const int outputSize = 32;
            var rng = new Random(99);

            var blocks = BuildRandomBlocks(rng, outputSize, inputSize);
            var input = RandomVector(rng, inputSize);
            var weight = new Q4KWeight(blocks, inputSize, outputSize);

            var actQuants = new sbyte[inputSize];
            var actScales = new float[1];
            var actBsums = new short[Q4KDotKernel.GroupsPerSuperBlock];

            var noBias = new float[outputSize];
            Q4KDotKernel.Project(input, weight, [], noBias, actQuants, actScales, actBsums);

            var bias = RandomVector(rng, outputSize);
            var withBias = new float[outputSize];
            Q4KDotKernel.Project(input, weight, bias, withBias, actQuants, actScales, actBsums);

            for (var o = 0; o < outputSize; o++)
            {
                Assert.Equal(noBias[o] + bias[o], withBias[o], 3);
            }
        }

        [Fact]
        public void QuantizeActivationQ8K_ReconstructsWithinQ8Noise()
        {
            const int n = 256;
            var rng = new Random(7);
            var source = RandomVector(rng, n);

            var quants = new sbyte[n];
            var scales = new float[1];
            var bsums = new short[Q4KDotKernel.GroupsPerSuperBlock];
            Q4KDotKernel.QuantizeActivationQ8K(source, quants, scales, bsums);

            // Reconstruct q8d·q8 and bound the error by the quantization step.
            for (var i = 0; i < n; i++)
            {
                var reconstructed = scales[0] * quants[i];
                Assert.True(MathF.Abs(reconstructed - source[i]) <= scales[0],
                    $"element {i}: |{reconstructed} - {source[i]}| > step {scales[0]}");
            }

            // bsums must equal the actual group sums of the quants.
            for (var g = 0; g < Q4KDotKernel.GroupsPerSuperBlock; g++)
            {
                var sum = 0;
                for (var k = 0; k < Q4KDotKernel.GroupSize; k++)
                {
                    sum += quants[g * Q4KDotKernel.GroupSize + k];
                }
                Assert.Equal(sum, bsums[g]);
            }
        }

        [Fact]
        public void QuantizeActivationQ8K_AllZeros_ProducesZeroBlock()
        {
            var quants = new sbyte[SuperBlockElements];
            var scales = new float[1];
            var bsums = new short[Q4KDotKernel.GroupsPerSuperBlock];

            Q4KDotKernel.QuantizeActivationQ8K(new float[SuperBlockElements], quants, scales, bsums);

            Assert.Equal(0f, scales[0]);
            foreach (var q in quants)
            {
                Assert.Equal(0, q);
            }
            foreach (var s in bsums)
            {
                Assert.Equal(0, s);
            }
        }

        // ── Helpers ──────────────────────────────────────────────────────────

        /// <summary>
        /// A synthetic Q4_K super-block: sane small positive FP16 d/dmin (random
        /// bits there could decode to NaN/Inf) over fully random scale/min/quant
        /// bytes — any such bytes are a valid Q4_K block.
        /// </summary>
        private static void WriteRandomBlock(Random rng, Span<byte> block144)
        {
            for (var i = 0; i < block144.Length; i++)
            {
                block144[i] = (byte)rng.Next(256);
            }

            var d = (Half)(rng.NextDouble() * 0.05 + 0.001);
            var dmin = (Half)(rng.NextDouble() * 0.03 + 0.001);
            BinaryPrimitives.WriteUInt16LittleEndian(block144[..2], BitConverter.HalfToUInt16Bits(d));
            BinaryPrimitives.WriteUInt16LittleEndian(block144.Slice(2, 2), BitConverter.HalfToUInt16Bits(dmin));
        }

        private static byte[] BuildRandomBlocks(Random rng, int outputSize, int inputSize)
        {
            var superBlocksPerRow = inputSize / SuperBlockElements;
            var blocks = new byte[outputSize * superBlocksPerRow * SuperBlockBytes];
            for (var b = 0; b < outputSize * superBlocksPerRow; b++)
            {
                WriteRandomBlock(rng, blocks.AsSpan(b * SuperBlockBytes, SuperBlockBytes));
            }
            return blocks;
        }

        private static float[] RandomVector(Random rng, int n)
        {
            var v = new float[n];
            for (var i = 0; i < n; i++)
            {
                v[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
            }
            return v;
        }

        /// <summary>F32 reference: decode each row's Q4_K blocks, plain dot with the input.</summary>
        private static float[] ReferenceProject(byte[] blocks, int inputSize, int outputSize, float[] input)
        {
            var superBlocksPerRow = inputSize / SuperBlockElements;
            var output = new float[outputSize];
            var row = new float[inputSize];

            for (var o = 0; o < outputSize; o++)
            {
                for (var sb = 0; sb < superBlocksPerRow; sb++)
                {
                    var blockIndex = o * superBlocksPerRow + sb;
                    GgmlDequant.DecodeQ4_KBlock(
                        blocks.AsSpan(blockIndex * SuperBlockBytes, SuperBlockBytes),
                        row.AsSpan(sb * SuperBlockElements, SuperBlockElements));
                }

                var sum = 0f;
                for (var i = 0; i < inputSize; i++)
                {
                    sum += row[i] * input[i];
                }
                output[o] = sum;
            }

            return output;
        }
    }
}
